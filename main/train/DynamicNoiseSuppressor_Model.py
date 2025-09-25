import warnings
warnings.filterwarnings('ignore')
import transformers
transformers.logging.set_verbosity_error()
import sys
sys.path.append('../main/')
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import torch
import torch.nn as nn
import os
from tqdm import tqdm
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer

from util import print_colored_text, read_jsonl_file, write_jsonl_file

class LlamaGuardWithNoiseProtection(nn.Module):
    def __init__(self, llama_model):
        super(LlamaGuardWithNoiseProtection, self).__init__()
        # Llama模型
        self.llama_model = llama_model
        # 噪声防护模块：自编码器
        self.denoising_autoencoder = DenoisingAdapter(input_dim=llama_model.config.hidden_size).to(device="cuda", dtype=self.llama_model.dtype)
        
    def forward(self, input_ids):
        # 获取Llama嵌入层输出
        embeddings = self.llama_model.model.embed_tokens(input_ids)
        # 通过噪声防护模块
        cleaned_embeddings = self.denoising_autoencoder(embeddings)
        # 后续transformer层
        outputs = self.llama_model(inputs_embeds=cleaned_embeddings)
        return outputs

# 自编码器：去噪模块
class DenoisingAdapter(nn.Module):
    """
    输出一个“增量”向量 Δ(x)，与原嵌入相加。
    默认权重初始化很小 → 刚开始几乎不影响模型。
    """
    def __init__(self, input_dim, hidden_dim=512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.delta   = nn.Linear(hidden_dim, input_dim)
        
        # 关键：把最后一层权重初始化为很小的值，确保 Δ≈0
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)

    def forward(self, x):          # x: [B, L, D]
        h = self.encoder(x)        # [B, L, H]
        delta = self.delta(h)      # [B, L, D]
        return x + delta           # 残差式输出


def save_noise_protection(model: nn.Module, save_path: str):
    state_dict = model.denoising_autoencoder.state_dict()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(state_dict, save_path)
    print(f"Saved noise protection module to: {save_path}")

def load_noise_protection(model: nn.Module, load_path: str, strict: bool = True):
    checkpoint = torch.load(load_path, map_location="cpu")
    model.denoising_autoencoder.load_state_dict(checkpoint, strict=strict)
    print(f"Loaded noise protection module from: {load_path}")

# 加载Llama模型
model_dir = '../../LLM/Llama-Guard-3-8B/models--meta-llama--Llama-Guard-3-8B/snapshots/7327bd9f6efbbe6101dc6cc4736302b3cbb6e425'
llama_model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="cuda:0", trust_remote_code=True, torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, use_fast=False)

# 创建LlamaGuard带噪声防护模块的模型
model_with_noise_protection = LlamaGuardWithNoiseProtection(llama_model)

# 训练过程
def train(train_loader, epochs=5, learning_rate=1e-4):
    for param in model_with_noise_protection.llama_model.parameters():
        param.requires_grad = False
    optimizer = optim.Adam(model_with_noise_protection.denoising_autoencoder.parameters(), lr=learning_rate)
    model_with_noise_protection.train()

    for epoch in tqdm(range(epochs)):
        total_loss = 0.0
        num_batches = 0

        for data in train_loader:

            chat_ori = [{"role": "user", "content": data['question']}, {"role": "assistant", "content": data['response']}]
            chat_atk = [{"role": "user", "content": data['question']}, {"role": "assistant", "content": data['attack_response']}]
            input_ids_ori = tokenizer.apply_chat_template(chat_ori, return_tensors="pt").cuda()
            input_ids_atk = tokenizer.apply_chat_template(chat_atk, return_tensors="pt").cuda()
            new_column_ori = torch.full((input_ids_ori.shape[0], 1), 198).cuda()
            input_ids_ori = torch.cat((input_ids_ori, new_column_ori), dim=1)
            new_column_atk = torch.full((input_ids_atk.shape[0], 1), 198).cuda()
            input_ids_atk = torch.cat((input_ids_atk, new_column_atk), dim=1)

            outputs_ori = llama_model(input_ids_ori)
            outputs_atk = model_with_noise_protection(input_ids_atk)
            logits_ori = outputs_ori.logits
            logits_atk = outputs_atk.logits
            logits_ori_last = logits_ori[:, -1, :]
            logits_atk_last = logits_atk[:, -1, :]
            log_probs_ori = torch.log_softmax(logits_ori_last, dim=-1)
            log_probs_atk = torch.log_softmax(logits_atk_last, dim=-1)
            kl_loss = torch.nn.functional.kl_div(log_probs_atk, log_probs_ori, reduction='batchmean', log_target=True)

            optimizer.zero_grad()
            loss = kl_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        tqdm.write(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")
    
    save_noise_protection(model_with_noise_protection, '../main/model/noise_protection.pt')


if __name__ == '__main__':
    datasets = read_jsonl_file('../datasets/attacks/common.jsonl')[:300]
    datasets.extend(read_jsonl_file('../datasets/attacks/suffix.jsonl')[:100])
    datasets.extend(read_jsonl_file('../datasets/attacks/rewrite.jsonl')[:100])
    datasets.extend(read_jsonl_file('../datasets/attacks/word_replace.jsonl')[:100])
    train(datasets, 10, 1e-4)