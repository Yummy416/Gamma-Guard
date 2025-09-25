import warnings
warnings.filterwarnings('ignore')
import transformers
transformers.logging.set_verbosity_error()
import sys
sys.path.append('./')
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from transformers import AutoTokenizer
import torch
from tqdm import tqdm
import torch.nn as nn
import pandas as pd
import random

from DynamicNoiseSuppressor import llama_guard
from util import print_colored_text, read_jsonl_file
from custom_models import custom_llama

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


def load_noise_protection(model: nn.Module, load_path: str, strict: bool = True):
    checkpoint = torch.load(load_path, map_location="cpu")
    model.load_state_dict(checkpoint, strict=strict)
    print(f"Loaded noise protection module from: {load_path}")
    return model

def sample_jsonl(file_path, sample_size):
    data = read_jsonl_file(file_path)
    return random.sample(data, min(sample_size, len(data)))

if __name__ == '__main__':
    model_dir = '../../LLM/Llama-Guard-3-8B/models--meta-llama--Llama-Guard-3-8B/snapshots/7327bd9f6efbbe6101dc6cc4736302b3cbb6e425'
    # model_dir = '../../LLM/Meta-Llama-Guard-2-8B/models--meta-llama--Meta-Llama-Guard-2-8B/snapshots/bb78080332eda00343dc37b0465b43bbf22c0251'
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, use_fast=False)
    model = custom_llama.from_pretrained(model_dir, device_map="auto", trust_remote_code=True, torch_dtype="auto")

    denoising_autoencoder = DenoisingAdapter(input_dim=model.config.hidden_size)
    denoising_autoencoder = load_noise_protection(denoising_autoencoder, 'model/noise_protection.pt')
    denoising_autoencoder.to(device="cuda", dtype=model.dtype)

    # 攻击数据
    datasets = read_jsonl_file('../datasets/attacks/suffix.jsonl')[0:50]
    # datasets = sample_jsonl('../datasets/attacks/suffix_6.jsonl', 50)
    attack_sum = len(datasets)
    attack_num_correct = 0
    for data in datasets:
        question = data['question']
        response = data['response']
        attack_response = data['attack_response']
        original_result = llama_guard(model, tokenizer, question, response, is_correct=False)
        correct_result = llama_guard(model, tokenizer, question, attack_response, is_correct=True, denoising_autoencoder=denoising_autoencoder)
        if original_result == correct_result:
            attack_num_correct += 1
    print(f"\n攻击数据上的修正成功率：{(attack_num_correct / attack_sum)*100:.4f}%")

    datasets = read_jsonl_file('../datasets/attacks/suffix.jsonl')[50:100]
    # datasets = sample_jsonl('../datasets/attacks/suffix_6.jsonl', 50)
    attack_sum = len(datasets)
    attack_num_correct = 0
    for data in datasets:
        question = data['question']
        response = data['response']
        attack_response = data['attack_response']
        original_result = llama_guard(model, tokenizer, question, response, is_correct=False)
        correct_result = llama_guard(model, tokenizer, question, attack_response, is_correct=True, denoising_autoencoder=denoising_autoencoder)
        if original_result == correct_result:
            attack_num_correct += 1
    print(f"\n攻击数据上的修正成功率：{(attack_num_correct / attack_sum)*100:.4f}%")

    datasets = read_jsonl_file('../datasets/attacks/word_replace.jsonl')[0:50]
    # datasets = sample_jsonl('../datasets/attacks/word_replace_6.jsonl', 20)
    attack_sum = len(datasets)
    attack_num_correct = 0
    for data in datasets:
        question = data['question']
        response = data['response']
        attack_response = data['attack_response']
        original_result = llama_guard(model, tokenizer, question, response, is_correct=False)
        correct_result = llama_guard(model, tokenizer, question, attack_response, is_correct=True, denoising_autoencoder=denoising_autoencoder)
        if original_result == correct_result:
            attack_num_correct += 1
    print(f"\n攻击数据上的修正成功率：{(attack_num_correct / attack_sum)*100:.4f}%")

    datasets = read_jsonl_file('../datasets/attacks/word_replace.jsonl')[50:100]
    # datasets = sample_jsonl('../datasets/attacks/word_replace_6.jsonl', 20)
    attack_sum = len(datasets)
    attack_num_correct = 0
    for data in datasets:
        question = data['question']
        response = data['response']
        attack_response = data['attack_response']
        original_result = llama_guard(model, tokenizer, question, response, is_correct=False)
        correct_result = llama_guard(model, tokenizer, question, attack_response, is_correct=True, denoising_autoencoder=denoising_autoencoder)
        if original_result == correct_result:
            attack_num_correct += 1
    print(f"\n攻击数据上的修正成功率：{(attack_num_correct / attack_sum)*100:.4f}%")

    datasets = read_jsonl_file('../datasets/attacks/word_replace.jsonl')[100:150]
    # datasets = sample_jsonl('../datasets/attacks/word_replace_6.jsonl', 20)
    attack_sum = len(datasets)
    attack_num_correct = 0
    for data in datasets:
        question = data['question']
        response = data['response']
        attack_response = data['attack_response']
        original_result = llama_guard(model, tokenizer, question, response, is_correct=False)
        correct_result = llama_guard(model, tokenizer, question, attack_response, is_correct=True, denoising_autoencoder=denoising_autoencoder)
        if original_result == correct_result:
            attack_num_correct += 1
    print(f"\n攻击数据上的修正成功率：{(attack_num_correct / attack_sum)*100:.4f}%")

    datasets = read_jsonl_file('../datasets/attacks/word_replace.jsonl')[150:200]
    # datasets = sample_jsonl('../datasets/attacks/word_replace_6.jsonl', 20)
    attack_sum = len(datasets)
    attack_num_correct = 0
    for data in datasets:
        question = data['question']
        response = data['response']
        attack_response = data['attack_response']
        original_result = llama_guard(model, tokenizer, question, response, is_correct=False)
        correct_result = llama_guard(model, tokenizer, question, attack_response, is_correct=True, denoising_autoencoder=denoising_autoencoder)
        if original_result == correct_result:
            attack_num_correct += 1
    print(f"\n攻击数据上的修正成功率：{(attack_num_correct / attack_sum)*100:.4f}%")

    datasets = read_jsonl_file('../datasets/attacks/word_replace.jsonl')[200:250]
    # datasets = sample_jsonl('../datasets/attacks/word_replace_6.jsonl', 20)
    attack_sum = len(datasets)
    attack_num_correct = 0
    for data in datasets:
        question = data['question']
        response = data['response']
        attack_response = data['attack_response']
        original_result = llama_guard(model, tokenizer, question, response, is_correct=False)
        correct_result = llama_guard(model, tokenizer, question, attack_response, is_correct=True, denoising_autoencoder=denoising_autoencoder)
        if original_result == correct_result:
            attack_num_correct += 1
    print(f"\n攻击数据上的修正成功率：{(attack_num_correct / attack_sum)*100:.4f}%")

    datasets = read_jsonl_file('../datasets/attacks/word_replace.jsonl')[250:300]
    # datasets = sample_jsonl('../datasets/attacks/word_replace_6.jsonl', 20)
    attack_sum = len(datasets)
    attack_num_correct = 0
    for data in datasets:
        question = data['question']
        response = data['response']
        attack_response = data['attack_response']
        original_result = llama_guard(model, tokenizer, question, response, is_correct=False)
        correct_result = llama_guard(model, tokenizer, question, attack_response, is_correct=True, denoising_autoencoder=denoising_autoencoder)
        if original_result == correct_result:
            attack_num_correct += 1
    print(f"\n攻击数据上的修正成功率：{(attack_num_correct / attack_sum)*100:.4f}%")

    datasets = read_jsonl_file('../datasets/attacks/word_replace.jsonl')[300:350]
    # datasets = sample_jsonl('../datasets/attacks/word_replace_6.jsonl', 20)
    attack_sum = len(datasets)
    attack_num_correct = 0
    for data in datasets:
        question = data['question']
        response = data['response']
        attack_response = data['attack_response']
        original_result = llama_guard(model, tokenizer, question, response, is_correct=False)
        correct_result = llama_guard(model, tokenizer, question, attack_response, is_correct=True, denoising_autoencoder=denoising_autoencoder)
        if original_result == correct_result:
            attack_num_correct += 1
    print(f"\n攻击数据上的修正成功率：{(attack_num_correct / attack_sum)*100:.4f}%")

    datasets = read_jsonl_file('../datasets/attacks/rewrite.jsonl')[0:50]
    # datasets = sample_jsonl('../datasets/attacks/rewrite_6.jsonl', 15)
    attack_sum = len(datasets)
    attack_num_correct = 0
    for data in datasets:
        question = data['question']
        response = data['response']
        attack_response = data['attack_response']
        original_result = llama_guard(model, tokenizer, question, response, is_correct=False)
        correct_result = llama_guard(model, tokenizer, question, attack_response, is_correct=True, denoising_autoencoder=denoising_autoencoder)
        if original_result == correct_result:
            attack_num_correct += 1
    print(f"\n攻击数据上的修正成功率：{(attack_num_correct / attack_sum)*100:.4f}%")

    datasets = read_jsonl_file('../datasets/attacks/rewrite.jsonl')[50:100]
    # datasets = sample_jsonl('../datasets/attacks/rewrite_6.jsonl', 15)
    attack_sum = len(datasets)
    attack_num_correct = 0
    for data in datasets:
        question = data['question']
        response = data['response']
        attack_response = data['attack_response']
        original_result = llama_guard(model, tokenizer, question, response, is_correct=False)
        correct_result = llama_guard(model, tokenizer, question, attack_response, is_correct=True, denoising_autoencoder=denoising_autoencoder)
        if original_result == correct_result:
            attack_num_correct += 1
    print(f"\n攻击数据上的修正成功率：{(attack_num_correct / attack_sum)*100:.4f}%")

    datasets = read_jsonl_file('../datasets/attacks/rewrite.jsonl')[50:100]
    # datasets = sample_jsonl('../datasets/attacks/rewrite_6.jsonl', 15)
    attack_sum = len(datasets)
    attack_num_correct = 0
    for data in datasets:
        question = data['question']
        response = data['response']
        attack_response = data['attack_response']
        original_result = llama_guard(model, tokenizer, question, response, is_correct=False)
        correct_result = llama_guard(model, tokenizer, question, attack_response, is_correct=True, denoising_autoencoder=denoising_autoencoder)
        if original_result == correct_result:
            attack_num_correct += 1
    print(f"\n攻击数据上的修正成功率：{(attack_num_correct / attack_sum)*100:.4f}%")

    # 正常数据
    datasets = pd.read_csv('../datasets/beavertails/prompt_response_classified.csv')
    questions = datasets['prompt'].tolist()[300:350]
    responses = datasets['response'].tolist()[300:350]
    original_sum = len(questions)
    original_num_correct = 0
    for question, response in zip(questions, responses):
        original_result = llama_guard(model, tokenizer, question, response, is_correct=False)
        correct_result = llama_guard(model, tokenizer, question, response, is_correct=True, denoising_autoencoder=denoising_autoencoder)
        if original_result == correct_result:
            original_num_correct += 1

    print(f"正常数据上的修正成功率：{(original_num_correct / original_sum)*100:.4f}%")