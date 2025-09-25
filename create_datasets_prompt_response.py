import warnings
warnings.filterwarnings('ignore')
import transformers
transformers.logging.set_verbosity_error()
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

from datasets import load_dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
from tqdm import tqdm
import random

def create_model(model_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", trust_remote_code=True, torch_dtype="auto")
    return model, tokenizer

def vicuna_chat(model, tokenizer, prompt, do_sample=True, max_new_tokens=512, is_instruction=True):
    if is_instruction:
        instruction = """USER: {} ASSISTANT: """.format(prompt)
    else:
        instruction = prompt
    generate_ids = model.generate(tokenizer(instruction, return_tensors='pt').input_ids.cuda(), do_sample=do_sample, temperature=0.9, top_p=0.9, max_new_tokens=max_new_tokens)
    response = tokenizer.decode(generate_ids[0], skip_special_tokens=True).split('ASSISTANT:')[1].strip()
    torch.cuda.empty_cache()
    return response
 
# 创建未经分类的advbench问答对数据集
model, tokenizer = create_model("../LLM/vicuna-13b-v1.5/models--lmsys--vicuna-13b-v1.5/snapshots/c8327bf999adbd2efe2e75f6509fa01436100dc2")
ds = pd.read_parquet("./cache/AdvBench/datasets--walledai--AdvBench/snapshots/9d4730540082fa4017450b65ca1c0e1d8d30446e/data/train-00000-of-00001.parquet")
prompts = ds['prompt'].tolist()
targets = ds['target'].tolist()
results = []
for prompt, target in zip(tqdm(prompts), targets):
    response = vicuna_chat(model, tokenizer, """USER: {} ASSISTANT: {}""".format(prompt, target), do_sample=False, is_instruction=False)
    results.append([prompt, response])
df = pd.DataFrame(results, columns=['prompt', 'response'])
df.to_csv('./datasets/advbench/prompt_response_unclassified.csv', index=False)


# 创建未经分类的PKU问答对数据集
ds2 = pd.read_json("./cache/BeaverTails/datasets--PKU-Alignment--BeaverTails/snapshots/8401fe609d288129cc684a9b3be6a93e41cfe678/round0/330k/train.jsonl.xz", lines=True)
datas = [[x, y] for x, y in zip(ds2["prompt"], ds2["response"])]
datas = list(set([tuple(i) for i in datas]))
prompts = [item[0] for item in datas]
responses = [item[1] for item in datas]
random_ids = random.sample(range(0, len(prompts)), 1500)
results = []
for i in tqdm(random_ids):
    results.append([prompts[i], responses[i]])
df = pd.DataFrame(results, columns=['prompt', 'response'])
df.to_csv('./datasets/beavertails/prompt_response_unclassified.csv', index=False)


# 创建未经分类的XSTEST问答对数据集
ds3 = pd.read_parquet("./cache/XSTest/datasets--walledai--XSTest/snapshots/f1d713187c61b6ae64e602d74f0b3d812cc2e8e8/data/train-00000-of-00001.parquet")
prompts = ds3['prompt'].tolist()
labels = ds3['label'].tolist()
model, tokenizer = create_model("../LLM/vicuna-13b-v1.5/models--lmsys--vicuna-13b-v1.5/snapshots/c8327bf999adbd2efe2e75f6509fa01436100dc2")
results = []
for prompt, label in zip(tqdm(prompts), labels):
    response = vicuna_chat(model, tokenizer, """USER: {} ASSISTANT: {}""".format(prompt, "Sure, here is"), do_sample=False, is_instruction=False)
    results.append([prompt, response, label])
df = pd.DataFrame(results, columns=['prompt', 'response', 'label'])
df.to_csv('./datasets/xstest/prompt_response_unclassified.csv', index=False)

# 创建未经分类的toxigen-data问答对数据集
ds4 = pd.read_csv('./cache/toxigen-data/datasets--toxigen--toxigen-data/snapshots/ea082a8973a287097b17973343576e0c8da8eb3b/toxigen.csv')
prompts = ds4['prompt'].tolist()
responses = ds4['generation'].tolist()
random_ids = random.sample(range(0, len(prompts)), 1500)
results = []
for i in tqdm(random_ids):
    results.append([prompts[i], responses[i]])
df = pd.DataFrame(results, columns=['prompt', 'response'])
df.to_csv('./datasets/toxigen/prompt_response_unclassified.csv', index=False)