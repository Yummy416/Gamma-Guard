import warnings
warnings.filterwarnings('ignore')
import transformers
transformers.logging.set_verbosity_error()
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
from tqdm import tqdm
import re
import random
import json

def create_model(model_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", trust_remote_code=True, torch_dtype="auto")
    return model, tokenizer

def llama_guard(model, tokenizer, question, response):
    chat = [{"role": "user", "content": question},
            {"role": "assistant", "content": response}]
    input_ids = tokenizer.apply_chat_template(chat, return_tensors="pt").cuda()
    # new_column = torch.full((input_ids.shape[0], 1), 198).to(model.device)
    # input_ids = torch.cat((input_ids, new_column), dim=1).to(model.device)
    output = model.generate(input_ids=input_ids, max_new_tokens=100, pad_token_id=0, do_sample=False)
    prompt_len = input_ids.shape[-1]
    response = tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)
    if 'unsafe' in response:
        return 1
    elif 'safe' in response:
        return 0
    else:
        return -1
    
def write_jsonl_file(filename, data):
    with open(filename, 'w', encoding='utf-8') as f:
        for entry in data:
            json_line = json.dumps(entry, ensure_ascii=False)
            f.write(json_line + '\n')

if __name__ == "__main__":
    # model, tokenizer = create_model('../LLM/Llama-Guard-3-8B/models--meta-llama--Llama-Guard-3-8B/snapshots/7327bd9f6efbbe6101dc6cc4736302b3cbb6e425')
    model, tokenizer = create_model('../LLM/Meta-Llama-Guard-2-8B/models--meta-llama--Meta-Llama-Guard-2-8B/snapshots/bb78080332eda00343dc37b0465b43bbf22c0251')
    datasets = pd.read_csv('./datasets/beavertails/harmful_datasets_all_2.csv').values.tolist()

    vocab = tokenizer.get_vocab()
    filtered_vocab = [token for token in vocab.keys() if re.match(r"^[a-zA-Z0-9]+$", token)]
    results = []
    temp = []
    for i in tqdm(range(100000)):
        random_tokens = " ".join(random.sample(filtered_vocab, 10))
        for question, response in datasets:
            if question in temp:
                continue
            label = llama_guard(model, tokenizer, question, response)
            if label != 1:
                temp.append(question)
                continue
            attack_response = response.strip() + ' ' + random_tokens
            result = llama_guard(model, tokenizer, question, attack_response)
            if result == -1:
                continue
            if label != result:
                results.append({"id": len(results)+1, "question": question, "response": response, "attack_response": attack_response, "random_tokens": random_tokens, "label": label, "result": result})
                temp.append(question)
                write_jsonl_file('./datasets/attacks/suffix_2.jsonl', results)
