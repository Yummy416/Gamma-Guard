import warnings
warnings.filterwarnings('ignore')
import transformers
transformers.logging.set_verbosity_error()
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
from tqdm import tqdm
import random
import nltk
import copy
import json
import string
from nltk.tokenize import word_tokenize
nltk.data.path.insert(0, './nltk_data')

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

def llama3_chat(model, tokenizer, prompt):
    chat = f"USER: {prompt} ASSISTANT: Here are 10 alternative synonyms:\n"
    input_ids = tokenizer(chat, return_tensors="pt").input_ids.cuda()
    output = model.generate(input_ids=input_ids, max_new_tokens=512, do_sample=True, temperature=0.6, top_p=0.9)
    prompt_len = input_ids.shape[-1]
    response = tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)
    return response

def write_jsonl_file(filename, data):
    with open(filename, 'w', encoding='utf-8') as f:
        for entry in data:
            json_line = json.dumps(entry, ensure_ascii=False)
            f.write(json_line + '\n')

if __name__ == "__main__":
    # model_guard, tokenizer_guard = create_model('../LLM/Llama-Guard-3-8B/models--meta-llama--Llama-Guard-3-8B/snapshots/7327bd9f6efbbe6101dc6cc4736302b3cbb6e425')
    model_guard, tokenizer_guard = create_model('../LLM/Meta-Llama-Guard-2-8B/models--meta-llama--Meta-Llama-Guard-2-8B/snapshots/bb78080332eda00343dc37b0465b43bbf22c0251')
    model_chat, tokenizer_chat = create_model('../LLM/vicuna-13b-v1.5/models--lmsys--vicuna-13b-v1.5/snapshots/c8327bf999adbd2efe2e75f6509fa01436100dc2')
    datasets = pd.read_csv('./datasets/beavertails/harmful_datasets_all_2.csv').values.tolist()

    results = []
    word_dict = dict()
    for question, response in tqdm(datasets):
        label = llama_guard(model_guard, tokenizer_guard, question, response)
        if label != 1:
            continue
        tokens = word_tokenize(response)
        flag = 0
        for k in range(1, 11):
            if flag == 1:
                break
            for _ in range(10):
                if flag == 1:
                    break
                tokens_temp = copy.deepcopy(tokens)
                meaningful_words = [(word, idx) for idx, word in enumerate(tokens) if word not in string.punctuation]
                selected_words = random.sample(meaningful_words, min(k, len(meaningful_words)))
                for word, idx in selected_words:
                    if word not in word_dict:
                        prompt = f"Please help me list 10 alternative synonyms for '{word}', separated by commas between each word. Remember not to say anything extra except for the 10 synonyms. \n\nPlease strictly follow the format of the example output, using ',' to separate each word without numbering. \n\nExample:\nHere are 10 alternative synonyms:\n A-word, B-word, C-word, D-word, E-word, F-word, G-word, H-word, I-word, J-word"
                        for _ in range(10):
                            like_words = [x.strip() for x in llama3_chat(model_chat, tokenizer_chat, prompt).strip().split(',')]
                            if len(like_words) == 10:
                                break
                        word_dict[word] = like_words
                    random_token = random.choice(word_dict[word])
                    tokens_temp[idx] = random_token
                attack_response = ' '.join(tokens_temp)
                result = llama_guard(model_guard, tokenizer_guard, question, attack_response)
                if result == -1:
                    continue
                if label != result:
                    results.append({"id": len(results)+1, "question": question, "response": response, "attack_response": attack_response, "label": label, "result": result})
                    write_jsonl_file('./datasets/attacks/word_replace_2.jsonl', results)
                    flag = 1
