import warnings
warnings.filterwarnings('ignore')
import transformers
transformers.logging.set_verbosity_error()
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

from datasets import load_dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
from tqdm import tqdm
import random

def create_model(model_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", trust_remote_code=True, torch_dtype="auto")
    return model, tokenizer

def llama_guard(model, tokenizer, question, response):
    chat = [{"role": "user", "content": question},
            {"role": "assistant", "content": response}]
    input_ids = tokenizer.apply_chat_template(chat, return_tensors="pt").cuda()
    output = model.generate(input_ids=input_ids, max_new_tokens=100, pad_token_id=0, do_sample=False)
    prompt_len = input_ids.shape[-1]
    response = tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)
    torch.cuda.empty_cache()
    if 'unsafe' in response:
        return 1
    else:
        return 0

if __name__ == "__main__":
    # 对beavertails数据集进行分类
    # model, tokenizer = create_model('../LLM/Llama-Guard-3-8B/models--meta-llama--Llama-Guard-3-8B/snapshots/7327bd9f6efbbe6101dc6cc4736302b3cbb6e425')
    # datasets = pd.read_csv('./datasets/beavertails/prompt_response_unclassified.csv')
    # questions = datasets['prompt'].tolist()
    # responses = datasets['response'].tolist()
    # results = []
    # num_0, num_1 = 0, 0
    # for question, response in zip(tqdm(questions), responses):
    #     label = llama_guard(model, tokenizer, question, response)
    #     if label == 0:
    #         num_0 += 1
    #     else:
    #         num_1 += 1
    #     results.append([question, response, label])
    # df = pd.DataFrame(results, columns=['prompt', 'response', 'label'])
    # df.to_csv('./datasets/beavertails/prompt_response_classified.csv', index=False)
    # print("num_0: ", num_0)
    # print("num_1: ", num_1)

    # 对advbench数据集进行分类
    # model, tokenizer = create_model('../LLM/Llama-Guard-3-8B/models--meta-llama--Llama-Guard-3-8B/snapshots/7327bd9f6efbbe6101dc6cc4736302b3cbb6e425')
    # datasets = pd.read_csv('./datasets/advbench/prompt_response_unclassified.csv')
    # questions = datasets['prompt'].tolist()
    # responses = datasets['response'].tolist()
    # results = []
    # num_0, num_1 = 0, 0
    # for question, response in zip(tqdm(questions), responses):
    #     label = llama_guard(model, tokenizer, question, response)
    #     if label == 0:
    #         num_0 += 1
    #     else:
    #         num_1 += 1
    #     results.append([question, response, label])
    # df = pd.DataFrame(results, columns=['prompt', 'response', 'label'])
    # df.to_csv('./datasets/advbench/prompt_response_classified.csv', index=False)
    # print("num_0: ", num_0)
    # print("num_1: ", num_1)

    # 对toxigen数据集进行分类
    # model, tokenizer = create_model('../LLM/Llama-Guard-3-8B/models--meta-llama--Llama-Guard-3-8B/snapshots/7327bd9f6efbbe6101dc6cc4736302b3cbb6e425')
    # datasets = pd.read_csv('./datasets/toxigen/prompt_response_unclassified.csv')
    # questions = datasets['prompt'].tolist()
    # responses = datasets['response'].tolist()
    # results = []
    # num_0, num_1 = 0, 0
    # for question, response in zip(tqdm(questions), responses):
    #     label = llama_guard(model, tokenizer, question, response)
    #     if label == 0:
    #         num_0 += 1
    #     else:
    #         num_1 += 1
    #     results.append([question, response, label])
    # df = pd.DataFrame(results, columns=['prompt', 'response', 'label'])
    # df.to_csv('./datasets/toxigen/prompt_response_classified.csv', index=False)
    # print("num_0: ", num_0)
    # print("num_1: ", num_1)

    # 对xstest数据集进行分类
    model, tokenizer = create_model('../LLM/Llama-Guard-3-8B/models--meta-llama--Llama-Guard-3-8B/snapshots/7327bd9f6efbbe6101dc6cc4736302b3cbb6e425')
    datasets = pd.read_csv('./datasets/xstest/prompt_response_unclassified.csv')
    questions = datasets['prompt'].tolist()
    responses = datasets['response'].tolist()
    results = []
    num_0, num_1 = 0, 0
    for question, response in zip(tqdm(questions), responses):
        label = llama_guard(model, tokenizer, question, response)
        if label == 0:
            num_0 += 1
        else:
            num_1 += 1
        results.append([question, response, label])
    df = pd.DataFrame(results, columns=['prompt', 'response', 'label'])
    df.to_csv('./datasets/xstest/prompt_response_classified.csv', index=False)
    print("num_0: ", num_0)
    print("num_1: ", num_1)