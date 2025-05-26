import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from tqdm import tqdm
import openai
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.utils import load_jsonl, dump_jsonl, make_needed_dir
import copy
from utils.utils import CodexTokenizer, CodeGenTokenizer, StarCoderTokenizer
import json
import requests
import ollama

device = "cuda"


def build_retrieval_prompt(case, tokenizer, max_num_tokens, max_top_k):
    context_max_tokens = max_num_tokens // 2
    comment = "Given following context: \n"
    context = ";".join(case["retrieved_snippets"])
    before = "and your need to complete following: \n"
    b = case["all_code"]
    context_prompt = comment + context + before+ b + "in one line:"
    return context_prompt

def build_prompt(case, tokenizer, max_num_tokens, max_top_k=10, mode='retrieval'):
    prompt = ""
    prompt = build_retrieval_prompt(case, tokenizer, max_num_tokens, max_top_k)
    return prompt

def parser_args():
    parser = argparse.ArgumentParser(description="Generate response from llm")
    parser.add_argument('--model', default='qwen-2.5-coder', type=str)
    parser.add_argument('--mode', default='retrieval', type=str, choices=['infile', 'retrieval'])
    parser.add_argument('--max_top_k', default=10, type=int)
    parser.add_argument('--max_new_tokens', default=100, type=int)

    return parser.parse_args()


def main(args, input_cases, responses_save_name):
    # set model and tokenizer# 设定模型和 tokenizer
    # if args.model == 'gpt-3.5-turbo-instruct':
    if args.model == 'gpt-3.5-turbo':

        model = openai.OpenAI(api_key="sk-vBVEDj8fJA7l7c5AhpXpqWABWy2iUaTLQmSloarMgb71tVRW",base_url="https://api.chatanywhere.tech/v1/chat/completions")

        tokenizer = CodexTokenizer()
        max_num_tokens = 4096
    elif args.model == 'starcoder':
        model = AutoModelForCausalLM.from_pretrained(f"./models_cache/{args.model}/")
        tokenizer_raw = AutoTokenizer.from_pretrained(f"./models_cache/{args.model}-tokenizer/", trust_remote_code=True)
        tokenizer = StarCoderTokenizer(tokenizer_raw)
        max_num_tokens = 8192
        generation_config = GenerationConfig(
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer_raw.eos_token_id,
            temperature=0,
            pad_token_id=tokenizer_raw.pad_token_id,
        )
    elif args.model in ['codegen2-16b', 'codegen2-7b', 'codegen2-1b']:
        model = AutoModelForCausalLM.from_pretrained(f"{args.model}/")
        tokenizer_raw = AutoTokenizer.from_pretrained(f"./models_cache/{args.model}-tokenizer/", trust_remote_code=True)
        tokenizer = CodeGenTokenizer(tokenizer_raw)
        max_num_tokens = 2048
        generation_config = GenerationConfig(
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer_raw.eos_token_id,
            temperature=0,
            pad_token_id=tokenizer_raw.pad_token_id,
        )
    print('Model loading finished')

    # generate response and save# 生成响应并保存
    responses = []
    # 原始
    max_num_tokens = 4096
    tokenizer = CodexTokenizer()
    max_prompt_tokens = max_num_tokens - args.max_new_tokens
    with tqdm(total=len(input_cases)) as pbar:
        for case in input_cases:
            pbar.set_description(f'Processing...')
            # 原始
            prompt = build_prompt(case, tokenizer, max_prompt_tokens, max_top_k=args.max_top_k,mode=args.mode)  # 待预测代码前面的+检索到的相关代码

            # if args.model == 'gpt-3.5-turbo-instruct':
            if args.model == 'gpt-3.5-turbo':
                url = "https://api.chatanywhere.tech/v1/chat/completions"
                # 我的免费的账号
                # OPENAI_KEY = "sk-vBVEDj8fJA7l7c5AhpXpqWABWy2iUaTLQmSloarMgb71tVRW"
                # 实验室付费账号
                OPENAI_KEY = "sk-E4Um08hki6npsebk4EKK1msCRZCgvPseUFrCFaykg0xXJeTY"
                payload = json.dumps({
                    "model": "gpt-3.5-turbo",
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a helpful assistant."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                })
                headers = {
                    'Authorization': f'Bearer {OPENAI_KEY}',
                    'Content-Type': 'application/json'
                }
                try:
                    response = requests.request("POST", url, headers=headers, data=payload)
                    response_dict = json.loads(response.text)
                    # 获取生成的内容并只返回第一行
                    generated_content = response_dict['choices'][0]['message']['content']
                    response = generated_content
                    # print(response)
                except Exception as e:  # 捕获所有其他异常
                    print(f"发生未知错误: {e}")
                    path = f"/data/wby/allcode/GraphCoder-main/repobench/generation_results/gpt-3.5-turbo/cross_file_first.gen_res111.jsonl"
                    make_needed_dir(path)
                    dump_jsonl(responses, path)
                    print("存储" + str(len(responses)) + "行结果")
                    return
            elif args.model == "starcoder":
                prompt_ids = tokenizer_raw(prompt, return_tensors="pt").to(device)
                response_ids = model.generate(prompt_ids['input_ids'],
                                              generation_config=generation_config,
                                              attention_mask=prompt_ids['attention_mask'])
                response = tokenizer.decode(response_ids[0])
                prompt_lines = prompt.splitlines(keepends=True)
                n_prompt_lines = len(prompt_lines)
                response_lines = response.splitlines(keepends=True)
                response = "".join(response_lines[n_prompt_lines:])
            elif args.model in ['codegen2-16b', 'codegen2-7b', 'codegen2-1b']:
                prompt_ids = tokenizer_raw(prompt, return_tensors="pt").to(device)
                response_ids = model.generate(prompt_ids['input_ids'],
                                              generation_config=generation_config,
                                              attention_mask=prompt_ids['attention_mask'])
                response = tokenizer.decode(response_ids[0])
                prompt_lines = prompt.splitlines(keepends=True)
                n_prompt_lines = len(prompt_lines)
                response_lines = response.splitlines(keepends=True)
                response = "".join(response_lines[n_prompt_lines:])
            elif args.model == 'qwen-2.5-coder':
                try:
                    response = ollama.chat(model='qwen2.5-coder:32b', messages=[
                        {
                            'role': 'user',
                            'content': prompt,
                        },
                    ],
                    options={
                             'num_predict': 100  # 限制最多生成 100 个 token
                             }
                    )
                    response=response['message']['content']
                    #print(response)
                    print("真实值：", case['next_line'])
                    print("预测值：", response)
                except Exception as e:  # 捕获所有其他异常
                    print(f"发生未知错误: {e}")
                    path=f"/data/wby/allcode/repohyper/GRACE/dataset/dataset_repoeval_updated/repoeval_to_repobench/generate_result/cross_file_random_gen_res111.jsonl"
                    make_needed_dir(path)
                    dump_jsonl(responses,path)
                    print("存储"+str(len(responses))+"行结果")
                    return
            # 记录每个输入案例的生成结果
            case_res = copy.deepcopy(case)
            case_res['generate_response'] = response
            responses.append(copy.deepcopy(case_res))
            pbar.update(1)

    dump_jsonl(responses, responses_save_name)


if __name__ == "__main__":
    args = parser_args()

    # 原始
    input_cases = load_jsonl("/data/wby/allcode/repohyper/GRACE/dataset/dataset_repoeval_updated/repoeval_to_repobench/grace_serach/line_python_search_results.jsonl")
    #input_cases=input_cases[1400:]

    print('Input loading finished')

    # responses_save_name = f"./generation_results/{args.model}/{args.input_file_name}.{args.mode}.{args.model}.gen_res.jsonl"
    responses_save_name = f"/data/wby/allcode/repohyper/GRACE/dataset/dataset_repoeval_updated/repoeval_to_repobench/generate_result/cross_file_random_gen_res.jsonl"
    make_needed_dir(responses_save_name)
    main(args, input_cases, responses_save_name)


