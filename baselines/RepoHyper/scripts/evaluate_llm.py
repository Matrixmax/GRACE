# Python
import torch
from torch.utils.data import DataLoader
import argparse
import os
from baselines.RepoHyper.src.utils import load_data
from baselines.RepoHyper.src.metrics import calc_metrics
from datasets import load_dataset
from baselines.RepoHyper.src.repo_graph.parse_source_code import parse_file, parse_source

# from langchain_community.chat_models import ChatOpenAI
# from langchain_community.embeddings import OpenAIEmbeddings
# from langchain_community.vectorstores import Chroma
# from langchain_community.llms import VLLM
import os
import json
import requests


language = "python"
task = "pipeline"
settings = ["cross_file_first"]

# cross_file_first_hard = load_data(task, language, "cross_file_first")["test"]["hard"]
cross_file_first_hard = load_dataset(f"/data/wxl/graphrag4se/GRACE/dataset/hf_datasets/repobench_{language}_v1.1",split=['cross_file_first'])[0]

def finding_context(contexts_files, relative_path):
    for file in contexts_files:
        if relative_path == file["relative_path"]:
            return file
    return None

class LLMModel:
    def __init__(self, model_name):
        openai_api_key = os.getenv("OPENAI_API_KEY")
        openai_base_url = os.getenv("OPENAI_BASE_URL")
        if model_name != "gpt3.5":
            pass
        else:
            pass

    # def complete(self, inputs, contexts):
    #     return self.llm(self.format(inputs, contexts))

    def complete(self, inputs, contexts):
        url = "https://api.chatanywhere.tech/v1/chat/completions"
        OPENAI_KEY = os.getenv("OPENAI_API_KEY")
        payload = json.dumps({
        "model": "gpt-3.5-turbo",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": self.format(inputs, contexts)
            }
        ]
        })
        headers = {
        'Authorization': f'Bearer {OPENAI_KEY}',
        'Content-Type': 'application/json'
        }

        response = requests.request("POST", url, headers=headers, data=payload)
        response_dict = json.loads(response.text)
        
        # 获取生成的内容并只返回第一行
        generated_content = response_dict['choices'][0]['message']['content']
        next_line = generated_content.split('\n')[0].strip()
        return next_line


    def format(self, inputs, contexts):
        return f"Given following context: {contexts} and your need to complete following {inputs} in one line:"




def evaluate_model(model_name, retrieved_contexts):
    # Load the model
    model = LLMModel(model_name)
    # model.eval()

    total_metric = 0
    filtered_data = cross_file_first_hard.filter(lambda x: x['repo_name'] == 'giaminhgist/3D-DAM')
    # Iterate over the data
    for batch in filtered_data:
        context = finding_context(retrieved_contexts, batch['file_path'])
        inputs, targets = batch["all_code"], batch["next_line"]

        # Forward pass
        outputs = model.complete(inputs, context)
        print(f'model predict next line code: {outputs}')

        break
        # # Compute accuracy
        # correct = calc_metrics(outputs, targets)
        # total_metric += correct

    # avg_accuracy = total_metric / len(cross_file_first_hard)

    # print(f'Average Metric: {avg_accuracy}')


def finding_context(contexts_files, relative_path):
    for file in contexts_files:
        if relative_path == file["relative_path"]:
            return file
    return None

def main():
    parser = argparse.ArgumentParser(description='Evaluate LLM model')
    parser.add_argument('--data', default="/data/wxl/graphrag4se/GRACE/dataset/hf_datasets/repobench_python_v1.1/cross_file_first/repos_graphs_labeled_link_with_called_imported_edges/3D-DAM_labeled.pkl", type=str, help='Path to the data')
    parser.add_argument('--model', default="gpt3.5", type=str, help='Model name')
    parser.add_argument('--num-workers', type=int, default=1, help='Number of workers')

    args = parser.parse_args()

    data_path = args.data
    model_name = args.model
    num_workers = args.num_workers

    if not os.path.exists(data_path):
        print(f"Data path {data_path} does not exist.")
        return
    # retrieved_contexts = load_data(args.data, language, "retrieved_contexts")
    # retrieved_contexts = load_dataset(f"/data/wxl/graphrag4se/GRACE/dataset/hf_datasets/repobench_{language}_v1.1")
    # TEMP

    contexts_files = parse_source("/data/wxl/graphrag4se/GRACE/dataset/hf_datasets/repobench_python_v1.1/cross_file_first/repos/3D-DAM", "/data/wxl/graphrag4se/GRACE/dataset/hf_datasets/repobench_python_v1.1/cross_file_first/repos_call_graphs/3D-DAM.json")

    # file = finding_context(contexts_files, "3d-dam/variational/study_test.py")
    # print(file)

    # TEMP

    # evaluate_model(model_name, retrieved_contexts)
    evaluate_model(model_name, contexts_files)
    

if __name__ == "__main__":
    main()