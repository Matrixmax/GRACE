import glob
import networkx as nx
import pickle
import json
import os
from networkx.readwrite import json_graph
import tiktoken

device = "cuda"


class CodeGenTokenizer:
    def __init__(self, tokenizer_raw):
        self.tokenizer = tokenizer_raw

    def tokenize(self, text):
        return self.tokenizer(text, return_tensors="pt")['input_ids'][0].to(device)

    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)


class StarCoderTokenizer:
    def __init__(self, tokenizer_raw):
        self.tokenizer = tokenizer_raw

    def tokenize(self, text):
        return self.tokenizer.encode(text, return_tensors="pt")[0].to(device)

    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)


class CONSTANTS:
    max_hop = 5
    max_search_top_k = 10
    max_statement = 20
    #原始graphcoder
    # repo_base_dir = f"./repositories"
    # graph_database_save_dir = f"./context_database"
    # query_graph_save_dir = "./graph_based_query"
    # search_results_save_dir = "./search_results"
    # generate_save_dir = "./generation_results"
    # dataset_dir = "./RepoEval-Updated"
    # repos = [
    #          "devchat",
    #          "nemo_aligner",
    #          "awslabs_fortuna",
    #          "task_weaver",
    #          "huggingface_diffusers",
    #          "opendilab_ACE",
    #          "metagpt",
    #          "apple_axlearn",
    #          "QingruZhang_AdaLoRA",
    #          "nerfstudio-project_nerfstudio",
    #          "itlemon_chatgpt4j",#
    #          # "Aelysium-Group_rusty-connector",#仓库路径很有问题
    #          # "neoforged_NeoGradle",#仓库路径很有问题
    #          "mybatis-flex_mybatis-flex",#
    #          "Guiqu1aixi_rocketmq",#
    #          "SimonHalvdansson_Harmonic-HN",#
    #          "Open-DBT_open-dbt",#
    #          "QuasiStellar_custom-pixel-dungeon",#
    #          "gentics_cms-oss",
    #          "FloatingPoint-MC_MIN"#
    #          ]
    # 
    # 
    # repos_language = {
    #      "devchat": "python",
    #      "nemo_aligner": "python",
    #      "awslabs_fortuna": "python",
    #      "task_weaver": "python",
    #      "huggingface_diffusers": "python",
    #      "opendilab_ACE": "python",
    #      "metagpt": "python",
    #      "apple_axlearn": "python",
    #      "QingruZhang_AdaLoRA": "python",
    #      "nerfstudio-project_nerfstudio": "python",
    #      "itlemon_chatgpt4j": "java",
    #      "Aelysium-Group_rusty-connector": "java",#仓库路径很有问题
    #      "neoforged_NeoGradle": "java",#仓库路径很有问题
    #      "mybatis-flex_mybatis-flex": "java",
    #      "Guiqu1aixi_rocketmq": "java",
    #      "SimonHalvdansson_Harmonic-HN": "java",
    #      "Open-DBT_open-dbt": "java",
    #      "QuasiStellar_custom-pixel-dungeon": "java",
    #      "gentics_cms-oss": "java",
    #      "FloatingPoint-MC_MIN": "java"
    # }

    #repobench
    repo_base_dir = f"/data/wby/allcode/repohyper/GRACE/dataset/hf_datasets/repobench_python_v1.1/cross_file_first/repos"
    graph_database_save_dir = f"/data/wby/allcode/GraphCoder-main/repobench/context_database"
    query_graph_save_dir = "/data/wby/allcode/GraphCoder-main/repobench/graph_based_query"
    search_results_save_dir= "/data/wby/allcode/GraphCoder-main/repobench/search_results"
    generate_save_dir = "/data/wby/allcode/GraphCoder-main/repobench/generation_results"
    dataset_dir = "/data/wby/allcode/GraphCoder-main/repobench/repobench_to_repoeval"
    repos = os.listdir(repo_base_dir)
    repos_language =  {item: "python" for item in repos}

    #crosscodeeval  python
    # repo_base_dir = f"/data/wby/allcode/repohyper/GRACE/dataset/crosscodeeval/repos"
    # graph_database_save_dir = f"/data/wby/allcode/GraphCoder-main/crosscodeeval/python/context_database"
    # query_graph_save_dir = "/data/wby/allcode/GraphCoder-main/crosscodeeval/python/graph_based_query"
    # search_results_save_dir = "/data/wby/allcode/GraphCoder-main/crosscodeeval/python/search_results"
    # generate_save_dir = "/data/wby/allcode/GraphCoder-main/crosscodeeval/python/generation_results"
    # dataset_dir = "/data/wby/allcode/GraphCoder-main/crosscodeeval/data"
    # repos = os.listdir(repo_base_dir)
    # repos_language = {item: "python" for item in repos}

    # crosscodeeval  java
    # repo_base_dir = f"/data/wby/allcode/repohyper/GRACE/dataset/crosscodeeval/repos"
    # graph_database_save_dir = f"/data/wby/allcode/GraphCoder-main/crosscodeeval/java/context_database"
    # query_graph_save_dir = "/data/wby/allcode/GraphCoder-main/crosscodeeval/java/graph_based_query"
    # search_results_save_dir = "/data/wby/allcode/GraphCoder-main/crosscodeeval/java/search_results"
    # generate_save_dir = "/data/wby/allcode/GraphCoder-main/crosscodeeval/java/generation_results"
    # dataset_dir = "/data/wby/allcode/GraphCoder-main/crosscodeeval/data"
    # repos = os.listdir(repo_base_dir)
    # repos_language = {item: "java" for item in repos}


class CodexTokenizer:
    def __init__(self):
        self.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo-instruct")
        # self.tokenizer = tiktoken.get_encoding("p50k_base")

    def tokenize(self, text):
        # return self.tokenizer.encode(text)
        return self.tokenizer.encode(text, allowed_special={'<|endoftext|>'})

    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids)


class CodeGenTokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def tokenize(self, text):
        return self.tokenizer.encode(text)

    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids)


def iterate_repository_file(base_dir, repo):
    if CONSTANTS.repos_language[repo] == "python":
        pattern = os.path.join(f'{base_dir}/{repo}', "**", f"*.py")
    elif CONSTANTS.repos_language[repo] == "java":
        pattern = os.path.join(f'{base_dir}/{repo}', "**", f"*.java")
    else:
        raise NotImplementedError
    files = glob.glob(pattern, recursive=True)
    return files


def iterate_repository_json_file(base_dir, repo):
    pattern = os.path.join(f'{base_dir}/{repo}', "**", "*.json")
    files = glob.glob(pattern, recursive=True)
    return files


def make_needed_dir(file_path):
    dir_path = os.path.dirname(file_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return


def load_pickle(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)


def set_default(obj):
    if isinstance(obj, set):
        return list(obj)
    raise TypeError


def dump_jsonl(obj, fname):
    with open(fname, 'w', encoding='utf8') as f:
        for item in obj:
            f.write(json.dumps(item, default=set_default) + '\n')


def load_jsonl(fname):
    with open(fname, 'r', encoding='utf8') as f:
        lines = []
        for line in f:
            lines.append(json.loads(line))
        return lines




def graph_to_json(obj: nx.MultiDiGraph):
    return json.dumps(json_graph.node_link_data(obj), default=set_default)


def json_to_graph(json_format):
    graph_js = json.loads(json_format)
    graph = json_graph.node_link_graph(graph_js)
    return graph


def tokenize(code):
    tokenizer = CodexTokenizer()
    return tokenizer.tokenize(code)
