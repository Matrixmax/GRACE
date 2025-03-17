from baselines.RepoHyper.src.repo_graph.repo_to_graph import load_contexts_then_embed, edge_dict_to_adjacency_tensor
from baselines.RepoHyper.src.repo_graph.parse_source_code import parse_source
import os
import pickle
from tqdm import tqdm
import json
from joblib import Parallel, delayed
import sys
import torch

# REPOS_FOLDER = "data/repobench/repos/"
REPOS_ROOT = "/data/wxl/graphrag4se/GRACE/dataset/hf_datasets/repobench_python_v1.1/cross_file_first"
REPOS_FOLDER = REPOS_ROOT + "/repos"
REPOS_TRANSLATED_FOLDER = REPOS_ROOT + "/python_repos_translated"
REPOS_CALL_GRAPHS_FOLDER = REPOS_ROOT + "/repos_call_graphs"
REPOS_GRAPH_UNIXCODER_FOLDER = REPOS_ROOT + "/repos_graphs_unixcoder"


def sizeof(obj):
    return len(pickle.dumps(obj))

def repo_to_graph(repo_path, call_graph_json_path):
    contexts_files = parse_source(repo_path, call_graph_json_path)
    embeddings, edges, type_edges, contexts, index_to_name, index_to_node_type = load_contexts_then_embed(contexts_files)
    edge_tensor = edge_dict_to_adjacency_tensor(edges, len(embeddings))
    type_edges_tensor = {k: edge_dict_to_adjacency_tensor(type_edges[k], len(embeddings)) for k in type_edges}

    return embeddings, edge_tensor, type_edges_tensor, contexts, index_to_name, index_to_node_type

def main(repo_name):
    try:
        call_graph_json_path = os.path.join(REPOS_CALL_GRAPHS_FOLDER, repo_name + ".json")
        with open(call_graph_json_path, "r") as f:
            data = json.load(f)
            if len(data) == 0:
                pass
        embeddings, edge_tensor, type_edges_tensor, contexts, index_to_name, index_to_node_type = repo_to_graph(os.path.join(REPOS_FOLDER, repo_name), call_graph_json_path)
        with open(os.path.join(REPOS_GRAPH_UNIXCODER_FOLDER, repo_name + ".pkl"), "wb") as f:
            pickle.dump({
                "embeddings": torch.concat(list(embeddings.values()), dim=0), 
                "edge_tensor": edge_tensor, 
                "type_edges_tensor": type_edges_tensor, 
                "contexts": contexts, 
                "index_to_name": index_to_name,
                "index_to_node_type": index_to_node_type
            }, f)
    
    except Exception as e:
        print(e)
        pass    
def fix():
    repo_names = os.listdir(REPOS_FOLDER)
    
    # TEMP 这里只使用一个来测试一下
    repo_names = ['3D-DAM']

    for name in tqdm(repo_names):
        main(name)
if __name__ == "__main__":
    # main()
    fix()