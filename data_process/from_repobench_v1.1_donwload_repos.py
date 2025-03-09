
import os
from joblib import Parallel, delayed
from tqdm import tqdm
import gzip
import pickle
from typing import Union
from pathlib import Path
import csv
import os
from datasets import load_dataset
import git


# def load_data(language:str):
#     """
#     Load data from the specified language.

#     Args:
#         language: the language to load
#         settings: the settings to load
    
#     Returns:
#         data: the loaded data
#     """
#     # ROOT = "GRACE/dataset/hf_datasets/repobench_r/data"
#     ROOT = f"../dataset/hf_datasets/repobench_{language}_v1.1/data"
    
#     if language.lower() == 'py':
#         language = 'python'
    
#     if isinstance(settings, str):
#         settings = [settings]
    
#     for i, setting in enumerate(settings):
#         if setting.lower() == 'xf-f':
#             settings[i] = 'cross_file_first'
#         elif setting.lower() == 'xf-r':
#             settings[i] = 'cross_file_random'
#         elif setting.lower() == 'if':
#             settings[i] = 'in_file'
        

#     # some assertions   

#     assert language.lower() in ['python', 'java', 'py'], \
#         "language must be one of python, java"


#     assert all([setting.lower() in ['cross_file_first', 'cross_file_random'] for setting in settings]), \
#         "For RepoBench-R, settings must be one of xf-f or xf-r"
    



#     data = load_dataset(f"GRACE/dataset/hf_datasets/repobench_{language}_v1.1")

#     # data = load_dataset("/data/wxl/graphrag4se/GRACE/dataset/hf_datasets/repobench_python_v1.1")

#     print("loaded ")

#     return data
#     # cff_file = f"{ROOT}/{language}_{setting_name}.gz"
#     # with gzip.open(cff_file, 'rb') as f:
#     #     data = pickle.load(f)
#     # dic['train'] = data['train']['easy']
    
#     # dic['test']['easy'] = data['test']['easy']


#     # all_data[setting] = dic
    
#     # if len(settings) == 1:
#     #     return all_data[settings[0]]
#     # else:
#     #     return [all_data[setting] for setting in settings]
    
# 获取当前SSH代理的socket路径
ssh_auth_sock = os.environ.get('SSH_AUTH_SOCK')

def clone_repo(url):
    # 为子进程设置环境变量
    env = {'SSH_AUTH_SOCK': ssh_auth_sock} if ssh_auth_sock else {}
    Repo.clone_from(url, env=env)  # 传递env参数（依gitpython版本调整）

def get_all_samples_from_repo(repo_name, dataset):
    samples = []
    for sample in dataset:
        if sample["repo_name"] == repo_name:
            samples.append(sample)
    return samples

def download_repo(repo):
    file_name = repo.split("/")[-1]
    repo_folder = f"/data/wxl/graphrag4se/GRACE/dataset/hf_datasets/repobench_{language}_v1.1/{settings[0]}/repos"
    
    # Check if repo_folder exists, if not create it
    if not os.path.exists(repo_folder):
        print(f"Creating directory: {repo_folder}")
        os.makedirs(repo_folder, exist_ok=True)
        
    if file_name not in os.listdir(repo_folder):
        try:

            # 构建 git URL 并克隆仓库
            # git_url = f"https://github.com/{repo}.git"
            git_url = f"https://{os.environ['GIT_USERNAME']}:{os.environ['GIT_PASSWORD']}@github.com/{repo}.git"
            git.Repo.clone_from(git_url, f"{repo_folder}/{file_name}", depth=1, single_branch=True)
            print(f"Successfully cloned {repo} with configured git settings")
        except Exception as e:
            print(f"Failed to clone {repo} using GitPython: {str(e)}")
    else:
        print(f"Already downloaded {repo}")

if __name__ =="__main__":

    language = "python"
    settings = ["cross_file_first"]

    print(f"language: {language}, settings: {settings}")

    unique_repo_names = set()

    data = load_dataset(f"/data/wxl/graphrag4se/GRACE/dataset/hf_datasets/repobench_{language}_v1.1")

    for sample in data[settings[0]]:
        unique_repo_names.add(sample["repo_name"])

    unique_repo_names = list(unique_repo_names)


    Parallel(n_jobs=4, prefer="threads")(
        delayed(download_repo)(name) for name in tqdm(unique_repo_names))

    # Parallel(n_jobs=4)(clone_repo(url) for url in repo_urls)