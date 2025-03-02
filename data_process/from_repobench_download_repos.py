
import os
from joblib import Parallel, delayed
from tqdm import tqdm
import gzip
import pickle
from tqdm import tqdm
from typing import Union
from pathlib import Path
import csv
import os


language = "python"
task = "retrieval"
settings = ["cross_file_first"]


def load_data(task:str, language:str, settings: Union[str, list]):
    """
    Load data from the specified task and language.

    Args:
        task: the task to load
        language: the language to load
        settings: the settings to load
    
    Returns:
        data: the loaded data
    """
    ROOT = "GRACE/dataset/hf_datasets/repobench_r/data"
    if task.lower() == 'r':
        task = 'retrieval'
    elif task.lower() == 'c':
        task = 'completion'
    elif task.lower() == 'p':
        task = 'pipeline'
    
    if language.lower() == 'py':
        language = 'python'
    
    if isinstance(settings, str):
        settings = [settings]
    
    for i, setting in enumerate(settings):
        if setting.lower() == 'xf-f':
            settings[i] = 'cross_file_first'
        elif setting.lower() == 'xf-r':
            settings[i] = 'cross_file_random'
        elif setting.lower() == 'if':
            settings[i] = 'in_file'
        

    # some assertions
    assert task.lower() in ['r', 'c', 'p', 'retrieval', 'completion', 'pipeline'], \
        "task must be one of R, C, or P"
    

    assert language.lower() in ['python', 'java', 'py'], \
        "language must be one of python, java"

    
    if task == "retrieval":
        assert all([setting.lower() in ['cross_file_first', 'cross_file_random'] for setting in settings]), \
            "For RepoBench-R, settings must be one of xf-f or xf-r"
    else:
        assert all([setting.lower() in ['cross_file_first', 'cross_file_random', 'in_file'] for setting in settings]), \
            "Settings must be one of xf-f, xf-r, or if"
    

    # load data
    data = {}
    # We split retrieval data into shards due to the github file size limit
    # if task == "retrieval":
    #     for setting in tqdm(settings, desc=f"Loading data"):
    #         # we only further split the cross_file_first setting for java
    #         if setting == "cross_file_first" and language == "java":
    #             dic = {
    #                 "train": {},
    #                 "test": {}
    #             }
    #             with gzip.open(f"{ROOT}/{task}/{language}/{setting}_train_easy_1.gz", 'rb') as f:
    #                 train_easy_1 = pickle.load(f)
    #             with gzip.open(f"{ROOT}/{task}/{language}/{setting}_train_easy_2.gz", 'rb') as f:
    #                 train_easy_2 = pickle.load(f)
    #             dic['train']['easy'] = train_easy_1['easy'] + train_easy_2['easy']

    #             with gzip.open(f"{ROOT}/{task}/{language}/{setting}_train_hard_1.gz", 'rb') as f:
    #                 train_hard_1 = pickle.load(f)
    #             with gzip.open(f"{ROOT}/{task}/{language}/{setting}_train_hard_2.gz", 'rb') as f:
    #                 train_hard_2 = pickle.load(f)
    #             dic['train']['hard'] = train_hard_1['hard'] + train_hard_2['hard']

    #             with gzip.open(f"{ROOT}/{task}/{language}/{setting}_test.gz", 'rb') as f:
    #                 test = pickle.load(f)
    #             dic['test'] = test['test']
        
            
    #             data[setting] = dic
        
    #         else:
    #             dic = {
    #                 "train": {},
    #                 "test": {}
    #             }
    #             with gzip.open(f"{ROOT}/{task}/{language}/{setting}_train_easy.gz", 'rb') as f:
    #                 train_easy = pickle.load(f)
    #             dic['train']['easy'] = train_easy['easy']

    #             with gzip.open(f"{ROOT}/{task}/{language}/{setting}_train_hard.gz", 'rb') as f:
    #                 train_hard = pickle.load(f)
    #             dic['train']['hard'] = train_hard['hard']

    #             with gzip.open(f"{ROOT}/{task}/{language}/{setting}_test.gz", 'rb') as f:
    #                 test = pickle.load(f)
    #             dic['test'] = test['test']
        
    #             data[setting] = dic

    # else:
    #     for setting in tqdm(settings, desc=f"Loading data"):
    #         with gzip.open(f"{ROOT}/{task}/{language}/{setting}.gz", 'rb') as f:
    #             data[setting] = pickle.load(f)
    
    if task == "retrieval":
        # 加载 cross_file_first 和 cross_file_random 时候的数据，python，java
        if setting == "cross_file_first":
            setting_name = "cff"
        elif setting == "cross_file_random":
            setting_name = "cfr"

            
        dic = {
            "train": {},
            "test": {}
        }
        cff_file = f"{ROOT}/{language}_{setting_name}.gz"
        with gzip.open(cff_file, 'rb') as f:
            data = pickle.load(f)
        dic['train']['easy'] = data['train']['easy']
        dic['train']['hard'] = data['train']['hard']
        
        dic['test']['easy'] = data['test']['easy']
        dic['test']['hard'] = data['test']['hard']

        data[setting] = dic
    else:
        print("Unknown task: ", task)
    
    if len(settings) == 1:
        return data[settings[0]]
    else:
        return [data[setting] for setting in settings]
    


def get_all_samples_from_repo(repo_name, dataset):
    samples = []
    for sample in dataset:
        if sample["repo_name"] == repo_name:
            samples.append(sample)
    return samples

def download_repo(repo):
    file_name = repo.split("/")[-1]
    if file_name not in os.listdir("baselines/RepoHyper_main/scripts/data/repobench/repos/"):
        os.system(f'git clone --depth 1 --single-branch https://github.com/{repo} data/repobench/repos/{file_name}')
    else:
        print(f"Already downloaded {repo}")

if __name__ =="__main__":

    cross_file_first_easy = load_data(task, language, "cross_file_first")["train"]["easy"]
    cross_file_first_hard = load_data(task, language, "cross_file_first")["train"]["hard"]
    cross_file_first = cross_file_first_hard + cross_file_first_easy

    unique_repo_names = set()

    for sample in cross_file_first:
        unique_repo_names.add(sample["repo_name"])

    unique_repo_names = list(unique_repo_names)


    Parallel(n_jobs=40, prefer="threads")(
        delayed(download_repo)(name) for name in tqdm(unique_repo_names))
