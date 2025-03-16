import gzip
import pickle
from tqdm import tqdm
from typing import Union
import errno
import torch
import sys
import logging
import json
from pathlib import Path
import torch.distributed as dist
import csv
import os
from datasets import load_dataset

# ROOT = "baselines/RepoHyper_main/scripts/data/repobench"
ROOT = "/data/wxl/graphrag4se/GRACE/dataset/hf_datasets/repobench_python_v1.1"

REPOS_ROOT = "/data/wxl/graphrag4se/GRACE/dataset/hf_datasets/repobench_python_v1.1/cross_file_first"
REPOS_FOLDER = REPOS_ROOT + "/repos"
REPOS_TRANSLATED_FOLDER = REPOS_ROOT + "/python_repos_translated"
REPOS_CALL_GRAPHS_FOLDER = REPOS_ROOT + "/repos_call_graphs"
REPOS_GRAPH_UNIXCODER_FOLDER = REPOS_ROOT + "/repos_graphs_unixcoder"


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
    if task == "retrieval":
        for setting in tqdm(settings, desc=f"Loading data"):
            # we only further split the cross_file_first setting for java
            if setting == "cross_file_first" and language == "java":
                dic = {
                    "train": {},
                    "test": {}
                }
                with gzip.open(f"{ROOT}/{task}/{language}/{setting}_train_easy_1.gz", 'rb') as f:
                    train_easy_1 = pickle.load(f)
                with gzip.open(f"{ROOT}/{task}/{language}/{setting}_train_easy_2.gz", 'rb') as f:
                    train_easy_2 = pickle.load(f)
                dic['train']['easy'] = train_easy_1['easy'] + train_easy_2['easy']

                with gzip.open(f"{ROOT}/{task}/{language}/{setting}_train_hard_1.gz", 'rb') as f:
                    train_hard_1 = pickle.load(f)
                with gzip.open(f"{ROOT}/{task}/{language}/{setting}_train_hard_2.gz", 'rb') as f:
                    train_hard_2 = pickle.load(f)
                dic['train']['hard'] = train_hard_1['hard'] + train_hard_2['hard']

                with gzip.open(f"{ROOT}/{task}/{language}/{setting}_test.gz", 'rb') as f:
                    test = pickle.load(f)
                dic['test'] = test['test']
        
            
                data[setting] = dic
        
            else:
                dic = {
                    "train": {},
                    "test": {}
                }
                with gzip.open(f"{ROOT}/{task}/{language}/{setting}_train_easy.gz", 'rb') as f:
                    train_easy = pickle.load(f)
                dic['train']['easy'] = train_easy['easy']

                with gzip.open(f"{ROOT}/{task}/{language}/{setting}_train_hard.gz", 'rb') as f:
                    train_hard = pickle.load(f)
                dic['train']['hard'] = train_hard['hard']

                with gzip.open(f"{ROOT}/{task}/{language}/{setting}_test.gz", 'rb') as f:
                    test = pickle.load(f)
                dic['test'] = test['test']
        
                data[setting] = dic

    else:
        for setting in tqdm(settings, desc=f"Loading data"):
            with gzip.open(f"{ROOT}/{task}/{language}/{setting}.gz", 'rb') as f:
                data[setting] = pickle.load(f)
    
    # TEMP

    if task == "retrieval":
        # 加载 cross_file_first 和 cross_file_random 时候的数据，python，java
        # if setting == "cross_file_first":
        #     setting_name = "cff"
        # elif setting == "cross_file_random":
        #     setting_name = "cfr"

            
        # dic = {
        #     "train": {},
        #     "test": {}
        # }
        # cff_file = f"{ROOT}/{language}_{setting_name}.gz"
        # with gzip.open(cff_file, 'rb') as f:
        #     data = pickle.load(f)
        # dic['train']['easy'] = data['train']['easy']
        # dic['train']['hard'] = data['train']['hard']
        
        # dic['test']['easy'] = data['test']['easy']
        # dic['test']['hard'] = data['test']['hard']

        # data[setting] = dic


        data = load_dataset(f"/data/wxl/graphrag4se/GRACE/dataset/hf_datasets/repobench_{language}_v1.1")

    else:
        print("Unknown task: ", task)

    # TEMP



    if len(settings) == 1:
        return data[settings[0]]
    else:
        return [data[setting] for setting in settings]
    
logger = logging.getLogger(__name__)

def init_logger(is_main=True, is_distributed=False, filename=None):
    if is_distributed:
        torch.distributed.barrier()
    handlers = [logging.StreamHandler(sys.stdout)]
    if filename is not None:
        handlers.append(logging.FileHandler(filename=filename))
    logging.basicConfig(
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main else logging.WARN,
        format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
        handlers=handlers,
    )
    logging.getLogger('transformers.tokenization_utils').setLevel(logging.ERROR)
    logging.getLogger('transformers.tokenization_utils_base').setLevel(logging.ERROR)
    return logger

def get_checkpoint_path(opt):
    checkpoint_path = Path(opt.checkpoint_dir) / opt.name
    checkpoint_exists = checkpoint_path.exists()
    if opt.is_distributed:
        torch.distributed.barrier()
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    return checkpoint_path, checkpoint_exists

def symlink_force(target, link_name):
    try:
        os.symlink(target, link_name)
    except OSError as e:
        if e.errno == errno.EEXIST:
            os.remove(link_name)
            os.symlink(target, link_name)
        else:
            raise e

def save(model, optimizer, scheduler, step, best_eval_metric, opt, dir_path, name):
    model_to_save = model.module if hasattr(model, "module") else model
    path = os.path.join(dir_path, "checkpoint")
    epoch_path = os.path.join(path, name) #"step-%s" % step)
    os.makedirs(epoch_path, exist_ok=True)
    model_to_save.save_pretrained(epoch_path)
    cp = os.path.join(path, "latest")
    fp = os.path.join(epoch_path, "optimizer.pth.tar")
    checkpoint = {
        "step": step,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "opt": opt,
        "best_eval_metric": best_eval_metric,
    }
    torch.save(checkpoint, fp)
    symlink_force(epoch_path, cp)


def load(model_class, dir_path, opt, reset_params=False):
    epoch_path = os.path.realpath(dir_path)
    optimizer_path = os.path.join(epoch_path, "optimizer.pth.tar")
    logger.info("Loading %s" % epoch_path)
    model = model_class.from_pretrained(epoch_path)
    model = model.to(opt.device)
    logger.info("loading checkpoint %s" %optimizer_path)
    checkpoint = torch.load(optimizer_path, map_location=opt.device)
    opt_checkpoint = checkpoint["opt"]
    step = checkpoint["step"]
    if "best_eval_metric" in checkpoint:
        best_eval_metric = checkpoint["best_eval_metric"]
    else:
        best_eval_metric = checkpoint["best_dev_em"]
    if not reset_params:
        optimizer, scheduler = set_optim(opt_checkpoint, model)
        scheduler.load_state_dict(checkpoint["scheduler"])
        optimizer.load_state_dict(checkpoint["optimizer"])
    else:
        optimizer, scheduler = set_optim(opt, model)

    return model, optimizer, scheduler, opt_checkpoint, step, best_eval_metric

class WarmupLinearScheduler(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, warmup_steps, scheduler_steps, min_ratio, fixed_lr, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.scheduler_steps = scheduler_steps
        self.min_ratio = min_ratio
        self.fixed_lr = fixed_lr
        super(WarmupLinearScheduler, self).__init__(
            optimizer, self.lr_lambda, last_epoch=last_epoch
        )

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return (1 - self.min_ratio)*step/float(max(1, self.warmup_steps)) + self.min_ratio

        if self.fixed_lr:
            return 1.0

        return max(0.0,
            1.0 + (self.min_ratio - 1) * (step - self.warmup_steps)/float(max(1.0, self.scheduler_steps - self.warmup_steps)),
        )


class FixedScheduler(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, last_epoch=-1):
        super(FixedScheduler, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)
    def lr_lambda(self, step):
        return 1.0


def set_dropout(model, dropout_rate):
    for mod in model.modules():
        if isinstance(mod, torch.nn.Dropout):
            mod.p = dropout_rate


def set_optim(opt, model):
    if opt.optim == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    elif opt.optim == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    if opt.scheduler == 'fixed':
        scheduler = FixedScheduler(optimizer)
    elif opt.scheduler == 'linear':
        if opt.scheduler_steps is None:
            scheduler_steps = opt.total_steps
        else:
            scheduler_steps = opt.scheduler_steps
        scheduler = WarmupLinearScheduler(optimizer, warmup_steps=opt.warmup_steps, scheduler_steps=scheduler_steps, min_ratio=0., fixed_lr=opt.fixed_lr)
    return optimizer, scheduler


def average_main(x, opt):
    if not opt.is_distributed:
        return x
    if opt.world_size > 1:
        dist.reduce(x, 0, op=dist.ReduceOp.SUM)
        if opt.is_main:
            x = x / opt.world_size
    return x


def sum_main(x, opt):
    if not opt.is_distributed:
        return x
    if opt.world_size > 1:
        dist.reduce(x, 0, op=dist.ReduceOp.SUM)
    return x


def weighted_average(x, count, opt):
    if not opt.is_distributed:
        return x, count
    t_loss = torch.tensor([x * count], device=opt.device)
    t_total = torch.tensor([count], device=opt.device)
    t_loss = sum_main(t_loss, opt)
    t_total = sum_main(t_total, opt)
    return (t_loss / t_total).item(), t_total.item()


def write_output(glob_path, output_path):
    files = list(glob_path.glob('*.txt'))
    files.sort()
    with open(output_path, 'w') as outfile:
        for path in files:
            with open(path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    outfile.write(line)
            path.unlink()
    glob_path.rmdir()


def save_distributed_dataset(data, opt):
    dir_path = Path(opt.checkpoint_dir) / opt.name
    write_path = dir_path / 'tmp_dir'
    write_path.mkdir(exist_ok=True)
    tmp_path = write_path / f'{opt.global_rank}.json'
    with open(tmp_path, 'w') as fw:
        json.dump(data, fw)
    if opt.is_distributed:
        torch.distributed.barrier()
    if opt.is_main:
        final_path = dir_path / 'dataset_wscores.json'
        logger.info(f'Writing dataset with scores at {final_path}')
        glob_path = write_path / '*'
        results_path = write_path.glob('*.json')
        alldata = []
        for path in results_path:
            with open(path, 'r') as f:
                data = json.load(f)
            alldata.extend(data)
            path.unlink()
        with open(final_path, 'w') as fout:
            json.dump(alldata, fout, indent=4)
        write_path.rmdir()

def load_passages(path):
    if not os.path.exists(path):
        logger.info(f'{path} does not exist')
        return
    logger.info(f'Loading passages from: {path}')
    passages = []
    with open(path) as fin:
        reader = csv.reader(fin, delimiter='\t')
        for k, row in enumerate(reader):
            if not row[0] == 'id':
                try:
                    passages.append((row[0], row[1], row[2]))
                except:
                    logger.warning(f'The following input line has not been correctly loaded: {row}')
    return passages