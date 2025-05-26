import editdistance
import numpy as np
from utils.utils import load_jsonl
from nltk.tokenize import RegexpTokenizer
from typing import FrozenSet
import keyword
import re

string_pattern = r'"([^"\\]*(\\.[^"\\]*)*)"|\'([^\'\\]*(\\.[^\'\\]*)*)\''
code_tokenizer = RegexpTokenizer(r'\w+')
IDENTIFIER_REGEX = re.compile('[_a-zA-Z][_a-zA-Z0-9]*')


def compute_EM(target, prediction, language):
    comment_prefix = ""
    if language == "python":
        comment_prefix = "#"
    elif language == "java":
        comment_prefix = "//"

    target_lines = [line.strip() for line in target.splitlines() if line.strip()]
    target_lines = [line for line in target_lines if not line.startswith(comment_prefix)]
    prediction_lines = [line.strip() for line in prediction.splitlines() if line.strip()]
    prediction_lines = [line for line in prediction_lines if not line.startswith(comment_prefix)][:len(target_lines)]
    target_lines_str = "".join(target_lines)
    prediction_lines_str = "".join(prediction_lines)
    if target_lines_str == prediction_lines_str:
        return 1
    else:
        return 0

def compute_ES(target, prediction, language):

    comment_prefix = ""
    if language == "python":
        comment_prefix = "#"
    elif language == "java":
        comment_prefix = "//"

    target_lines = [line.strip() for line in target.splitlines() if line.strip()]
    target_lines = [line for line in target_lines if not line.startswith(comment_prefix)]
    prediction_lines = [line.strip() for line in prediction.splitlines() if line.strip()]
    prediction_lines = [line for line in prediction_lines if not line.startswith(comment_prefix)][:len(target_lines)]

    target_str = ''.join(target_lines)
    prediction_str = ''.join(prediction_lines)
    ES_score = 1 - (editdistance.eval(target_str, prediction_str) / max(len(target_str), len(prediction_str)))

    return ES_score

def compute_batch_EM(generation_res_file_path, language):
    res = load_jsonl(generation_res_file_path)
    em_val = 0
    cnt=0
    for i in range(0, len(res)):
        cnt=cnt+1
        pred_str = res[i]['generate_response']

        # 使用 split 分割字符串，最多分割一次
        parts = pred_str.split(f"```{language}", 1)
        if(len(parts) <=1): parts = pred_str.split(f"```", 1)
        # 如果存在前缀，则取第二部分，否则保持原样
        pred_str = parts[1] if len(parts) > 1 else pred_str

        gt_str = res[i]['next_line']
        em_val += compute_EM(gt_str, pred_str, language)
    return em_val / cnt


def compute_batch_ES( generation_res_file_path, language):
    res = load_jsonl(generation_res_file_path)
    es_val = 0
    cnt=0
    for i in range(0, len(res)):

        cnt = cnt + 1
        pred_str = res[i]['generate_response']

        # 使用 split 分割字符串，最多分割一次
        parts = pred_str.split(f"```{language}", 1)
        if (len(parts) <= 1): parts = pred_str.split(f"```", 1)
        # 如果存在前缀，则取第二部分，否则保持原样
        pred_str = parts[1] if len(parts) > 1 else pred_str

        gt_str = res[i]['next_line']
        es_val += compute_ES(gt_str, pred_str, language)
    return es_val / cnt


def compute_recall(target, prediction, language):
    """
    计算 Recall（召回率）: TP / (TP + FN)
    """
    # 去掉注释行
    comment_prefix = "#" if language == "python" else "//"
    target_lines = [line.strip() for line in target.splitlines() if line.strip()]
    target_lines = [line for line in target_lines if not line.startswith(comment_prefix)]
    target_tokens = set(code_tokenizer.tokenize(" ".join(target_lines)))
    prediction_lines = [line.strip() for line in prediction.splitlines() if line.strip()]
    prediction_lines = [line for line in prediction_lines if not line.startswith(comment_prefix)][:len(target_lines)]
    prediction_tokens = set(code_tokenizer.tokenize(" ".join(prediction_lines)))

    if not target_tokens:
        return 1.0 if not prediction_tokens else 0.0  # 避免 0 除错误

    tp = len(target_tokens & prediction_tokens)  # 预测正确的 token
    fn = len(target_tokens - prediction_tokens)  # ground truth 里有但预测缺失的 token

    return tp / (tp + fn) if (tp + fn) > 0 else 0.0

def compute_precision(target, prediction, language):
    """
    计算 Precision（精确率）: TP / (TP + FP)
    """
    comment_prefix = "#" if language == "python" else "//"
    target_lines = [line.strip() for line in target.splitlines() if line.strip()]
    target_lines = [line for line in target_lines if not line.startswith(comment_prefix)]
    target_tokens = set(code_tokenizer.tokenize(" ".join(target_lines)))
    prediction_lines = [line.strip() for line in prediction.splitlines() if line.strip()]
    prediction_lines = [line for line in prediction_lines if not line.startswith(comment_prefix)][:len(target_lines)]
    prediction_tokens = set(code_tokenizer.tokenize(" ".join(prediction_lines)))

    if not prediction_tokens:
        return 1.0 if not target_tokens else 0.0  # 避免 0 除错误

    tp = len(target_tokens & prediction_tokens)  # 预测正确的 token
    fp = len(prediction_tokens - target_tokens)  # 预测里多出的 token

    return tp / (tp + fp) if (tp + fp) > 0 else 0.0

def compute_f1(target, prediction, language):
    """
    计算 F1-score: 2 * (precision * recall) / (precision + recall)
    """
    recall = compute_recall(target, prediction, language)
    precision = compute_precision(target, prediction, language)

    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0


def compute_batch_recall(generation_res_file_path, language):
    """
    批量计算 Recall
    """
    res = load_jsonl(generation_res_file_path)
    recall_val = 0
    cnt=0
    for i in range(0,len(res)):

        cnt = cnt + 1
        pred_str = res[i]['generate_response']

        # 使用 split 分割字符串，最多分割一次
        parts = pred_str.split(f"```{language}", 1)
        if (len(parts) <= 1): parts = pred_str.split(f"```", 1)
        # 如果存在前缀，则取第二部分，否则保持原样
        pred_str = parts[1] if len(parts) > 1 else pred_str

        gt_str = res[i]['next_line']
        recall_val += compute_recall(gt_str, pred_str, language=language)

    return recall_val / cnt


def compute_batch_f1(generation_res_file_path, language):
    """
    批量计算 F1-score
    """
    res = load_jsonl(generation_res_file_path)
    f1_val = 0
    cnt=0

    for i in range(0,len(res)):

        cnt = cnt + 1
        pred_str = res[i]['generate_response']

        # 使用 split 分割字符串，最多分割一次
        parts = pred_str.split(f"```{language}", 1)
        if (len(parts) <= 1): parts = pred_str.split(f"```", 1)
        # 如果存在前缀，则取第二部分，否则保持原样
        pred_str = parts[1] if len(parts) > 1 else pred_str

        gt_str = res[i]['next_line']
        f1_val += compute_f1(gt_str, pred_str, language=language)

    return f1_val / cnt


def main():
    generation_path="/data/wby/allcode/repohyper/GRACE/dataset/dataset_repoeval_updated/repoeval_to_repobench/generate_result/cross_file_random_gen_res.jsonl"
    language="python"
    em=compute_batch_EM(generation_path,language)
    es=compute_batch_ES(generation_path,language)
    recall=compute_batch_recall(generation_path, language)
    f1=compute_batch_f1(generation_path, language)
    print("em:",em)
    print("es:", es)
    print("recall:", recall)
    print("f1:", f1)


if __name__ == "__main__":
    main()
