import os
import torch
import networkx as nx
import numpy as np
import hnswlib
import json
import requests
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import argparse
from transformers import AutoModel,AutoTokenizer, AutoModelForCausalLM
from transformers import PreTrainedTokenizer, PreTrainedModel
from datasets import load_dataset
import dgl
from dgl import LapPE
import logging
import pickle
import ast
import parso
import graph_text_encoders


from multilevel_graph_builder import (
    MultiLevelGraph, 
    MultiLevelGraphBuilder, 
    process_repobench_repo, 
    preprocess_repobench_data
)

from codebleu import calc_codebleu


def calc_metrics(hypotheses, references):
    # 计算指标：EM、ES、recall、F1
    result = {}
    result['codebleu'] = calc_codebleu(hypotheses, references, lang="python")
    # Calculate Exact Match (EM)
    exact_matches = sum(1 for hyp, ref in zip(hypotheses, references) if hyp.strip() == ref.strip())
    result['em'] = exact_matches / len(hypotheses) if hypotheses else 0
    
    # Calculate Edit Similarity (ES)
    result['es'] = np.mean([1 - levenshtein(hyp, ref) / max(len(hyp), len(ref)) 
                           for hyp, ref in zip(hypotheses, references)]) if hypotheses else 0
    
    # Calculate Recall
    result['recall'] = sum(1 for hyp, ref in zip(hypotheses, references) 
                          if ref.strip() in hyp.strip()) / len(hypotheses) if hypotheses else 0
    
    # Calculate F1 Score
    precision = sum(1 for hyp, ref in zip(hypotheses, references) 
                   if hyp.strip() in ref.strip()) / len(hypotheses) if hypotheses else 0
    recall = result['recall']
    result['f1'] = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return result

def _calculate_laplacian_pe(graph: nx.Graph, node_id: str) -> np.ndarray:
    """
    Placeholder: Calculates Laplacian Positional Encoding for a node.
    The actual LPE dimension should be self.graph_node_embedding_dim - self.code_embedding_dim.
    """
    
    return dgl.lap_pe(graph, k=5, padding=True)

def _parse_code_to_ast_graph(code_snippet: str) -> nx.Graph:
    """构建AST图，并为节点添加code属性"""

    # Node ID 计数器，使用列表使其可以在递归中被修改
    node_id_counter = [0]
    # 存储 parso 节点到 networkx 节点 ID 的映射，避免重复处理（主要用于更复杂的图，对树形结构非必需）
    # parso_to_nx_map = {} # 对于简单树形转换，递增ID即可

    def parso_to_networkx_recursive(p_node, nx_graph, parent_nx_id=None):
        """
        递归地将 parso 节点及其子节点添加到 NetworkX 图中。

        :param p_node: 当前 parso 节点。
        :param nx_graph: NetworkX DiGraph 对象。
        :param parent_nx_id: 父节点在 NetworkX 图中的 ID。
        """
        current_nx_id = node_id_counter[0]
        node_id_counter[0] += 1

        # 获取节点类型和对应的代码片段
        node_type = p_node.type
        try:
            node_code = p_node.get_code().strip()
        except Exception: # 有些非常底层的节点可能没有直接的 get_code() 或返回空
            node_code = f"<{node_type}>"


        # 向 NetworkX 图中添加节点
        nx_graph.add_node(current_nx_id, type=node_type, code=node_code, parso_type=str(type(p_node)))

        # 如果有父节点，则添加从父节点到当前节点的边
        if parent_nx_id is not None:
            nx_graph.add_edge(parent_nx_id, current_nx_id)

        # 递归处理子节点
        if hasattr(p_node, 'children') and p_node.children:
            for child_node in p_node.children:
                # 可以选择性地过滤掉一些不重要的节点类型，如 'newline', 'indent', 'dedent'
                # if child_node.type not in ('newline', 'indent', 'dedent', 'endmarker'):
                parso_to_networkx_recursive(child_node, nx_graph, current_nx_id)
        
        # 对于某些特定类型的节点，它们可能没有 .children 属性，
        # 而是将子元素直接作为属性（例如 Name 节点的 value 属性是字符串，不是子 Parso 节点）
        # 这个基本版本主要关注 .children 属性。
    
    G = nx.DiGraph()
    module_node = parso.parse(code_snippet)
    # 从 parso 的根节点开始转换
    parso_to_networkx_recursive(module_node, G)
    return G

def filter_llm_response(code:str, language:str="python"): 
    #llm的回复可能不是那么标准，例如，首行可能会出现“'''”或者“python”，或者“the next code is:”这种不规范的输出
    # 我们需要将这种输出给过滤一下。

    # check if the language is valid
    assert language in ["python", "java"], "language must be one of [python, java]"


    # first remove the \n at the beginning of the code
    code = code.lstrip('\n')

    lines = code.split('\n')
    
    # --- Start: Added noise filtering ---
    filtered_lines = []
    start_processing = False
    # Define patterns for noise lines to ignore at the beginning
    # Matches lines like: ```, ```python, python, java, ''', """, the next code is:
    noise_patterns = [
        re.compile(r"^\s*```(\s*\w*)?\s*$"), # Code fence start (optional language)
        re.compile(r"^\s*(python|java)\s*$", re.IGNORECASE), # Language name only
        re.compile(r"^\s*('''|\"\"\")\s*$"), # Triple quotes only
        re.compile(r"^\s*the next code is:", re.IGNORECASE), # Introductory phrase
    ]
    
    line_index = 0
    while line_index < len(lines):
        line = lines[line_index]
        stripped_line = line.strip()

        # If we already started processing, keep the line
        if start_processing:
            filtered_lines.append(line)
            line_index += 1
            continue
            
        # Skip empty lines before code starts
        if not stripped_line:
            line_index += 1
            continue

        # Check if the line matches any noise pattern
        is_noise = False
        for pattern in noise_patterns:
            if pattern.match(stripped_line):
                is_noise = True
                break
        
        if is_noise:
            # It's a noise line, skip it
            line_index += 1
            continue
        else:
            # Not noise, not empty -> start processing from here
            start_processing = True
            filtered_lines.append(line)
            line_index += 1
            
    # If all lines were filtered out as noise or empty
    if not filtered_lines:
        # Return empty string or maybe the original first line if that's preferred?
        # Let's return empty string for now.
        return ""
        
    # --- End: Added noise filtering ---

    # Now process the filtered lines
    lines = filtered_lines # Use the filtered lines for the rest of the logic
    in_multiline_comment = False

    if language == "python":
        for line in lines:
            # if the line is empty, then skip (already handled by initial filter, but keep for safety)
            if not line.strip():
                continue
            # if the line is a start of a multiline comment, then set the in_multiline_comment to True and skip
            # Check if it STARTS with quotes but isn't JUST quotes (handled by noise filter)
            if not in_multiline_comment and (line.strip().startswith('"""') or line.strip().startswith("'''")):
                 # Check if it also ends on the same line and isn't just the quotes
                if not (line.strip().endswith('"""') or line.strip().endswith("'''")) or len(line.strip()) <= 3:
                    in_multiline_comment = True
                # If it's a single-line docstring/comment like """ comment """, don't enter multi-line mode
                continue 
            # if the line is the end of a multiline comment, then set the in_multiline_comment to False and skip
            if in_multiline_comment and (line.strip().endswith('"""') or line.strip().endswith("'''")):
                in_multiline_comment = False
                continue
            # if the line is in a multiline comment, then skip
            if in_multiline_comment:
                continue
            # if the line is a single line comment, then skip
            if line.strip().startswith('#'):
                continue
            # if the line is not a comment, then return the line
            return line

    elif language == "java":
        for line in lines:
            # if the line is empty, then skip
            if not line.strip():
                continue
            # if the line is a start of a multiline comment, then set the in_multiline_comment to True and skip
            if not in_multiline_comment and line.strip().startswith('/*'):
                 # Check if it also ends on the same line
                if not line.strip().endswith('*/') or len(line.strip()) <= 2:
                     in_multiline_comment = True
                # If it's a single-line block comment /* comment */, don't enter multi-line mode
                continue
            # if the line is the end of a multiline comment, then set the in_multiline_comment to False and skip
            if in_multiline_comment and line.strip().endswith('*/'):
                in_multiline_comment = False
                continue
            # if the line is in a multiline comment, then skip
            if in_multiline_comment:
                continue
            # if the line is a single line comment, then skip
            if line.strip().startswith('//'):
                continue
            # if the line is not a comment, then return the line
            return line


    # if we cannot find a line that is not a comment or noise, return the first line of the *filtered* list
    # (or empty string if filtering removed everything)
    return lines[0] if lines else ""


# def retrieved_graph_serialized(self, retrieved_graph: nx.DiGraph) -> List[str]:
#     """
#     Serialize the retrieved graph into a list of code snippets for the LLM.
    
#     Args:
#         retrieved_graph: The fused graph with query and snippets
        
#     Returns:
#         List[str]: Serialized code snippets from the graph
#     """
#     serialized = []
    
#     # Add node code attributes to the serialized output
#     for node, data in retrieved_graph.nodes(data=True):
#         if 'code' in data:
#             code = data['code']
#             # Add node type prefix if available
#             if 'type' in data:
#                 prefix = f"[{data['type']}] "
#             else:
#                 prefix = ""
            
#             serialized.append(f"{prefix}{code}")
    
#     # Optionally, we could add edge information as well
#     # for u, v, data in retrieved_graph.edges(data=True):
#     #     if 'weight' in data and data['weight'] > 0.7:  # Only add high-confidence edges
#     #         serialized.append(f"Relationship: {u} -> {v} (confidence: {data['weight']:.2f})")
    
#     return serialized

class LLMModel:
    def __init__(self, model_name: str="gpt3.5"):
        openai_api_key = os.getenv("OPENAI_API_KEY")
        openai_base_url = os.getenv("OPENAI_BASE_URL")
        if model_name != "gpt3.5":
            pass
        else:
            pass

    def gpt_complete(self, inputs, contexts):
        url = "https://api.chatanywhere.tech/v1/chat/completions"
        OPENAI_KEY = os.getenv("OPENAI_API_KEY")
        payload = json.dumps({
        # "model": "gpt-3.5-turbo",
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
        
        # print(response_dict)
        # 获取生成的内容并只返回第一行
        generated_content = response_dict['choices'][0]['message']['content']
        next_line = generated_content.split('\n')[0].strip()
        return generated_content


    def format(self, inputs, contexts):
        return f"Given following context: {contexts} and your need to complete following {inputs} in one line:"
    
    def get_embeddings(self, text):
        """Get embeddings for a text using a language model or a mock for testing"""
        # TODO 这里需要换成其他的 code LLM 对于 code 进行 embedding，例如 codeT5,看一下 repohyper 用的啥
        import numpy as np
        import torch
        from hashlib import md5
        
        # 使用哈希方法创建一个确定性的但看似随机的嵌入向量
        hash_object = md5(text.encode())
        hash_hex = hash_object.hexdigest()
        
        # 将哈希字符串转换为数字序列
        hash_ints = [int(hash_hex[i:i+2], 16) for i in range(0, len(hash_hex), 2)]
        
        # 生成一个嵌入向量（使用哈希值初始化一个随机数生成器）
        np.random.seed(sum(hash_ints))
        embedding = torch.tensor(np.random.randn(1, 768)).float() # 假设维度是768
        
        # 归一化嵌入向量
        embedding = embedding / torch.norm(embedding)
        
        return embedding
    
    def _encode_text_netease(self, text: str) -> np.ndarray:
        """使用网易API的embedding模型编码文本
        
        Args:
            text: 输入文本
            
        Returns:
            np.ndarray: 文本的向量表示
        """
        url = "https://api.siliconflow.cn/v1/embeddings"
    
        payload = {
            "input": text,
            "model": "netease-youdao/bce-embedding-base_v1",
            "encoding_format": "float"
        }
        headers = {
            "Authorization": "Bearer " + os.getenv("SILICIONFLOW_API_KEY", ""),
            "Content-Type": "application/json"
        }
        
        response = requests.post(url, json=payload, headers=headers)
        if response.status_code == 200:
            return np.array(response.json()['data'][0]['embedding'])
        else:
            raise RuntimeError(f"Failed to get embedding from Netease API: {response.text}")
    
    def load_codet5(self, checkpoint: str = "Salesforce/codet5p-110m-embedding"):
        """
        Load the CodeT5 model from the specified checkpoint.
        
        Args:
            checkpoint (str): Path to the CodeT5 model checkpoint.
            
        Returns:
            Tuple[AutoTokenizer, AutoModel]: The tokenizer and model.
        """
        device = "cuda:1" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
        model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True).to(device)
        return tokenizer, model

    def _encode_text_codet5(self, text: str, tokenizer, model) -> np.ndarray:
        """
        Encodes text using the loaded CodeT5 model to get embeddings.

        Args:
            text: The input text (code snippet).

        Returns:
            np.ndarray: The embedding vector for the text.
        """

        with torch.no_grad(): # Ensure no gradients are calculated
            device = "cuda:1" if torch.cuda.is_available() else "cpu"
            inputs = tokenizer.encode(text, return_tensors="pt").to(device)
            embedding = model(inputs)[0]

            return embedding.cpu().numpy().astype(np.float32)


class GraphRAGPipeline:
    def __init__(
        self,
        args,
        model_name: str,
        index_path: str,
        dim: int = 768,
        ef_construction: int = 400,
        M: int = 64
    ):
        self.model = LLMModel(model_name)
        self.dim = dim
        self.index_path = Path(index_path)
        self.args = args
        
        # # 只创建索引主目录，不再初始化全局索引
        # # 现在每个仓库都有自己的索引目录和文件
        # if not self.index_path.exists():
        #     Path(self.index_path).mkdir(parents=True, exist_ok=True)
        #     print(f'Created main index directory at {self.index_path}')
    
    # def mean_encode_graph(self, graph: nx.DiGraph) -> torch.Tensor:
    #     """Encode graph structure into embeddings by averaging node embeddings"""
    #     # NOTE 现在使用所有节点的 embedding 的 mean 作为 graph 的 embedding
    #     # Get all nodes from the graph
    #     nodes = list(graph.nodes())
        
    #     if not nodes:
    #         # Return zero embedding if graph is empty
    #         return torch.zeros(1, self.dim)
        
    #     # Get node attributes if they exist, otherwise encode node names
    #     node_embeddings = []
    #     for node in nodes:
    #         if 'embedding' in graph.nodes[node]:
    #             emb = graph.nodes[node]['embedding']
    #             if isinstance(emb, (list, np.ndarray)):
    #                 emb = torch.tensor(emb)
    #             node_embeddings.append(emb)
    #         else:
    #             node_content = str(node)
    #             if 'content' in graph.nodes[node]:
    #                 node_content = graph.nodes[node]['content']
    #             node_emb = self.encode_code(node_content)
    #             node_embeddings.append(node_emb)
        
    #     # Stack all embeddings and compute mean
    #     node_embeddings = torch.stack(node_embeddings)
    #     graph_embedding = torch.mean(node_embeddings, dim=0, keepdim=True)
        
    #     return graph_embedding
    
    # def encode_graph(self, graph: nx.Graph) -> np.ndarray:
    #     """ TODO 有大问题
    #     1. 对节点中的代码/标识符进行语义嵌入 (使用 self.encode_code)。
    #     2. 计算结构化嵌入 (例如，如果图足够复杂，可以使用 LPE；或者通过图神经网络 GNN 学习)。
    #     3. 将这些嵌入聚合为一个单一的向量代表整个图。

    #     参数:
    #         graph (nx.Graph): 输入的图。

    #     返回:
    #         np.ndarray: 图的嵌入向量 (目前是基于简单规则和随机数的占位符)。
    #     """
    #     if graph.number_of_nodes() == 0: # 检查图是否有节点
    #         logging.warning("Graph is empty. Returning zero vector.")
    #         return np.zeros(self.graph_node_embedding_dim, dtype='float32')

    #     # 提取图中所有节点 'code' 属性的值，如果不存在则为空字符串 TODO 这里有大问题，是不是应该，使用code_lines的 embedding 来替代呢？
    #     node_codes = [data.get('code', "") for _, data in graph.nodes(data=True) if data.get('code')]
    #     # 初始化语义嵌入向量为一个符合代码嵌入维度的零向量
    #     semantic_emb_np = np.zeros(self.code_embedding_dim, dtype='float32')
    #     if not node_codes: # 如果没有从图节点中提取到代码
    #         logging.warning("No 'code' attribute found in graph nodes for semantic embedding.")
    #     else:
    #         # (简化处理) 只对第一个节点的代码进行编码
    #         semantic_emb_tensor = self.encode_code(node_codes[0])
    #         # 将 PyTorch 张量转换为 NumPy 数组，并展平为一维，确保类型为 float32
    #         temp_semantic_emb_np = semantic_emb_tensor.cpu().numpy().flatten().astype('float32')
    #         # 保证语义嵌入向量的维度与 self.code_embedding_dim 一致 (填充或截断)
    #         copy_len = min(temp_semantic_emb_np.shape[0], self.code_embedding_dim)
    #         semantic_emb_np[:copy_len] = temp_semantic_emb_np[:copy_len]

    #     # 计算结构嵌入部分应有的维度
    #     lpe_dim_part = self.graph_node_embedding_dim - self.code_embedding_dim
    #     # 初始化结构嵌入向量
    #     structural_emb_np = np.array([], dtype=np.float32)
    #     if lpe_dim_part > 0: # 如果结构嵌入维度大于0
    #         # 生成随机的结构嵌入作为占位符
    #         structural_emb_np = np.random.rand(lpe_dim_part).astype('float32')
        
    #     # 准备合并语义嵌入和结构嵌入
    #     combined_emb_list = []
    #     if semantic_emb_np.size > 0: # 如果语义嵌入非空
    #         combined_emb_list.append(semantic_emb_np)
    #     if structural_emb_np.size > 0: # 如果结构嵌入非空
    #         combined_emb_list.append(structural_emb_np)

    #     # 拼接语义和结构嵌入
    #     combined_emb = np.concatenate(combined_emb_list).astype('float32')
        
    #     # 创建最终的图嵌入向量，并用0初始化
    #     final_emb = np.zeros(self.graph_node_embedding_dim, dtype='float32')
    #     # 保证最终嵌入向量的维度与 self.graph_node_embedding_dim 一致
    #     copy_len = min(combined_emb.shape[0], self.graph_node_embedding_dim)
    #     final_emb[:copy_len] = combined_emb[:copy_len]
        
    #     # 维度检查，确保最终嵌入维度正确
    #     if final_emb.shape[0] != self.graph_node_embedding_dim:
    #          logging.error(f"Critical dimension mismatch in _embed_query_ast_graph: expected {self.graph_node_embedding_dim}, got {final_emb.shape[0]}")
            
    #     return final_emb

    # def old_encode_code(self, code: str) -> torch.Tensor:
    #     """Encode code snippet into embeddings using the language model"""
    #     try:
    #         if not isinstance(code, str):
    #             code = str(code) if code is not None else ""
    #         return self.model.get_embeddings(code)
    #     except Exception as e:
    #         print(f"Error encoding code: {e}")
    #         # 返回一个零向量作为安全涵盖
    #         return torch.zeros(1, self.dim)


    def _chunk_code(self, file_path: Path, chunk_size: int = 50, overlap: int = 5):
        """
        Chunks code from a file. Simple line-based chunking.
        Yields tuples of (chunk_content, chunk_id).
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except Exception as e:
            logging.warning(f"Could not read file {file_path}: {e}")
            return

        for i in range(0, len(lines), chunk_size - overlap):
            chunk_lines = lines[i : i + chunk_size]
            if not chunk_lines:
                continue
            chunk_content = "".join(chunk_lines)
            start_line = i + 1
            end_line = i + len(chunk_lines)
            chunk_id = f"{file_path}:{start_line}-{end_line}"
            yield chunk_content, chunk_id

    def encode_code(self, code: str) -> torch.Tensor:
        """
        使用提供的语言模型将代码片段编码为嵌入向量。

        参数:
            code (str): 需要编码的代码片段字符串。

        返回:
            torch.Tensor: 代码的嵌入向量 (PyTorch 张量)。
        """
        try:
            # 类型检查与转换：确保输入是字符串
            if not isinstance(code, str):
                code = str(code) if code is not None else "" # 如果非字符串，尝试转换；如果是None，则为空字符串
            
            # 获取模型所在的设备 (如 'cpu' 或 'cuda')，如果模型没有 device 属性，则默认为 'cpu'
            device = self.model.device if hasattr(self.model, 'device') else 'cpu'

            if not code.strip(): # 如果代码去除首尾空格后为空
                # 返回一个符合代码嵌入维度的零向量
                return torch.zeros(self.code_embedding_dim, device=device)

            # 使用模型的 get_embeddings 方法获取嵌入
            embedding = self.model.get_embeddings(code)
            
            # 如果嵌入是二维的且第一个维度是1 (例如形状是 [1, dim])，则去除多余的维度，变为一维 [dim]
            if embedding.ndim == 2 and embedding.shape[0] == 1:
                embedding = embedding.squeeze(0)
            
            # 维度检查与调整：确保嵌入维度与 self.code_embedding_dim 一致
            if embedding.shape[0] != self.code_embedding_dim:
                # 如果维度不匹配，创建一个目标维度的零向量
                final_embedding = torch.zeros(self.code_embedding_dim, device=embedding.device)
                # 计算需要复制的长度 (取原始嵌入长度和目标长度的较小者)
                copy_len = min(embedding.shape[0], self.code_embedding_dim)
                # 复制内容
                final_embedding[:copy_len] = embedding[:copy_len]
                return final_embedding
            return embedding # 返回维度正确的嵌入
        except Exception as e: # 捕获编码过程中可能发生的任何异常
            logging.error(f"Error encoding code: '{code[:50]}...': {e}", exc_info=True) # 记录错误信息，包括代码片段前50字符和异常堆栈
            # 出错时，获取设备信息并返回一个零向量
            device = self.model.device if hasattr(self.model, 'device') else 'cpu'
            return torch.zeros(self.code_embedding_dim, device=device)


    def _encode_graph_nodes(self,graph: nx.Graph, llm_tools: LLMModel, tokenizer, model, calculate_lappe: bool = True, lappe_k: int = 5) -> (np.ndarray, list):
        dgl_graph = dgl.from_networkx(graph)
        lappe_embeddings = dgl.lap_pe(dgl_graph, k=5, padding=True)
        i = 0
        graph_node_embeddings_for_hnsw = np.array([]) # 存储图节点嵌入向量，用于添加到 HNSW 索引
        graph_node_ids = [] # 存储图节点ID，用于映射到HNSW索引
        graph_node_snippets = [] # Added this line
        for node_id, node_data in graph.nodes.items():
            node_emb = llm_tools._encode_text_codet5(node_data['code'], tokenizer, model) # Use CodeT5
            struct_emb = lappe_embeddings[i] # Use LapPE # NOTE 后面再检查一下这个 index 是不是正确
            i += 1
            final_emb = np.concatenate((node_emb, struct_emb)).astype(np.float32)
            # graph_node_embeddings_for_hnsw.append(final_emb)
            if graph_node_embeddings_for_hnsw.size != 0:
                graph_node_embeddings_for_hnsw = np.concatenate([graph_node_embeddings_for_hnsw,[final_emb]], axis=0)
            else:
                graph_node_embeddings_for_hnsw = np.array([final_emb])
            graph_node_ids.append(node_id) # Store the original node ID
            graph_node_snippets.append(node_data['code']) # Added this line
        return graph_node_embeddings_for_hnsw, graph_node_ids, graph_node_snippets # Added graph_node_snippets to return


    def build_index(self, graph_path: str, repo_name: str, repo_path: Path, index_save_path: str, code_snippet: str = None):
        """
        Build HNSW indexes for code and graph embeddings from processed graph files
        
        Args:
            graph_path: 图结构文件路径
            repo_name: 仓库名称
            repo_path: 仓库路径
            index_save_path: 索引保存路径
            code_snippet: 数据集中的代码片段
        """
        # 创建索引保存目录
        Path(index_save_path).mkdir(parents=True, exist_ok=True)
        
        # 针对这个仓库初始化新的索引
        
        
        # 读取处理好的图结构（graphml格式）
        graph_file_path = Path(graph_path) / "repo_multi_graph.graphml"
        if not graph_file_path.exists(): # 检查文件是否存在
            logging.error(f"GraphML file not found at {graph_file_path}")
            return # 文件不存在则直接返回

        combined_graph = nx.read_graphml(graph_file_path)
        print(f'Successfully loaded graph from {graph_file_path}')

        # 转换为 DGL 图
        # dgl_graph = dgl.from_networkx(combined_graph, node_attrs=['code', 'graph_type', 'original_id'])
        dgl_graph = dgl.from_networkx(combined_graph)
        # 计算Laplacian PE
        lappe_embeddings = dgl.lap_pe(dgl_graph, k=5, padding=True)

        # 初始化用于存储元数据和嵌入向量的列表
        code_metadata_list = []         # 存储代码行元数据
        graph_node_metadata_list = []   # 存储图节点元数据
        
        code_embeddings_for_hnsw = []   # 存储代码行嵌入向量，用于添加到 HNSW 索引
        
        current_code_id = 0         # HNSW 索引中代码项的当前ID计数器
        current_graph_node_id = 0   # HNSW 索引中图节点项的当前ID计数器

        # 构建 graph_index
        # 遍历图中的每一个节点及其关联数据
        i = 0
        llm_tools = LLMModel()
        tokenizer, model = llm_tools.load_codet5()

        graph_node_embeddings_for_hnsw, graph_node_ids, graph_node_snippets = self._encode_graph_nodes(combined_graph, llm_tools, tokenizer, model)

        # 4. Build HNSW index for graph nodes
        if graph_node_embeddings_for_hnsw.size != 0:
            graph_index = hnswlib.Index(space='cosine', dim=graph_node_embeddings_for_hnsw.shape[-1])
            graph_index.init_index(max_elements=1000000, ef_construction=400, M=64)
            graph_index.add_items(graph_node_embeddings_for_hnsw, np.arange(len(graph_node_embeddings_for_hnsw))) # Use sequential IDs for HNSW

            # Save the graph index and the mapping from HNSW ID to original node ID
            graph_index_file = f"{index_save_path}/{repo_name}_graph_index.bin"
            graph_map_file = f"{index_save_path}/{repo_name}_graph_map.pkl"
            graph_snippets_file = f"{index_save_path}/{repo_name}_graph_node_snippets.pkl" # Added this line
            graph_index.save_index(str(graph_index_file))
            with open(graph_map_file, 'wb') as f_map:
                pickle.dump(graph_node_ids, f_map) # Save the list mapping index 0..N-1 to original node IDs
            with open(graph_snippets_file, 'wb') as f_snippets: # Added this block
                pickle.dump(graph_node_snippets, f_snippets) # Added this line
            logging.info(f"Graph index saved to {graph_index_file}")
            logging.info(f"Graph node ID map saved to {graph_map_file}")
            logging.info(f"Graph node snippets saved to {graph_snippets_file}") # Added this line
        else:
            logging.warning("No valid graph node embeddings generated. Skipping graph index creation.")

        
        # 5. Find source files, chunk, embed, and add to index
        # Assuming Python files for now
        logging.info("Starting code chunk index building...")
        code_chunk_embeddings = np.array([])
        code_chunk_ids = []
        all_code_snippets = [] # Added this line

        source_files = list(repo_path.rglob('*.py')) 
        logging.info(f"Found {len(source_files)} Python files for code chunking.")

        for py_file in source_files:
            for chunk_content, chunk_id in self._chunk_code(py_file):
                # Compute semantic embedding for the code chunk
                chunk_emb = llm_tools._encode_text_codet5(chunk_content, tokenizer, model) # Use CodeT5
                if code_chunk_embeddings.size != 0:
                    code_chunk_embeddings = np.concatenate([code_chunk_embeddings,[chunk_emb]], axis=0)
                else:
                    code_chunk_embeddings = np.array([chunk_emb])
                code_chunk_ids.append(chunk_id) # Store file:start-end ID
                all_code_snippets.append(chunk_content) # Added this line

        # 6. Build HNSW index for code chunks
        if code_chunk_embeddings.size != 0:
            code_index = hnswlib.Index(space='cosine', dim=chunk_emb.shape[-1])
            code_index.init_index(max_elements=1000000, ef_construction=400, M=64)
            code_index.add_items(code_chunk_embeddings, np.arange(len(code_chunk_embeddings))) # Use sequential IDs

            # Save the code index and the mapping from HNSW ID to chunk ID
            code_index_file = f"{index_save_path}/{repo_name}_code_index.bin"
            code_map_file = f"{index_save_path}/{repo_name}_code_map.pkl"
            code_snippets_file = f"{index_save_path}/{repo_name}_code_snippets.pkl" # Added this line
            code_index.save_index(str(code_index_file))
            with open(code_map_file, 'wb') as f_map:
                 pickle.dump(code_chunk_ids, f_map) # Save list mapping index 0..N-1 to chunk IDs
            with open(code_snippets_file, 'wb') as f_snippets: # Added this block
                pickle.dump(all_code_snippets, f_snippets) # Added this line
            logging.info(f"Code index saved to {code_index_file}")
            logging.info(f"Code chunk ID map saved to {code_map_file}")
            logging.info(f"Code snippets saved to {code_snippets_file}") # Added this line
        else:
            logging.warning("No valid code chunk embeddings generated. Skipping code index creation.")
            
        logging.info(f"Index building finished for repo: {repo_name}")


    def retrieve(
        self,
        query_code: str,
        query_graph: nx.DiGraph,
        repo_name: str,
        k: int = 5,
        alpha: float = 0.5,
        llm_tools: LLMModel = None,
        tokenizer = None,
        model = None,
        sample_idx: int = None
    ) -> Tuple[List[str], List[float]]:
        """Retrieve relevant code snippets using both code and graph similarity from a specific repository"""
        # 确定要使用的仓库索引路径
        repo_sample_name = f'{sample_idx}_{repo_name}' if sample_idx is not None else repo_name
        
        # 仓库名称路径
        repo_index_path = Path(self.args.index_path) / repo_sample_name
        
        # 检查所需的索引文件是否存在
        if not (repo_index_path / f'{repo_name}_code_index.bin').exists() or not (repo_index_path / f'{repo_name}_graph_index.bin').exists():
            print(f'Index files missing for repository {repo_name} at {repo_index_path}')
            return [], []
        
        # 加载该仓库的索引
        code_index = hnswlib.Index(space='cosine', dim=256)
        graph_index = hnswlib.Index(space='cosine', dim=261)

        code_index.load_index(str(repo_index_path / f'{repo_name}_code_index.bin'))
        graph_index.load_index(str(repo_index_path / f'{repo_name}_graph_index.bin'))
                
        # 获取查询嵌入向量
        code_emb = llm_tools._encode_text_codet5(query_code, tokenizer, model) # Use CodeT5
        graph_emb, _, _ = self._encode_graph_nodes(query_graph, llm_tools, tokenizer, model)
        
        # 在两个索引中搜索
        k = min(k, graph_index.get_current_count())
        k = min(k, code_index.get_current_count())
        code_ids, code_distances = code_index.knn_query(code_emb, k=k)
        graph_ids, graph_distances = graph_index.knn_query(np.mean(graph_emb, axis=0), k=k)
        
        # 加载代码片段和图节点片段
        code_snippets_file = f"{repo_index_path}/{repo_name}_code_snippets.pkl"
        graph_snippets_file = f"{repo_index_path}/{repo_name}_graph_node_snippets.pkl"
        
        # 加载代码片段
        all_code_snippets = []
        if os.path.exists(code_snippets_file):
            try:
                with open(code_snippets_file, 'rb') as f_snippets:
                    all_code_snippets = pickle.load(f_snippets)
                print(f"Loaded {len(all_code_snippets)} code snippets from {code_snippets_file}")
            except Exception as e:
                print(f"Error loading code snippets: {e}")
        else:
            print(f"Warning: Code snippets file not found: {code_snippets_file}")
        
        # 加载图节点片段
        all_graph_snippets = []
        if os.path.exists(graph_snippets_file):
            try:
                with open(graph_snippets_file, 'rb') as f_snippets:
                    all_graph_snippets = pickle.load(f_snippets)
                print(f"Loaded {len(all_graph_snippets)} graph snippets from {graph_snippets_file}")
            except Exception as e:
                print(f"Error loading graph snippets: {e}")
        else:
            print(f"Warning: Graph snippets file not found: {graph_snippets_file}")
        

        
        # 分别处理代码和图的检索结果
        code_results = []
        for i in range(len(code_ids[0])):
            code_idx = code_ids[0][i]
            # 直接使用索引访问代码片段
            if code_idx < len(all_code_snippets):
                code_content = all_code_snippets[code_idx]
                code_results.append({
                    'id': str(code_idx),  # 使用索引作为 ID
                    'content': code_content,
                    'score': alpha * (1 - code_distances[0][i]),
                    'type': 'code'
                })
        
        graph_results = []
        for i in range(len(graph_ids[0])):
            graph_idx = graph_ids[0][i]
            # 直接使用索引访问图片段
            if graph_idx < len(all_graph_snippets):
                graph_content = all_graph_snippets[graph_idx]
                graph_results.append({
                    'id': str(graph_idx),  # 使用索引作为 ID
                    'content': graph_content,
                    'score': (1 - alpha) * (1 - graph_distances[0][i]),
                    'type': 'graph'
                })
        
        print(f'Retrieved {len(code_results)} code snippets and {len(graph_results)} graph snippets for {repo_name}')
        
        # 重排序：不再使用混合权重，而是分别对code和graph的结果进行排序，然后交替选择
        # 这样能保证结果中既有代码检索也有图检索的结果
        code_results.sort(key=lambda x: x['score'], reverse=True)
        graph_results.sort(key=lambda x: x['score'], reverse=True)
        
        # 交替选择结果
        merged_results = []
        for i in range(k):
            if i < len(code_results):
                merged_results.append(code_results[i])
            if i < len(graph_results):
                merged_results.append(graph_results[i])
        
        # 根据总分再次排序并截取前k个
        merged_results.sort(key=lambda x: x['score'], reverse=True)
        top_results = merged_results[:k] if len(merged_results) >= k else merged_results
        
        retrieved_snippets = []
        scores = []
        
        # 处理排序后的结果
        for result in top_results:
            retrieved_snippets.append(result['content'])
            scores.append(result['score'])
        
        return retrieved_snippets, scores
    
    def graph_fusion(
        self,
        snippets: List[str],
        query_graph: nx.DiGraph,
        fusion_type: str = 'attention',
        llm_tools: LLMModel = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        model: Optional[PreTrainedModel] = None
    ) -> torch.Tensor:
        """Fuse code and graph embeddings using different fusion strategies"""
        if fusion_type == 'concat':
            # Simple concatenation followed by linear projection
            fused = torch.cat([code_emb, graph_emb], dim=-1)
            # Project back to original dimension
            projection = torch.nn.Linear(self.dim * 2, self.dim).to(code_emb.device)
            return projection(fused)
            
        elif fusion_type == 'attention':
            # 两部分：emb 融合+graph 图结构，使用邻接矩阵
            # TODO 这里应该是 query_graph 与 retrieved_graph 的图结构融合，不是 code_emb和graph_emb计算 attention
            # 1. emb融合
            # Calculate cross-attention between query and retrieved graph embeddings
            # Convert graph structures to node feature matrices
            # 1. Build snippets graph
            snippets_graph = nx.DiGraph()
            for i, snippet in enumerate(snippets):
                snippets_graph.add_node(i, code=snippet)

            snippet_embeddings = []
            
            for i, snippet in enumerate(snippets):
                emb = llm_tools._encode_text_codet5(snippet, tokenizer, model)  # Get code embedding
                if isinstance(emb, np.ndarray):
                    emb = torch.from_numpy(emb).to(torch.float32)
                snippet_embeddings.append(emb)
            
            # Create edges between similar snippets (similarity > 0.5)
            for i in range(len(snippets)-1):
                for j in range(i+1, len(snippets)):
                    sim = torch.cosine_similarity(snippet_embeddings[i], snippet_embeddings[j], dim=0)
                    if sim > 0.5:
                        snippets_graph.add_edge(f"snippet_{i}", f"snippet_{j}", weight=sim.item())
                        snippets_graph.add_edge(f"snippet_{j}", f"snippet_{i}", weight=sim.item())
            
            # 2. Encode query graph nodes
            query_node_embeddings = {}
            for node in query_graph.nodes():
                if 'code' in query_graph.nodes[node]:
                    code = query_graph.nodes[node]['code']
                    emb = llm_tools._encode_text_codet5(code, tokenizer, model)
                    if isinstance(emb, np.ndarray):
                        emb = torch.from_numpy(emb).to(torch.float32)
                    query_node_embeddings[node] = emb

            # 3. Create fusion graph by copying query_graph
            fused_graph = query_graph.copy()
            
            # 4. Compute cross-attention between query nodes and snippet nodes
            device = next(iter(query_node_embeddings.values())).device
            
            # Prepare query features
            query_nodes = list(query_node_embeddings.keys())
            query_features = torch.stack([query_node_embeddings[node] for node in query_nodes])
            
            # Prepare snippet features
            snippets_nodes = list(snippets_graph.nodes())
            snippets_features = torch.stack(snippet_embeddings)
            
            # Calculate cross-attention
            cross_attention = torch.matmul(query_features, snippets_features.transpose(-2, -1)) / np.sqrt(self.dim)
            attention_weights = torch.softmax(cross_attention, dim=-1)
            
            # 5. Add snippet nodes to fused graph based on attention weights
            for i, q_node in enumerate(query_nodes):
                for j, s_node in enumerate(snippets_nodes[:len(snippets_nodes) // 2]):
                    if attention_weights[i, j] > 0.5:  # Only add if similarity > 0.5
                        # Add snippet node to fused graph if not already there
                        s_node_id = f"snippet_{j}"
                        if s_node_id not in fused_graph:
                            fused_graph.add_node(s_node_id, code=snippets_graph.nodes[s_node]['code'],type='retrieved_snippet')
                        
                        # Add edge from query node to snippet node
                        fused_graph.add_edge(q_node, s_node_id,weight=attention_weights[i, j].item(),type='query_to_snippet')
            
            # 6. Add edges between snippets in the fused graph
            for u, v, data in snippets_graph.edges(data=True):
                if u in fused_graph and v in fused_graph:
                    fused_graph.add_edge(u, v, weight=data['weight'], type='snippet_to_snippet')
            
            return fused_graph
            
        elif fusion_type == 'sum':
            # Simple element-wise sum
            return code_emb + graph_emb
            
        else:
            raise ValueError(f'Unknown fusion type: {fusion_type}')
    
    def complete_code(
        self,
        code_context: str,
        retrieved_contexts: List[str]
    ) -> str:
        """Complete code using retrieved contexts and LLM"""
        pass
        
def load_jsonl(fname):
    with open(fname, 'r', encoding='utf8') as f:
        lines = []
        for line in f:
            lines.append(json.loads(line))
        return lines

def convert(obj):
    if isinstance(obj, dict):
        return {k: convert(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert(i) for i in obj]
    elif isinstance(obj, np.float32) or isinstance(obj, np.float64):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    else:
        return obj

def main():
    
    parser = argparse.ArgumentParser(description='Run the Graph-guided RAG pipeline')
    parser.add_argument('--model', type=str, default='gpt3.5', help='Name of the LLM model to use')
    parser.add_argument('--repo-graph-path', type=str,  default='/data/wby/allcode/repohyper/GRACE/dataset/dataset_repoeval_updated/repoeval_to_repobench/graphs', help='Path to the repository')
    parser.add_argument('--repo-path', type=str,  default='/data/wby/allcode/repohyper/GRACE/dataset/dataset_repoeval_updated/repos', help='Path to the repository')
    parser.add_argument('--index-path', type=str,  default='/data/wby/allcode/repohyper/GRACE/dataset/dataset_repoeval_updated/repoeval_to_repobench/index', help='Path to store/load HNSW indexes')
    parser.add_argument('--fusion-type', type=str, default='attention', choices=['concat', 'attention', 'sum'],
                        help='Type of fusion to use for code and graph embeddings')
    parser.add_argument('--k', type=int, default=5, help='Number of neighbors to retrieve')
    parser.add_argument('--alpha', type=float, default=0.5, help='Weight for code vs graph similarity')
    parser.add_argument('--eval-dataset', type=str,default = "/data/wby/allcode/repohyper/GRACE/dataset/hf_datasets/repobench_python_v1.1" , help='Path to dataset for evaluation')
    parser.add_argument('--device', type=str, default='cuda:3' if torch.cuda.is_available() else 'cpu', help='Device to use for model')
    
    args = parser.parse_args()
    
    pipeline = GraphRAGPipeline(
        args = args,
        model_name=args.model,
        index_path=args.index_path
    )
    
    # 确保索引目录存在
    Path(args.index_path).mkdir(parents=True, exist_ok=True)

    #原始
    # 加载RepoBench数据集
    # dataset = load_dataset(args.eval_dataset, split=['cross_file_first'])[0]
    # 加载repoeval-updated数据集
    repobench_root = Path(f"/data/wby/allcode/repohyper/GRACE/dataset/dataset_repoeval_updated/repoeval_to_repobench")
    repobench_path = Path(f"/data/wby/allcode/repohyper/GRACE/dataset/dataset_repoeval_updated/repos")

    dataset = load_jsonl(f"{repobench_root}/line_level.python.test.jsonl")
    dataset=dataset[:40]

    print(f'Loaded dataset with {len(dataset)} samples')
    
    # 初始化一个集合来跟踪已处理过的仓库
    # 遍历 index_path 目录，获取所有已存在索引的仓库
    processed_repos = set()

    
    # 遍历索引目录中的所有子目录（每个子目录代表一个仓库）
    for repo_dir in Path(args.index_path).iterdir():
        if repo_dir.is_dir():
            # 检查所需的索引文件是否存在
            # if (repo_dir / 'code_index.bin').exists() and (repo_dir / 'graph_index.bin').exists() and (repo_dir / 'code_snippets.json').exists():
            if (repo_dir / 'graph_index.bin').exists():
                # 将这个仓库添加到已处理列表
                processed_repos.add(repo_dir.name)
                print(f'Found existing index for repository: {repo_dir.name}')
    print(f'Found {len(processed_repos)} repositories with existing indexes')
    
    # 初始化评估指标
    total_samples = 0
    correct_samples = 0
    bleu_scores = []
    total_metric = 0.0
    results = []

    search_result = []
    # 遍历数据集中的每个样本
    for idx, sample in enumerate(dataset):
        
        #原始
        #repo_author, repo_name = sample['repo_name'].split('/')
        repo_name = sample['repo_name']
        repo_path = repobench_path / repo_name
        temp_repo_path = repobench_root / 'temp' / repo_name

        file_path = sample['file_path']
        code_context = sample['all_code']  # 使用all_code作为上下文
        next_line = sample['next_line']  # 这是我们要预测的行
        cropped_code = sample['cropped_code']
        all_code = sample['all_code']


        print(f'\nProcessing sample {idx+1}/{len(dataset)}: {repo_name} - {file_path}')
        
        # 构建索引
        try:
            if str(idx) + str("_") + repo_name not in processed_repos:
                processed_repos.add(str(idx) + str("_") + repo_name)
                repo_index_path = Path(args.index_path) / Path(str(idx) + str("_") + repo_name)
                print(f'Building index for idx {idx} repository: {repo_name}')
                # 获取仓库的图结构路径
                repo_graph_path = Path(args.repo_graph_path) / Path(str(idx) + str("_") + repo_name)
                if repo_graph_path.exists() and (repo_graph_path / "repo_multi_graph.graphml").exists():
                    print(f'Found processed graph for {repo_name} at {repo_graph_path}')
                    # 读取图结构并构建索引
                    pipeline.build_index(
                        str(repo_graph_path),
                        repo_name,
                        Path(args.repo_path + "/" + repo_name),
                        str(repo_index_path),
                        code_snippet=code_context  # 添加代码片段参数
                    )
                else:
                    print(f'Warning: No processed graph found for {repo_name} at {repo_graph_path}')

            # 根据仓库名确定仓库图路径
            repo_dir = Path(args.repo_graph_path) / Path(str(idx) + str("_") + repo_name)
            if not repo_dir.exists():
                print(f'Repository graph {repo_name} not found at {repo_dir}')
                continue

            # 预处理样本，过滤next_line及其后续内容
            # ?processed_sample = preprocess_repobench_data(sample, repo_path,temp_repo_path)
            # ?filtered_code = processed_sample['all_code']

            # 为当前样本构建图结构
            # ?graphs = process_repobench_repo(str(repo_dir), processed_sample)
            query_graph = _parse_code_to_ast_graph(cropped_code)

            llm_tools = LLMModel()
            tokenizer, model = llm_tools.load_codet5()
            # 检索相关代码片段
            snippets, scores = pipeline.retrieve(
                query_code=cropped_code,  # 使用过滤后的代码
                query_graph=query_graph,
                repo_name=repo_name,  # 添加仓库名
                k=args.k,
                alpha=args.alpha,
                llm_tools=llm_tools,
                tokenizer=tokenizer,
                model=model,
                sample_idx=str(idx)
            )

            # graph fusion

            # snippets = [
            #     'import numpy as np\nimport torch\nfrom torch import nn\nfrom lib.model.attention_block import SpatialAttention3D, ChannelAttention3D, residual_block\n\n\nclass DAM(nn.Module):\n    def __init__(self, channels=64):\n        super(DAM, self).__init__()\n\n        self.sa = SpatialAttention3D(out_channel=channels)\n        self.ca = ChannelAttention3D(in_planes=channels)\n\n    def forward(self, x):\n        residual = x\n        out = self.ca(x)\n        out = self.sa(out)\n        out = out + residual\n        return out\n\n\nclass Duo_Attention(nn.Module):\n    def __init__(\n            self, input_size=(1, 169, 208, 179), num_classes=3, dropout=0\n    ):\n        super().__init__()\n        self.conv = nn.Sequential(\n            nn.Conv3d(input_size[0], 8, 3, padding=1),\n            nn.BatchNorm3d(8),\n            nn.ReLU(),\n            # nn.MaxPool3d(2, 2),\n\n            nn.Conv3d(8, 16, 3, padding=1, stride=2),\n            nn.BatchNorm3d(16),\n            nn.ReLU(),\n            residual_block(channel_size=16),\n            nn.MaxPool3d(2, 2),\n\n            nn.Conv3d(16, 32, 3, padding=1, stride=2),\n            nn.BatchNorm3d(32),\n            nn.ReLU(),\n            residual_block(channel_size=32),\n            DAM(channels=32),\n            nn.MaxPool3d(2, 2),\n\n            nn.Conv3d(32, 64, 3, padding=1, stride=2),\n            nn.BatchNorm3d(64),\n            nn.ReLU(),\n            residual_block(channel_size=64),\n            nn.MaxPool3d(2, 2),\n',
            #     'import numpy as np\nimport torch\nfrom torch import nn\nfrom lib.model.attention_block import SpatialAttention3D, ChannelAttention3D, residual_block\n\n\nclass DAM(nn.Module):\n    def __init__(self, channels=64):\n        super(DAM, self).__init__()\n\n        self.sa = SpatialAttention3D(out_channel=channels)\n        self.ca = ChannelAttention3D(in_planes=channels)\n\n    def forward(self, x):\n        residual = x\n        out = self.ca(x)\n        out = self.sa(out)\n        out = out + residual\n        return out\n\n\nclass Duo_Attention(nn.Module):\n    def __init__(\n            self, input_size=(1, 169, 208, 179), num_classes=3, dropout=0\n    ):\n        super().__init__()\n        self.conv = nn.Sequential(\n            nn.Conv3d(input_size[0], 8, 3, padding=1),\n            nn.BatchNorm3d(8),\n            nn.ReLU(),\n            # nn.MaxPool3d(2, 2),\n\n            nn.Conv3d(8, 16, 3, padding=1, stride=2),\n            nn.BatchNorm3d(16),\n            nn.ReLU(),\n',
            #     'import numpy as np\nimport torch\nfrom torch import nn\nfrom lib.model.attention_block import SpatialAttention3D, ChannelAttention3D, residual_block\n\n\nclass DAM(nn.Module):\n    def __init__(self, channels=64):\n        super(DAM, self).__init__()\n\n        self.sa = SpatialAttention3D(out_channel=channels)\n        self.ca = ChannelAttention3D(in_planes=channels)\n\n    def forward(self, x):\n        residual = x\n        out = self.ca(x)\n        out = self.sa(out)\n        out = out + residual\n        return out\n\n\nclass Duo_Attention(nn.Module):\n    def __init__(\n            self, input_size=(1, 169, 208, 179), num_classes=3, dropout=0\n    ):\n        super().__init__()\n        self.conv = nn.Sequential(\n            nn.Conv3d(input_size[0], 8, 3, padding=1),\n            nn.BatchNorm3d(8),\n            nn.ReLU(),\n            # nn.MaxPool3d(2, 2),\n\n            nn.Conv3d(8, 16, 3, padding=1, stride=2),\n            nn.BatchNorm3d(16),\n            nn.ReLU(),\n',
            #     '1',
            #     '1'
            # ]

            fused_graph = pipeline.graph_fusion(
                snippets=snippets,
                query_graph=query_graph,
                fusion_type='attention',
                llm_tools=llm_tools,
                tokenizer=tokenizer,
                model=model
            )

            new_context = graph_text_encoders.encode_graph(fused_graph, node_encoder="integer", edge_encoder="incident")
            search_result.append({
                "repo_name": repo_name,
                "file_path": file_path,
                "all_code": all_code,
                "new_context": new_context,
                "next_line": next_line,
                "retrieved_snippets": snippets,
                "retrieval_scores": scores
            })
        except Exception as e:
            print(e)
            continue
    output_file = "/data/wby/allcode/repohyper/GRACE/dataset/dataset_repoeval_updated/repoeval_to_repobench/grace_serach/line_python_search_results.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        for item in search_result:
            f.write(json.dumps(convert(item), ensure_ascii=False) + "\n")


        # 使用检索到的片段完成代码
    #     completion = pipeline.complete_code(all_code, new_context)
    #
    #     # 评估结果
    #     is_correct = completion.strip() == next_line.strip()
    #     if is_correct:
    #         correct_samples += 1
    #
    #     # 计算代码BLEU得分
    #     bleu = calc_codebleu([completion], [next_line], lang="python")
    #     bleu_scores.append(bleu['codebleu'])
    #
    #     # 记录结果
    #     result = {
    #         'sample_id': idx,
    #         'repo_name': repo_name,
    #         'file_path': file_path,
    #         'filtered_code': filtered_code,
    #         'completion': completion,
    #         'next_line': next_line,
    #         'is_correct': is_correct,
    #         'bleu_score': bleu['codebleu'],
    #         'retrieved_snippets': snippets,
    #         'retrieval_scores': scores
    #     }
    #     results.append(result)
    #
    #     total_samples += 1
    #
    #     # 打印当前样本的结果
    #     print(f'Predicted: {completion}')
    #     print(f'Expected: {next_line}')
    #     print(f'Correct: {is_correct}, BLEU: {bleu["codebleu"]:.4f}')
    #     print('Retrieved contexts:')
    #     for snippet, score in zip(snippets[:3], scores[:3]):
    #         print(f'Score: {score:.3f}')
    #         print(snippet[:200] + '...' if len(snippet) > 200 else snippet)
    #         print('---')
    #
    #     # 每10个样本保存一次结果
    #     if idx % 10 == 0 and idx > 0:
    #         with open(f'results_{idx}.json', 'w') as f:
    #             json.dump(results, f, indent=2)
    #
    #
    # # 计算最终指标
    # accuracy = correct_samples / total_samples if total_samples > 0 else 0
    # avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0
    #
    # # 计算其他统计数据
    # processed_samples = len(results)
    # skipped_samples = len(dataset) - processed_samples
    # success_rate = processed_samples / len(dataset) if len(dataset) > 0 else 0
    #
    # print(f'\nEvaluation complete:')
    # print(f'Total samples: {total_samples}')
    # print(f'Accuracy: {accuracy:.4f}')
    # print(f'Average CodeBLEU: {avg_bleu:.4f}')
    #
    # # 保存最终结果
    # results_file = f'final_results_{Path(args.eval_dataset).stem}.json'
    # print(f'Saving results to {results_file}')
    # try:
    #     with open(results_file, 'w') as f:
    #         json.dump({
    #             'dataset': args.eval_dataset,
    #             'total_dataset_samples': len(dataset),
    #             'processed_samples': processed_samples,
    #             'skipped_samples': skipped_samples,
    #             'success_rate': success_rate,
    #             'total_completed_samples': total_samples,
    #             'correct_samples': correct_samples,
    #             'accuracy': accuracy,
    #             'avg_bleu': avg_bleu,
    #             'results': results
    #         }, f, indent=2)
    #     print(f'Results saved successfully')
    # except Exception as e:
    #     print(f'Error saving results: {e}')
        


if __name__ == '__main__':
    main()
