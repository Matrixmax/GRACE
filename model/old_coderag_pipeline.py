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
from datasets import load_dataset

from GRACE.data_process.multilevel_graph_builder import (
    MultiLevelGraph, 
    MultiLevelGraphBuilder, 
    process_repobench_repo, 
    preprocess_repobench_data
)

from codebleu import calc_codebleu

def calc_metrics(hypotheses, references):
    # TODO 计算指标：EM、ES、recall、F1
    # 1. EM 
    # 2. ES
    # 3. recall
    # 4. F1
    # 5. CodeBLEU
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

def get_real_first_line(code:str, language:str="python"):
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



class LLMModel:
    def __init__(self, model_name):
        openai_api_key = os.getenv("OPENAI_API_KEY")
        openai_base_url = os.getenv("OPENAI_BASE_URL")
        if model_name != "gpt3.5":
            pass
        else:
            pass

    def complete(self, inputs, contexts):
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



@dataclass
class GraphEmbedding:
    code_emb: torch.Tensor  # Code embeddings
    graph_emb: torch.Tensor  # Graph structure embeddings
    combined_emb: Optional[torch.Tensor] = None  # Combined embeddings after fusion
    
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
    
    def encode_graph(self, graph: nx.DiGraph) -> torch.Tensor:
        """Encode graph structure into embeddings by averaging node embeddings"""
        # NOTE 现在使用所有节点的 embedding 的 mean 作为 graph 的 embedding
        # Get all nodes from the graph
        nodes = list(graph.nodes())
        
        if not nodes:
            # Return zero embedding if graph is empty
            return torch.zeros(1, self.dim)
        
        # Get node attributes if they exist, otherwise encode node names
        node_embeddings = []
        for node in nodes:
            if 'embedding' in graph.nodes[node]:
                emb = graph.nodes[node]['embedding']
                if isinstance(emb, (list, np.ndarray)):
                    emb = torch.tensor(emb)
                node_embeddings.append(emb)
            else:
                node_content = str(node)
                if 'content' in graph.nodes[node]:
                    node_content = graph.nodes[node]['content']
                node_emb = self.encode_code(node_content)
                node_embeddings.append(node_emb)
        
        # Stack all embeddings and compute mean
        node_embeddings = torch.stack(node_embeddings)
        graph_embedding = torch.mean(node_embeddings, dim=0, keepdim=True)
        
        return graph_embedding
    
    def encode_code(self, code: str) -> torch.Tensor:
        """Encode code snippet into embeddings using the language model"""
        try:
            if not isinstance(code, str):
                code = str(code) if code is not None else ""
            return self.model.get_embeddings(code)
        except Exception as e:
            print(f"Error encoding code: {e}")
            # 返回一个零向量作为安全涵盖
            return torch.zeros(1, self.dim)
    
    def build_index_from_processed_graph(self, graph_path: str, repo_name: str, index_save_path: str, code_snippet: str = None):
        """Build HNSW indexes for code and graph embeddings from processed graph files
        
        Args:
            graph_path: 图结构文件路径
            repo_name: 仓库名称
            index_save_path: 索引保存路径
            code_snippet: 数据集中的代码片段
        """
        # 创建索引保存目录
        Path(index_save_path).mkdir(parents=True, exist_ok=True)
        
        # 针对这个仓库初始化新的索引
        code_index = hnswlib.Index(space='cosine', dim=self.dim)
        graph_index = hnswlib.Index(space='cosine', dim=self.dim)
        code_index.init_index(max_elements=1000000, ef_construction=400, M=64)
        graph_index.init_index(max_elements=1000000, ef_construction=400, M=64)
        
        # 读取处理好的图结构（graphml格式）
        graph_file_path = Path(graph_path) / "repo_multi_graph.graphml"
        combined_graph = nx.read_graphml(graph_file_path)
            
        print(f'Successfully loaded graph from {graph_file_path}')

        
        # 生成嵌入向量
        # code_embeddings = []
        # graph_embeddings = []
        
        # 应该是把code_snippet 的每一行，即 code_lines 做 emb。可以参考一下
        code_lines = code_snippet.lstrip('\n').split('\n')
        # outputs.logits.shapetorch.Size([29, 36, 32256])，29 行代码，32256 应该是词表？
        # tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-1.3b-base")
        # model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-coder-1.3b-base")
        # inputs = tokenizer(code_lines, padding=True, return_tensors="pt")
        # outputs = model.generate(**inputs, max_length=128)
        # print(tokenizer.decode(outputs[0], skip_special_tokens=True))
        # outputs = model(**inputs)
        # code_embeddings = outputs.last_hidden_state.mean(dim=1)
        # code_embeddings = outputs.last_hidden_state.mean(dim=1)
        checkpoint = "Salesforce/codet5p-110m-embedding"
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
        model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True).to(self.args.device)

        # 对于graph来说，应该是获得每一个节点的 embedding

        code_embeddings = []
        # TODO 对 code_lines 做 embedding
        for code in code_lines:
            # 编码代码片段
            inputs = tokenizer.encode(code, return_tensors="pt").to(self.args.device)
            code_emb = model(inputs)[0]
            # print(f'Dimension of the embedding: {embedding.size()[0]}, with norm={embedding.norm().item()}')
            code_embeddings.append(code_emb)
            
            node = combined_graph['ast_graph']
            # 获取该节点的相关子图（例如节点及其相邻节点）
            neighbors = list(combined_graph.neighbors(node))
            subgraph_nodes = [node] + neighbors
            subgraph = combined_graph.subgraph(subgraph_nodes)
            
            # 编码子图
            graph_emb = self.encode_graph(subgraph)
            graph_embeddings.append(graph_emb)
            
            if i % 100 == 0:
                print(f'Processed {i+1}/{len(code_snippets)} code snippets for {repo_name}')

        # TODO 对 graph的节点 做 embedding

        
        if not code_embeddings or not graph_embeddings:
            print(f'Failed to generate embeddings for {repo_name}')
            return
    
        # 添加到索引
        code_embeddings_array = np.vstack([e.numpy() for e in code_embeddings])
        # TODO 这里应该存储graph 中的节点的 embedding，而不是代码的 embedding
        graph_embeddings_array = np.vstack([e.numpy() for e in graph_embeddings])
        
        code_index.add_items(
            code_embeddings_array,
            np.arange(len(code_embeddings))
        )
        graph_index.add_items(
            graph_embeddings_array,
            np.arange(len(graph_embeddings))
        )
        
        # 保存索引
        code_index.save_index(str(Path(index_save_path) / 'code_index.bin'))
        graph_index.save_index(str(Path(index_save_path) / 'graph_index.bin'))
        
        # 保存代码片段用于后续检索
        with open(str(Path(index_save_path) / 'code_snippets.json'), 'w') as f:
            json.dump(code_snippets, f)
            
        print(f'Successfully built and saved index for {repo_name} at {index_save_path}')

    def retrieve(
        self,
        query_code: str,
        query_graph: nx.DiGraph,
        repo_name: str,
        k: int = 5,
        alpha: float = 0.5
    ) -> Tuple[List[str], List[float]]:
        """Retrieve relevant code snippets using both code and graph similarity from a specific repository"""
        # 确定要使用的仓库索引路径
        repo_short_name = repo_name.split('/')[-1]
        
        # 先检查简单的仓库名称路径
        repo_index_path = Path(self.args.index_path) / repo_short_name
        
        # 如果简单路径不存在，尝试查找带索引格式的路径 (idx_reponame)
        if not repo_index_path.exists() or not (repo_index_path / 'code_index.bin').exists():
            # 遍历索引目录查找匹配的索引
            found = False
            for index_dir in Path(self.args.index_path).iterdir():
                if index_dir.is_dir() and repo_short_name in index_dir.name:
                    if (index_dir / 'code_index.bin').exists() and (index_dir / 'graph_index.bin').exists():
                        repo_index_path = index_dir
                        found = True
                        break
            
            if not found:
                print(f'No index found for repository {repo_name} in any format')
                return [], []
        
        # 再次检查所需的索引文件是否存在
        if not (repo_index_path / 'code_index.bin').exists() or not (repo_index_path / 'graph_index.bin').exists():
            print(f'Index files missing for repository {repo_name} at {repo_index_path}')
            return [], []
        
        # 加载该仓库的索引
        code_index = hnswlib.Index(space='cosine', dim=self.dim)
        graph_index = hnswlib.Index(space='cosine', dim=self.dim)

        code_index.load_index(str(repo_index_path / 'code_index.bin'))
        graph_index.load_index(str(repo_index_path / 'graph_index.bin'))
        
        # 加载存储的代码片段
        with open(str(repo_index_path / 'code_snippets.json'), 'r') as f:
            code_snippets = json.load(f)
            
        print(f'Loaded index and {len(code_snippets)} code snippets for {repo_name}')

            
        # 获取查询嵌入向量
        code_emb = self.encode_code(query_code)
        graph_emb = self.encode_graph(query_graph)
        
        # 在两个索引中搜索
        code_ids, code_distances = code_index.knn_query(code_emb.numpy(), k=k)
        graph_ids, graph_distances = graph_index.knn_query(graph_emb.numpy(), k=k)
        
        # 对结果进行加权计分
        combined_scores = {}
        
        # 先添加所有代码相似度高的结果
        for i in range(len(code_ids[0])):
            code_id = code_ids[0][i]
            combined_scores[code_id] = alpha * (1 - code_distances[0][i])
            
        # 再增加图相似度的分数
        for i in range(len(graph_ids[0])):
            graph_id = graph_ids[0][i]
            if graph_id in combined_scores:
                combined_scores[graph_id] += (1 - alpha) * (1 - graph_distances[0][i])
            else:
                combined_scores[graph_id] = (1 - alpha) * (1 - graph_distances[0][i])
        
        # 按照组合分数排序
        sorted_ids = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        
        # 取前k个结果
        top_k = sorted_ids[:k] if len(sorted_ids) >= k else sorted_ids
        
        retrieved_snippets = []
        scores = []
        
        for idx, score in top_k:
            if 0 <= idx < len(code_snippets):
                retrieved_snippets.append(code_snippets[idx])
                scores.append(score)
            else:
                print(f'Warning: index {idx} out of range for code_snippets')
        
        return retrieved_snippets, scores
    
    def fuse_embeddings(
        self,
        code_emb: torch.Tensor,
        graph_emb: torch.Tensor,
        query_graph: nx.DiGraph,
        retrieved_graph: nx.DiGraph,
        fusion_type: str = 'attention'
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
            # 1. emb融合
            attention_scores = torch.matmul(code_emb, graph_emb.transpose(-2, -1)) / np.sqrt(self.dim)
            attention_weights = torch.softmax(attention_scores, dim=-1)
            attended_graph = torch.matmul(attention_weights, graph_emb)
            
            # 2. 图结构融合
            query_adj = nx.adjacency_matrix(query_graph).todense()
            retrieved_adj = nx.adjacency_matrix(retrieved_graph).todense()
            query_adj = torch.tensor(query_adj, dtype=torch.float32).to(code_emb.device)
            retrieved_adj = torch.tensor(retrieved_adj, dtype=torch.float32).to(code_emb.device)

            query_deg = torch.sum(query_adj, dim=1, keepdim=True).clamp(min=1)
            retrieved_deg = torch.sum(retrieved_adj, dim=1, keepdim=True).clamp(min=1)
            norm_query_adj = query_adj / query_deg
            norm_retrieved_adj = retrieved_adj / retrieved_deg
            
            # Message passing to incorporate structure
            def graph_conv(features, adj):
                return torch.matmul(adj, features)
            
            # Apply graph convolution to both graphs
            query_conv = graph_conv(code_emb, norm_query_adj)
            retrieved_conv = graph_conv(attended_graph, norm_retrieved_adj)
            
            # Combine structure-aware features
            gamma = torch.sigmoid(torch.nn.Linear(self.dim, 1).to(code_emb.device)(query_conv))
            structure_fused = gamma * query_conv + (1 - gamma) * retrieved_conv
            
            # 3. Final fusion combining embedding and structure
            beta = torch.sigmoid(torch.nn.Linear(self.dim, 1).to(code_emb.device)(code_emb))
            final_fused = beta * structure_fused + (1 - beta) * attended_graph
            return final_fused
            
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
        return self.model.complete(code_context, retrieved_contexts)


def main():
    
    parser = argparse.ArgumentParser(description='Run the Graph-guided RAG pipeline')
    parser.add_argument('--model', type=str, default='gpt3.5', help='Name of the LLM model to use')
    parser.add_argument('--repo-graph-path', type=str,  default='GRACE/dataset/hf_datasets/repobench_python_v1.1/cross_file_first/graphs', help='Path to the repository')
    parser.add_argument('--index-path', type=str,  default='GRACE/dataset/hf_datasets/repobench_python_v1.1/cross_file_first/index', help='Path to store/load HNSW indexes')
    parser.add_argument('--fusion-type', type=str, default='attention', choices=['concat', 'attention', 'sum'],
                        help='Type of fusion to use for code and graph embeddings')
    parser.add_argument('--k', type=int, default=5, help='Number of neighbors to retrieve')
    parser.add_argument('--alpha', type=float, default=0.5, help='Weight for code vs graph similarity')
    parser.add_argument('--eval-dataset', type=str,default = "GRACE/dataset/hf_datasets/repobench_python_v1.1" , help='Path to dataset for evaluation')
    parser.add_argument('--device', type=str, default='cuda:3' if torch.cuda.is_available() else 'cpu', help='Device to use for model')
    
    args = parser.parse_args()
    
    pipeline = GraphRAGPipeline(
        args = args,
        model_name=args.model,
        index_path=args.index_path
    )
    
    # 确保索引目录存在
    Path(args.index_path).mkdir(parents=True, exist_ok=True)
    
    # 加载RepoBench数据集
    dataset = load_dataset(args.eval_dataset, split=['cross_file_first'])[0]
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
    
    # 遍历数据集中的每个样本
    for idx, sample in enumerate(dataset):
        
        repo_author, repo_name = sample['repo_name'].split('/')
        file_path = sample['file_path']
        code_context = sample['all_code']  # 使用all_code作为上下文
        next_line = sample['next_line']  # 这是我们要预测的行

        # 临时过滤一下，只使用几个 repo 进行测试
        # 如果有测试仓库列表，只处理其中的仓库
        if repo_name not in ["3D-DAM", "4dfy", "4k4d", "AA", "A3FL", "ace"]:
            continue

        
        # 检查该仓库的索引是否已存在，不存在则构建（只检查每个仓库一次）
        if  str(idx)+str("_")+repo_name not in processed_repos:
            processed_repos.add(str(idx)+str("_")+repo_name)
            repo_index_path = Path(args.index_path) / Path(str(idx)+str("_")+repo_name)
            print(f'Building index for idx {idx} repository: {repo_name}')
            # 获取仓库的图结构路径
            repo_graph_path = Path(args.repo_graph_path) / Path(str(idx)+str("_")+repo_name)
            if repo_graph_path.exists() and (repo_graph_path / "repo_multi_graph.graphml").exists():
                print(f'Found processed graph for {repo_name} at {repo_graph_path}')
                # 读取图结构并构建索引
                pipeline.build_index_from_processed_graph(
                    str(repo_graph_path), 
                    repo_name, 
                    str(repo_index_path),
                    code_snippet=code_context  # 添加代码片段参数
                )
            else:
                print(f'Warning: No processed graph found for {repo_name} at {repo_graph_path}')
        
        if not code_context or not next_line:
            print(f'Sample {idx+1} has empty code_context or next_line, skipping.')
            continue
    
        print(f'\nProcessing sample {idx+1}/{len(dataset)}: {repo_name} - {file_path}')
        
        # 根据仓库名确定仓库图路径
        repo_dir = Path(args.repo_graph_path) / repo_name.split('/')[-1]
        if not repo_dir.exists():
            print(f'Repository graph {repo_name} not found at {repo_dir}')
            continue
        
        # 预处理样本，过滤next_line及其后续内容
        processed_sample = preprocess_repobench_data(sample)
        filtered_code = processed_sample['all_code']
        
        # 为当前样本构建图结构
        graphs = process_repobench_repo(str(repo_dir), processed_sample)
        query_graph = graphs.combined_graph
        
        # 检索相关代码片段
        snippets, scores = pipeline.retrieve(
            query_code=filtered_code,  # 使用过滤后的代码
            query_graph=query_graph,
            repo_name=repo_name,  # 添加仓库名
            k=args.k,
            alpha=args.alpha
        )
        
        # 使用检索到的片段完成代码
        completion = pipeline.complete_code(filtered_code, snippets)
        
        # 评估结果
        is_correct = completion.strip() == next_line.strip()
        if is_correct:
            correct_samples += 1
        
        # 计算代码BLEU得分
        bleu = calc_codebleu([completion], [next_line], lang="python")
        bleu_scores.append(bleu['codebleu'])
        
        # 记录结果
        result = {
            'sample_id': idx,
            'repo_name': repo_name,
            'file_path': file_path,
            'filtered_code': filtered_code,
            'completion': completion,
            'next_line': next_line,
            'is_correct': is_correct,
            'bleu_score': bleu['codebleu'],
            'retrieved_snippets': snippets,
            'retrieval_scores': scores
        }
        results.append(result)
        
        total_samples += 1
        
        # 打印当前样本的结果
        print(f'Predicted: {completion}')
        print(f'Expected: {next_line}')
        print(f'Correct: {is_correct}, BLEU: {bleu["codebleu"]:.4f}')
        print('Retrieved contexts:')
        for snippet, score in zip(snippets[:3], scores[:3]):
            print(f'Score: {score:.3f}')
            print(snippet[:200] + '...' if len(snippet) > 200 else snippet)
            print('---')
        
        # 每10个样本保存一次结果
        if idx % 10 == 0 and idx > 0:
            with open(f'results_{idx}.json', 'w') as f:
                json.dump(results, f, indent=2)
        
    
    # 计算最终指标
    accuracy = correct_samples / total_samples if total_samples > 0 else 0
    avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0
    
    # 计算其他统计数据
    processed_samples = len(results)
    skipped_samples = len(dataset) - processed_samples
    success_rate = processed_samples / len(dataset) if len(dataset) > 0 else 0
    
    print(f'\nEvaluation complete:')
    print(f'Total samples: {total_samples}')
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Average CodeBLEU: {avg_bleu:.4f}')
    
    # 保存最终结果
    results_file = f'final_results_{Path(args.eval_dataset).stem}.json'
    print(f'Saving results to {results_file}')
    try:
        with open(results_file, 'w') as f:
            json.dump({
                'dataset': args.eval_dataset,
                'total_dataset_samples': len(dataset),
                'processed_samples': processed_samples,
                'skipped_samples': skipped_samples,
                'success_rate': success_rate,
                'total_completed_samples': total_samples,
                'correct_samples': correct_samples,
                'accuracy': accuracy,
                'avg_bleu': avg_bleu,
                'results': results
            }, f, indent=2)
        print(f'Results saved successfully')
    except Exception as e:
        print(f'Error saving results: {e}')
        


if __name__ == '__main__':
    main()
