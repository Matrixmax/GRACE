import os
import torch
import networkx as nx
import numpy as np
import hnswlib
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import argparse

# 数据集处理相关
try:
    from datasets import load_dataset
except ImportError:
    pass

from GRACE.data_process.multilevel_graph_builder import (
    MultiLevelGraph, 
    MultiLevelGraphBuilder, 
    process_repobench_repo, 
    preprocess_repobench_data
)

from codebleu import calc_codebleu

def calc_metrics(hypotheses, references):
    result = calc_codebleu(hypotheses, references, lang="python")
    return result


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



@dataclass
class GraphEmbedding:
    code_emb: torch.Tensor  # Code embeddings
    graph_emb: torch.Tensor  # Graph structure embeddings
    combined_emb: Optional[torch.Tensor] = None  # Combined embeddings after fusion
    
class GraphRAGPipeline:
    def __init__(
        self,
        model_name: str,
        index_path: str,
        dim: int = 768,
        ef_construction: int = 400,
        M: int = 64
    ):
        self.model = LLMModel(model_name)
        self.dim = dim
        self.index_path = Path(index_path)
        
        # Initialize HNSW indexes for both code and graph embeddings
        self.code_index = hnswlib.Index(space='cosine', dim=dim)
        self.graph_index = hnswlib.Index(space='cosine', dim=dim)
        
        # Create if not exists
        if not self.index_path.exists():
            Path(self.index_path).mkdir(parents=True, exist_ok=True)
            self.code_index.init_index(max_elements=1000000, ef_construction=ef_construction, M=M)
            self.graph_index.init_index(max_elements=1000000, ef_construction=ef_construction, M=M)
            print(f'Initialized new indices at {self.index_path}')
        elif not (self.index_path / 'code_index.bin').exists() or not (self.index_path / 'graph_index.bin').exists():
            # If directory exists but index files don't
            print(f'Index directory exists but missing index files. Initializing new indices.')
            self.code_index.init_index(max_elements=1000000, ef_construction=ef_construction, M=M)
            self.graph_index.init_index(max_elements=1000000, ef_construction=ef_construction, M=M)
        else:
            print(f'Loading existing indices from {self.index_path}')
            self.code_index.load_index(str(self.index_path / 'code_index.bin'))
            self.graph_index.load_index(str(self.index_path / 'graph_index.bin'))
    
    def encode_graph(self, graph: nx.DiGraph) -> torch.Tensor:
        """Encode graph structure into embeddings by averaging node embeddings"""
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
        return self.model.get_embeddings(code)
    
    def build_index(self, repo_path: str):
        """Build HNSW indexes for code and graph embeddings from repository"""
        # Build multi-level graph
        builder = MultiLevelGraphBuilder(repo_path)
        builder.build_repo_level()
        builder.build_module_level() 
        builder.build_function_level()
        graphs = builder.graphs
        
        # Get all code snippets from repository
        code_snippets = []  # TODO: Extract code snippets from repo
        
        # Generate embeddings
        code_embeddings = []
        graph_embeddings = []
        
        for code in code_snippets:
            code_emb = self.encode_code(code)
            code_embeddings.append(code_emb)
            
            # Get corresponding graph for the code
            graph = graphs.combined_graph  # TODO: Get relevant subgraph
            graph_emb = self.encode_graph(graph)
            graph_embeddings.append(graph_emb)
            
        # Add to indexes
        self.code_index.add_items(
            np.vstack(code_embeddings),
            np.arange(len(code_embeddings))
        )
        self.graph_index.add_items(
            np.vstack(graph_embeddings), 
            np.arange(len(graph_embeddings))
        )
        
        # Save indexes
        self.code_index.save_index(str(self.index_path / 'code_index.bin'))
        self.graph_index.save_index(str(self.index_path / 'graph_index.bin'))
    
    def retrieve(
        self,
        query_code: str,
        query_graph: nx.DiGraph,
        k: int = 5,
        alpha: float = 0.5
    ) -> Tuple[List[str], List[float]]:
        """Retrieve relevant code snippets using both code and graph similarity"""
        # Get query embeddings
        code_emb = self.encode_code(query_code)
        graph_emb = self.encode_graph(query_graph)
        
        # Search both indexes
        code_ids, code_distances = self.code_index.knn_query(code_emb.numpy(), k=k)
        graph_ids, graph_distances = self.graph_index.knn_query(graph_emb.numpy(), k=k)
        
        # Combine results with weighted scoring
        combined_scores = {}
        for i in range(k):
            code_id = code_ids[0][i]
            graph_id = graph_ids[0][i]
            
            # Combine scores for same items
            if code_id == graph_id:
                score = alpha * (1 - code_distances[0][i]) + (1 - alpha) * (1 - graph_distances[0][i])
                combined_scores[code_id] = score
                
        # Sort by combined score
        sorted_ids = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        
        retrieved_snippets = []
        scores = []
        
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
    parser.add_argument('--model', type=str, required=True, default='gpt3.5', help='Name of the LLM model to use')
    parser.add_argument('--repo-path', type=str, required=True, default='.', help='Path to the repository')
    parser.add_argument('--index-path', type=str, required=True, default='GRACE/data_process/index', help='Path to store/load HNSW indexes')
    parser.add_argument('--fusion-type', type=str, default='attention', choices=['concat', 'attention', 'sum'],
                        help='Type of fusion to use for code and graph embeddings')
    parser.add_argument('--k', type=int, default=5, help='Number of neighbors to retrieve')
    parser.add_argument('--alpha', type=float, default=0.5, help='Weight for code vs graph similarity')
    parser.add_argument('--eval-dataset', type=str, help='Path to dataset for evaluation (e.g., "/data/wxl/graphrag4se/GRACE/dataset/hf_datasets/repobench_python_v1.1")')
    
    args = parser.parse_args()
    
    pipeline = GraphRAGPipeline(
        model_name=args.model,
        index_path=args.index_path
    )
    
    # 构建索引
    if not Path(args.index_path).exists():
        print(f'Building index from repository: {args.repo_path}')
        pipeline.build_index(args.repo_path)
    else:
        print(f'Loading existing index from: {args.index_path}')
    
    # 数据集评估
    print(f'\nEvaluating on dataset: {args.eval_dataset}')
    
    # 加载RepoBench数据集
    dataset = load_dataset(args.eval_dataset, split=['cross_file_first'])[0]
    print(f'Loaded dataset with {len(dataset)} samples')
    
    # 初始化评估指标
    total_samples = 0
    correct_samples = 0
    bleu_scores = []
    total_metric = 0.0
    results = []
    
    # 遍历数据集中的每个样本
    for idx, sample in enumerate(dataset):
        try:
            repo_name = sample['repo_name']
            file_path = sample['file_path']
            code_context = sample['all_code']  # 使用all_code作为上下文
            next_line = sample['next_line']  # 这是我们要预测的行
            
            if not code_context or not next_line:
                print(f'Sample {idx+1} has empty code_context or next_line, skipping.')
                continue
        
        print(f'\nProcessing sample {idx+1}/{len(dataset)}: {repo_name} - {file_path}')
        
        # 根据仓库名确定仓库路径
        repo_dir = Path(args.repo_path) / repo_name.split('/')[-1]
        if not repo_dir.exists():
            print(f'Repository {repo_name} not found at {repo_dir}')
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
