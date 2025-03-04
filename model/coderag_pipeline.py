import os
import torch
import networkx as nx
import numpy as np
import hnswlib
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

from GRACE.data_process.multilevel_graph_builder import MultiLevelGraph, MultiLevelGraphBuilder
from src.llm import LLMModel
from src.metrics import calc_metrics

from codebleu import calc_codebleu

def calc_metrics(hypotheses, references):
    result = calc_codebleu(hypotheses, references, lang="python")

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
            self.code_index.init_index(max_elements=1000000, ef_construction=ef_construction, M=M)
            self.graph_index.init_index(max_elements=1000000, ef_construction=ef_construction, M=M)
        else:
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
        
        # TODO: Convert ids back to code snippets
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
            # 1. Embedding-level fusion using attention
            attention_scores = torch.matmul(code_emb, graph_emb.transpose(-2, -1)) / np.sqrt(self.dim)
            attention_weights = torch.softmax(attention_scores, dim=-1)
            attended_graph = torch.matmul(attention_weights, graph_emb)
            
            # 2. Structure-level fusion using graph matching
            # Get adjacency matrices
            query_adj = nx.adjacency_matrix(query_graph).todense()
            retrieved_adj = nx.adjacency_matrix(retrieved_graph).todense()
            query_adj = torch.tensor(query_adj, dtype=torch.float32).to(code_emb.device)
            retrieved_adj = torch.tensor(retrieved_adj, dtype=torch.float32).to(code_emb.device)
            
            # Graph structure similarity using normalized adjacency
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
    import argparse
    
    parser = argparse.ArgumentParser(description='Run the Graph-guided RAG pipeline')
    parser.add_argument('--model', type=str, required=True, help='Name of the LLM model to use')
    parser.add_argument('--repo-path', type=str, required=True, help='Path to the repository')
    parser.add_argument('--index-path', type=str, required=True, help='Path to store/load HNSW indexes')
    parser.add_argument('--fusion-type', type=str, default='attention', choices=['concat', 'attention', 'sum'],
                        help='Type of fusion to use for code and graph embeddings')
    parser.add_argument('--k', type=int, default=5, help='Number of neighbors to retrieve')
    parser.add_argument('--alpha', type=float, default=0.5, help='Weight for code vs graph similarity')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = GraphRAGPipeline(
        model_name=args.model,
        index_path=args.index_path
    )
    
    # Build or load index
    if not Path(args.index_path).exists():
        print(f'Building index from repository: {args.repo_path}')
        pipeline.build_index(args.repo_path)
    else:
        print(f'Loading existing index from: {args.index_path}')
    
    # Interactive code completion loop
    print('\nEnter code context (or "quit" to exit):')
    while True:
        try:
            code_context = input('> ')
            if code_context.lower() == 'quit':
                break
                
            # Get current file context for graph construction
            builder = MultiLevelGraphBuilder(args.repo_path)
            builder.build_module_level()
            query_graph = builder.graphs.combined_graph
            
            # Retrieve similar code snippets
            snippets, scores = pipeline.retrieve(
                query_code=code_context,
                query_graph=query_graph,
                k=args.k,
                alpha=args.alpha
            )
            
            # Complete the code
            completion = pipeline.complete_code(code_context, snippets)
            print('\nCompletion:')
            print(completion)
            print('\nRetrieved contexts:')
            for snippet, score in zip(snippets, scores):
                print(f'Score: {score:.3f}')
                print(snippet)
                print('---')
                
        correct = calc_metrics(outputs, targets)
        total_metric += correct
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f'Error: {e}')
            continue

if __name__ == '__main__':
    main()
