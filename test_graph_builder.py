#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试多层次图构建器的功能
"""

import os
import sys
import logging
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
from data_process.multilevel_graph_builder import MultiLevelGraphBuilder, process_repobench_repo

def test_on_small_repo(repo_path: str):
    """
    在一个小型代码仓库上测试图构建功能
    """
    logging.basicConfig(level=logging.INFO)
    logging.info(f"Testing graph builder on repository: {repo_path}")
    
    try:
        # 构建图
        graphs = process_repobench_repo(repo_path)
        
        # 打印各个图的基本信息
        print("\n=== 各层次图的基本信息 ===")
        print(f"文件夹结构图: {len(graphs.folder_structure.nodes())} 节点, {len(graphs.folder_structure.edges())} 边")
        print(f"跨文件依赖图: {len(graphs.cross_file_deps.nodes())} 节点, {len(graphs.cross_file_deps.edges())} 边")
        print(f"调用图: {len(graphs.call_graph.nodes())} 节点, {len(graphs.call_graph.edges())} 边")
        print(f"类型依赖图: {len(graphs.type_deps.nodes())} 节点, {len(graphs.type_deps.edges())} 边")
        print(f"类继承图: {len(graphs.class_inheritance.nodes())} 节点, {len(graphs.class_inheritance.edges())} 边")
        print(f"AST图: {len(graphs.ast_graph.nodes())} 节点, {len(graphs.ast_graph.edges())} 边")
        print(f"控制流图: {len(graphs.cfg.nodes())} 节点, {len(graphs.cfg.edges())} 边")
        print(f"数据流图: {len(graphs.dfg.nodes())} 节点, {len(graphs.dfg.edges())} 边")
        
        # 打印组合图的信息
        print("\n=== 组合图信息 ===")
        print(f"组合图: {len(graphs.combined_graph.nodes())} 节点, {len(graphs.combined_graph.edges())} 边")
        
        # 分析边的类型
        edge_types = {}
        for _, _, data in graphs.combined_graph.edges(data=True):
            edge_type = data.get('edge_type', 'unknown')
            edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
        
        print("\n=== 组合图边类型统计 ===")
        for edge_type, count in edge_types.items():
            print(f"{edge_type}: {count} 条边")
        
        # 保存可视化结果
        output_dir = Path("GRACE/test_output")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 尝试可视化组合图的一部分（完整图可能太大）
        try:
            # 只可视化前100个节点和它们之间的边
            subgraph_nodes = list(graphs.combined_graph.nodes())[:min(100, len(graphs.combined_graph.nodes()))]
            subgraph = graphs.combined_graph.subgraph(subgraph_nodes)
            
            plt.figure(figsize=(15, 15))
            pos = nx.spring_layout(subgraph)
            
            # 按边类型绘制不同颜色
            edge_colors = []
            for u, v, data in subgraph.edges(data=True):
                if 'internal' in data.get('edge_type', ''):
                    edge_colors.append('gray')
                else:
                    edge_colors.append('red')
            
            nx.draw(subgraph, pos, with_labels=False, 
                    node_size=50, node_color="blue", 
                    edge_color=edge_colors,
                    font_size=8, alpha=0.7)
            
            plt.title("Combined Graph Visualization (Subset)")
            plt.savefig(output_dir / "combined_graph_subset.png", dpi=300)
            plt.close()
            print(f"\n可视化结果已保存到 {output_dir / 'combined_graph_subset.png'}")
        except Exception as e:
            logging.error(f"Failed to visualize graph: {str(e)}")
        
        return True
    except Exception as e:
        logging.error(f"Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # 使用当前目录作为测试仓库
    # 也可以指定其他小型Python仓库路径
    repo_path = os.path.dirname(os.path.abspath(__file__))
    success = test_on_small_repo(repo_path)
    
    if success:
        print("\n测试成功完成！")
    else:
        print("\n测试失败，请查看日志。")
