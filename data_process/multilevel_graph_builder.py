import os
import ast
import networkx as nx
from typing import Dict, List, Set, Tuple, Optional
import ast
from dataclasses import dataclass
from pathlib import Path
import logging
import matplotlib.pyplot as plt
from staticfg import CFGBuilder
import astroid
# from pydeps.py2depgraph import py2dep  # 不再使用pydeps
from mypy import build
from mypy.nodes import MypyFile

@dataclass
class MultiLevelGraph:
    # Repo Level
    folder_structure: nx.DiGraph  # 文件目录结构图
    cross_file_deps: nx.DiGraph   # 跨文件依赖图
    
    # Module Level
    call_graph: nx.DiGraph       # 调用图
    type_deps: nx.DiGraph       # 类型依赖图
    class_inheritance: nx.DiGraph # 类继承图
    
    # Function Level
    ast_graph: nx.DiGraph       # AST图
    cfg: nx.DiGraph            # 控制流图
    dfg: nx.DiGraph           # 数据流图
    
    # Combined Graph (整合所有层次的最终图结构)
    combined_graph: Optional[nx.DiGraph] = None

class MultiLevelGraphBuilder:
    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path)
        self.graphs = MultiLevelGraph(
            folder_structure=nx.DiGraph(),
            cross_file_deps=nx.DiGraph(),
            call_graph=nx.DiGraph(),
            type_deps=nx.DiGraph(),
            class_inheritance=nx.DiGraph(),
            ast_graph=nx.DiGraph(),
            cfg=nx.DiGraph(),
            dfg=nx.DiGraph()
        )
        
    def build_repo_level(self):
        """构建仓库级别的图结构"""
        self._build_folder_structure()
        self._build_cross_file_deps()
        print("build_repo_level success")
        
    def build_module_level(self):
        """构建模块级别的图结构"""
        self._build_call_graph()
        self._build_type_deps()
        self._build_class_inheritance()
        print("build_module_level success")
        
    def build_function_level(self):
        """构建函数级别的图结构"""
        self._build_ast_graph()
        self._build_cfg()
        self._build_dfg()
        print("build_function_level success")

    def _build_folder_structure(self):
        """构建文件夹结构图"""
        for root, dirs, files in os.walk(self.repo_path):
            root_node = str(Path(root).relative_to(self.repo_path))
            for d in dirs:
                child_node = str(Path(root, d).relative_to(self.repo_path))
                self.graphs.folder_structure.add_edge(root_node, child_node)
            for f in files:
                if f.endswith('.py'):
                    child_node = str(Path(root, f).relative_to(self.repo_path))
                    self.graphs.folder_structure.add_edge(root_node, child_node)

    def _build_cross_file_deps(self):
        """构建跨文件依赖图"""
        # 使用AST来分析文件之间的导入关系
        python_files = {}
        
        # 第一遍：收集所有Python文件的路径
        for py_file in self.repo_path.rglob('*.py'):
            rel_path = str(py_file.relative_to(self.repo_path))
            python_files[rel_path] = py_file
        
        # 第二遍：分析每个文件的导入
        for rel_path, py_file in python_files.items():
            try:
                with open(py_file) as f:
                    tree = ast.parse(f.read())
                
                # 寻找导入语句
                for node in ast.walk(tree):
                    # 处理 import x 类型的导入
                    if isinstance(node, ast.Import):
                        for name in node.names:
                            imported_module = name.name
                            self._add_dependency(rel_path, imported_module, python_files)
                    
                    # 处理 from x import y 类型的导入
                    elif isinstance(node, ast.ImportFrom) and node.module:
                        self._add_dependency(rel_path, node.module, python_files)
            
            except Exception as e:
                logging.warning(f"Failed to analyze imports in {py_file}: {str(e)}")
    
    def _add_dependency(self, source_file, imported_module, python_files):
        """将模块导入添加为文件依赖关系"""
        # 将模块路径转换为文件路径
        parts = imported_module.split('.')
        
        # 尝试多种可能的文件匹配
        potential_paths = []
        
        # 尝试直接匹配文件名
        potential_paths.append(f"{'/'.join(parts)}.py")
        
        # 尝试匹配包下的__init__.py
        potential_paths.append(f"{'/'.join(parts)}/__init__.py")
        
        # 尝试去掉最后一个组件并将其作为文件名
        if len(parts) > 1:
            potential_paths.append(f"{'/'.join(parts[:-1])}/{parts[-1]}.py")
        
        # 检查是否有任何匹配的文件
        for target_file in potential_paths:
            if target_file in python_files:
                self.graphs.cross_file_deps.add_edge(source_file, target_file)
                return
        
        # 如果没有完全匹配，尝试前缀匹配
        prefix = '/'.join(parts)
        for file_path in python_files.keys():
            if file_path.startswith(prefix + '/') or file_path == prefix + '.py':
                self.graphs.cross_file_deps.add_edge(source_file, file_path)
                return

    def _build_call_graph(self):
        """构建调用图"""
        for py_file in self.repo_path.rglob('*.py'):
            try:
                with open(py_file) as f:
                    tree = ast.parse(f.read())
                    
                class CallVisitor(ast.NodeVisitor):
                    def __init__(self):
                        self.calls = []
                        self.current_func = None
                        
                    def visit_FunctionDef(self, node):
                        old_func = self.current_func
                        self.current_func = node.name
                        self.generic_visit(node)
                        self.current_func = old_func
                        
                    def visit_Call(self, node):
                        if isinstance(node.func, ast.Name) and self.current_func:
                            self.calls.append((self.current_func, node.func.id))
                        self.generic_visit(node)
                
                visitor = CallVisitor()
                visitor.visit(tree)
                
                for caller, callee in visitor.calls:
                    self.graphs.call_graph.add_edge(caller, callee)
            except:
                logging.warning(f"Failed to process {py_file}")

    def _build_type_deps(self):
        """构建类型依赖图"""
        # 使用mypy进行类型分析
        for py_file in self.repo_path.rglob('*.py'):
            try:
                result = build.build(sources=[build.BuildSource(str(py_file), None, None)])
                tree = result.files[str(py_file)]
                self._process_mypy_tree(tree)
            except:
                logging.warning(f"Failed to process types in {py_file}")

    def _process_mypy_tree(self, tree: MypyFile):
        """处理mypy分析树以提取类型依赖"""
        for node in tree.defs:
            if hasattr(node, 'type'):
                for typ in self._get_referenced_types(node.type):
                    self.graphs.type_deps.add_edge(node.name, typ)

    def _build_class_inheritance(self):
        """构建类继承图"""
        for py_file in self.repo_path.rglob('*.py'):
            try:
                with open(py_file) as f:
                    tree = ast.parse(f.read())
                    
                class InheritanceVisitor(ast.NodeVisitor):
                    def __init__(self):
                        self.inheritance = []
                        
                    def visit_ClassDef(self, node):
                        for base in node.bases:
                            if isinstance(base, ast.Name):
                                self.inheritance.append((node.name, base.id))
                        self.generic_visit(node)
                
                visitor = InheritanceVisitor()
                visitor.visit(tree)
                
                for child, parent in visitor.inheritance:
                    self.graphs.class_inheritance.add_edge(child, parent)
            except:
                logging.warning(f"Failed to process inheritance in {py_file}")

    def _build_ast_graph(self):
        """构建AST图"""
        for py_file in self.repo_path.rglob('*.py'):
            try:
                with open(py_file) as f:
                    tree = ast.parse(f.read())
                    
                def add_ast_edges(node, parent_id):
                    node_id = id(node)
                    self.graphs.ast_graph.add_edge(parent_id, node_id)
                    for child in ast.iter_child_nodes(node):
                        add_ast_edges(child, node_id)
                
                add_ast_edges(tree, id(tree))
            except:
                logging.warning(f"Failed to build AST for {py_file}")

    def _build_cfg(self):
        """构建控制流图"""
        for py_file in self.repo_path.rglob('*.py'):
            try:
                cfg = CFGBuilder().build_from_file(str(py_file))
                for block in cfg.blocks:
                    for succ in block.successors:
                        self.graphs.cfg.add_edge(id(block), id(succ))
            except:
                logging.warning(f"Failed to build CFG for {py_file}")

    def _build_dfg(self):
        """构建数据流图"""
        for py_file in self.repo_path.rglob('*.py'):
            try:
                with open(py_file) as f:
                    tree = ast.parse(f.read())
                    
                class DataFlowVisitor(ast.NodeVisitor):
                    def __init__(self):
                        self.dfg_edges = []
                        self.definitions = {}
                        
                    def visit_Assign(self, node):
                        for target in node.targets:
                            if isinstance(target, ast.Name):
                                self.definitions[target.id] = id(node)
                        self.generic_visit(node)
                        
                    def visit_Name(self, node):
                        if isinstance(node.ctx, ast.Load) and node.id in self.definitions:
                            self.dfg_edges.append((self.definitions[node.id], id(node)))
                        self.generic_visit(node)
                
                visitor = DataFlowVisitor()
                visitor.visit(tree)
                
                for src, dst in visitor.dfg_edges:
                    self.graphs.dfg.add_edge(src, dst)
            except Exception as e:
                logging.warning(f"Failed to build DFG for {py_file}: {str(e)}")
                
    def _get_referenced_types(self, type_obj):
        """从mypy类型对象中提取引用的类型"""
        # 这是一个简化的实现，实际应根据mypy的API进行完善
        referenced_types = []
        if hasattr(type_obj, 'name'):
            referenced_types.append(type_obj.name)
        return referenced_types
    
    def build_combined_graph(self):
        """构建组合图，将各个层次的图连接起来"""
        # 创建一个新的有向图作为组合图
        self.combined_graph = nx.DiGraph()
        # 同时更新MultiLevelGraph对象中的combined_graph属性
        self.graphs.combined_graph = self.combined_graph
        
        # 添加所有子图的节点和边到组合图
        for graph_name, graph in [
            ('folder_structure', self.graphs.folder_structure),
            ('cross_file_deps', self.graphs.cross_file_deps),
            ('call_graph', self.graphs.call_graph),
            ('type_deps', self.graphs.type_deps),
            ('class_inheritance', self.graphs.class_inheritance),
            ('ast_graph', self.graphs.ast_graph),
            ('cfg', self.graphs.cfg),
            ('dfg', self.graphs.dfg)
        ]:
            # 为每个子图的节点添加图类型属性
            for node in graph.nodes():
                self.combined_graph.add_node(f"{graph_name}:{node}", 
                                            graph_type=graph_name,
                                            original_id=node)
            
            # 添加子图中的边，保留原始图的结构
            for src, dst in graph.edges():
                self.combined_graph.add_edge(f"{graph_name}:{src}", 
                                           f"{graph_name}:{dst}",
                                           edge_type=f"internal_{graph_name}")
        
        # 建立跨层次的连接
        self._connect_repo_to_module_level()
        self._connect_module_to_function_level()
        self._connect_function_level_internally()
        
        return self.combined_graph
    
    def _connect_repo_to_module_level(self):
        """连接仓库级别和模块级别的图"""
        # 1. 文件位置约束方法调用关系
        for file_node in self.graphs.folder_structure.nodes():
            if file_node.endswith('.py'):
                # 查找该文件中定义的方法
                file_path = self.repo_path / file_node
                try:
                    with open(file_path) as f:
                        tree = ast.parse(f.read())
                        
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            # 文件位置约束方法
                            self.combined_graph.add_edge(
                                f"folder_structure:{file_node}",
                                f"call_graph:{node.name}",
                                edge_type="file_location_constraint"
                            )
                except Exception as e:
                    logging.warning(f"Failed to connect file to methods for {file_path}: {str(e)}")
        
        # 2. 跨模块调用关系
        for src, dst in self.graphs.cross_file_deps.edges():
            # 查找源文件中的方法调用目标文件中的方法
            for caller in self.graphs.call_graph.nodes():
                for callee in self.graphs.call_graph.successors(caller):
                    # 简化实现：假设如果存在跨文件依赖，则文件中的方法可能相互调用
                    self.combined_graph.add_edge(
                        f"cross_file_deps:{src}",
                        f"call_graph:{caller}",
                        edge_type="cross_module_call"
                    )
        
        # 3. 类型跨文件引用
        for src, dst in self.graphs.cross_file_deps.edges():
            for type_src, type_dst in self.graphs.type_deps.edges():
                # 简化实现：假设跨文件依赖可能导致类型依赖
                self.combined_graph.add_edge(
                    f"cross_file_deps:{src}",
                    f"type_deps:{type_src}",
                    edge_type="type_cross_file_reference"
                )
        
        # 4. 接口继承关系
        for src, dst in self.graphs.cross_file_deps.edges():
            for child, parent in self.graphs.class_inheritance.edges():
                # 简化实现：假设跨文件依赖可能涉及继承关系
                self.combined_graph.add_edge(
                    f"cross_file_deps:{src}",
                    f"class_inheritance:{child}",
                    edge_type="interface_inheritance"
                )
    
    def _connect_module_to_function_level(self):
        """连接模块级别和函数级别的图"""
        # 1. 方法体结构
        for func_node in self.graphs.call_graph.nodes():
            # 查找AST中对应的函数定义节点
            for ast_node in self.graphs.ast_graph.nodes():
                # 简化实现：使用节点ID不太可靠，实际应存储更多元数据
                # 这里仅作为示例
                self.combined_graph.add_edge(
                    f"call_graph:{func_node}",
                    f"ast_graph:{ast_node}",
                    edge_type="method_body_structure"
                )
                # 只连接一个示例，实际应该更精确
                break
        
        # 2. 变量类型约束
        for type_src, type_dst in self.graphs.type_deps.edges():
            for dfg_src, dfg_dst in self.graphs.dfg.edges():
                # 简化实现：假设类型依赖可能影响数据流
                self.combined_graph.add_edge(
                    f"type_deps:{type_src}",
                    f"dfg:{dfg_src}",
                    edge_type="variable_type_constraint"
                )
                # 只连接一个示例，实际应该更精确
                break
        
        # 3. 方法继承
        for child, parent in self.graphs.class_inheritance.edges():
            for caller, callee in self.graphs.call_graph.edges():
                # 简化实现：假设继承关系可能影响方法调用
                self.combined_graph.add_edge(
                    f"class_inheritance:{child}",
                    f"call_graph:{caller}",
                    edge_type="method_inheritance"
                )
                # 只连接一个示例，实际应该更精确
                break
    
    def _connect_function_level_internally(self):
        """连接函数级别内部的图"""
        # 1. 语法结构影响控制流
        for ast_node in self.graphs.ast_graph.nodes():
            for cfg_node in self.graphs.cfg.nodes():
                # 简化实现：假设AST节点可能对应控制流节点
                self.combined_graph.add_edge(
                    f"ast_graph:{ast_node}",
                    f"cfg:{cfg_node}",
                    edge_type="syntax_structure"
                )
                # 只连接一个示例，实际应该更精确
                break
        
        # 2. 控制依赖影响数据流
        for cfg_node in self.graphs.cfg.nodes():
            for dfg_node in self.graphs.dfg.nodes():
                # 简化实现：假设控制流节点可能影响数据流节点
                self.combined_graph.add_edge(
                    f"cfg:{cfg_node}",
                    f"dfg:{dfg_node}",
                    edge_type="control_dependency"
                )
                # 只连接一个示例，实际应该更精确
                break

def filter_next_line_and_after(code: str, next_line: str) -> str:
    """
    从代码中过滤掉 next_line 及其后续内容。
    
    Args:
        code (str): 原始代码字符串
        next_line (str): 需要过滤的下一行代码（以及之后的所有内容）
    
    Returns:
        str: 过滤后的代码
    """
    if not next_line or not code:
        return code
    
    # 在代码中查找 next_line 的位置
    next_line_pos = code.find(next_line)
    if next_line_pos == -1:
        return code  # 如果没找到，返回原始代码
    
    # 返回 next_line 之前的代码部分
    return code[:next_line_pos].rstrip()

def preprocess_repobench_data(data_sample):
    """
    处理 RepoBench 数据集的样本，过滤掉 next_line 及其后续内容。
    
    Args:
        data_sample: RepoBench 数据集的一个样本
    
    Returns:
        dict: 处理后的样本，其中 all_code 被过滤后的版本替代
    """
    # 创建数据样本的副本，避免修改原始数据
    processed_sample = dict(data_sample)
    
    # 获取 next_line 和 all_code
    next_line = data_sample.get('next_line', '')
    all_code = data_sample.get('all_code', '')
    
    # 过滤 all_code，移除 next_line 及其后续内容
    filtered_code = filter_next_line_and_after(all_code, next_line)
    
    # 用过滤后的代码替换原始的 all_code
    processed_sample['all_code'] = filtered_code
    
    return processed_sample

def process_repobench_repo(repo_path: str, data_sample=None) -> MultiLevelGraph:
    """
    处理 RepoBench 仓库，构建多层次图结构。
    如果提供了数据样本，会先过滤掉 next_line 及其后续内容。
    
    Args:
        repo_path (str): 仓库路径
        data_sample (dict, optional): RepoBench 数据样本，包含 next_line 和对应的代码
    
    Returns:
        MultiLevelGraph: 构建的图结构
    """
    # 如果提供了数据样本，先预处理它
    if data_sample:
        processed_sample = preprocess_repobench_data(data_sample)
        # 这里可以根据需要对仓库中的文件进行修改，去掉 next_line 及后续内容
        # 例如：写入临时文件并建立图结构，或者修改特定文件
    
    builder = MultiLevelGraphBuilder(repo_path)
    
    # 构建所有层次的图
    builder.build_repo_level()
    builder.build_module_level()
    builder.build_function_level()
    
    # 构建组合图
    builder.build_combined_graph()
    
    return builder.graphs


def visualize_and_save_graphs(graphs, output_dir):
    def save_and_visualize_graph(graph, name, file_ext=".gpickle", figsize=(20, 20)):
        # 保存图结构
        nx.write_graphml(graph, output_dir / f"{name}{file_ext}")
        
        # 尝试可视化
        try:
            plt.figure(figsize=figsize)
            pos = nx.spring_layout(graph)
            nx.draw(graph, pos, with_labels=True, node_size=30,
                    node_color="blue", font_size=8, alpha=0.6)
            plt.title(name)
            plt.savefig(output_dir / f"{name}_viz.png", dpi=100)
            plt.close()
        except Exception as e:
            logging.warning(f"Failed to visualize {name} graph: {str(e)}")
    
    # 保存并可视化各个子图
    graph_mapping = {
        "folder_structure": graphs.folder_structure,
        "cross_file_deps": graphs.cross_file_deps,
        "call_graph": graphs.call_graph,
        "type_deps": graphs.type_deps,
        "class_inheritance": graphs.class_inheritance,
        "ast_graph": graphs.ast_graph,
        "cfg": graphs.cfg,
        "dfg": graphs.dfg
    }
    
    for name, graph in graph_mapping.items():
        save_and_visualize_graph(graph, name)
    
    # 保存并可视化组合图
    # 使用save_and_visualize_graph函数，但指定不同的文件扩展名和图形大小
    save_and_visualize_graph(graphs.combined_graph, "combined_graph", file_ext=".graphml", figsize=(100, 100))


def main():
    language = "python"
    logging.basicConfig(level=logging.INFO)
    
    # RepoBench数据集路径
    repobench_path = f"GRACE/dataset/hf_datasets/repobench_{language}_v1.1/cross_file_first/repos"
    
    # 加载 RepoBench 数据集
    try:
        from datasets import load_dataset
        dataset = load_dataset(f"/data/wxl/graphrag4se/GRACE/dataset/hf_datasets/repobench_{language}_v1.1", split=['cross_file_first'])[0]
        logging.info(f"Successfully loaded RepoBench dataset with {len(dataset)} samples")
    except Exception as e:
        logging.error(f"Failed to load RepoBench dataset: {str(e)}")
        dataset = None
       
    # 1. 获取数据集中所有的repo_name
    repo_names = set()
    if dataset is not None:
        for sample in dataset:
            # 从样本的repo_name中提取实际仓库名称
            repo_name = sample['repo_name'].split('/')[-1]
            repo_names.add(repo_name)
        logging.info(f"Found {len(repo_names)} unique repositories in the dataset")
    
    # 用于测试的仓库列表 (可选择性过滤)
    test_repos = ["3D-DAM", "4dfy", "4k4d", "AA", "A3FL", "ace"]
    if test_repos:
        repo_names = {name for name in repo_names if name in test_repos}
        logging.info(f"Filtered to {len(repo_names)} test repositories: {repo_names}")
    
    # 创建图结构存储的根目录
    repo_graphs_dir = Path(repobench_path) / "repo_graphs"
    repo_graphs_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. 遍历这些repo对应的仓库
    for repo_name in repo_names:
        repo_dir = Path(repobench_path) / repo_name
        if not repo_dir.is_dir():
            logging.warning(f"Repository directory not found: {repo_dir}")
            continue
        
        logging.info(f"Processing repository: {repo_dir}")
        try:
            # 找出该仓库相关的所有样本
            repo_samples = [sample for sample in dataset if repo_name in sample['repo_name']]
            logging.info(f"Found {len(repo_samples)} samples for repo {repo_name}")
            
            # 为仓库创建处理结果和图结构目录
            processed_dir = repo_dir / "processed"
            processed_dir.mkdir(parents=True, exist_ok=True)
            
            repo_graph_dir = repo_graphs_dir / repo_name
            repo_graph_dir.mkdir(parents=True, exist_ok=True)
            
            # 3. 为每个样本进行过滤和图结构构建
            for i, sample in enumerate(repo_samples):
                sample_id = f"sample_{i+1}"
                logging.info(f"Processing sample {i+1}/{len(repo_samples)} for repo {repo_name}")
                
                # 预处理样本，过滤next_line及其后续内容
                processed_sample = preprocess_repobench_data(sample)
                
                # 为当前样本构建图结构
                graphs = process_repobench_repo(str(repo_dir), processed_sample)
                
                # 4. 保存处理结果到processed目录
                sample_dir = processed_dir / sample_id
                sample_dir.mkdir(parents=True, exist_ok=True)
                
                # 保存过滤后的代码
                with open(sample_dir / "filtered_code.py", "w") as f:
                    f.write(processed_sample['all_code'])
                
                # 保存原始代码和预期的next_line（用于评估）
                with open(sample_dir / "original_code.py", "w") as f:
                    f.write(sample['all_code'])
                with open(sample_dir / "next_line.txt", "w") as f:
                    f.write(sample['next_line'])
                
                # 保存图结构到repo_graphs目录
                graph_dir = repo_graph_dir / sample_id
                graph_dir.mkdir(parents=True, exist_ok=True)
                
                # 调用可视化函数并保存图结构
                visualize_and_save_graphs(graphs, graph_dir)
            
            logging.info(f"Successfully processed repo: {repo_name}")
        except Exception as e:
            logging.error(f"Failed to process repo: {repo_name}: {str(e)}")



if __name__ == "__main__":
    main()