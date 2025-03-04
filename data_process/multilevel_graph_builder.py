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

def process_repobench_repo(repo_path: str) -> MultiLevelGraph:
    
    builder = MultiLevelGraphBuilder(repo_path)
    
    # 构建所有层次的图
    builder.build_repo_level()
    builder.build_module_level()
    builder.build_function_level()
    
    # 构建组合图
    builder.build_combined_graph()
    
    return builder.graphs

def main():
    language = "python"
    logging.basicConfig(level=logging.INFO)
    
    # RepoBench数据集路径
    repobench_path = f"GRACE/dataset/hf_datasets/repobench_r/{language}_repos"
    
    # 处理每个仓库
    for repo_dir in Path(repobench_path).iterdir():
        if repo_dir.is_dir():
            logging.info(f"Processing repository: {repo_dir}")
            try:
                graphs = process_repobench_repo(str(repo_dir))
                
                # 保存图结构
                output_dir = Path("GRACE/dataset/processed_graphs") / repo_dir.name
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # # 保存各个子图
                # nx.write_gpickle(graphs.folder_structure, output_dir / "folder_structure.gpickle")
                # nx.write_gpickle(graphs.cross_file_deps, output_dir / "cross_file_deps.gpickle")
                # nx.write_gpickle(graphs.call_graph, output_dir / "call_graph.gpickle")
                # nx.write_gpickle(graphs.type_deps, output_dir / "type_deps.gpickle")
                # nx.write_gpickle(graphs.class_inheritance, output_dir / "class_inheritance.gpickle")
                # nx.write_gpickle(graphs.ast_graph, output_dir / "ast_graph.gpickle")
                # nx.write_gpickle(graphs.cfg, output_dir / "cfg.gpickle")
                # nx.write_gpickle(graphs.dfg, output_dir / "dfg.gpickle")
                
                # 保存组合图
                nx.write_gpickle(graphs.combined_graph, output_dir / "combined_graph.gpickle")
                
                # 保存可视化版本（仅适用于小型图）
                try:
                    plt.figure(figsize=(20, 20))
                    pos = nx.spring_layout(graphs.combined_graph)
                    nx.draw(graphs.combined_graph, pos, with_labels=False, node_size=10, 
                            node_color="blue", font_size=8, alpha=0.6)
                    plt.savefig(output_dir / "combined_graph_viz.png", dpi=300)
                    plt.close()
                except Exception as viz_error:
                    logging.warning(f"Failed to visualize graph: {str(viz_error)}")
                
                logging.info(f"Successfully processed repo: {repo_dir}")
            except Exception as e:
                logging.error(f"Failed to process repo: {repo_dir}: {str(e)}")

if __name__ == "__main__":
    main()