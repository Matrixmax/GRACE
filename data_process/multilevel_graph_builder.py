import os
import ast
import networkx as nx
from typing import Dict, List, Set, Tuple
import ast
from dataclasses import dataclass
from pathlib import Path
import logging
from staticfg import CFGBuilder
import astroid
from pydeps.py2depgraph import py2dep
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
        
    def build_module_level(self):
        """构建模块级别的图结构"""
        self._build_call_graph()
        self._build_type_deps()
        self._build_class_inheritance()
        
    def build_function_level(self):
        """构建函数级别的图结构"""
        self._build_ast_graph()
        self._build_cfg()
        self._build_dfg()

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
        # 使用pydeps分析模块依赖
        dep_graph = py2dep(str(self.repo_path))
        for src, dst in dep_graph.edges():
            self.graphs.cross_file_deps.add_edge(src, dst)

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
            except:
                logging.warning(f"Failed to build DFG for {py_file}")

def process_repobench_repo(repo_path: str) -> MultiLevelGraph:
    
    builder = MultiLevelGraphBuilder(repo_path)
    
    # 构建所有层次的图
    builder.build_repo_level()
    builder.build_module_level()
    builder.build_function_level()
    
    return builder.graphs

def main():
    
    logging.basicConfig(level=logging.INFO)
    
    # RepoBench数据集路径
    repobench_path = "GRACE/dataset/repobench"
    
    # 处理每个仓库
    for repo_dir in Path(repobench_path).iterdir():
        if repo_dir.is_dir():
            logging.info(f"Processing repository: {repo_dir}")
            try:
                graphs = process_repobench_repo(str(repo_dir))
                # 添加图的保存逻辑
                logging.info(f"Successfully processed {repo_dir}")
            except Exception as e:
                logging.error(f"Failed to process {repo_dir}: {str(e)}")

if __name__ == "__main__":
    main()