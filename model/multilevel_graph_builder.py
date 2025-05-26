import os
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
import json

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
    ast: nx.DiGraph       # AST图
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
            ast=nx.DiGraph(),
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
        # self._build_type_deps()
        self._build_class_inheritance()
        print("build_module_level success")
        
    def build_function_level(self):
        """构建函数级别的图结构"""
        self._build_ast_graph()
        # self._build_cfg()
        self._build_dfg()
        print("build_function_level success")
        
    def _add_node_with_code(self, graph, node_id, code_content, file_path=None, extra_attrs=None):
        """添加节点到图中，并确保节点具有code属性
        
        Args:
            graph: 要添加节点的图
            node_id: 节点的唯一标识符
            code_content: 节点的代码内容
            file_path: 可选，节点所属的文件路径
            extra_attrs: 可选，要添加到节点的其他属性的字典
        """
        attrs = {'code': code_content}
        if file_path:
            attrs['file_path'] = file_path
        if extra_attrs:
            attrs.update(extra_attrs)
            
        if not graph.has_node(node_id):
            graph.add_node(node_id, **attrs)
        else:
            # 更新现有节点的属性
            graph.nodes[node_id].update(attrs)

    def _build_folder_structure(self):
        """构建文件夹结构图，节点的code属性存储路径名"""
        for root, dirs, files in os.walk(self.repo_path):
            root_path_str = str(Path(root).relative_to(self.repo_path))
            
            if not self.graphs.folder_structure.has_node(root_path_str):
                self.graphs.folder_structure.add_node(root_path_str, code=root_path_str)
            else:
                if 'code' not in self.graphs.folder_structure.nodes[root_path_str]:
                    self.graphs.folder_structure.nodes[root_path_str]['code'] = root_path_str

            for d in dirs:
                child_path_str = str(Path(root, d).relative_to(self.repo_path))
                if not self.graphs.folder_structure.has_node(child_path_str):
                    self.graphs.folder_structure.add_node(child_path_str, code=child_path_str)
                self.graphs.folder_structure.add_edge(root_path_str, child_path_str)
                self.graphs.folder_structure.nodes[root_path_str]['code'] = root_path_str
                self.graphs.folder_structure.nodes[child_path_str]['code'] = child_path_str

            for f in files:
                # if f.endswith('.py'): # Keep or remove based on whether all files or only .py are needed
                child_path_str = str(Path(root, f).relative_to(self.repo_path))
                if not self.graphs.folder_structure.has_node(child_path_str):
                    self.graphs.folder_structure.add_node(child_path_str, code=child_path_str)
                self.graphs.folder_structure.add_edge(root_path_str, child_path_str)
                self.graphs.folder_structure.nodes[root_path_str]['code'] = root_path_str
                self.graphs.folder_structure.nodes[child_path_str]['code'] = child_path_str

    def _build_cross_file_deps(self):
        """构建跨文件依赖图，节点的code属性存储文件名"""
        # 使用AST来分析文件之间的导入关系
        python_files = {}
        
        # 第一遍：收集所有Python文件的路径
        for py_file in self.repo_path.rglob('*.py'):
            rel_path = str(py_file.relative_to(self.repo_path))
            python_files[rel_path] = py_file
            if not self.graphs.cross_file_deps.has_node(rel_path):
                self.graphs.cross_file_deps.add_node(rel_path, code=rel_path)
        
        # 第二遍：分析每个文件的导入
        for rel_path, py_file in python_files.items():
            try:
                with open(py_file) as f:
                    try:
                        tree = ast.parse(f.read())
                    except SyntaxError as se:
                        logging.warning(f"Syntax error in {py_file}: {se}. Skipping this file for cross-file dependencies.")
                        continue
                
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
        """将模块导入添加为文件依赖关系，并为节点添加code属性"""
        parts = imported_module.split('.')
        potential_paths = []
        potential_paths.append(f"{'/'.join(parts)}.py")
        potential_paths.append(f"{'/'.join(parts)}/__init__.py")
        if len(parts) > 1:
            potential_paths.append(f"{'/'.join(parts[:-1])}/{parts[-1]}.py")
        
        for target_file_candidate in potential_paths:
            if target_file_candidate in python_files:
                if not self.graphs.cross_file_deps.has_node(source_file):
                    self.graphs.cross_file_deps.add_node(source_file, code=source_file)
                else:
                     self.graphs.cross_file_deps.nodes[source_file]['code'] = source_file
                
                if not self.graphs.cross_file_deps.has_node(target_file_candidate):
                    self.graphs.cross_file_deps.add_node(target_file_candidate, code=target_file_candidate)
                else:
                    self.graphs.cross_file_deps.nodes[target_file_candidate]['code'] = target_file_candidate
                
                self.graphs.cross_file_deps.add_edge(source_file, target_file_candidate)
                return
        
        prefix = '/'.join(parts)
        for file_path_candidate in python_files.keys():
            if file_path_candidate.startswith(prefix + '/') or file_path_candidate == prefix + '.py':
                if not self.graphs.cross_file_deps.has_node(source_file):
                    self.graphs.cross_file_deps.add_node(source_file, code=source_file)
                else:
                    self.graphs.cross_file_deps.nodes[source_file]['code'] = source_file

                if not self.graphs.cross_file_deps.has_node(file_path_candidate):
                    self.graphs.cross_file_deps.add_node(file_path_candidate, code=file_path_candidate)
                else:
                    self.graphs.cross_file_deps.nodes[file_path_candidate]['code'] = file_path_candidate

                self.graphs.cross_file_deps.add_edge(source_file, file_path_candidate)
                return

    def _build_call_graph(self):
        """构建调用图，节点的code属性存储函数名"""
        for py_file in self.repo_path.rglob('*.py'):
            try:
                with open(py_file, encoding='utf-8') as f:
                    try:
                        tree = ast.parse(f.read())
                    except SyntaxError as se:
                        logging.warning(f"Syntax error in {py_file}: {se}. Skipping this file for call graph.")
                        continue
                    
                class CallVisitor(ast.NodeVisitor):
                    def __init__(self, graph, file_path_str):
                        self.calls = []
                        self.current_func = None
                        self.graph = graph
                        self.file_path_str = file_path_str 

                    def visit_FunctionDef(self, node):
                        old_func = self.current_func
                        self.current_func = node.name 
                        
                        if not self.graph.has_node(self.current_func):
                            self.graph.add_node(self.current_func, code=self.current_func)
                        else: 
                            if 'code' not in self.graph.nodes[self.current_func]:
                                 self.graph.nodes[self.current_func]['code'] = self.current_func

                        self.generic_visit(node)
                        self.current_func = old_func
                        
                    def visit_Call(self, node):
                        callee_name = None
                        if isinstance(node.func, ast.Name):
                            callee_name = node.func.id
                        elif isinstance(node.func, ast.Attribute):
                            callee_name = node.func.attr 
                        
                        if callee_name and self.current_func:
                            self.calls.append((self.current_func, callee_name))
                        self.generic_visit(node)
                
                visitor = CallVisitor(self.graphs.call_graph, str(py_file.name))
                visitor.visit(tree)
                
                for caller, callee in visitor.calls:
                    if not self.graphs.call_graph.has_node(callee):
                        self.graphs.call_graph.add_node(callee, code=callee)
                    else:
                        if 'code' not in self.graphs.call_graph.nodes[callee]:
                            self.graphs.call_graph.nodes[callee]['code'] = callee
                    
                    self.graphs.call_graph.add_edge(caller, callee)
            except Exception as e: 
                logging.warning(f"Failed to process call graph for {py_file}: {e}", exc_info=True)

    def _build_type_deps(self):
        """构建类型依赖图，节点的code属性存储类型名"""
        # Ensure mypy is installed and relevant types are imported
        # from mypy.nodes import MypyFile, ClassDef, FuncDef (adjust as needed)
        # from mypy.types import Type, Instance, UnionType, TupleType, CallableType, AnyType, NoneType
        # from mypy.build import build, BuildSource (or appropriate API for your mypy version)

        for py_file in self.repo_path.rglob('*.py'):
            try:
                # This is a placeholder for actual mypy invocation
                # You'll need to replace this with your mypy API usage if it's different
                # For example, using mypy.api.run or a similar newer API if available
                # The `build.build` API might be older or specific to certain mypy versions.
                # result = build.build(sources=[BuildSource(str(py_file), None, None)], options=self.mypy_options) # Pass options if needed
                # tree = result.files.get(str(py_file.resolve())) # Use resolved absolute path for key
                
                # Placeholder: to avoid breaking if mypy isn't set up, we'll log and skip.
                # Replace with your actual mypy integration.
                logging.info(f"Mypy type analysis for {py_file} - Placeholder, actual mypy call needed.")
                # if tree:
                #    self._process_mypy_tree(tree)
                # else:
                #    logging.warning(f"Mypy did not produce a tree for {py_file}")
                pass # Remove this pass when mypy integration is active

            except Exception as e:
                logging.warning(f"Failed to process types in {py_file} (mypy integration): {e}", exc_info=True)

    def _build_class_inheritance(self):
        """构建类继承图，节点的code属性存储类名"""
        for py_file in self.repo_path.rglob('*.py'):
            try:
                with open(py_file, encoding='utf-8') as f:
                    try:
                        tree = ast.parse(f.read())
                    except SyntaxError as se:
                        logging.warning(f"Syntax error in {py_file}: {se}. Skipping this file for class inheritance.")
                        continue
                    
                class InheritanceVisitor(ast.NodeVisitor):
                    def __init__(self, graph):
                        self.inheritance = []
                        self.graph = graph
                        
                    def visit_ClassDef(self, node):
                        child_class_name = node.name
                        if not self.graph.has_node(child_class_name):
                            self.graph.add_node(child_class_name, code=child_class_name)
                        else:
                            if 'code' not in self.graph.nodes[child_class_name]:
                                self.graph.nodes[child_class_name]['code'] = child_class_name

                        for base in node.bases:
                            parent_class_name = None
                            if isinstance(base, ast.Name):
                                parent_class_name = base.id
                            elif isinstance(base, ast.Attribute):
                                parent_class_name = base.attr 
                            
                            if parent_class_name:
                                self.inheritance.append((child_class_name, parent_class_name))
                        self.generic_visit(node)
                
                visitor = InheritanceVisitor(self.graphs.class_inheritance)
                visitor.visit(tree)
                
                for child, parent in visitor.inheritance:
                    if not self.graphs.class_inheritance.has_node(parent):
                        self.graphs.class_inheritance.add_node(parent, code=parent)
                    else:
                        if 'code' not in self.graphs.class_inheritance.nodes[parent]:
                             self.graphs.class_inheritance.nodes[parent]['code'] = parent
                    
                    self.graphs.class_inheritance.add_edge(child, parent)
            except Exception as e:
                logging.warning(f"Failed to process inheritance in {py_file}: {e}", exc_info=True)

    def _build_ast_graph(self):
        """构建AST图，并为节点添加code属性"""
        for py_file in self.repo_path.rglob('*.py'):
            with open(py_file, 'r', encoding='utf-8') as f:
                source_code = f.read()
                try:
                    tree = ast.parse(source_code)
                except SyntaxError as se:
                    logging.warning(f"Syntax error in {py_file}: {se}. Skipping this file for AST graph.")
                    continue
            
            # 存储文件的源码行，方便后续基于行号提取（如果需要）
            # source_lines = source_code.splitlines()

            def get_node_uid(node, filename_str):
                # 为AST节点生成一个唯一的、包含信息的ID
                if isinstance(node, ast.Module): # 根Module节点
                    return f"ast:{filename_str}:module"
                # 确保节点有行列号属性，对于某些AST节点，它们可能不存在
                lineno = getattr(node, 'lineno', 0)
                col_offset = getattr(node, 'col_offset', 0)
                return f"ast:{filename_str}:{lineno}:{col_offset}:{type(node).__name__}"

            def get_code_snippet_for_node(node, full_source_code):
                # 优先使用 ast.get_source_segment (Python 3.8+)
                if hasattr(ast, 'get_source_segment'):
                    segment = ast.get_source_segment(full_source_code, node)
                    return segment if segment is not None else ""
                return "" # 如果无法获取代码，返回空

            # 1. 遍历所有AST节点，为它们添加属性并加入图
            for node_obj in ast.walk(tree):
                node_identifier = get_node_uid(node_obj, py_file.name)
                
                attributes = {
                    'node_type': type(node_obj).__name__,
                    'file_path': str(py_file.relative_to(self.repo_path.parent)), # 存储相对路径
                    'code': get_code_snippet_for_node(node_obj, source_code)
                }
                if hasattr(node_obj, 'name'): # 如 FunctionDef, ClassDef
                    attributes['name'] = node_obj.name
                elif hasattr(node_obj, 'id'): # 如 Name (变量名)
                    attributes['name'] = node_obj.id
                elif hasattr(node_obj, 'attr'): # 如 Attribute (obj.attr)
                    attributes['name'] = node_obj.attr
                
                if isinstance(node_obj, ast.Constant) and not isinstance(node_obj.value, ast.AST): # Python 3.8+
                    attributes['value'] = str(node_obj.value)
                elif isinstance(node_obj, ast.Str): # Python < 3.8
                    attributes['value'] = node_obj.s
                elif isinstance(node_obj, ast.Num): # Python < 3.8
                    attributes['value'] = str(node_obj.n)

                self._add_node_with_code(self.graphs.ast, node_identifier, attributes['code'], str(py_file.relative_to(self.repo_path.parent)), extra_attrs=attributes)

            # 2. 添加边
            def add_edges_recursively(parent_ast_node):
                parent_uid = get_node_uid(parent_ast_node, py_file.name)
                for child_ast_node in ast.iter_child_nodes(parent_ast_node):
                    child_uid = get_node_uid(child_ast_node, py_file.name)
                    # 确保子节点也已添加（在上面的 ast.walk 中已完成）
                    self.graphs.ast.add_edge(parent_uid, child_uid)
                    add_edges_recursively(child_ast_node)
            
            add_edges_recursively(tree) # 从根节点开始递归添加边

    def _build_cfg(self):
        """构建控制流图，节点的code属性存储块对应的代码行"""
        try:
            from staticfg import CFGBuilder
        except ImportError:
            logging.error("staticfg library not found. CFG will not be built. pip install staticfg")
            return

        for py_file in self.repo_path.rglob('*.py'):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    source_code = f.read()
                source_lines = source_code.splitlines()

                try:
                    # Using str(py_file.resolve()) for path consistency
                    cfg = CFGBuilder().build_from_file(py_file.name, str(py_file.resolve())) 
                except SyntaxError as se:
                    logging.warning(f"Syntax error when building CFG for {py_file}: {se}")
                    continue 
                except Exception as e_cfg:
                    logging.warning(f"Failed to build CFG (initialization) for {py_file}: {e_cfg}", exc_info=True)
                    continue

                file_rel_path_str = str(py_file.relative_to(self.repo_path.parent))
                
                # In staticfg, the CFG object has blocks as a dictionary, not a list
                # The keys are the block IDs and the values are the Block objects
                for block_id, block in cfg.blocks.items():
                    # Extract code content for this block
                    block_code_content = ""
                    if hasattr(block, 'source') and block.source:
                        # In staticfg, blocks have a 'source' attribute with the code
                        block_code_content = block.source
                    else:
                        # Fallback: try to extract code from line numbers if available
                        if hasattr(block, 'at') and block.at:
                            try:
                                # 'at' attribute contains line numbers (1-based)
                                block_code_lines = [source_lines[line_num - 1] for line_num in sorted(block.at) 
                                                   if 0 < line_num <= len(source_lines)]
                                block_code_content = "\n".join(block_code_lines)
                            except (IndexError, AttributeError) as e:
                                logging.warning(f"Error extracting code for block {block_id} in {py_file}: {e}")
                    
                    # Create a unique ID for this block
                    block_uid = f"cfg:{file_rel_path_str}:{block_id}"

                    # Add the block as a node in our graph
                    node_attrs = {
                        'code': block_code_content, 
                        'file_path': file_rel_path_str, 
                        'block_id_orig': block_id
                    }
                    
                    if not self.graphs.cfg.has_node(block_uid):
                        self.graphs.cfg.add_node(block_uid, **node_attrs)
                    else: 
                        self.graphs.cfg.nodes[block_uid].update(node_attrs)

                    # Add edges to successor blocks
                    # In staticfg, blocks have 'exits' which are the edges to successor blocks
                    if hasattr(block, 'exits'):
                        for exit_edge in block.exits:
                            if hasattr(exit_edge, 'target') and exit_edge.target:
                                # Get the target block ID
                                succ_block_id = exit_edge.target.id
                                succ_block = exit_edge.target
                                
                                # Create a unique ID for the successor block
                                succ_block_uid = f"cfg:{file_rel_path_str}:{succ_block_id}"
                                
                                # Extract code content for the successor block
                                succ_block_code_content = ""
                                if hasattr(succ_block, 'source') and succ_block.source:
                                    succ_block_code_content = succ_block.source
                                else:
                                    # Fallback: try to extract code from line numbers
                                    if hasattr(succ_block, 'at') and succ_block.at:
                                        try:
                                            succ_block_code_lines = [source_lines[line_num - 1] 
                                                                    for line_num in sorted(succ_block.at) 
                                                                    if 0 < line_num <= len(source_lines)]
                                            succ_block_code_content = "\n".join(succ_block_code_lines)
                                        except (IndexError, AttributeError) as e:
                                            logging.warning(f"Error extracting code for successor block {succ_block_id} in {py_file}: {e}")
                                
                                # Add the successor block as a node in our graph
                                succ_node_attrs = {
                                    'code': succ_block_code_content, 
                                    'file_path': file_rel_path_str, 
                                    'block_id_orig': succ_block_id
                                }
                                
                                if not self.graphs.cfg.has_node(succ_block_uid):
                                    self.graphs.cfg.add_node(succ_block_uid, **succ_node_attrs)
                                else:
                                    self.graphs.cfg.nodes[succ_block_uid].update(succ_node_attrs)
                                
                                # Add an edge from the current block to the successor block
                                self.graphs.cfg.add_edge(block_uid, succ_block_uid)
            
            except Exception as e:
                logging.warning(f"Generic error in _build_cfg for {py_file}: {e}", exc_info=True)

    def _build_dfg(self):
        """构建数据流图，并为节点添加code属性"""
        for py_file in self.repo_path.rglob('*.py'):
            with open(py_file, 'r', encoding='utf-8') as f:
                source_code = f.read()
            try:
                tree = ast.parse(source_code)
            except SyntaxError as se:
                logging.warning(f"Syntax error in {py_file}: {se}. Skipping this file for DFG.")
                continue

            # (与_build_ast_graph中类似的 get_node_uid 和 get_code_snippet_for_node 辅助函数)
            def get_node_uid(node, filename_str): # (复用或重新定义)
                if isinstance(node, ast.Module): return f"dfg:{filename_str}:module" # DFG节点前缀
                lineno = getattr(node, 'lineno', 0)
                col_offset = getattr(node, 'col_offset', 0)
                return f"dfg:{filename_str}:{lineno}:{col_offset}:{type(node).__name__}"

            def get_code_snippet_for_node(node, full_source_code): # (复用或重新定义)
                if hasattr(ast, 'get_source_segment'):
                    segment = ast.get_source_segment(full_source_code, node)
                    return segment if segment is not None else ""
                return ""
            
            # 确保DFG中涉及的AST节点先被添加到图中并带有属性
            # 这部分可能需要与AST图的节点共享，或者为DFG创建独立的节点（如果它们的属性需求不同）
            # 假设我们复用AST节点的UID和属性添加逻辑，只是边表示数据流
            
            # 先确保所有可能成为DFG节点的AST节点都已通过类似AST构建的方式被“知道”其属性
            # (一个简化的方式是，假设AST图已经构建完毕，并且节点已包含所需属性)
            # 或者，在DFG构建时也提取这些属性

            class DataFlowVisitor(ast.NodeVisitor):
                def __init__(self, graph_obj, filename_str, full_source_code, file_path_for_nodes, builder):
                    self.graph = graph_obj
                    self.filename = filename_str
                    self.source_code = full_source_code
                    self.file_path = file_path_for_nodes
                    self.builder = builder  # Reference to the MultiLevelGraphBuilder instance
                    self.definitions = {} # 存储变量定义处的AST节点对象

                def _ensure_node_in_graph(self, ast_node_obj):
                    """确保AST节点作为DFG节点存在于图中，并拥有属性"""
                    node_uid = get_node_uid(ast_node_obj, self.filename)
                    if not self.graph.has_node(node_uid):
                        attributes = {
                            'node_type': type(ast_node_obj).__name__,
                            'file_path': self.file_path,
                            'code': get_code_snippet_for_node(ast_node_obj, self.source_code)
                        }
                        if hasattr(ast_node_obj, 'name'): attributes['name'] = ast_node_obj.name
                        elif hasattr(ast_node_obj, 'id'): attributes['name'] = ast_node_obj.id
                        elif hasattr(ast_node_obj, 'attr'): attributes['name'] = ast_node_obj.attr
                        if isinstance(ast_node_obj, ast.Constant) and not isinstance(ast_node_obj.value, ast.AST):
                            attributes['value'] = str(ast_node_obj.value)
                        # ... (其他属性)
                        self.builder._add_node_with_code(self.graph, node_uid, attributes['code'], self.file_path, extra_attrs=attributes)
                    return node_uid

                def visit_Assign(self, node):
                    # node.value 是赋值的右侧表达式的根AST节点
                    # node.targets 是赋值左侧的目标列表
                    rhs_node_obj = node.value 
                    rhs_uid = self._ensure_node_in_graph(rhs_node_obj)

                    for target in node.targets:
                        if isinstance(target, ast.Name): # 简单变量赋值 a = ...
                            target_uid = self._ensure_node_in_graph(target)
                            self.definitions[target.id] = target_uid # 记录变量名到其定义节点的UID
                            # DFG边：从表达式结果流向变量定义处 (或者反之，取决于DFG定义)
                            # 这里假设从 RHS 流向 LHS 的 target 节点
                            self.graph.add_edge(rhs_uid, target_uid, type='assignment_flow')
                        # (可以处理更复杂的赋值目标，如 a.b, a[0] 等)
                    self.generic_visit(node) # 继续访问子节点

                def visit_Name(self, node): # 当代码中使用一个变量时
                    if isinstance(node.ctx, ast.Load) and node.id in self.definitions:
                        # 这是一个变量的读取 (use)
                        use_uid = self._ensure_node_in_graph(node)
                        def_uid = self.definitions[node.id] # 获取该变量定义处的UID
                        # DFG边：从定义处流向使用处
                        self.graph.add_edge(def_uid, use_uid, type='data_flow')
                    self.generic_visit(node)
            
            file_path_str = str(py_file.relative_to(self.repo_path.parent))
            dfg_visitor = DataFlowVisitor(self.graphs.dfg, py_file.name, source_code, file_path_str, self)
            dfg_visitor.visit(tree)

    def _get_referenced_types(self, type_obj):
        """从mypy类型对象中提取引用的类型名称字符串 (需要完善以处理所有Mypy类型)"""
        # Ensure relevant mypy types are imported for isinstance checks if not using quoted types
        # e.g. from mypy.types import Instance, UnionType, TupleType, CallableType, AnyType, NoneType, TypeVarType
        referenced_types = [] 
        if type_obj is None: return []

        # Simple name attribute check (might be too generic, but a fallback)
        if hasattr(type_obj, 'name') and isinstance(type_obj.name, str):
            referenced_types.append(type_obj.name)
        
        if hasattr(type_obj, 'fullname') and isinstance(type_obj.fullname, str):
            referenced_types.append(type_obj.fullname)

        # Specific Mypy type handling (requires mypy.types to be importable)
        try:
            from mypy.types import Instance, UnionType, TupleType, CallableType, AnyType, NoneType as MypyNoneType, TypeVarType
            if isinstance(type_obj, Instance):
                if type_obj.type and hasattr(type_obj.type, 'fullname'): # type_obj.type is TypeInfo
                    referenced_types.append(type_obj.type.fullname)
                for arg_type in type_obj.args:
                    referenced_types.extend(self._get_referenced_types(arg_type))
            elif isinstance(type_obj, UnionType):
                for item_type in type_obj.items:
                    referenced_types.extend(self._get_referenced_types(item_type))
            elif isinstance(type_obj, TupleType):
                for item_type in type_obj.items:
                    referenced_types.extend(self._get_referenced_types(item_type))
            elif isinstance(type_obj, CallableType):
                for arg_t in type_obj.arg_types:
                    referenced_types.extend(self._get_referenced_types(arg_t))
                referenced_types.extend(self._get_referenced_types(type_obj.ret_type))
            elif isinstance(type_obj, TypeVarType): # e.g. _T = TypeVar('_T')
                if type_obj.fullname:
                    referenced_types.append(type_obj.fullname)
            elif isinstance(type_obj, (AnyType, MypyNoneType)):
                pass
        except ImportError:
            logging.debug("Mypy types not available for detailed type parsing in _get_referenced_types.")
            # Fallback to basic name if mypy types can't be imported (e.g. mypy not fully installed/available)
            if hasattr(type_obj, 'name') and isinstance(type_obj.name, str) and type_obj.name not in referenced_types:
                 referenced_types.append(type_obj.name)
            if hasattr(type_obj, 'fullname') and isinstance(type_obj.fullname, str) and type_obj.fullname not in referenced_types:
                 referenced_types.append(type_obj.fullname)

        # Filter out None, empty strings and non-string types, then unique
        return list(set(name for name in referenced_types if name and isinstance(name, str)))
    
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
            ('ast', self.graphs.ast),
            ('cfg', self.graphs.cfg),
            ('dfg', self.graphs.dfg)
        ]:
            # 为每个子图的节点添加图类型属性
            for node in graph.nodes():
                self.combined_graph.add_node(f"{graph_name}:{node}", 
                                            code=graph.nodes[node]['code'],
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
            for ast_node in self.graphs.ast.nodes():
                # 简化实现：使用节点ID不太可靠，实际应存储更多元数据
                # 这里仅作为示例
                self.combined_graph.add_edge(
                    f"call_graph:{func_node}",
                    f"ast:{ast_node}",
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
        for ast_node in self.graphs.ast.nodes():
            for cfg_node in self.graphs.cfg.nodes():
                # 简化实现：假设AST节点可能对应控制流节点
                self.combined_graph.add_edge(
                    f"ast:{ast_node}",
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


def preprocess_repobench_data(data_sample, repo_path, temp_repo_path):
    """
    处理 RepoBench 数据集的样本，
    1. 过滤掉 next_line 及其后续内容，防止数据泄露。具体处理方式是：
        读取每个代码文件，定位到 next_line，
        然后删除 next_line 以及后面的内容
        然后将内存中的代码，写入到文件中
        存储在
    如果提供了repo_path，会在原始仓库文件中定位并修改相应文件。
    
    Args:
        data_sample: RepoBench 数据集的一个样本
        repo_path: 仓库根目录路径，如果提供，将在实际文件中定位并过滤代码
    
    """
    # 获取必要信息
    next_line = data_sample.get('next_line', '')
    all_code = data_sample.get('all_code', '')
    file_path = data_sample.get('file_path', '')
    
    # 创建数据样本的副本
    processed_sample = dict(data_sample) if data_sample else {}

    try:
        # 检查仓库路径是否存在
        repo_path = Path(repo_path)
        if not repo_path.exists() or not repo_path.is_dir():
            logging.error(f"Repository path does not exist or is not a directory: {repo_path}")
            return processed_sample
            
        # 创建临时目录来存放过滤后的文件
        # 现在的 repo_path ，应该放在 cross_file_first 里面的一个 temp
        temp_repo_path.mkdir(exist_ok=True, parents=True)
        
        # 用于记录找到的文件
        found_files = []
        
        # 遍历仓库中的所有Python文件
        logging.info(f"Scanning repository {repo_path} for next_line: {next_line[:50]}...")
        
        for py_file in repo_path.rglob('*.py'):
            try:
                # 读取文件内容
                with open(py_file, 'r', encoding='utf-8') as f:
                    file_content = f.read()
                
                # 如果文件中包含 next_line
                if next_line in file_content:
                    logging.info(f"Found next_line in file: {py_file}")

                    # TODO 再找一下这个 next_line 前面的东西是不是 all_code
                    
                    # 定位 next_line 的位置
                    next_line_pos = file_content.find(next_line)
                    
                    # 创建过滤后的文件内容 (只保留 next_line 之前的内容)
                    filtered_content = file_content[:next_line_pos].rstrip() + '\n'
                    
                    # 写入过滤后的内容到临时文件
                    rel_path = py_file.relative_to(repo_path)
                    temp_file_path = temp_repo_path /rel_path
                    temp_file_path.parent.mkdir(exist_ok=True, parents=True)
                    
                    with open(temp_file_path, 'w', encoding='utf-8') as f:
                        f.write(filtered_content)
                    
                    # found_files.append({
                    #     'original_file': str(py_file),
                    #     'filtered_file': str(temp_file_path),
                    #     'relative_path': str(rel_path)
                    # })
                    
                    logging.info(f"Created filtered file: {temp_file_path}")
            except Exception as e:
                logging.warning(f"Error processing file {py_file}: {e}")
        
        # if found_files:
        #     # 添加所有找到的文件信息到返回结果中
        #     processed_sample['filtered_files'] = found_files
            
        #     # 如果想保持向后兼容，还可以设置第一个找到的文件作为 filtered_file_path
        #     processed_sample['filtered_file_path'] = found_files[0]['filtered_file']
            
        #     logging.info(f"Found and filtered {len(found_files)} files containing next_line")
        # else:
        #     logging.warning(f"Could not find any files containing next_line: {next_line[:50]}...")
    except Exception as e:
        logging.error(f"Error processing repository: {e}")


def process_repobench_repo(repo_path: str) -> MultiLevelGraph:
    """
    处理 RepoBench 仓库，构建多层次图结构。
    
    Args:
        repo_path (str): 仓库路径    
    Returns:
        MultiLevelGraph: 构建的图结构
    """
    
    # 初始化图构建器
    builder = MultiLevelGraphBuilder(repo_path)
    
    # 构建仓库层级和模块层级的图
    builder.build_repo_level()
    builder.build_module_level()
    builder.build_function_level()
    
    # 构建组合图
    builder.build_combined_graph()
    
    return builder.graphs

def load_jsonl(fname):
    with open(fname, 'r', encoding='utf8') as f:
        lines = []
        for line in f:
            lines.append(json.loads(line))
        return lines

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
        "ast": graphs.ast,
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

    #原始
    # RepoBench数据集路径
    repobench_path = Path(f"/data/wby/allcode/repohyper/GRACE/dataset/hf_datasets/repobench_{language}_v1.1/cross_file_first/repos")
    repobench_root = Path(f"/data/wby/allcode/repohyper/GRACE/dataset/hf_datasets/repobench_{language}_v1.1/in_file")

    # repoeval-updated
    # repobench_path = Path(f"/data/wby/allcode/repohyper/GRACE/dataset/dataset_repoeval_updated/repos")
    # repobench_root = Path(f"/data/wby/allcode/repohyper/GRACE/dataset/dataset_repoeval_updated/repoeval_to_repobench")

    #crosscodeeval
    # repobench_path = Path(f"/data/wby/allcode/repohyper/GRACE/dataset/dataset_crosscodeeval/repos")
    # repobench_root = Path(f"/data/wby/allcode/repohyper/GRACE/dataset/dataset_crosscodeeval/data/python")

    #原始

    # 加载 RepoBench 数据集
    try:
        from datasets import load_dataset
        dataset = load_dataset(f"/data/wby/allcode/repohyper/GRACE/dataset/hf_datasets/repobench_{language}_v1.1", split=['in_file'])[0]
        logging.info(f"Successfully loaded RepoBench dataset with {len(dataset)} samples")
    except Exception as e:
        logging.error(f"Failed to load RepoBench dataset: {str(e)}")
        dataset = None

    #加载repoeval-updated数据集
    #dataset=load_jsonl(f"{repobench_root}/line_level.python.test.jsonl")

    # 加载crosscodeeval数据集
    #dataset = load_jsonl(f"{repobench_root}/line_completion.jsonl")

    # 用于测试的仓库列表 (可选择性过滤)
    test_repos = ["3D-DAM", "4dfy", "4k4d", "AA", "A3FL", "ace"]


    # 创建根目录
    processed_dir = repobench_root / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    graphs_dir = repobench_root / "graphs"
    graphs_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 直接遍历数据集中的所有样本
    if dataset is None:
        logging.error("No dataset provided for processing")
        return
    
    logging.info(f"Processing {len(dataset)} samples from dataset")
    
    # 统计信息
    processed_count = 0
    skipped_count = 0
    error_count = 0
            
    # 处理每个样本
    for idx, sample in enumerate(dataset):
        # 从样本中提取仓库名称
        repo_full_name = sample['repo_name']
        #原始
        repo_author, repo_name = repo_full_name.split('/')
        #repo_name=repo_full_name

        repo_path = repobench_path / repo_name
        temp_repo_path = repobench_root / 'temp' / repo_name
        
        # 如果有测试仓库列表，只处理其中的仓库
        # if test_repos and repo_name not in test_repos:
        #     skipped_count += 1
        #     continue
        
        # 根据样本的repo_name和索引创建文件夹
        sample_id = f"{idx}_{repo_name}"
        # 创建该样本的处理后的目录
        sample_processed_dir = processed_dir / sample_id
        sample_processed_dir.mkdir(parents=True, exist_ok=True)
        sample_graph_dir = graphs_dir / sample_id
        sample_graph_dir.mkdir(parents=True, exist_ok=True)

        # 如果已经处理过了这个样本，那么跳过，判断是否处理过的方式是：查看一下是否存在sample_graph_dir / "repo_multi_graph.pkl"
        # 如果存在，则跳过
        if (sample_graph_dir / "repo_multi_graph.pkl").exists():
            print(f"Sample {sample_id} has already been processed, skipping...")
            continue


        try:    
            # 预处理样本，过滤next_line及其后续内容
            preprocess_repobench_data(sample, repo_path,temp_repo_path)
            repo_path = sample_processed_dir
            
            # 为当前 repo 构建图结构
            # graphs = process_repobench_repo(str(repo_path))
            graphs = process_repobench_repo(str(temp_repo_path))

            # 保存networkx结构
            nx.write_graphml_lxml(graphs.combined_graph, sample_graph_dir / "repo_multi_graph.graphml")

            # # 可视化并保存图结构 TODO 这一步耗时非常长
            # visualize_and_save_graphs(graphs, sample_graph_dir)
            
            processed_count += 1
            logging.info(f"Successfully processed sample {idx}")
            
            # 每几个样本输出一下当前进度
            if idx % 10 == 0 and idx > 0:
                logging.info(f"Progress: {idx}/{len(dataset)} samples processed")
                
        except Exception as e:
            error_count += 1
            logging.error(f"Failed to process sample {idx}: {str(e)}")
    
    # 输出最终处理统计
    logging.info(f"Finished processing dataset")
    logging.info(f"Total samples: {len(dataset)}")
    logging.info(f"Successfully processed: {processed_count}")
    logging.info(f"Skipped: {skipped_count}")
    logging.info(f"Errors: {error_count}")



if __name__ == "__main__":
    main()