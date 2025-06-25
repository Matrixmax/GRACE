# GRACE
GRACE: Graph-guided Repository-Aware Code Completion through Hierarchical Code Fusion


## 整体代码流程
### 1. 数据获取
1. repoeval_updated 数据集:运行代码 data_process/from_repobench_v1.1_donwload_repos.py 下载对应的 repo。cceval数据集不用获取

### 2. graph 构建
1. 运行代码 data_process/multilevel_graph_builder.py 生成对应的 graph

### 3. 运行 RAG 流程
1. 运行代码 model/coderag_pipeline.py 进行测试。


