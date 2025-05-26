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

# 在两个索引中搜索
code_ids, code_distances = code_index.knn_query(code_emb, k=k)
graph_ids, graph_distances = graph_index.knn_query(np.mean(graph_emb, axis=0), k=k)

# 分别处理代码和图的检索结果
code_results = []
for i in range(len(code_ids[0])):
    code_idx = code_ids[0][i]
    if code_idx < len(code_chunk_ids):
        code_id = code_chunk_ids[code_idx]
        # 获取实际代码内容
        if code_idx < len(all_code_snippets):
            code_content = all_code_snippets[code_idx]
            code_results.append({
                'id': code_id,
                'content': code_content,
                'score': alpha * (1 - code_distances[0][i]),
                'type': 'code'
            })

graph_results = []
for i in range(len(graph_ids[0])):
    graph_idx = graph_ids[0][i]
    if graph_idx < len(graph_node_ids):
        node_id = graph_node_ids[graph_idx]
        # 获取实际图节点内容
        if graph_idx < len(all_graph_snippets):
            graph_content = all_graph_snippets[graph_idx]
            graph_results.append({
                'id': node_id,
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