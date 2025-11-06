import torch
import torch.nn.functional as F
import networkx as nx

# ===== 1️⃣ 读取保存的结果 =====
data = torch.load("video_features_results.pt")

features = torch.cat([d["image_features"] for d in data], dim=0)   # [N, D]
video_paths = [d["video_path"] for d in data]
num_videos = len(video_paths)
print(f"Loaded {num_videos} videos with feature dim = {features.shape[1]}")

# ===== 2️⃣ 初始化图结构，并嵌入节点属性 =====
G = nx.Graph()
for i, (path, feat) in enumerate(zip(video_paths, features)):
    G.add_node(i, path=path, feature=feat)  # 每个节点包含路径和特征向量

# ===== 3️⃣ 添加时间边 =====
for i in range(num_videos - 1):
    G.add_edge(i, i + 1, type="temporal", weight=1.0)

# ===== 4️⃣ 计算余弦相似度矩阵 =====
sim_matrix = F.cosine_similarity(
    features.unsqueeze(1),  # [N, 1, D]
    features.unsqueeze(0),  # [1, N, D]
    dim=-1
)  # [N, N]

# ===== 5️⃣ 添加空间边（全连接版本） =====
for i in range(num_videos):
    for j in range(num_videos):
        if i == j:
            continue  # 不连接自己
        if abs(i - j) == 1:
            continue  # 排除相邻时间节点（保留 temporal edge 独立）
        sim = float(sim_matrix[i, j].item())
        G.add_edge(i, j, type="spatial", weight=sim)

print(f"\n[✓] Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges\n")

# ===== ✅ 示例：打印某个节点的属性 =====
node_id = 0
print(f"Node {node_id} info:")
print("  path:", G.nodes[node_id]['path'])
print("  feature vector shape:", G.nodes[node_id]['feature'].shape)

import pickle

# 保存图结构到本地
with open("spatio_temporal_graph.pkl", "wb") as f:
    pickle.dump(G, f)

print("✅ Graph saved to spatio_temporal_graph.pkl")


# 加载
# import pickle

# with open("spatio_temporal_graph.pkl", "rb") as f:
#     G_loaded = pickle.load(f)

# print(f"✅ Graph loaded: {G_loaded.number_of_nodes()} nodes, {G_loaded.number_of_edges()} edges")

# # 示例：验证节点属性
# node_id = 0
# print(f"Node {node_id} info:")
# print("  path:", G_loaded.nodes[node_id]['path'])
# print("  feature vector shape:", G_loaded.nodes[node_id]['feature'].shape)
