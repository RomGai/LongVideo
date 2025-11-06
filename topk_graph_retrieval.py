import torch
import torch.nn.functional as F
import networkx as nx
import core.vision_encoder.pe as pe
import core.vision_encoder.transforms as transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== 1️⃣ 加载图结构 =====
import pickle

with open("spatio_temporal_graph.pkl", "rb") as f:
    G = pickle.load(f)

print(f"Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges\n")

# ===== 2️⃣ 初始化模型与文本查询 =====
model_name = 'PE-Core-G14-448'
model = pe.CLIP.from_config(model_name, pretrained=True).to(device)
tokenizer = transforms.get_text_tokenizer(model.context_length)

query = "Chopping Tree in Minecraft"
text = tokenizer([query]).to(device)

with torch.no_grad():
    text_features = model.encode_text(text)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

# ===== 3️⃣ 计算每个节点与文本的相似度 =====
similarities = []
for i in range(G.number_of_nodes()):
    img_feat = G.nodes[i]["feature"].to(device)
    img_feat = img_feat / img_feat.norm()
    sim = torch.dot(img_feat, text_features.squeeze(0)).item()
    similarities.append((i, sim))

# 按相似度降序排列
similarities.sort(key=lambda x: x[1], reverse=True)
top1_node, top1_sim = similarities[0]

print("=== 🔍 Top-1 Most Similar Node ===")
print(f"Node {top1_node} | path={G.nodes[top1_node]['path']}")
print(f"Similarity: {top1_sim:.4f}")
print("-" * 80)

# ===== 4️⃣ 找出 Top-3 空间边 =====
spatial_neighbors = [
    (j, G.edges[top1_node, j]["weight"])
    for j in G.neighbors(top1_node)
    if G.edges[top1_node, j]["type"] == "spatial"
]
spatial_neighbors = sorted(spatial_neighbors, key=lambda x: x[1], reverse=True)[:3]

print("=== 🧭 Top-3 Spatial Neighbors (by edge weight) ===")
for rank, (j, w) in enumerate(spatial_neighbors, 1):
    print(f"[{rank}] Node {j} | weight={w:.4f} | path={G.nodes[j]['path']}")
print("-" * 80)

# ===== 5️⃣ 收集这些节点及它们的 temporal 邻居 =====
selected_nodes = {top1_node}
selected_nodes.update([j for j, _ in spatial_neighbors])

# 加入 temporal 邻居
for n in list(selected_nodes):
    for j in G.neighbors(n):
        if G.edges[n, j]["type"] == "temporal":
            selected_nodes.add(j)

print("=== 🕓 Final Collected Nodes (Top-1 + 3 spatial + temporal neighbors) ===")
for nid in sorted(selected_nodes):
    tag = " <-- [Top-1]" if nid == top1_node else ""
    print(f"Node {nid} | {G.nodes[nid]['path']}{tag}")

# 可选：返回对应路径列表
# selected_paths = [G.nodes[nid]["path"] for nid in sorted(selected_nodes)]

import json

# ===== ✅ 仅保存被选中节点的视频路径 =====
selected_paths = [G.nodes[nid]["path"] for nid in sorted(selected_nodes)]

# 打印查看
print(f"\n[✓] 共选出 {len(selected_paths)} 个视频：")
for p in selected_paths:
    print("  ", p)

# 保存为 json（更方便阅读）
with open("selected_video_paths.json", "w", encoding="utf-8") as f:
    json.dump(selected_paths, f, indent=4, ensure_ascii=False)

print("\n[✓] 已保存到 'selected_video_paths.json'")

# ===== 之后加载 =====
# with open("selected_video_paths.json", "r", encoding="utf-8") as f:
#     paths = json.load(f)