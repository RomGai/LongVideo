from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import cv2
import os
from peft import PeftModel
adapter_dir = "./result"  # 👈 替换为你训练保存LoRA的目录
# ===== 1️⃣ 模型加载 =====
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct", torch_dtype="auto", device_map="auto"
)

# 2. 加载 LoRA adapter
model = PeftModel.from_pretrained(model, adapter_dir)

# 3. 合并权重（可选，但推荐推理前执行）
model = model.merge_and_unload()
model.eval()

processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
tokenizer = processor.tokenizer

id_1 = tokenizer.convert_tokens_to_ids("1")
id_2 = tokenizer.convert_tokens_to_ids("2")
id_3 = tokenizer.convert_tokens_to_ids("3")
id_4 = tokenizer.convert_tokens_to_ids("4")
id_5 = tokenizer.convert_tokens_to_ids("5")

# ===== 2️⃣ 计算logits概率 =====
@torch.no_grad()
def compute_logits(inputs, **kwargs):
    batch_scores = model(**inputs).logits[:, -1, :]
    id1_vector = batch_scores[:, id_1]
    id2_vector = batch_scores[:, id_2]
    id3_vector = batch_scores[:, id_3]
    id4_vector = batch_scores[:, id_4]
    id5_vector = batch_scores[:, id_5]
    batch_scores = torch.stack([id1_vector, id2_vector, id3_vector, id4_vector, id5_vector], dim=1)
    batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
    scores = batch_scores.exp().tolist()
    return scores

# ===== 3️⃣ 格式化消息 =====
def format_message(query, image_path):
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": f"""Given the image, which is a frame from a video, rate how relevant this frame is for answering the question: '{query}'.
                Output only one number from 1 to 5, where:
                1 = completely irrelevant — the frame provides no visual or contextual information related to the question or its answer.
                2 = slightly relevant — the frame shows general background or context, but it is unlikely to contribute to answering.
                3 = moderately relevant — the frame includes partial clues or indirect context that might help infer the answer, but the key evidence is missing.
                4 = mostly relevant — the frame provides substantial visual or contextual information that can be used to answer the question, though not fully decisive.
                5 = highly relevant — the frame clearly contains the decisive evidence or strong contextual cues that directly or indirectly support the correct answer."""},
            ],
        }
    ]

# ===== 4️⃣ 评估函数（接受视频） =====
def evaluate_video(video_path, query, frame_interval=10, temp_dir="frames_tmp"):
    """
    对视频的每一帧（间隔取样）与 query 组合打分。
    frame_interval: 每隔多少帧取一帧
    """
    os.makedirs(temp_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video loaded: {video_path} | {total_frames} frames at {fps} FPS")

    frame_paths = []
    idx = 0
    success, frame = cap.read()
    while success:
        if idx % frame_interval == 0:
            frame_path = os.path.join(temp_dir, f"frame_{idx:05d}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_paths.append(frame_path)
        success, frame = cap.read()
        idx += 1
    cap.release()
    print(f"Extracted {len(frame_paths)} frames")

    results = []
    for frame_path in frame_paths:
        messages = format_message(query, frame_path)
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to("cuda")

        probs = compute_logits(inputs)[0]
        weighted_sum = sum((i + 1) * p for i, p in enumerate(probs))
        results.append({"frame": frame_path, "score": weighted_sum})

    # 按相似度排序
    results.sort(key=lambda x: x["score"], reverse=True)

    return results

# ===== 5️⃣ 示例运行 =====
if __name__ == "__main__":
    query = "why did the boy stopped walking and looked up"
    video_path = "2503404966.mp4"  # 改成你的视频路径
    results = evaluate_video(video_path, query, frame_interval=5)
    # for r in results:
    #     print(f"Frame: {r['frame']}\nRelevance Score (1–5): {r['score']:.3f}\n")
    for r in results[:30]:  # 打印前10个最相关帧
        print(f"Frame: {r['frame']}\nRelevance Score (1–5): {r['score']:.3f}\n")
