import core.vision_encoder.pe as pe
import core.vision_encoder.transforms as transforms
import os
import torch
from PIL import Image
import decord

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = 'PE-Core-G14-448'
model = pe.CLIP.from_config(model_name, pretrained=True).to(device)

preprocess = transforms.get_image_transform(model.image_size)

def preprocess_video(video_path, frame_interval=5, transform=None):
    vr = decord.VideoReader(video_path)
    total_frames = len(vr)
    frame_indices = list(range(0, total_frames, frame_interval))
    frames = vr.get_batch(frame_indices).asnumpy()
    preprocessed_frames = [transform(Image.fromarray(frame)) for frame in frames]
    return torch.stack(preprocessed_frames, dim=0)

# ====== 批处理整个文件夹 ======
video_dir = "../segments"

results = []

with torch.no_grad():
    for fname in os.listdir(video_dir):
        if not fname.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            continue

        video_path = os.path.join(video_dir, fname)
        try:
            video = preprocess_video(video_path, frame_interval=5, transform=preprocess)
            video = video.unsqueeze(0).to(device)

            image_features = model.encode_video(video)
            image_features /= image_features.norm(dim=-1, keepdim=True) #1,1280

            results.append({
                "video_path": video_path,
                "image_features": image_features.cpu(),
            })
            print(f"[✓] Processed {fname}")

        except Exception as e:
            print(f"[x] Failed on {fname}: {e}")

# 保存为 .pt
torch.save(results, "video_features_results.pt")
print(f"Saved {len(results)} videos to video_features_results.pt")
