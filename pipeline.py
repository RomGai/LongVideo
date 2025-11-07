import argparse
import json
import os
import shutil
import tempfile
from typing import Any, List, Dict

import numpy as np
import torch

from split import extract_visual_embeddings, cluster_and_segment, export_segments
from chunk_embedding import compute_video_features
from build_graph import build_spatiotemporal_graph
from topk_graph_retrieval import retrieve_topk_segments
from reranker import rerank_segments


def _to_serializable(obj: Any) -> Any:
    """Recursively convert objects containing numpy/torch types for JSON dumping."""

    if isinstance(obj, dict):
        return {key: _to_serializable(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_to_serializable(item) for item in obj]
    if isinstance(obj, tuple):
        return [_to_serializable(item) for item in obj]

    if isinstance(obj, (np.generic,)):
        return obj.item()

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, torch.Tensor):
        if obj.ndim == 0:
            return obj.item()
        return obj.detach().cpu().tolist()

    return obj


def run_pipeline(
    video_path: str,
    query: str,
    output_dir: str,
    frame_interval: int = 30,
    n_clusters: int = 30,
    min_segment_sec: float = 0.4,
    embedding_frame_interval: int = 10,
    top_k: int = 3,
    spatial_k: int = 3,
    rerank_frame_interval: int = 5,
    top_frames: int = 128,
    temporal_weight: float = 1.0,
) -> List[Dict]:
    """执行完整的长视频检索与重排序流程。"""

    temp_root = tempfile.mkdtemp(prefix="pipeline_tmp_")
    segments_dir = os.path.join(temp_root, "segments")
    os.makedirs(segments_dir, exist_ok=True)

    output_dir = os.path.abspath(output_dir)
    # 确保最终输出目录不在临时工作区中，避免清理临时目录时误删最终结果
    if os.path.commonpath([output_dir, temp_root]) == temp_root:
        raise ValueError(
            "Output directory must be outside the pipeline's temporary workspace."
        )

    os.makedirs(output_dir, exist_ok=True)

    try:
        embeddings, frames = extract_visual_embeddings(video_path, frame_interval=frame_interval)
        change_points, _ = cluster_and_segment(
            video_path,
            embeddings,
            frames,
            method="kmeans",
            n_clusters=n_clusters,
        )

        segment_infos = export_segments(
            video_path,
            change_points,
            output_dir=segments_dir,
            min_segment_sec=min_segment_sec,
        )

        if not segment_infos:
            raise RuntimeError("No segments were generated from the input video.")

        segment_features = compute_video_features(segment_infos, frame_interval=embedding_frame_interval)
        graph = build_spatiotemporal_graph(segment_features, temporal_weight=temporal_weight)

        selected_segments = retrieve_topk_segments(
            graph,
            query,
            top_k=top_k,
            spatial_k=spatial_k,
        )

        if not selected_segments:
            raise RuntimeError("No segments were selected from the retrieval stage.")

        final_frames = rerank_segments(
            selected_segments,
            query=query,
            frame_interval=rerank_frame_interval,
            top_frames=top_frames,
            output_dir=output_dir,
        )

        results_path = os.path.join(output_dir, "rerank_results.json")
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(_to_serializable(final_frames), f, ensure_ascii=False, indent=2)

        return final_frames

    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the full long-video retrieval pipeline")
    parser.add_argument("--video", type=str, default="./videos/f44gpGR4uWU.mp4",help="Path to the input video")
    parser.add_argument("--query", type=str, default="Question: In the opening of the video, there's a man wearing a black top and a gray hat in the car. In which of the following scenes does he appear later? A. In the water. B. In the car, on the sofa. C. On the mountain. D. In the bathroom. Answer with the option's letter from the given choices directly.",help="Text query for retrieval")
    parser.add_argument("--output", type=str, default="./output/",help="Directory to save the final ranked frames")
    parser.add_argument("--frame-interval", type=int, default=30, dest="frame_interval")
    parser.add_argument("--clusters", type=int, default=30, dest="n_clusters")
    parser.add_argument("--min-segment-sec", type=float, default=1, dest="min_segment_sec")
    parser.add_argument("--embed-frame-interval", type=int, default=10, dest="embedding_frame_interval")
    parser.add_argument("--top-k", type=int, default=3, dest="top_k")
    parser.add_argument("--spatial-k", type=int, default=3, dest="spatial_k")
    parser.add_argument("--rerank-frame-interval", type=int, default=5, dest="rerank_frame_interval")
    parser.add_argument("--top-frames", type=int, default=256, dest="top_frames")
    parser.add_argument("--temporal-weight", type=float, default=1.0, dest="temporal_weight")

    args = parser.parse_args()


    results = run_pipeline(
        video_path=args.video,
        query=args.query,
        output_dir=args.output,
        frame_interval=args.frame_interval,
        n_clusters=args.n_clusters,
        min_segment_sec=args.min_segment_sec,
        embedding_frame_interval=args.embedding_frame_interval,
        top_k=args.top_k,
        spatial_k=args.spatial_k,
        rerank_frame_interval=args.rerank_frame_interval,
        top_frames=args.top_frames,
        temporal_weight=args.temporal_weight,
    )

    print(json.dumps(results, ensure_ascii=False, indent=2))