import argparse
import json
import os
import shutil
import tempfile
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

from split import extract_visual_embeddings, cluster_and_segment, export_segments
from chunk_embedding import compute_video_features
from build_graph import build_spatiotemporal_graph
from topk_graph_retrieval import retrieve_topk_segments
from reranker import rerank_segments
from query_intent import analyze_query_intent
from text_embedding import compute_similarities


_NEAR_ZERO_DURATION = 1e-3


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


def _seconds_to_timestamp(seconds: float) -> str:
    seconds = max(float(seconds), 0.0)
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _normalize_subtitle_time(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if ":" in text:
            parts = text.split(":")
            try:
                parts = [float(p) for p in parts]
            except ValueError:
                return None
            seconds = 0.0
            for part in parts:
                seconds = seconds * 60.0 + part
            return seconds
        try:
            return float(text)
        except ValueError:
            return None
    return None


def _load_subtitle_entries(path: Optional[str]) -> List[Dict[str, Any]]:
    if not path:
        return []
    if not os.path.exists(path):
        raise FileNotFoundError(f"Subtitle file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, dict):
        candidates = raw.get("subtitles") or raw.get("segments") or []
    else:
        candidates = raw

    entries: List[Dict[str, Any]] = []
    for item in candidates:
        if not isinstance(item, dict):
            continue
        start = _normalize_subtitle_time(item.get("start") or item.get("start_time"))
        end = _normalize_subtitle_time(item.get("end") or item.get("end_time"))
        text = item.get("text") or item.get("content") or item.get("line") or ""
        if start is None or end is None:
            continue
        entries.append({"start": float(start), "end": float(end), "text": str(text)})

    entries.sort(key=lambda x: x["start"])
    return entries


def _attach_subtitles(
    segment_infos: Iterable[Dict[str, Any]], subtitle_entries: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    subtitle_entries = list(subtitle_entries)
    enriched_segments: List[Dict[str, Any]] = []
    for info in segment_infos:
        start = float(info.get("start_sec", 0.0) or 0.0)
        end = float(info.get("end_sec", start))
        matched_texts: List[str] = []
        matched_entries: List[Dict[str, Any]] = []
        for entry in subtitle_entries:
            entry_start = float(entry.get("start", 0.0) or 0.0)
            entry_end = float(entry.get("end", entry_start) or entry_start)
            if entry_end < entry_start:
                entry_start, entry_end = entry_end, entry_start

            duration = entry_end - entry_start
            if duration <= _NEAR_ZERO_DURATION:
                if entry_start < start:
                    continue
                if entry_start > end:
                    break
            else:
                if entry_end < start:
                    continue
                if entry_start > end:
                    break
            matched_texts.append(entry["text"])
            matched_entries.append(entry)

        enriched = dict(info)
        enriched["subtitle_text"] = " ".join(t for t in matched_texts if t).strip()
        enriched["subtitle_entries"] = matched_entries
        enriched_segments.append(enriched)

    return enriched_segments


def _time_attribute_text(attrs: Dict[str, Any]) -> str:
    start = attrs.get("start_sec")
    end = attrs.get("end_sec")
    if start is None or end is None:
        return ""
    timestamp_start = _seconds_to_timestamp(start)
    timestamp_end = _seconds_to_timestamp(end)
    duration = max(float(end) - float(start), 0.0)
    return (
        f"Clip spanning {timestamp_start} to {timestamp_end} (duration {duration:.2f} seconds)."
    )


def _subtitle_attribute_text(attrs: Dict[str, Any]) -> str:
    text = attrs.get("subtitle_text")
    if not text:
        return ""
    return str(text)


def _retrieve_nodes_by_attribute(
    graph, query: str, text_getter, top_k: int
) -> List[Tuple[int, float, str]]:
    documents: List[str] = []
    node_ids: List[int] = []
    for node_id, attrs in graph.nodes(data=True):
        text = text_getter(attrs)
        if not text:
            continue
        node_ids.append(node_id)
        documents.append(text)

    if not documents:
        return []

    sims = compute_similarities(query, documents)
    scored = list(zip(node_ids, sims, documents))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[: max(top_k, 1)]


def _combine_segment_scores(info: Dict[str, Any]) -> float:
    vision_score = float(info.get("similarity") or 0.0)
    subtitle_score = float(info.get("subtitle_similarity") or 0.0)
    time_score = float(info.get("time_similarity") or 0.0)
    combined = vision_score + subtitle_score + time_score
    info["combined_score"] = combined
    return combined

def _merge_attribute_results(
    graph,
    base_segments: Iterable[Dict[str, Any]],
    intent: Dict[str, Any],
    query: str,
    attribute_top_k: int,
) -> List[Dict[str, Any]]:
    aggregated = {info["node_id"]: dict(info) for info in base_segments if "node_id" in info}

    for info in aggregated.values():
        info.setdefault("subtitle_similarity", 0.0)
        info.setdefault("time_similarity", 0.0)

    if intent.get("subtitle_search"):
        subtitle_hits = _retrieve_nodes_by_attribute(
            graph, query, _subtitle_attribute_text, attribute_top_k
        )
        for node_id, sim, _ in subtitle_hits:
            info = aggregated.get(node_id)
            if info is None:
                attrs = dict(graph.nodes[node_id])
                attrs["node_id"] = node_id
                attrs["similarity"] = 0.0
                attrs["subtitle_similarity"] = 0.0
                attrs["time_similarity"] = 0.0
                aggregated[node_id] = attrs
                info = attrs
            info["subtitle_similarity"] = float(sim)

    if intent.get("time_search"):
        time_hits = _retrieve_nodes_by_attribute(
            graph, query, _time_attribute_text, attribute_top_k
        )
        for node_id, sim, _ in time_hits:
            info = aggregated.get(node_id)
            if info is None:
                attrs = dict(graph.nodes[node_id])
                attrs["node_id"] = node_id
                attrs["similarity"] = 0.0
                attrs["subtitle_similarity"] = 0.0
                attrs["time_similarity"] = 0.0
                aggregated[node_id] = attrs
                info = attrs
            info["time_similarity"] = float(sim)

    merged = list(aggregated.values())
    for info in merged:
        _combine_segment_scores(info)

    merged.sort(key=lambda x: x.get("combined_score", 0.0), reverse=True)
    return merged


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
    subtitle_json: Optional[str] = None,
    attribute_top_k: int = 3,
    min_frames_per_clip: int = 6,
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

        subtitle_entries = _load_subtitle_entries(subtitle_json)
        if subtitle_entries:
            segment_infos = _attach_subtitles(segment_infos, subtitle_entries)
        else:
            segment_infos = list(segment_infos)

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

        intent = analyze_query_intent(query)
        print(
            "Query intent:",
            json.dumps({
                "subtitle_search": intent.get("subtitle_search"),
                "time_search": intent.get("time_search"),
                "reason": intent.get("reason"),
            }, ensure_ascii=False),
        )

        merged_segments = _merge_attribute_results(
            graph,
            selected_segments,
            intent=intent,
            query=query,
            attribute_top_k=attribute_top_k,
        )

        max_segments = max(top_k + attribute_top_k, len(selected_segments))
        selected_segments = merged_segments[:max_segments]

        final_frames = rerank_segments(
            selected_segments,
            query=query,
            frame_interval=rerank_frame_interval,
            top_frames=top_frames,
            output_dir=output_dir,
            min_frames_per_clip=min_frames_per_clip,
        )

        results_path = os.path.join(output_dir, "rerank_results.json")
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(_to_serializable(final_frames), f, ensure_ascii=False, indent=2)

        plan_path = os.path.join(output_dir, "retrieval_plan.json")
        with open(plan_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "intent": intent,
                    "selected_segments": _to_serializable(selected_segments),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

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
    parser.add_argument("--subtitle-json", type=str, default=None, dest="subtitle_json")
    parser.add_argument("--attribute-top-k", type=int, default=3, dest="attribute_top_k")
    parser.add_argument("--min-frames-per-clip", type=int, default=6, dest="min_frames_per_clip")

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
        subtitle_json=args.subtitle_json,
        attribute_top_k=args.attribute_top_k,
        min_frames_per_clip=args.min_frames_per_clip,
    )

    print(json.dumps(results, ensure_ascii=False, indent=2))
