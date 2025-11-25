from torch.utils.data import Dataset
import os
import decord
from decord import VideoReader, cpu
import numpy as np
from PIL import Image
import torch

import json
from typing import Dict, List, Any, Tuple

def timestamp_to_seconds(timestamp):
    # Split the timestamp into hours, minutes, and seconds
    h, m, s = timestamp.split(':')
    # Convert hours, minutes, and total seconds (including fractions) to float and compute total seconds
    total_seconds = int(h) * 3600 + int(m) * 60 + float(s)
    return total_seconds

def load_video(video_file, duration, max_num_frames=16):
    from decord import VideoReader
    vr = VideoReader(video_file, ctx=cpu(0), num_threads=1)
    fps = vr.get_avg_fps()
    total_valid_frames = int(duration * fps)
    num_frames = min(max_num_frames, int(duration))

    frame_indices = [int(total_valid_frames / num_frames) * i for i in range(num_frames)]
    
    frames = vr.get_batch(frame_indices)
    if isinstance(frames, torch.Tensor):
        frames = frames.numpy()
    else:
        frames = frames.asnumpy()
    frame_timestamps = [frame_index / fps for frame_index in frame_indices]
    
    return [Image.fromarray(fr).convert("RGB") for fr in frames], frame_timestamps

def insert_subtitles(subtitles):
    interleaved_list = []
    cur_i = 0
    
    for subtitle in subtitles:
        if "timestamp" in subtitle:
            subtitle_text = subtitle["text"]
        else:
            subtitle_text = subtitle["line"]

        interleaved_list.append(subtitle_text)

    return interleaved_list
        
def insert_subtitles_into_frames(
    frames,
    frame_timestamps,
    subtitles,
    starting_timestamp_for_subtitles,
    duration,
    deduplicate_adjacent: bool = True,
):
    def _append_item(target_list: List[Any], item: Any):
        if not deduplicate_adjacent:
            target_list.append(item)
            return

        if not target_list:
            target_list.append(item)
            return

        last_item = target_list[-1]
        if isinstance(last_item, str) and isinstance(item, str):
            if last_item.strip() == item.strip():
                return
        target_list.append(item)

    interleaved_list: List[Any] = []
    cur_i = 0

    for subtitle in subtitles:
        if "timestamp" in subtitle:
            start, end = subtitle["timestamp"]

            if not isinstance(end, float):
                end = duration

            start -= starting_timestamp_for_subtitles
            end -= starting_timestamp_for_subtitles


            subtitle_timestamp = (start + end) / 2
            subtitle_text = subtitle["text"]
        else:
            start, end = subtitle["start"], subtitle["end"]
            start = timestamp_to_seconds(start)
            end = timestamp_to_seconds(end)
            start -= starting_timestamp_for_subtitles
            end -= starting_timestamp_for_subtitles

            subtitle_timestamp = (start + end) / 2
            subtitle_text = subtitle["line"]


        for i, (frame, frame_timestamp) in enumerate(zip(frames[cur_i:], frame_timestamps[cur_i:])):
                if frame_timestamp <= subtitle_timestamp:
                    #print("frame:", frame_timestamp)
                    _append_item(interleaved_list, frame)
                    cur_i += 1
                else:
                    break

        if end - start < 1:
            end = subtitle_timestamp + 0.5
            start = subtitle_timestamp - 0.5

        covering_frames = False
        for frame, frame_timestamp in zip(frames, frame_timestamps):
            if frame_timestamp < end and frame_timestamp > start:
                covering_frames = True
                break
        #
        if covering_frames:
            #print("subtitle:", subtitle_timestamp, start, end)
            _append_item(interleaved_list, subtitle_text)
        else:
            pass
            #print("leaving out subtitle:", start, end)

    for i, (frame, frame_timestamp) in enumerate(zip(frames[cur_i:], frame_timestamps[cur_i:])):
        #print(frame_timestamp)
        _append_item(interleaved_list, frame)

    return interleaved_list
    
def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _gather_subtitles(selected_segments: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], float]:
    subtitles: List[Dict[str, Any]] = []
    max_end = 0.0
    for segment in selected_segments:
        end = float(segment.get("end_sec") or 0.0)
        max_end = max(max_end, end)
        for entry in segment.get("subtitle_entries", []) or []:
            entry_start = float(entry.get("start") or 0.0)
            entry_end = float(entry.get("end") or entry_start)
            subtitles.append(
                {
                    "timestamp": (entry_start, entry_end),
                    "text": entry.get("text") or entry.get("line") or "",
                }
            )
    subtitles.sort(key=lambda x: x["timestamp"][0])
    return subtitles, max_end


def _load_frames_from_results(paths: List[Dict[str, Any]], max_num_frames: int) -> Tuple[List[Image.Image], List[float]]:
    sorted_frames = sorted(paths, key=lambda x: float(x.get("timestamp") or 0.0))
    if max_num_frames > 0 and len(sorted_frames) > max_num_frames:
        indices = np.linspace(0, len(sorted_frames) - 1, max_num_frames, dtype=int)
        sorted_frames = [sorted_frames[i] for i in indices]

    frames: List[Image.Image] = []
    timestamps: List[float] = []
    for item in sorted_frames:
        frame_path = item.get("output_path") or item.get("path")
        if not frame_path:
            continue
        frame_path = os.path.abspath(frame_path)
        if not os.path.exists(frame_path):
            continue
        with open(frame_path, "rb") as f:
            frame = Image.open(f).convert("RGB")
        frames.append(frame)
        timestamps.append(float(item.get("timestamp") or 0.0))

    return frames, timestamps


class LongVideoBenchDataset(Dataset):
    def __init__(
        self,
        output_root,
        max_num_frames=256,
        insert_text=True,
        insert_frame=True,
    ):
        super().__init__()
        self.output_root = output_root
        self.insert_text = insert_text
        self.max_num_frames = max_num_frames

        self.data: List[Dict[str, Any]] = []
        for name in sorted(os.listdir(output_root)):
            video_dir = os.path.join(output_root, name)
            if not os.path.isdir(video_dir):
                continue

            rerank_path = os.path.join(video_dir, "rerank_results.json")
            plan_path = os.path.join(video_dir, "retrieval_plan.json")
            time_focus_path = os.path.join(video_dir, "time_focus_results.json")

            if not os.path.exists(rerank_path) or not os.path.exists(plan_path):
                continue

            rerank_results = _load_json(rerank_path)
            retrieval_plan = _load_json(plan_path)
            time_focus_results = _load_json(time_focus_path) if os.path.exists(time_focus_path) else []

            selected_segments = retrieval_plan.get("selected_segments") or []
            subtitles, max_segment_end = _gather_subtitles(selected_segments)

            duration = float(retrieval_plan.get("video_metadata", {}).get("duration_sec") or max_segment_end)

            self.data.append(
                {
                    "id": name,
                    "rerank_results": rerank_results,
                    "time_focus_results": time_focus_results,
                    "subtitles": subtitles,
                    "duration": duration,
                    "question": retrieval_plan.get("query_variants", {}).get("original")
                    or retrieval_plan.get("query_variants", {}).get("vision")
                    or "",
                    "candidates": retrieval_plan.get("candidates", []),
                    "correct_choice": retrieval_plan.get("correct_choice", -1),
                }
            )

    def __getitem__(self, index):
        di = self.data[index]

        def _format_correct_choice() -> str:
            choice_index = int(di.get("correct_choice", -1))
            if choice_index < 0:
                return "@"
            return chr(ord("A") + choice_index)

        if self.max_num_frames == 0:
            inputs = []
            inputs += ["Question: " + (di.get("question") or "")]
            inputs += [". ".join([chr(ord("A") + i), candidate]) for i, candidate in enumerate(di.get("candidates", []))]
            inputs += ["Answer with the option's letter from the given choices directly."]
            return {"inputs": inputs, "correct_choice": _format_correct_choice(), "id": di["id"]}

        frames_info = list(di.get("rerank_results", [])) + list(di.get("time_focus_results", []))
        frames, frame_timestamps = _load_frames_from_results(frames_info, self.max_num_frames)

        subtitles = di.get("subtitles", [])
        inputs: List[Any] = []
        if self.insert_text:
            inputs = insert_subtitles_into_frames(
                frames,
                frame_timestamps,
                subtitles,
                starting_timestamp_for_subtitles=0.0,
                duration=di.get("duration", 0.0),
            )
        else:
            inputs = frames

        ##### YOU MAY MODIFY THE FOLLOWING PART TO ADAPT TO YOUR MODEL #####
        if di.get("question"):
            inputs += ["Question: " + di["question"]]
        if di.get("candidates"):
            inputs += [". ".join([chr(ord("A") + i), candidate]) for i, candidate in enumerate(di["candidates"])]
        inputs += ["Answer with the option's letter from the given choices directly."]
        ##### YOU MAY MODIFY THE PREVIOUS PART TO ADAPT TO YOUR MODEL #####

        ##### CORRECT CHOICE WILL BE "@" FOR TEST SET SAMPLES #####
        return {"inputs": inputs, "correct_choice": _format_correct_choice(), "id": di["id"]}

    def __len__(self):
        return len(self.data)

    def get_id(self, index):
        return self.data[index]["id"]


if __name__ == "__main__":
    db = LongVideoBenchDataset("./output")
    def _describe_inputs(inputs: List[Any]) -> List[str]:
        descriptions: List[str] = []
        for idx, item in enumerate(inputs):
            if isinstance(item, str):
                descriptions.append(f"{idx:02d} text: {item}")
            elif isinstance(item, Image.Image):
                descriptions.append(
                    f"{idx:02d} frame: size={item.size} mode={item.mode}"
                )
            else:
                descriptions.append(f"{idx:02d} unknown type: {type(item)}")
        return descriptions

    for i in range(min(10, len(db))):
        sample = db[i]
        print("\nSample ID:", sample["id"])
        frame_count = len([ele for ele in sample["inputs"] if not isinstance(ele, str)])
        print("Frame count:", frame_count)
        print("Detailed inputs:")
        for desc in _describe_inputs(sample["inputs"]):
            print(desc)
                     

            
            