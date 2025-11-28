"""Run LongVideoBench samples with LLaVA-Video-7B-Qwen2.

This script mirrors ``run_refined_samples.py`` but targets the LLaVA-Video
model family. It builds prompts that respect the interleaved frame + subtitle
order produced by ``LongVideoBenchDataset`` and sends the resulting video
context to the model for reasoning.
"""

import copy
import os
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from decord import cpu, VideoReader  # noqa: F401 (kept for parity with reference snippet)
from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token

from longvideobench_dataset import LongVideoBenchDataset

PRETRAINED_MODEL = os.getenv("LLAVA_PRETRAINED", "lmms-lab/LLaVA-Video-7B-Qwen2")
MODEL_NAME = os.getenv("LLAVA_MODEL_NAME", "llava_qwen")
DEVICE = os.getenv("LLAVA_DEVICE", "cuda")
DEVICE_MAP = os.getenv("LLAVA_DEVICE_MAP", "auto")
MAX_FRAMES = int(os.getenv("MAX_FRAMES", "64"))
NUM_SAMPLES = int(os.getenv("NUM_SAMPLES", "3"))
CONV_TEMPLATE = os.getenv("LLAVA_CONV_TEMPLATE", "qwen_1_5")


def _split_inputs(
    inputs: Sequence[Any],
) -> Tuple[List[np.ndarray], List[str], str, List[str]]:
    """Separate frames, subtitles, question, and candidates while keeping order."""

    frames: List[np.ndarray] = []
    timeline: List[str] = []
    question: str = ""
    candidates: List[str] = []

    for item in inputs:
        if isinstance(item, str):
            if item.startswith("Question: "):
                question = item[len("Question: ") :].strip()
            elif item.strip().lower().startswith("answer with"):
                # Terminal instruction; omit from timeline.
                continue
            elif len(item) >= 3 and item[1:3] == ". ":
                candidates.append(item.strip())
            else:
                timeline.append(f"Subtitle: {item.strip()}")
        else:
            frames.append(np.array(item))
            timeline.append(f"Frame {len(frames)}")

    return frames, timeline, question, candidates


def _build_prompt(timeline: List[str], question: str, candidates: List[str], frame_count: int) -> str:
    timeline_text = "\n".join(f"{idx + 1}. {entry}" for idx, entry in enumerate(timeline))

    option_text = "" if not candidates else "\nOptions:\n" + "\n".join(candidates)

    return (
        f"{DEFAULT_IMAGE_TOKEN}\n"
        f"The following visual input contains {frame_count} uniformly sampled frames.\n"
        "Subtitles are interleaved in the timeline below to preserve their order relative to the frames.\n"
        f"Timeline:\n{timeline_text}\n"
        f"Question: {question}{option_text}\n"
        "Answer with the option's letter if choices are provided."
    )


def _prepare_video_tensor(image_processor, frames: List[np.ndarray], device: str) -> torch.Tensor:
    if not frames:
        frames = [np.zeros((336, 336, 3), dtype=np.uint8)]

    video = image_processor.preprocess(np.stack(frames, axis=0), return_tensors="pt")[
        "pixel_values"
    ]
    return video.to(device)


def _run_sample(
    tokenizer, model, image_processor, sample: Dict[str, Any], device: str
) -> str:
    frames, timeline, question, candidates = _split_inputs(sample["inputs"])
    prompt = _build_prompt(timeline, question, candidates, len(frames))

    video = _prepare_video_tensor(image_processor, frames, device)
    images = [video]

    conv = copy.deepcopy(conv_templates[CONV_TEMPLATE])
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    final_prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(
        final_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(device)

    output = model.generate(
        input_ids,
        images=images,
        modalities=["video"],
        do_sample=False,
        temperature=0,
        max_new_tokens=512,
    )

    return tokenizer.batch_decode(output, skip_special_tokens=True)[0].strip()


def run_samples() -> None:
    tokenizer, model, image_processor, _ = load_pretrained_model(
        PRETRAINED_MODEL,
        None,
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map=DEVICE_MAP,
    )
    model.eval()

    dataset = LongVideoBenchDataset(output_root="./output", max_num_frames=MAX_FRAMES)

    for idx in range(min(NUM_SAMPLES, len(dataset))):
        sample = dataset[idx]
        reply = _run_sample(tokenizer, model, image_processor, sample, DEVICE)

        print(f"\n=== Sample {idx} (ID: {sample.get('id')}) ===")
        print("Correct choice:", sample.get("correct_choice", "@"))
        print("Model reply:", reply)


if __name__ == "__main__":
    run_samples()
