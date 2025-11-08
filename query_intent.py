"""Utilities for analyzing query intent with a Qwen VL model."""

from __future__ import annotations

import json
import re
from typing import Any, Dict

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

_processor = AutoProcessor.from_pretrained(_MODEL_ID)
_intent_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    _MODEL_ID, torch_dtype="auto", device_map="auto"
)
_intent_model.eval()


INTENT_PROMPT = (
    "You are helping with long-video retrieval. Analyze the user's query and decide "
    "whether the search should activate extra subtitle-based retrieval and/or time-range "
    "retrieval.\n"
    "Return a strict JSON object with the following keys: 'subtitle_search' (true or false), "
    "'time_search' (true or false), and 'reason' (a short sentence that justifies your decision).\n"
    "Focus only on explicit or strongly implied cues in the query (for example, emphasizing the beginning, end, or a specific time segment of the video; or pointing out a particular subtitle).\n"
    "Query: {query}\n"
    "JSON:"
)


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "y", "1"}:
            return True
        if lowered in {"false", "no", "n", "0"}:
            return False
    return False


def analyze_query_intent(query: str) -> Dict[str, Any]:
    """Infer whether subtitle or temporal retrieval is required for a query."""

    formatted_prompt = INTENT_PROMPT.format(query=query.strip())
    messages = [
        {"role": "system", "content": "You decide retrieval strategies for video search."},
        {"role": "user", "content": formatted_prompt},
    ]

    chat_text = _processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = _processor(text=[chat_text], return_tensors="pt").to(_intent_model.device)
    with torch.no_grad():
        generated = _intent_model.generate(**inputs, max_new_tokens=256)

    # Only keep newly generated tokens (excluding the prompt part)
    new_tokens = generated[:, inputs["input_ids"].shape[-1] :]
    response = _processor.batch_decode(new_tokens, skip_special_tokens=True)[0]

    json_match = re.search(r"\{.*\}", response, re.DOTALL)
    payload: Dict[str, Any]
    if json_match:
        try:
            payload = json.loads(json_match.group())
        except json.JSONDecodeError:
            payload = {}
    else:
        payload = {}

    subtitle_needed = _to_bool(payload.get("subtitle_search"))
    time_needed = _to_bool(payload.get("time_search"))
    reason = payload.get("reason") if isinstance(payload.get("reason"), str) else ""

    return {
        "subtitle_search": subtitle_needed,
        "time_search": time_needed,
        "reason": reason,
        "raw_response": response.strip(),
    }

