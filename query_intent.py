"""Utilities for analyzing query intent with a Qwen VL model."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

_processor = AutoProcessor.from_pretrained(_MODEL_ID)
_intent_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    _MODEL_ID, torch_dtype=torch_dtype=torch.float16, device_map="auto"
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

SUBTITLE_REWRITE_PROMPT = (
    "You assist with long-video question answering. The user query may contain quoted "
    "or paraphrased subtitles mixed with other instructions. Extract only the subtitle "
    "text that should be matched against a subtitle index, and rewrite the query so "
    "that it no longer contains any literal subtitle-related text (for example: "After the subtitle '......'"; "When the phrase '.....'") while keeping all other "
    "statements and the final question.\n"
    "Return a strict JSON object with keys: 'subtitle_text' (a single string with the "
    "subtitle text separated by spaces, or an empty string if none), 'cleaned_query' "
    "(the query rewritten without subtitle-related text but preserving the rest), and 'reason' "
    "(briefly explain your extraction).\n"
    "Query: {query}\n"
    "JSON:"
)


def _generate_response(messages: List[Dict[str, Any]], max_new_tokens: int = 256) -> str:
    chat_text = _processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = _processor(text=[chat_text], return_tensors="pt").to(_intent_model.device)
    with torch.no_grad():
        generated = _intent_model.generate(**inputs, max_new_tokens=max_new_tokens)

    new_tokens = generated[:, inputs["input_ids"].shape[-1] :]
    return _processor.batch_decode(new_tokens, skip_special_tokens=True)[0]


def _extract_json_object(response: str) -> Dict[str, Any]:
    json_match = re.search(r"\{.*\}", response, re.DOTALL)
    if not json_match:
        return {}
    try:
        return json.loads(json_match.group())
    except json.JSONDecodeError:
        return {}


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

    response = _generate_response(messages)
    payload = _extract_json_object(response)

    subtitle_needed = _to_bool(payload.get("subtitle_search"))
    time_needed = _to_bool(payload.get("time_search"))
    reason = payload.get("reason") if isinstance(payload.get("reason"), str) else ""

    return {
        "subtitle_search": subtitle_needed,
        "time_search": time_needed,
        "reason": reason,
        "raw_response": response.strip(),
    }


def rewrite_query_and_extract_subtitles(query: str) -> Dict[str, Any]:
    """Use the VL model to split subtitle text from the rest of the query."""

    formatted_prompt = SUBTITLE_REWRITE_PROMPT.format(query=query.strip())
    messages = [
        {
            "role": "system",
            "content": "You extract subtitle text and rewrite queries for video retrieval.",
        },
        {"role": "user", "content": formatted_prompt},
    ]

    response = _generate_response(messages, max_new_tokens=384)
    payload = _extract_json_object(response)

    subtitle_text = payload.get("subtitle_text")
    if isinstance(subtitle_text, str):
        subtitle_text = subtitle_text.strip()
    else:
        subtitle_text = ""

    cleaned_query = payload.get("cleaned_query")
    if isinstance(cleaned_query, str):
        cleaned_query = cleaned_query.strip()
    else:
        cleaned_query = ""

    reason = payload.get("reason") if isinstance(payload.get("reason"), str) else ""

    return {
        "subtitle_text": subtitle_text,
        "cleaned_query": cleaned_query,
        "reason": reason,
        "raw_response": response.strip(),
    }

