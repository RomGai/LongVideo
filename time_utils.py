"""Utility helpers for working with video timestamps."""

from __future__ import annotations

import math


def seconds_to_timestamp(seconds: float) -> str:
    """Format seconds into ``HH:MM:SS`` string."""

    seconds = max(float(seconds), 0.0)
    whole_seconds = int(seconds)
    minutes, secs = divmod(whole_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def timestamp_label(seconds: float) -> str:
    """Generate a filesystem-friendly label for a timestamp."""

    seconds = max(float(seconds), 0.0)
    timestamp = seconds_to_timestamp(seconds)

    fractional = seconds - math.floor(seconds)
    if fractional <= 1e-3:
        return timestamp.replace(":", "h", 1).replace(":", "m", 1) + "s"

    millis = int(round(fractional * 1000))
    base = timestamp.replace(":", "h", 1).replace(":", "m", 1) + "s"
    return f"{base}_{millis:03d}ms"


def sortable_timestamp(seconds: float, precision: int = 3) -> str:
    """Return a zero-padded numeric timestamp suitable for lexicographic sorting."""

    seconds = max(float(seconds), 0.0)
    precision = max(int(precision), 0)

    # ``width`` ensures we have enough padding for typical long videos while
    # remaining compact. ``precision + 6`` supports durations up to ~10,000 hours
    # with millisecond precision without overflowing.
    width = max(precision + 6, 8)
    formatted = f"{seconds:0{width}.{precision}f}"
    return f"{formatted}s"

