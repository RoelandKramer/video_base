"""Extract video clips from match videos using ffmpeg."""

import subprocess
import hashlib
import tempfile
from pathlib import Path

CLIP_CACHE_DIR = Path(tempfile.gettempdir()) / "kkd_clip_cache"


def get_clip_path(video_url: str, start_sec: float, end_sec: float) -> Path:
    key = f"{video_url}_{start_sec:.1f}_{end_sec:.1f}"
    clip_hash = hashlib.md5(key.encode()).hexdigest()[:12]
    return CLIP_CACHE_DIR / f"{clip_hash}.mp4"


def extract_clip(
    video_url: str,
    video_time_sec: float,
    pad_before: float = 5.0,
    pad_after: float = 12.0,
) -> Path:
    """Extract a clip centred on a video timestamp.

    pad_before: seconds before the event to include
    pad_after: seconds after the event to include
    """
    actual_start = max(0, video_time_sec - pad_before)
    actual_end = video_time_sec + pad_after
    duration = actual_end - actual_start

    clip_path = get_clip_path(video_url, actual_start, actual_end)
    if clip_path.exists():
        return clip_path

    CLIP_CACHE_DIR.mkdir(exist_ok=True)

    cmd = [
        "ffmpeg",
        "-ss", str(actual_start),
        "-i", video_url,
        "-t", str(duration),
        "-c", "copy",
        "-y",
        str(clip_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr[-500:]}")

    return clip_path
