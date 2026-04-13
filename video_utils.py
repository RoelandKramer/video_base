"""Extract video clips from remote match videos using ffmpeg."""

import subprocess
import hashlib
import tempfile
from pathlib import Path

CLIP_CACHE_DIR = Path(tempfile.gettempdir()) / "kkd_clip_cache"


def get_clip_path(video_url: str, start_sec: float, end_sec: float) -> Path:
    """Return the cached clip path for a given video URL + time range."""
    key = f"{video_url}_{start_sec}_{end_sec}"
    clip_hash = hashlib.md5(key.encode()).hexdigest()[:12]
    return CLIP_CACHE_DIR / f"{clip_hash}.mp4"


def extract_clip(
    video_url: str,
    video_start_sec: float,
    video_end_sec: float,
    pad_before: float = 2.0,
    pad_after: float = 30.0,
) -> Path:
    """Extract a clip from a video. Returns path to the cached clip mp4.

    video_start_sec: video timestamp where corner is awarded
    video_end_sec: video timestamp where corner possession ends

    The clip starts 2s before the award (context) and runs 30s past
    the possession end to capture the full delivery + outcome sequence.
    Corner delivery typically happens 15-25s after the award in the video.
    """
    actual_start = max(0, video_start_sec - pad_before)
    actual_end = video_end_sec + pad_after
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
