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
    start_sec: float,
    end_sec: float,
    pad_before: float = -25.0,
    pad_after: float = 35.0,
) -> Path:
    """Extract a clip from a remote video URL. Returns path to the clip mp4.

    Uses ffmpeg with input seeking (-ss before -i) so it issues an HTTP range
    request and only downloads the bytes it needs. Requires the remote mp4 to
    be faststart-encoded (moov atom at the beginning of the file).

    Clips are cached in the system temp directory so repeated requests are instant.
    """
    actual_start = max(0, start_sec - pad_before)
    duration = (end_sec + pad_after) - actual_start

    clip_path = get_clip_path(video_url, actual_start, actual_start + duration)

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
