"""Parse SciSports JSON event files and XML video timestamp files."""

import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path


# ================================================================
# DATA CLASSES
# ================================================================

@dataclass
class Event:
    """A single match event with both game-clock and video timestamps."""
    event_type: str        # e.g. "corner", "shot", "goal_kick"
    sub_type: str          # e.g. "CORNER_CROSSED", "ON_TARGET"
    team: str
    player: str
    game_time_ms: int
    video_time_sec: float  # seconds into the mp4 file
    result: str            # SUCCESSFUL / UNSUCCESSFUL
    receiver: str = ""
    start_x: float = 0.0
    start_y: float = 0.0
    end_x: float = 0.0
    end_y: float = 0.0
    xg: float = 0.0
    body_part: str = ""
    sequence_id: int = -1
    labels: list = field(default_factory=list)
    part_id: int = 1        # 1 = first half, 2 = second half
    jersey_number: int = 0
    receiver_jersey: int = 0
    shot_type: str = ""           # SciSports shotTypeName: ON_TARGET/WIDE/BLOCKED
    sequence_start: bool = False
    sequence_end: bool = False
    start_zone: str = ""          # SciSports descriptives.startZone (e.g. "3C")
    end_zone: str = ""            # SciSports descriptives.endZone
    start_third: int = 0          # 1=defensive, 2=middle, 3=attacking (SciSports)
    end_third: int = 0
    goal_progression: float = 0.0
    forward_displacement: float = 0.0
    xa: float = 0.0
    player_id: int = 0
    receiver_id: int = 0

    @property
    def game_time_display(self) -> str:
        m, s = divmod(self.game_time_ms // 1000, 60)
        return f"{m}:{s:02d}"

    @property
    def label(self) -> str:
        return f"{self.game_time_display} — {self.team} — {self.player}"


@dataclass
class Match:
    name: str
    match_id: int
    home_team: str
    away_team: str
    cameras: dict[str, str]
    events: list[Event] = field(default_factory=list)
    video_offset: float = 0.0  # seconds to add to game-clock for video seeking
    players: dict = field(default_factory=dict)   # playerName -> {'shirt': int, 'team': str}
    attack_dir: dict = field(default_factory=dict)  # (team, part_id) -> +1 (right) / -1 (left)


# ================================================================
# EVENT TYPE MAPPING
# ================================================================

# Map JSON (baseTypeName, subTypeName) to our event categories
EVENT_MAP = {
    # Corners
    ("CROSS", "CORNER_CROSSED"): "corner",
    ("PASS", "CORNER_SHORT"): "corner",
    # Goal kicks
    ("PASS", "GOAL_KICK"): "goal_kick",
    # Free kicks (the pass/cross/shot from the free kick, not the award)
    ("PASS", "FREE_KICK"): "free_kick",
    ("CROSS", "FREE_KICK_CROSSED"): "free_kick",
    ("SHOT", "SHOT_FREE_KICK"): "free_kick",
    # Crosses (excluding corners and FK crosses)
    ("CROSS", "CROSS"): "cross",
    ("CROSS", "CROSS_CUTBACK"): "cross",
    # Shots (all types)
    ("SHOT", "SHOT"): "shot",
    ("SHOT", "SHOT_FREE_KICK"): "shot",
    # Recoveries
    ("INTERCEPTION", "RECOVERY"): "recovery",
    # Interceptions
    ("INTERCEPTION", "INTERCEPTION"): "interception",
    # Generic passes (needed for Build-up & Final Third views)
    ("PASS", "PASS"): "pass",
    ("PASS", "LAUNCH_PASS"): "pass",
    # Carries (needed for penalty-box entries)
    ("DRIBBLE", "CARRY"): "carry",
    ("DRIBBLE", "TAKE_ON"): "carry",
}


def _classify_shot(event: dict) -> list[str]:
    """Return all shot categories this event belongs to."""
    cats = ["shot"]
    result = event.get("resultName", "")
    shot_type = event.get("shotTypeName", "")
    xg = event.get("metrics", {}).get("xG", 0)

    if result == "SUCCESSFUL":
        cats.append("goal")
    if shot_type == "ON_TARGET" or result == "SUCCESSFUL":
        cats.append("shot_on_target")
    if xg >= 0.3:
        cats.append("big_chance")
    return cats


# ================================================================
# PARSING
# ================================================================

def _find_key_pass_keys(raw_events: list[dict]) -> set[tuple[int, int]]:
    """Return set of (sequenceId, startTimeMs) for passes/crosses that preceded a shot.

    A key pass is the most recent PASS or CROSS by the same team in the same
    sequence as a SHOT, walked backwards from the shot.
    """
    sorted_events = sorted(raw_events, key=lambda e: e.get("startTimeMs", 0))
    keys = set()
    for i, ev in enumerate(sorted_events):
        if ev.get("baseTypeName") != "SHOT":
            continue
        seq_id = ev.get("sequenceId", -1)
        if seq_id < 0:
            continue
        team = ev.get("teamName", "")
        for j in range(i - 1, -1, -1):
            prev = sorted_events[j]
            if prev.get("sequenceId", -1) != seq_id:
                break  # sequence ended
            if prev.get("teamName", "") != team:
                continue
            if prev.get("baseTypeName") in ("PASS", "CROSS"):
                keys.add((seq_id, prev.get("startTimeMs", 0)))
                break
    return keys


def _build_event_from_raw(ev: dict, event_type: str, player_shirts: dict = None) -> Event:
    metrics = ev.get("metrics", {})
    descr = ev.get("descriptives", {})
    player_shirts = player_shirts or {}
    player = ev.get("playerName", "")
    receiver = ev.get("receiverName", "")
    return Event(
        event_type=event_type,
        sub_type=ev.get("subTypeName", ""),
        team=ev.get("teamName", ""),
        player=player,
        game_time_ms=ev.get("startTimeMs", 0),
        video_time_sec=0.0,
        result=ev.get("resultName", ""),
        receiver=receiver,
        start_x=ev.get("startPosXM", 0.0),
        start_y=ev.get("startPosYM", 0.0),
        end_x=ev.get("endPosXM", 0.0),
        end_y=ev.get("endPosYM", 0.0),
        xg=metrics.get("xG", 0.0),
        body_part=ev.get("bodyPartName", ""),
        sequence_id=ev.get("sequenceId", -1),
        labels=ev.get("labels", []),
        part_id=ev.get("partId", 1) if ev.get("partId", 1) in (1, 2) else 1,
        jersey_number=player_shirts.get(player, 0),
        receiver_jersey=player_shirts.get(receiver, 0),
        shot_type=ev.get("shotTypeName", ""),
        sequence_start=bool(ev.get("sequenceStart", False)),
        sequence_end=bool(ev.get("sequenceEnd", False)),
        start_zone=descr.get("startZone", "") or "",
        end_zone=descr.get("endZone", "") or "",
        start_third=descr.get("startThird", 0) or 0,
        end_third=descr.get("endThird", 0) or 0,
        goal_progression=metrics.get("goalProgression", 0.0) or 0.0,
        forward_displacement=metrics.get("forwardDisplacement", 0.0) or 0.0,
        xa=metrics.get("xA", 0.0) or 0.0,
        player_id=ev.get("playerId", 0) or 0,
        receiver_id=ev.get("receiverId", 0) or 0,
    )


def _load_json_events(json_path: Path) -> tuple[list[Event], dict]:
    """Parse all relevant events from a SciSports JSON events file."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    meta = data.get("metaData", {})
    events = []
    raw_events = data.get("data", [])

    # Player -> shirt number lookup
    players_info = {}
    player_shirts = {}
    for p in data.get("players", []):
        name = p.get("playerName", "")
        if not name:
            continue
        players_info[name] = {
            "shirt": p.get("shirtNumber", 0) or 0,
            "team": p.get("teamName", ""),
        }
        player_shirts[name] = p.get("shirtNumber", 0) or 0

    # Build corner sequence IDs for linking shots/clearances to corners
    corner_sequences = set()
    for ev in raw_events:
        if "CORNER" in ev.get("subTypeName", ""):
            if ev.get("sequenceId", -1) >= 0:
                corner_sequences.add(ev["sequenceId"])

    # Detect key passes (pass/cross that preceded a shot)
    key_pass_keys = _find_key_pass_keys(raw_events)

    for ev in raw_events:
        base = ev.get("baseTypeName", "")
        sub = ev.get("subTypeName", "")
        key = (base, sub)

        if key not in EVENT_MAP:
            continue

        event_type = EVENT_MAP[key]
        events.append(_build_event_from_raw(ev, event_type, player_shirts))

        # For shots, also add to goal/on_target/big_chance categories
        if base == "SHOT":
            for cat in _classify_shot(ev):
                if cat != "shot":
                    events.append(_build_event_from_raw(ev, cat, player_shirts))

    # Add key-pass category events (passes/crosses that led to a shot)
    for ev in raw_events:
        if ev.get("baseTypeName") not in ("PASS", "CROSS"):
            continue
        sig = (ev.get("sequenceId", -1), ev.get("startTimeMs", 0))
        if sig in key_pass_keys:
            events.append(_build_event_from_raw(ev, "key_pass", player_shirts))

    # Compute per-team, per-period attacking direction from SHOT start_x sign.
    attack_dir = _compute_attack_directions(raw_events)

    return events, {
        "meta": meta,
        "corner_sequences": corner_sequences,
        "players": players_info,
        "attack_dir": attack_dir,
        "raw": data,
    }


def _compute_attack_directions(raw_events: list[dict]) -> dict:
    """For each (team, part_id), determine attacking direction (+1=right, -1=left).

    Uses mean sign of SHOT start_x — shots go toward opponent goal, so the sign
    of the shot's x tells us which way that team is attacking.
    """
    buckets = {}  # (team, part_id) -> [signs]
    for ev in raw_events:
        if ev.get("baseTypeName") != "SHOT":
            continue
        team = ev.get("teamName", "")
        part = ev.get("partId", 1)
        if part not in (1, 2):
            continue
        x = ev.get("startPosXM", 0.0)
        if x == 0:
            continue
        buckets.setdefault((team, part), []).append(1 if x > 0 else -1)
    dirs = {}
    for k, signs in buckets.items():
        avg = sum(signs) / len(signs)
        dirs[k] = 1 if avg >= 0 else -1
    return dirs


def _compute_video_times(events: list[Event], video_offset: float):
    """Convert game-clock ms to video timestamps using the offset."""
    for ev in events:
        ev.video_time_sec = ev.game_time_ms / 1000.0 + video_offset


# ================================================================
# MATCH DISCOVERY
# ================================================================

def _load_video_config(data_dir: Path) -> dict:
    videos_file = data_dir / "videos.json"
    if not videos_file.exists():
        return {}
    return json.loads(videos_file.read_text(encoding="utf-8"))


def load_positions(json_path: Path) -> list:
    """Load SciSports tracking data (positions.json) — list of frames.
    Each frame: {"t": ms, "h": [{x,y,p,s}], "a": [...], "b": {x,y,z}}.
    """
    if not json_path.exists():
        return []
    data = json.loads(json_path.read_text(encoding="utf-8"))
    return data.get("data", [])


def avg_positions_in_window(frames: list, start_ms: int, end_ms: int) -> dict:
    """Compute mean (x, y) per player ID in the time window.
    Returns {'h': {player_id: (x, y, shirt)}, 'a': {...}}."""
    buckets = {"h": {}, "a": {}}  # side -> pid -> [xs, ys, shirt]
    for f in frames:
        t = f.get("t", 0)
        if t < start_ms:
            continue
        if t > end_ms:
            break  # frames are sorted by t
        for side in ("h", "a"):
            for pl in f.get(side, []):
                pid = pl.get("p")
                if pid is None:
                    continue
                d = buckets[side].setdefault(pid, {"xs": [], "ys": [], "s": pl.get("s", 0)})
                d["xs"].append(pl.get("x", 0))
                d["ys"].append(pl.get("y", 0))
    out = {"h": {}, "a": {}}
    for side in ("h", "a"):
        for pid, d in buckets[side].items():
            if not d["xs"]:
                continue
            out[side][pid] = (sum(d["xs"])/len(d["xs"]),
                              sum(d["ys"])/len(d["ys"]),
                              d["s"])
    return out


def discover_matches(data_dir: Path) -> list[Match]:
    """Scan directory for JSON event files and build match objects."""
    video_config = _load_video_config(data_dir)
    matches = []

    for json_file in sorted(data_dir.glob("*SciSportsEvents - *.json")):
        stem = json_file.stem
        match_prefix = stem.split("SciSportsEvents")[0].strip()

        # Extract match ID
        match_id = 0
        parts = stem.split(" - ")
        if len(parts) >= 2:
            try:
                match_id = int(parts[-1])
            except ValueError:
                pass

        # Load config for this match
        match_config = video_config.get(match_prefix, {})
        cameras = {k: v for k, v in match_config.items() if k != "offset"}
        video_offset = match_config.get("offset", 0.0)

        # Parse events
        events, extra = _load_json_events(json_file)
        _compute_video_times(events, video_offset)

        meta = extra["meta"]

        matches.append(Match(
            name=match_prefix,
            match_id=match_id,
            home_team=meta.get("homeTeamName", ""),
            away_team=meta.get("awayTeamName", ""),
            cameras=cameras,
            events=events,
            video_offset=video_offset,
            players=extra.get("players", {}),
            attack_dir=extra.get("attack_dir", {}),
        ))

    return matches


def get_events_by_type(match: Match, event_type: str) -> list[Event]:
    """Filter match events by type, sorted by game time."""
    return sorted(
        [e for e in match.events if e.event_type == event_type],
        key=lambda e: e.game_time_ms,
    )


def get_corner_sequences(json_path: Path) -> set[int]:
    """Get sequence IDs that belong to corners."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    seqs = set()
    for ev in data.get("data", []):
        if "CORNER" in ev.get("subTypeName", ""):
            if ev.get("sequenceId", -1) >= 0:
                seqs.add(ev["sequenceId"])
    return seqs
