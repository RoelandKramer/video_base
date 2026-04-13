"""Parse SciSports XML/JSON event files to extract corner kick events with video timestamps."""

import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Corner:
    team: str
    player: str
    video_start_sec: float  # video timestamp: corner awarded
    video_end_sec: float    # video timestamp: possession ends
    game_time_ms: int       # JSON startTimeMs (used for role detection)
    label: str

    @property
    def start_display(self) -> str:
        """Display the game-clock time of the delivery."""
        total_sec = self.game_time_ms // 1000
        m, s = divmod(total_sec, 60)
        return f"{m}:{s:02d}"


@dataclass
class Match:
    name: str
    match_id: int
    xml_path: Path
    json_path: Path | None
    cameras: dict[str, str]  # camera name -> video URL/path
    corners: list[Corner] = field(default_factory=list)
    home_team: str = ""
    away_team: str = ""


def _sanitize_xml(content: str) -> str:
    return re.sub(r"[^\x09\x0A\x0D\x20-\x7E\xA0-\xFF\u0100-\uFFFF]", "", content)


def _parse_xml_corners(xml_path: Path) -> list[dict]:
    """Extract Set Piece: Corner events from XML with video timestamps."""
    content = xml_path.read_text(encoding="utf-8")
    content = _sanitize_xml(content)
    root = ET.fromstring(content)

    # Get Possession: corner events (wider time window than Set Piece)
    possession_windows = []
    for inst in root.findall(".//instance"):
        code_el = inst.find("code")
        if code_el is None or code_el.text is None:
            continue
        if "Possession: corner" in code_el.text:
            start = float(inst.find("start").text)
            end = float(inst.find("end").text)
            team = code_el.text.split(" - Possession")[0]
            possession_windows.append({"start": start, "end": end, "team": team})

    # Get Set Piece: Corner events (have player info)
    setpiece_corners = []
    for inst in root.findall(".//instance"):
        code_el = inst.find("code")
        if code_el is None or code_el.text is None:
            continue
        if "Set Piece: Corner" not in code_el.text:
            continue

        start = float(inst.find("start").text)
        team = code_el.text.split(" - Set Piece")[0]

        player = ""
        for lbl in inst.findall("label"):
            group_el = lbl.find("group")
            if group_el is not None and group_el.text == "Player":
                text_el = lbl.find("text")
                if text_el is not None and text_el.text:
                    player = text_el.text

        # Find matching possession window (same team, close start time)
        poss_end = start + 10  # fallback
        for pw in possession_windows:
            if pw["team"] == team and abs(pw["start"] - start) < 2:
                poss_end = pw["end"]
                break

        setpiece_corners.append({
            "video_start": start,
            "video_end": poss_end,
            "team": team,
            "player": player,
        })

    return setpiece_corners


def _parse_json_corners(json_path: Path) -> list[dict]:
    """Extract CORNER_CROSSED/SHORT from JSON with game-clock timestamps."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    corners = []
    for ev in data.get("data", []):
        if ev.get("subTypeName") in ("CORNER_CROSSED", "CORNER_SHORT"):
            corners.append({
                "game_time_ms": ev["startTimeMs"],
                "team": ev["teamName"],
                "player": ev["playerName"],
            })
    return corners


def _match_xml_json_corners(xml_corners: list[dict], json_corners: list[dict]) -> list[Corner]:
    """Match XML (video timestamps) with JSON (game-clock timestamps) by order.

    Both sources list corners in chronological order for the same match,
    so we align them by index. The ~3s systematic offset confirms they
    describe the same events.
    """
    corners = []
    for idx, xc in enumerate(xml_corners):
        # Find matching JSON corner (same team, closest by index)
        jc = None
        if idx < len(json_corners):
            jc = json_corners[idx]

        game_time_ms = jc["game_time_ms"] if jc else 0
        total_sec = game_time_ms // 1000
        m, s = divmod(total_sec, 60)

        label = f"Corner #{idx+1} — {xc['team']} — {xc['player']} ({m}:{s:02d})"

        corners.append(Corner(
            team=xc["team"],
            player=xc["player"],
            video_start_sec=xc["video_start"],
            video_end_sec=xc["video_end"],
            game_time_ms=game_time_ms,
            label=label,
        ))

    return corners


def _load_video_config(data_dir: Path) -> dict[str, dict[str, str]]:
    videos_file = data_dir / "videos.json"
    if not videos_file.exists():
        return {}
    return json.loads(videos_file.read_text(encoding="utf-8"))


def _extract_match_id(filename: str) -> int | None:
    """Extract numeric match ID from filename like '... - 2561462.json'."""
    parts = filename.split(" - ")
    if len(parts) >= 2:
        try:
            return int(parts[-1].replace(".json", "").replace(".xml", "").strip())
        except ValueError:
            pass
    return None


def discover_matches(data_dir: Path) -> list[Match]:
    """Scan a directory for XML + JSON + video config and build match objects."""
    video_config = _load_video_config(data_dir)
    matches = []

    for xml_file in sorted(data_dir.glob("*SciSportsEvents XML*.xml")):
        stem = xml_file.stem
        match_prefix = stem.split("SciSportsEvents")[0].strip()
        match_id = _extract_match_id(stem)

        # Find matching JSON events file
        json_path = None
        for jp in data_dir.glob(f"{match_prefix} SciSportsEvents - *.json"):
            json_path = jp
            break

        # Parse corners
        xml_corners = _parse_xml_corners(xml_file)
        json_corners = _parse_json_corners(json_path) if json_path else []
        corners = _match_xml_json_corners(xml_corners, json_corners)

        # Load match metadata from JSON
        home_team = away_team = ""
        if json_path:
            with open(json_path, "r", encoding="utf-8") as f:
                meta = json.load(f).get("metaData", {})
            home_team = meta.get("homeTeamName", "")
            away_team = meta.get("awayTeamName", "")

        cameras = video_config.get(match_prefix, {})
        name = match_prefix.strip()

        matches.append(Match(
            name=name,
            match_id=match_id or 0,
            xml_path=xml_file,
            json_path=json_path,
            cameras=cameras,
            corners=corners,
            home_team=home_team,
            away_team=away_team,
        ))

    return matches
