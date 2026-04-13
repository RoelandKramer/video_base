"""Parse SciSports XML event files to extract corner kick events with video timestamps."""

import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Corner:
    team: str
    player: str
    start_sec: float
    end_sec: float
    label: str  # e.g. "Eindhoven - Corner #1 (0:58)"

    @property
    def start_display(self) -> str:
        m, s = divmod(int(self.start_sec), 60)
        return f"{m}:{s:02d}"


@dataclass
class Match:
    name: str  # e.g. "28-03-2026 FC Eindhoven vs FC Emmen"
    xml_path: Path
    cameras: dict[str, str]  # camera name -> video URL/path
    corners: list[Corner] = field(default_factory=list)


def _sanitize_xml(content: str) -> str:
    """Remove invalid XML characters."""
    return re.sub(r"[^\x09\x0A\x0D\x20-\x7E\xA0-\xFF\u0100-\uFFFF]", "", content)


def parse_corners_from_xml(xml_path: Path) -> list[Corner]:
    """Extract corner kick events from a SciSports XML v2 file.

    Uses 'Set Piece: Corner' events which contain exact video timestamps.
    """
    content = xml_path.read_text(encoding="utf-8")
    content = _sanitize_xml(content)
    root = ET.fromstring(content)

    corners: list[Corner] = []
    idx = 1

    for inst in root.findall(".//instance"):
        code_el = inst.find("code")
        if code_el is None or code_el.text is None:
            continue
        code = code_el.text

        if "Set Piece: Corner" not in code:
            continue

        start = float(inst.find("start").text)
        end = float(inst.find("end").text)

        # Extract team name (before " - Set Piece")
        team = code.split(" - Set Piece")[0]

        # Extract player from label
        player = ""
        for lbl in inst.findall("label"):
            group_el = lbl.find("group")
            if group_el is not None and group_el.text == "Player":
                text_el = lbl.find("text")
                if text_el is not None and text_el.text:
                    player = text_el.text

        m, s = divmod(int(start), 60)
        label = f"Corner #{idx} — {team} — {player} ({m}:{s:02d})"
        corners.append(Corner(team=team, player=player, start_sec=start, end_sec=end, label=label))
        idx += 1

    return corners


def _load_video_config(data_dir: Path) -> dict[str, dict[str, str]]:
    """Load match-prefix -> {camera_name: url} mapping from videos.json."""
    videos_file = data_dir / "videos.json"
    if not videos_file.exists():
        return {}
    return json.loads(videos_file.read_text(encoding="utf-8"))


def discover_matches(data_dir: Path) -> list[Match]:
    """Scan a directory for XML files and resolve video URLs from videos.json."""
    video_config = _load_video_config(data_dir)
    matches = []

    for xml_file in sorted(data_dir.glob("*SciSportsEvents XML*.xml")):
        stem = xml_file.stem
        match_prefix = stem.split("SciSportsEvents")[0].strip()

        cameras = video_config.get(match_prefix, {})
        corners = parse_corners_from_xml(xml_file)
        name = match_prefix.strip()

        matches.append(Match(name=name, xml_path=xml_file, cameras=cameras, corners=corners))

    return matches
