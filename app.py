"""D&V Team Analysis - Streamlit app for match event analysis with video clips."""

import functools
import os

import streamlit as st
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from PIL import Image
from collections import Counter
from pathlib import Path

from event_parser import (discover_matches, get_events_by_type, Event, Match,
                            load_positions, avg_positions_in_window)
from video_utils import extract_clip


@st.cache_data(show_spinner=False)
def _cached_positions(path_str: str):
    from pathlib import Path as _P
    return load_positions(_P(path_str))


@st.cache_data(show_spinner=False, max_entries=200)
def _cached_extract_clip(source_path: str, video_time_sec: float) -> str:
    """Memoised clip extraction. Streamlit re-runs ``show_video_for_event`` on
    every page rerun; without this wrapper each rerun would hash, stat, and
    return the same path. With the wrapper, repeat calls for the same event
    hit the in-memory cache (essentially free)."""
    return str(extract_clip(source_path, video_time_sec))


@st.cache_resource(show_spinner=False)
def _cached_image(path_str: str):
    """PIL.Image.open is fast but does a fresh file read every call. Each
    corner panel renders 2 images per side, so the saved I/O adds up over a
    full Defending Corners page."""
    return Image.open(path_str)


# Compatibility shim: st.fragment (1.37+) -> st.experimental_fragment (1.33+).
# Falls back to a no-op decorator on older Streamlit so the app still runs.
def _fragment(fn=None, **kwargs):
    decorator = getattr(st, "fragment", None) or getattr(st, "experimental_fragment", None)
    if decorator is None:
        return fn if fn is not None else (lambda f: f)
    return decorator(fn, **kwargs) if fn is not None else decorator(**kwargs)


def _positions_path_for_match(match):
    """Derive SciSportsPositions JSON path from the match prefix."""
    for f in DATA_DIR.glob(f"{match.name}*SciSportsPositions*.json"):
        return f
    return None


def _events_path_for_match(match):
    """Derive SciSportsEvents JSON path from the match prefix."""
    for f in DATA_DIR.glob(f"{match.name}*SciSportsEvents*.json"):
        return f
    return None

# DATA_DIR holds match files (events JSON, positions JSON, videos via
# videos.json). Priority:
#   1. explicit DATA_DIR env var (e.g. a Railway-mounted rclone path) wins
#   2. ONEDRIVE_* env vars present → run the Graph sync once, cache locally
#   3. otherwise local dev: use the repo folder
_BOOTSTRAP_ERROR: list[str] = []   # surfaced inside main() after set_page_config


@st.cache_resource(show_spinner="Syncing match metadata from OneDrive…")
def _bootstrap_data_dir() -> Path:
    explicit = os.environ.get("DATA_DIR", "").strip()
    if explicit:
        return Path(explicit)
    # Check env vars BEFORE attempting to import onedrive_sync — that module
    # pulls in msal, which isn't installed in local-dev environments. Skip
    # the import entirely when the OneDrive feature is off.
    onedrive_keys = ("ONEDRIVE_TENANT_ID", "ONEDRIVE_CLIENT_ID",
                     "ONEDRIVE_CLIENT_SECRET", "ONEDRIVE_USER_EMAIL",
                     "ONEDRIVE_BASE_FOLDER")
    if all(os.environ.get(k) for k in onedrive_keys):
        try:
            import onedrive_sync
            return onedrive_sync.get_sync().sync_metadata()
        except Exception as e:
            # DO NOT call st.error here — Streamlit would treat that as the
            # "first command" and break the later set_page_config call.
            # Stash and surface inside main() instead.
            _BOOTSTRAP_ERROR.append(str(e))
    return Path(__file__).parent


DATA_DIR = _bootstrap_data_dir()
# Model joblibs ship with the repo, so they always live next to app.py.
MODELS_DIR = Path(__file__).parent
LABELING_SHEET_PATH = DATA_DIR / "corner_role_labels.csv"


# ---------- CornerAnalyser singleton + cached match analysis ----------

_CORNER_ANALYSER = None
_CORNER_ANALYSER_ERROR = None


def _get_corner_analyser():
    """Lazy-load CornerAnalyser once. Returns (analyser, error_str)."""
    global _CORNER_ANALYSER, _CORNER_ANALYSER_ERROR
    if _CORNER_ANALYSER is not None or _CORNER_ANALYSER_ERROR is not None:
        return _CORNER_ANALYSER, _CORNER_ANALYSER_ERROR
    try:
        from fcdb_corner_inference import CornerAnalyser
        _CORNER_ANALYSER = CornerAnalyser(MODELS_DIR)
    except Exception as e:  # missing artefacts, sklearn version mismatch, etc.
        _CORNER_ANALYSER_ERROR = f"{type(e).__name__}: {e}"
    return _CORNER_ANALYSER, _CORNER_ANALYSER_ERROR


@st.cache_data(show_spinner=False)
def _cached_corner_analysis(events_path_str: str, positions_path_str: str,
                              defending_team: str):
    """Run CornerAnalyser.analyse_match once per (events, positions, team)
    triple. Returns (rows_list, aggregates_dict) or (None, error_string)."""
    analyser, err = _get_corner_analyser()
    if analyser is None:
        return None, err or "Corner analyser unavailable."
    try:
        rows = analyser.analyse_match(events_path_str, positions_path_str,
                                       defending_team)
        aggregates = analyser.compute_aggregates(rows) if rows else {}
        return {"rows": rows, "aggregates": aggregates}, None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"

EVENT_TYPES = {
    "Defending Corners": "def_corner",
    "Attacking Corners": "att_corner",
    "Goal Kicks": "goal_kick",
    "Free Kicks": "free_kick",
    "Crosses": "cross",
    "Key Passes": "key_pass",
    "Build-Up": "build_up",
    "Final 3rd": "final_3rd",
    "Offensive Transitions": "off_transition",
    "Defensive Transitions": "def_transition",
    "Shots": "shots_all",
    "Big Chances": "big_chance",
    "Ball Recoveries": "recovery",
    "Interceptions": "interception",
}

BOTH_LABEL = "Both"

# SciSports pitch half-lengths (standard)
PITCH_X = 52.5
PITCH_Y = 34.0


# ================================================================
# SESSION STATE HELPERS
# ================================================================

def _find_event_index(events, target):
    for i, e in enumerate(events):
        if e.game_time_ms == target.game_time_ms and e.player == target.player:
            return i
    return -1


def _jump_to_event(event, all_events):
    """Trigger jump to event on next rerun (safe - applied before widget render)."""
    idx = _find_event_index(all_events, event)
    if idx >= 0:
        st.session_state["pending_event_idx"] = idx
        st.session_state.pop("direct_play_event", None)
    else:
        # Event is a different type (cross-type navigation)
        st.session_state["direct_play_event"] = event
    st.rerun()


def _clear_event_filter():
    """Remove any active click-driven event filter."""
    for k in ("event_filter_key", "event_filter_sigs", "event_filter_label"):
        st.session_state.pop(k, None)


def _apply_filter_from_click(group, filter_key, label):
    """Filter the main event dropdown + timeline to just `group` (a list of
    Event objects). Clicking the same `filter_key` again toggles it off and
    restores the full list."""
    if not group:
        return
    active = st.session_state.get("event_filter_key")
    if active == filter_key:
        # Toggle off: restore full event list
        _clear_event_filter()
        st.session_state.pop("event_selector", None)
        st.session_state["selected_event_idx"] = 0
        st.rerun()
        return
    sigs = {(e.game_time_ms, e.player) for e in group}
    st.session_state["event_filter_key"] = filter_key
    st.session_state["event_filter_sigs"] = sigs
    st.session_state["event_filter_label"] = label
    st.session_state["pending_event_idx"] = 0
    st.rerun()


def _jump_button(label, event, all_events, key):
    if st.button(label, key=key, use_container_width=True):
        _jump_to_event(event, all_events)


def _event_buttons(events_to_show, all_events, prefix):
    for i, e in enumerate(events_to_show):
        _jump_button(
            f"{e.game_time_display} - {_pname(e)}",
            e, all_events, key=f"{prefix}_{i}",
        )


def _handle_plotly_click(result, key, index_to_event, all_events):
    """Handle a click on a plotly chart that uses customdata=event_index."""
    if not result:
        return
    sel = result.get("selection") if isinstance(result, dict) else getattr(result, "selection", None)
    if not sel:
        return
    points = sel.get("points") if isinstance(sel, dict) else getattr(sel, "points", None)
    if not points:
        return
    custom = points[0].get("customdata") if isinstance(points[0], dict) else None
    if custom is None:
        return
    if isinstance(custom, list):
        custom = custom[0]
    click_sig = str(custom)
    consumed_key = f"__consumed_{key}"
    if st.session_state.get(consumed_key) == click_sig:
        return
    st.session_state[consumed_key] = click_sig
    event = index_to_event.get(int(custom))
    if event is None:
        return
    _jump_to_event(event, all_events)


def _normalize_pos(x, y):
    """SciSports events are already per-team normalized (each team attacks to +x),
    so positions are used as-is. Kept as a helper for clarity/future flexibility."""
    return x, y


def _pname(event) -> str:
    """Format player name with jersey number: '7. Daniel de Ridder'."""
    if event.jersey_number and event.jersey_number > 0:
        return f"{event.jersey_number}. {event.player}"
    return event.player


def _rname(event) -> str:
    """Format receiver name with jersey number, empty if no receiver."""
    if not event.receiver or event.receiver == "NOT_APPLICABLE":
        return ""
    if event.receiver_jersey and event.receiver_jersey > 0:
        return f"{event.receiver_jersey}. {event.receiver}"
    return event.receiver


def _match_player_label(match, player_name: str) -> str:
    info = match.players.get(player_name) if hasattr(match, "players") else None
    if info and info.get("shirt"):
        return f"{info['shirt']}. {player_name}"
    return player_name


# ================================================================
# PITCH IMAGE OVERLAY (16m area)
# ================================================================

_PITCH_IMG_PATH = Path(__file__).parent / "16m area.png"
_LEFT_CORNER_IMG = Path(__file__).parent / "left_side_corner.png"
_RIGHT_CORNER_IMG = Path(__file__).parent / "right_side_corner.png"

_PX_PER_M = 36.0 / 16.5  # 2.1818
_GOAL_COL = 75.0
_GOAL_X = 52.5
_CENTER_ROW = 79.0


def _sci_to_pixel(x_m, y_m):
    col = _GOAL_COL - (_GOAL_X - x_m) * _PX_PER_M
    row = _CENTER_ROW + y_m * _PX_PER_M
    return col, row


# ================================================================
# FULL-PITCH PLOTLY HELPERS
# ================================================================

def _plotly_pitch(fig_height=420, xrange=(-PITCH_X, PITCH_X), yrange=(-PITCH_Y, PITCH_Y),
                  outline=True):
    """Build an empty plotly full pitch. With outline=False, the outer rect,
    halfway line and centre circle are not drawn (used for zoomed views)."""
    fig = go.Figure()
    green = "#2d7a3a"
    line = "#ffffff"
    # Pitch background (fill only when outline is hidden, to avoid a visible frame)
    outer_line = dict(color=line, width=2) if outline else dict(color=green, width=0)
    fig.add_shape(type="rect", x0=xrange[0], y0=yrange[0], x1=xrange[1], y1=yrange[1],
                  fillcolor=green, line=outer_line, layer="below")
    if outline:
        # Halfway line
        fig.add_shape(type="line", x0=0, y0=yrange[0], x1=0, y1=yrange[1],
                      line=dict(color=line, width=2))
        # Centre circle
        fig.add_shape(type="circle", x0=-9.15, y0=-9.15, x1=9.15, y1=9.15,
                      line=dict(color=line, width=2))
    # Boxes + penalty spots (always drawn; they sit at the goal ends)
    for side in (-1, 1):
        fig.add_shape(type="rect",
                      x0=side*52.5, y0=-20.16, x1=side*(52.5-16.5), y1=20.16,
                      line=dict(color=line, width=2))
        fig.add_shape(type="rect",
                      x0=side*52.5, y0=-9.16, x1=side*(52.5-5.5), y1=9.16,
                      line=dict(color=line, width=2))
        fig.add_shape(type="circle",
                      x0=side*41.5-0.3, y0=-0.3, x1=side*41.5+0.3, y1=0.3,
                      line=dict(color=line), fillcolor=line)
        # Goal line segment (tiny marker by the goal)
        fig.add_shape(type="line",
                      x0=side*52.5, y0=-3.66, x1=side*52.5, y1=3.66,
                      line=dict(color=line, width=4))
    fig.update_layout(
        xaxis=dict(range=list(xrange), visible=False),
        yaxis=dict(range=list(yrange), visible=False, scaleanchor="x"),
        plot_bgcolor=green, paper_bgcolor=green,
        margin=dict(l=0, r=0, t=0, b=0),
        height=fig_height,
        showlegend=False,
    )
    return fig


def _plotly_pitch_vertical(fig_height=520):
    """Vertical pitch: length along y-axis (team attacks upward toward +y).
    Note: Our in-data convention is x = length (±52.5), y = width (±34).
    For the vertical view we MAP: display_x = -y, display_y = x.
    Caller is responsible for applying this transform to points."""
    fig = go.Figure()
    green = "#2d7a3a"; line = "#ffffff"
    # Pitch rect
    fig.add_shape(type="rect", x0=-PITCH_Y, y0=-PITCH_X, x1=PITCH_Y, y1=PITCH_X,
                  fillcolor=green, line=dict(color=line, width=2), layer="below")
    # Halfway line
    fig.add_shape(type="line", x0=-PITCH_Y, y0=0, x1=PITCH_Y, y1=0,
                  line=dict(color=line, width=2))
    # Centre circle
    fig.add_shape(type="circle", x0=-9.15, y0=-9.15, x1=9.15, y1=9.15,
                  line=dict(color=line, width=2))
    # Goal boxes (top and bottom)
    for side in (-1, 1):
        # 18-yd box: width 40.32, depth 16.5
        fig.add_shape(type="rect",
                      x0=-20.16, y0=side*52.5, x1=20.16, y1=side*(52.5-16.5),
                      line=dict(color=line, width=2))
        # 6-yd box
        fig.add_shape(type="rect",
                      x0=-9.16, y0=side*52.5, x1=9.16, y1=side*(52.5-5.5),
                      line=dict(color=line, width=2))
        # Pen spot
        fig.add_shape(type="circle", x0=-0.3, y0=side*41.5-0.3, x1=0.3, y1=side*41.5+0.3,
                      line=dict(color=line), fillcolor=line)
        # Goal line
        fig.add_shape(type="line", x0=-3.66, y0=side*52.5, x1=3.66, y1=side*52.5,
                      line=dict(color=line, width=4))
    fig.update_layout(
        xaxis=dict(range=[-PITCH_Y, PITCH_Y], visible=False),
        yaxis=dict(range=[-PITCH_X, PITCH_X], visible=False, scaleanchor="x"),
        plot_bgcolor=green, paper_bgcolor=green,
        margin=dict(l=0, r=0, t=0, b=0), height=fig_height, showlegend=False,
    )
    return fig


def _v(x, y):
    """Project (x, y) to vertical-pitch display coords (dx, dy)."""
    return (-y, x)


def _plotly_pitch_attacking_half(fig_height=520):
    """Horizontal attacking-half pitch: x in [0, 52.5] (halfway line -> goal line).
    Fills the entire frame (no side padding), locked (no pan/zoom), and drawn
    so the backline runs fully sideline-to-sideline."""
    fig = go.Figure()
    green = "#2d7a3a"
    line = "#ffffff"

    # Full-frame green (ensures the whole canvas is pitch)
    fig.add_shape(type="rect", x0=0, y0=-PITCH_Y, x1=PITCH_X, y1=PITCH_Y,
                  fillcolor=green, line=dict(color=line, width=2), layer="below")

    # Halfway line (left edge of the attacking half)
    fig.add_shape(type="line", x0=0, y0=-PITCH_Y, x1=0, y1=PITCH_Y,
                  line=dict(color=line, width=2))

    # Centre-circle arc (half of it, since we only render the attacking half)
    fig.add_shape(type="circle", x0=-9.15, y0=-9.15, x1=9.15, y1=9.15,
                  line=dict(color=line, width=2))

    # Penalty box (18-yard): width 40.32 (y: \u00b120.16), depth 16.5
    fig.add_shape(type="rect",
                  x0=PITCH_X - 16.5, y0=-20.16, x1=PITCH_X, y1=20.16,
                  line=dict(color=line, width=2))

    # Six-yard box: width 18.32 (y: \u00b19.16), depth 5.5
    fig.add_shape(type="rect",
                  x0=PITCH_X - 5.5, y0=-9.16, x1=PITCH_X, y1=9.16,
                  line=dict(color=line, width=2))

    # Penalty spot (11 m from goal line)
    fig.add_shape(type="circle",
                  x0=PITCH_X - 11 - 0.25, y0=-0.25, x1=PITCH_X - 11 + 0.25, y1=0.25,
                  line=dict(color=line), fillcolor=line)

    # Goal (a small rectangle at the goal line, 7.32 m wide)
    fig.add_shape(type="rect",
                  x0=PITCH_X, y0=-3.66, x1=PITCH_X + 0.6, y1=3.66,
                  fillcolor=line, line=dict(color=line, width=2))

    fig.update_layout(
        xaxis=dict(range=[0, PITCH_X + 0.6], visible=False, fixedrange=True),
        yaxis=dict(range=[-PITCH_Y, PITCH_Y], visible=False, fixedrange=True),
        plot_bgcolor=green, paper_bgcolor=green,
        margin=dict(l=0, r=0, t=0, b=0),
        height=fig_height,
        showlegend=False,
        dragmode=False,
    )
    return fig


# ================================================================
# ATTACKING HALF ZONES (shared by shots, crosses, final 3rd, passes)
# ================================================================

# Zones covering the attacking half (x: 0..52.5). Each zone is a rect
# (x0, x1, y0, y1). Designed to mirror the reference image divisions.
# Names are short for display. Y positive = team's left side (since SciSports
# events are per-team normalized, each team attacks toward +x).

def _build_att_half_zones():
    """Return dict of zone_name -> (x0, x1, y0, y1) for the attacking half."""
    # Penalty box = x in (36, 52.5), y in (-20.16, 20.16)
    # 6-yard box = x in (47, 52.5), y in (-9.16, 9.16)
    zones = {
        # 6-yard box (3 across)
        "6yd L":       (47,   52.5,  3.05,  9.16),
        "6yd C":       (47,   52.5, -3.05,  3.05),
        "6yd R":       (47,   52.5, -9.16, -3.05),
        # Inside 18yd (excluding 6yd), 4 across to match ref image
        "Box L":       (36,   47,   10,   20.16),
        "Box CL":      (36,   47,   0,    10),
        "Box CR":      (36,   47,  -10,   0),
        "Box R":       (36,   47,  -20.16, -10),
        # Edge of box (center D + wide edges), y matches penalty width
        "Edge L":      (30,   36,   10,    20.16),
        "Edge C":      (30,   36,  -10,    10),
        "Edge R":      (30,   36,  -20.16, -10),
        # Wings / deep attacking third (above/below box width)
        "Wing L":      (17.5, 52.5, 20.16, 34),
        "Deep L":      (17.5, 30,   0,    20.16),
        "Deep R":      (17.5, 30,  -20.16, 0),
        "Wing R":      (17.5, 52.5, -34,  -20.16),
    }
    return zones


def _att_zone_of(x, y):
    """Return the attacking-half zone name for (x, y), or None if outside."""
    zones = _build_att_half_zones()
    for name, (x0, x1, y0, y1) in zones.items():
        if x0 <= x <= x1 and y0 <= y <= y1:
            return name
    return None


def _draw_att_zone_counts(fig, counts, color="rgba(255, 215, 0, 0.18)",
                          text_color="white"):
    """Overlay zone rectangles with counts on the figure (at the zone center)."""
    zones = _build_att_half_zones()
    max_cnt = max(counts.values()) if counts else 1
    for name, (x0, x1, y0, y1) in zones.items():
        cnt = counts.get(name, 0)
        if cnt == 0:
            continue
        alpha = 0.12 + 0.45 * (cnt / max_cnt)
        fig.add_shape(
            type="rect", x0=x0, y0=y0, x1=x1, y1=y1,
            fillcolor=f"rgba(255, 215, 0, {alpha:.2f})",
            line=dict(color="rgba(255,255,255,0.5)", width=1),
            layer="below",
        )
        fig.add_annotation(
            x=(x0+x1)/2, y=(y0+y1)/2, text=f"<b>{cnt}</b>",
            showarrow=False, font=dict(color=text_color, size=14),
        )


# ================================================================
# SHOT PHASE / OUTCOME CLASSIFICATION
# ================================================================

SHOT_PHASE_COLORS = {
    "Open Play":        "#3498db",  # blue
    "Set Piece Phase":  "#f39c12",  # orange
    "Direct Set Piece": "#9b59b6",  # purple
}

SHOT_OUTCOME_COLORS = {
    "Goal":        "#27ae60",  # green
    "Save":        "#e67e22",  # orange
    "Blocked":     "#95a5a6",  # grey
    "Hit Woodwork":"#3498db",  # blue
    "Miss":        "#e74c3c",  # red
}


def _shot_outcome(event) -> str:
    """Classify shot by outcome for the phase bar chart."""
    if event.result == "SUCCESSFUL":
        return "Goal"
    st_name = (event.shot_type or "").upper()
    if "WOODWORK" in st_name or "POST" in st_name or "CROSSBAR" in st_name:
        return "Hit Woodwork"
    if "BLOCK" in st_name:
        return "Blocked"
    if st_name == "ON_TARGET":
        return "Save"
    if "WIDE" in st_name or "OFF_TARGET" in st_name or "HIGH" in st_name:
        return "Miss"
    return "Miss"


def _shot_phase(event, match) -> str:
    """Open Play / Set Piece Phase / Direct Set Piece."""
    if event.sub_type == "SHOT_FREE_KICK":
        return "Direct Set Piece"
    if event.sub_type == "PENALTY":
        return "Direct Set Piece"
    # Set-piece phase = shot whose sequence started with a set piece
    set_piece_types = ("corner", "free_kick", "goal_kick")
    seq = event.sequence_id
    if seq >= 0:
        start = [e for e in match.events
                 if e.sequence_id == seq and e.team == event.team
                 and e.event_type in set_piece_types
                 and e.game_time_ms <= event.game_time_ms]
        if start:
            return "Set Piece Phase"
    return "Open Play"


# ================================================================
# CORNER ZONE GEOMETRY
# ================================================================

@functools.lru_cache(maxsize=4)
def _build_corner_zones(top_x=PITCH_X, left_y=-PITCH_Y):
    length_PA = 16.5
    width_PA = 40.32
    length_PB = 5.49
    width_PB = 18.29
    PA_x = top_x - length_PA
    PB_x = top_x - length_PB
    half_PA = width_PA / 2
    half_PB = width_PB / 2
    PS_x = top_x - 11
    edge_x = top_x - (length_PA + 4)
    GA_band = width_PB / 3

    # For TL corner (left_y=-34): Short_Corner_Zone is the strip from the corner flag
    # to the near edge of the 18-yd box (y in [-34, -20.16]), NOT spanning to +half_PA.
    zones_TL = {
        "Short_Corner_Zone": [(top_x, left_y), (top_x, -half_PA), (PA_x, left_y), (PA_x, -half_PA)],
        "Front_Zone": [(top_x, half_PA), (top_x, half_PB), (PA_x, half_PB), (PA_x, half_PA)],
        "Back_Zone": [(top_x, -half_PB), (top_x, -half_PA), (PA_x, -half_PA), (PA_x, -half_PB)],
        "GA1": [(top_x, half_PB), (top_x, half_PB - GA_band), (PB_x, half_PB - GA_band), (PB_x, half_PB)],
        "GA2": [(top_x, half_PB - GA_band), (top_x, half_PB - 2*GA_band), (PB_x, half_PB - 2*GA_band), (PB_x, half_PB - GA_band)],
        "GA3": [(top_x, half_PB - 2*GA_band), (top_x, -half_PB), (PB_x, -half_PB), (PB_x, half_PB - 2*GA_band)],
        "CA1": [(PB_x, half_PB), (PB_x, half_PB - GA_band), (PS_x, half_PB - GA_band), (PS_x, half_PB)],
        "CA2": [(PB_x, half_PB - GA_band), (PB_x, half_PB - 2*GA_band), (PS_x, half_PB - 2*GA_band), (PS_x, half_PB - GA_band)],
        "CA3": [(PB_x, half_PB - 2*GA_band), (PB_x, -half_PB), (PS_x, -half_PB), (PS_x, half_PB - 2*GA_band)],
        "Edge_Zone": [(PS_x, half_PB), (PS_x, -half_PB), (edge_x, -half_PB), (edge_x, half_PB)],
    }
    zones_BR = {n: [(-x, -y) for x, y in r] for n, r in zones_TL.items()}
    zones_TR = {n: [(x, -y) for x, y in r] for n, r in zones_TL.items()}
    zones_BL = {n: [(-x, y) for x, y in r] for n, r in zones_TL.items()}
    return {"top_left": zones_TL, "top_right": zones_TR,
            "bottom_left": zones_BL, "bottom_right": zones_BR}


def _point_in_rect(px, py, rect):
    xs = [p[0] for p in rect]
    ys = [p[1] for p in rect]
    return min(xs) <= px <= max(xs) and min(ys) <= py <= max(ys)


def _assign_zone(ex, ey, zones):
    for name, rect in zones.items():
        if _point_in_rect(ex, ey, rect):
            return name
    return None


def _get_corner_position(start_x, start_y, top_x=PITCH_X, left_y=-PITCH_Y, thresh=12.0):
    corners = {
        "top_left": (top_x, left_y),
        "top_right": (top_x, -left_y),
        "bottom_left": (-top_x, left_y),
        "bottom_right": (-top_x, -left_y),
    }
    best_name, best_d = None, float("inf")
    for n, (cx, cy) in corners.items():
        d = ((start_x - cx) ** 2 + (start_y - cy) ** 2) ** 0.5
        if d < best_d:
            best_d, best_name = d, n
    return best_name if best_d <= thresh else None


# Corner image zone pixel coordinates (from opp_analysis_new.py)
_ATT_L_ZONES = {
    "Front_Zone": [(218, 135), (508, 135), (508, 575), (218, 575)],
    "Back_Zone": [(970, 135), (1260, 135), (1260, 575), (970, 575)],
    "Short_Corner_Zone": [(30, 135), (218, 135), (218, 575), (30, 575)],
    "GA1": [(505, 135), (660, 135), (660, 280), (505, 280)],
    "GA2": [(660, 135), (822, 135), (822, 280), (660, 280)],
    "GA3": [(822, 135), (970, 135), (970, 280), (822, 280)],
    "CA1": [(505, 285), (660, 285), (660, 425), (505, 425)],
    "CA2": [(664, 285), (822, 285), (822, 425), (664, 425)],
    "CA3": [(822, 285), (972, 285), (972, 425), (822, 425)],
    "Edge_Zone": [(505, 425), (970, 425), (970, 700), (505, 700)],
}
_ATT_L_CENTERS = {
    "GA1": (590, 245), "GA2": (745, 250), "GA3": (900, 250),
    "CA1": (590, 400), "CA2": (745, 400), "CA3": (900, 400),
    "Edge_Zone": (745, 520), "Front_Zone": (350, 380),
    "Back_Zone": (1127, 380), "Short_Corner_Zone": (130, 380),
}
_ATT_R_ZONES = {
    "Back_Zone": [(218, 130), (508, 130), (508, 560), (218, 560)],
    "Front_Zone": [(965, 130), (1260, 130), (1260, 560), (965, 560)],
    "Short_Corner_Zone": [(1260, 130), (1430, 130), (1430, 560), (1260, 560)],
    "GA3": [(510, 130), (660, 130), (660, 280), (510, 280)],
    "GA2": [(660, 130), (812, 130), (812, 280), (660, 280)],
    "GA1": [(812, 130), (960, 130), (960, 280), (812, 280)],
    "CA3": [(505, 284), (660, 284), (660, 425), (505, 425)],
    "CA2": [(660, 285), (812, 285), (812, 425), (660, 425)],
    "CA1": [(812, 285), (960, 285), (960, 425), (812, 425)],
    "Edge_Zone": [(505, 425), (960, 425), (960, 690), (505, 690)],
}
_ATT_R_CENTERS = {
    "GA3": (590, 250), "GA2": (745, 250), "GA1": (900, 250),
    "CA3": (590, 400), "CA2": (745, 400), "CA1": (900, 400),
    "Edge_Zone": (745, 520), "Back_Zone": (350, 380),
    "Front_Zone": (1120, 380), "Short_Corner_Zone": (1355, 400),
}

# ---- DEF-side schematic images (no_names_left.png / no_names_right.png) ----
# Pixel polygons taken from the example opp_analysis_new.get_visualization_coords().
_DEF_L_ZONES = {
    "Front_Zone":        [(265, 15), (644, 15), (644, 600), (265, 600)],
    "Back_Zone":         [(1287, 15), (1667, 15), (1667, 600), (1287, 600)],
    "Short_Corner_Zone": [(10, 15), (260, 15), (260, 605), (10, 605)],
    "GA1": [(655, 14), (865, 14), (865, 207), (655, 207)],
    "GA2": [(867, 14), (1080, 14), (1080, 207), (867, 207)],
    "GA3": [(1080, 14), (1275, 14), (1275, 207), (1080, 207)],
    "CA1": [(650, 212), (865, 212), (865, 397), (650, 397)],
    "CA2": [(867, 212), (1080, 212), (1080, 397), (867, 397)],
    "CA3": [(1080, 212), (1284, 212), (1284, 397), (1080, 397)],
    "Edge_Zone": [(650, 401), (1284, 401), (1284, 776), (650, 776)],
}
_DEF_R_ZONES = {
    "Front_Zone":        _DEF_L_ZONES["Back_Zone"],
    "Back_Zone":         _DEF_L_ZONES["Front_Zone"],
    "Short_Corner_Zone": [(1672, 15), (1922, 15), (1922, 605), (1672, 605)],
    "GA1": _DEF_L_ZONES["GA3"],
    "GA2": _DEF_L_ZONES["GA2"],
    "GA3": _DEF_L_ZONES["GA1"],
    "CA1": _DEF_L_ZONES["CA3"],
    "CA2": _DEF_L_ZONES["CA2"],
    "CA3": _DEF_L_ZONES["CA1"],
    "Edge_Zone": _DEF_L_ZONES["Edge_Zone"],
}
_NO_NAMES_L = Path(__file__).parent / "no_names_left.png"
_NO_NAMES_R = Path(__file__).parent / "no_names_right.png"

# ----------------------------------------------------------------
# New "Zone Figures" pitch images (measured anchor points from white-line
# detection on each PNG). Each entry maps a metric position (attacking-right
# normalised: goal at +52.5, sidelines at +-34) to pixel coords on the image.
#
#   pixel_x = px_x_center + metric_y * scale_h
#   pixel_y = px_y_attacking_goal + (52.5 - metric_x) * scale_v        (half-pitch)
#   pixel_y = px_y_attacking_goal + (52.5 - metric_x) * scale_v_full   (full pitch,
#       valid for the WHOLE field; own goal then sits at scale_v_full * 105)
# ----------------------------------------------------------------
_ZONE_FIG_DIR = Path(__file__).parent / "Zone Figures"


def _zf(name):
    return _ZONE_FIG_DIR / name


# Each entry: dict with image path + transform anchors.
# half_pitch=True means the image only shows the attacking half (~50-55m of
# length); full_pitch=True means full 105m field.
_PITCH_IMAGES = {
    # Half-pitch (attacking goal at top, corner-flag artwork pre-rendered)
    "sg_left_nozones": dict(
        path=_zf("nieuw_SG_Links_NoZones.png"),
        size=(1288, 507),
        px_x_center=632, scale_h=16.94,
        px_y_attacking_goal=35, scale_v=20.61, half_pitch=True,
    ),
    "sg_right_nozones": dict(
        path=_zf("nieuw_SG_Rechts_NoZones.png"),
        size=(1288, 507),
        px_x_center=632, scale_h=16.94,
        px_y_attacking_goal=35, scale_v=20.61, half_pitch=True,
    ),
    "sg_left_nonames": dict(
        path=_zf("nieuw_SG_links_NoNames.png"),
        size=(1288, 507),
        px_x_center=632, scale_h=16.94,
        px_y_attacking_goal=35, scale_v=20.61, half_pitch=True,
    ),
    "sg_right_nonames": dict(
        path=_zf("nieuw_SG_rechts_NoNames.png"),
        size=(1288, 507),
        px_x_center=632, scale_h=16.94,
        px_y_attacking_goal=35, scale_v=20.61, half_pitch=True,
    ),

    # Half-pitch: penalty-box entries (1285 x 987). Penalty area: 290..972 left
    # of the goal posts at 580/684 (so 6-yd box ±9.16m at 469/790 — verified).
    "pb_entries": dict(
        path=_zf("penalty_box_entries_zones.png"),
        size=(1285, 987),
        px_x_center=631, scale_h=16.92,
        px_y_attacking_goal=38, scale_v=20.55, half_pitch=True,
    ),

    # Half-pitch: cross zones (1283 x 982). Same pixel layout as pb_entries.
    "cross_zones": dict(
        path=_zf("cross_zones.png"),
        size=(1283, 982),
        px_x_center=631, scale_h=16.92,
        px_y_attacking_goal=33, scale_v=20.55, half_pitch=True,
    ),

    # Full-pitch vertical (attacking goal at top). 1213 x 1832.
    # Goal posts at px_x 538/642; sidelines at 54/1144.
    # px_y_top ~ 30 (top edge) ~ attacking goal line.
    # Halfway line at px_y ~ 927; own goal line ~ 1791 (1832 - 41 frame).
    "full_pp": dict(
        path=_zf("full_field_zones_ProgrPasses-CC-More.png"),
        size=(1213, 1832),
        px_x_center=599, scale_h=16.03,
        px_y_attacking_goal=30, scale_v=15.84, full_pitch=True,
        px_y_own_goal=1693,
    ),
    # Full-pitch vertical (cleaner / zonal). 1227 x 1835. Sidelines at 66/1156.
    "full_zo": dict(
        path=_zf("Full_field_zo_zones.png"),
        size=(1227, 1835),
        px_x_center=611, scale_h=16.03,
        px_y_attacking_goal=34, scale_v=15.84, full_pitch=True,
        px_y_own_goal=1697,
    ),
    # Horizontal version: Full_field_zo_zones.png rotated 90 deg CW.
    # Attacking goal now sits on the RIGHT (high pixel_x), own goal on LEFT.
    # Image size becomes (orig_height, orig_width) = (1835, 1227).
    "full_zo_h": dict(
        path=_zf("Full_field_zo_zones.png"),
        rotate_cw_deg=90,
        size=(1835, 1227),
        # After CW rotation: original_y becomes (height - new_x); original_x becomes new_y
        # So a metric (x_m, y_m) -> pixel (new_x, new_y) where:
        #   new_x = px_x_attacking_goal - (52.5 - metric_x) * scale_h
        #   new_y = px_y_center        + metric_y      * scale_v
        px_x_attacking_goal=1801, scale_h=15.84,
        px_y_center=611,           scale_v=16.03,
        full_pitch=True, horizontal=True,
    ),
}


@st.cache_resource(show_spinner=False)
def _cached_rotated_image(path_str: str, angle_cw: int):
    """Cache the rotated PIL image. ``angle_cw`` is clockwise degrees; PIL
    ``rotate`` is counter-clockwise, so we negate."""
    return Image.open(path_str).rotate(-angle_cw, expand=True)


# ----------------------------------------------------------------
# Piecewise anchor points — for images where the painted pitch features
# (penalty area, 6-yard box, halfway line) are not at uniform scale.
# Each list is an ascending sequence of (metric_value, pixel_value) pairs.
# Used by ``_metric_to_pixel`` when the image's PITCH_IMAGES config has a
# matching ``anchors_x`` / ``anchors_y`` key.
# ----------------------------------------------------------------

# full_pp anchors (vertical full pitch, attacking goal at TOP).
# IMPORTANT: the TRUE goal line is the WIDE white line that runs sideline-to-
# sideline (detected at the corner where sideline column first turns white),
# NOT the narrow goal-net back bar that sits a bit above it. Detection:
#   L-corner top y=67, R-corner top y=68 → goal line ≈ 67
#   L-corner bottom y=1793, R-corner bottom y=1793 → own goal ≈ 1793
_FULL_PP_ANCHORS_Y = [   # metric_y -> pixel_x (horizontal axis)
    # SciSports per-team normalised coords: y=+34 is TV-LEFT, y=-34 is TV-RIGHT.
    # We map y=+34 → LOW pixel_x (image LEFT) so renders match the TV view.
    (-34.0,  1148),     # TV-RIGHT → image right sideline
    (-20.16,  919),
    (-9.16,   746),
    ( 0.0,    593),
    ( 9.16,   440),
    (20.16,   270),
    (34.0,     56),     # TV-LEFT → image left sideline
]
_FULL_PP_ANCHORS_X = [   # metric_x -> pixel_y (vertical axis, ascending pixel)
    ( 52.5,    67),     # attacking goal line (TRUE backline, sideline corner)
    ( 47.0,   165),     # 6-yard line
    ( 36.0,   357),     # penalty area bottom
    ( 17.5,   644),     # painted "edge of attacking third" line
    (  0.0,   930),     # halfway line
    (-17.5,  1216),
    (-36.0,  1503),     # own penalty area top
    (-47.0,  1697),     # own 6-yard line
    (-52.5,  1793),     # own goal line (TRUE bottom backline)
]

# full_zo anchors — pitch outlines look identical to full_pp but image is
# 1227 x 1835. Sideline corner top y=67/68, bottom y=1793.
_FULL_ZO_ANCHORS_Y = [
    # y=+34 → TV-LEFT → LOW pixel_x (image LEFT)
    (-34.0,  1154),    # TV-RIGHT → image right
    (-20.16,  927),
    (-9.16,   754),
    ( 0.0,    602),
    ( 9.16,   449),
    (20.16,   278),
    (34.0,     64),    # TV-LEFT → image left
]
_FULL_ZO_ANCHORS_X = [
    ( 52.5,    67),    # attacking goal line (TRUE backline)
    ( 47.0,   164),    # 6-yard line (measured)
    ( 36.0,   358),    # penalty area bottom (measured)
    ( 17.5,   644),    # close-to-PA bottom
    (  0.0,   930),    # halfway line (measured)
    (-17.5,  1216),
    (-36.0,  1503),    # own penalty area top
    (-47.0,  1696),    # own 6-yard line
    (-52.5,  1793),    # own goal line (TRUE bottom backline)
]

# Attach the anchor lists to the image configs that use them.
_PITCH_IMAGES["full_pp"]["anchors_x"] = _FULL_PP_ANCHORS_X
_PITCH_IMAGES["full_pp"]["anchors_y"] = _FULL_PP_ANCHORS_Y
_PITCH_IMAGES["full_zo"]["anchors_x"] = _FULL_ZO_ANCHORS_X
_PITCH_IMAGES["full_zo"]["anchors_y"] = _FULL_ZO_ANCHORS_Y

# ---------- SG half-pitch images (1288x507) ----------
# Goal line = the WIDE white line spanning sideline-to-sideline at y≈80
# (NOT the narrow goal-net back bar at y≈35).
_SG_ANCHORS_X = [
    (52.5,  80),   # goal line (TRUE backline, sideline corner)
    (47.0, 173),   # 6-yard line (measured)
    (36.0, 376),   # penalty area bottom (measured)
    (17.5, 447),   # image bottom
]
# Y anchors INVERTED: in SciSports per-team normalised coords, y=+34 is the
# TV-LEFT touchline. The SG images use a TV-top-down view (left of image = TV
# left = where the corner flag sits in nieuw_SG_Links_NoZones), so y=+34 must
# map to the LOW pixel_x (left of image), not the high pixel_x.
_SG_ANCHORS_Y = [
    (-34.0,  1212),
    (-20.16,  973),
    (-9.16,   791),
    ( 0.0,    632),
    ( 9.16,   473),
    (20.16,   291),
    (34.0,     67),
]

for _key in ("sg_left_nozones", "sg_right_nozones",
              "sg_left_nonames",  "sg_right_nonames"):
    _PITCH_IMAGES[_key]["anchors_x"] = _SG_ANCHORS_X
    _PITCH_IMAGES[_key]["anchors_y"] = _SG_ANCHORS_Y

# ---------- Penalty-box entries (1285x987) ----------
_PB_ANCHORS_X = [
    # Real goal line is where the 6-yd-box verticals START (col-scan detected
    # GA_L top at y=72). Earlier y=77 was the corner-arc top — slightly BELOW
    # the actual painted goal line, which made passes from x≈52.5 render
    # below the backline. y=72 matches the painted goal line exactly.
    (52.5,  72),   # goal line (TRUE backline)
    (47.0, 173),   # 6-yard line (detected)
    (36.0, 378),   # PA bottom (detected)
]
_PB_ANCHORS_Y = [
    (-34.0,  1211),  # TV-RIGHT → image right
    (-20.16,  972),
    (-9.16,   790),
    ( 0.0,    631),
    ( 9.16,   468),
    (20.16,   290),
    (34.0,     66),  # TV-LEFT → image left
]
_PITCH_IMAGES["pb_entries"]["anchors_x"] = _PB_ANCHORS_X
_PITCH_IMAGES["pb_entries"]["anchors_y"] = _PB_ANCHORS_Y

# ---------- Cross zones (1283x982) ----------
_CZ_ANCHORS_X = [
    (52.5,  72),   # goal line (TRUE backline)
    (47.0, 167),   # 6-yard line (detected)
    (36.0, 370),   # PA bottom (detected)
    (17.5, 671),   # close-to-PA bottom (detected)
]
_CZ_ANCHORS_Y = [
    (-34.0,  1211),  # TV-RIGHT → image right
    (-20.16,  972),
    (-9.16,   790),
    ( 0.0,    631),
    ( 9.16,   468),
    (20.16,   290),
    (34.0,     66),  # TV-LEFT → image left
]
_PITCH_IMAGES["cross_zones"]["anchors_x"] = _CZ_ANCHORS_X
_PITCH_IMAGES["cross_zones"]["anchors_y"] = _CZ_ANCHORS_Y

# ---------- Opp_half_no_zones (1288x984) — attacking half, goal at top ----------
_PITCH_IMAGES["opp_half"] = dict(
    path=_zf("Opp_half_no_zones.png"),
    size=(1288, 984),
    px_x_center=640, scale_h=16.94,
    px_y_attacking_goal=35, scale_v=20.55, half_pitch=True,
    anchors_x=[
        (52.5,  75),   # goal line (TRUE backline)
        (47.0, 173),   # 6-yard line (measured)
        (36.0, 376),   # penalty area bottom (measured)
        (17.5, 446),   # close-to-PA line (measured)
        ( 0.0, 817),   # halfway line (measured)
    ],
    anchors_y=[
        (-34.0,  1214),  # TV-RIGHT → image right
        (-20.16,  975),
        (-9.16,   792),
        ( 0.0,    640),
        ( 9.16,   486),
        (20.16,   293),
        (34.0,     68),  # TV-LEFT → image left
    ],
)


def _piecewise_interp(value, anchors):
    """Piecewise-linear interpolation. ``anchors`` is a list of (m, p) pairs
    sorted by m. Extrapolates linearly outside the anchor range."""
    if not anchors:
        return None
    if len(anchors) == 1:
        return anchors[0][1]
    # Sorted ascending by metric — but X anchors are descending (metric x from
    # +52.5 down to -52.5). Detect and handle both directions.
    ascending = anchors[0][0] <= anchors[-1][0]
    if not ascending:
        anchors = list(reversed(anchors))
    if value <= anchors[0][0]:
        m0, p0 = anchors[0]; m1, p1 = anchors[1]
        return p0 + (value - m0) * (p1 - p0) / (m1 - m0)
    if value >= anchors[-1][0]:
        m0, p0 = anchors[-2]; m1, p1 = anchors[-1]
        return p1 + (value - m1) * (p1 - p0) / (m1 - m0)
    for i in range(len(anchors) - 1):
        m0, p0 = anchors[i]; m1, p1 = anchors[i + 1]
        if m0 <= value <= m1:
            return p0 + (value - m0) * (p1 - p0) / (m1 - m0)
    return None


def _metric_to_pixel(x_m, y_m, image_key):
    """Map (metric_x, metric_y) -> (pixel_x, pixel_y) for the named image.
    Uses piecewise anchors when the image config has ``anchors_x`` /
    ``anchors_y``, otherwise falls back to a single linear scale."""
    cfg = _PITCH_IMAGES.get(image_key)
    if cfg is None:
        return None
    if cfg.get("horizontal"):
        # Horizontal rotation: re-use vertical anchors of the source image but
        # SWAP axes — metric_x drives pixel_x, metric_y drives pixel_y.
        # For now, horizontal images keep the simple linear form (only used for
        # the dual-media viewer where exact zone alignment isn't required).
        px = cfg["px_x_attacking_goal"] - (52.5 - x_m) * cfg["scale_h"]
        py = cfg["px_y_center"]         + y_m         * cfg["scale_v"]
        return px, py
    ax = cfg.get("anchors_x")
    ay = cfg.get("anchors_y")
    if ax and ay:
        py = _piecewise_interp(x_m, ax)
        px = _piecewise_interp(y_m, ay)
        if px is None or py is None:
            return None
        return px, py
    # Fallback: linear
    px = cfg["px_x_center"]        + y_m         * cfg["scale_h"]
    py = cfg["px_y_attacking_goal"] + (52.5 - x_m) * cfg["scale_v"]
    return px, py


# ----------------------------------------------------------------
# Pixel-coord zones for full_pp / full_zo. Defined by the painted white-line
# boundaries (NOT by FIFA-distance metrics). Layout:
#   - 6 zones inside the penalty area (3 cols x 2 rows)
#   - 2 zones flanking the penalty area (left wing, right wing)
#   - 3 zones just outside the penalty area (closer-to-PA row)
#   - 3 zones in the middle third (further out)
# Total = 14 attacking-half zones. The own half / centre circle are never
# part of zoning per user spec.
# ----------------------------------------------------------------

def _build_full_pp_pixel_zones():
    """Return list of (zone_name, px0, py0, px1, py1) for the full_pp image.

    Pixel positions anchored to the TRUE painted pitch lines:
      goal_y = the WIDE backline running sideline-to-sideline (NOT the narrow
      goal-net back bar above it).

    Zone layout (14 zones, attacking half only):
      • GA-L / GA-C / GA-R: inside the 6-yd box (3 cols, top row)
      • Front: full LEFT flank of the PA, from goal line to PA bottom
                (between PA-left edge and 6-yd-box left edge)
      • PA: centre of the PA below the 6-yd box
      • Back: full RIGHT flank of the PA (mirror of Front)
      • Wide_L / Wide_R: outside the PA, between sideline and PA edge,
                         from goal line to PA bottom
      • Close-L / Close-C / Close-R: row just below PA bottom
      • Mid-L / Mid-C / Mid-R: middle-third row
    Together GA-L, GA-C, GA-R, Front, PA, Back cover the entire penalty area.
    """
    L_side   =  56   # left sideline pixel (detected at corner)
    PA_L     = 270   # penalty area left edge (detected)
    GA_L     = 440   # 6-yard box left edge (detected)
    # GA-L / GA-C / GA-R dividers split the 6-yd box into 3 EQUAL columns
    # (matches the painted dividers in this image at col 542 and 644).
    GP_L     = 542   # left divider — 1/3 of 6-yd box width
    GP_R     = 644   # right divider — 2/3 of 6-yd box width
    GA_R     = 746   # 6-yard box right edge (detected)
    PA_R     = 919   # penalty area right edge (detected)
    R_side   =1148   # right sideline pixel

    goal_y   =  67   # TRUE backline (sideline-to-sideline goal line)
    GA_bot   = 165   # 6-yard line (= 6-yd box bottom)
    PA_bot   = 357   # penalty area bottom (strong white line)
    closer_y = 644   # painted "close-to-PA" zone bottom
    mid_y    = 930   # halfway line

    return [
        # 1-3: Goal area (inside 6-yard box) — 3 cols
        ("GA-L",       GA_L,   goal_y, GP_L,   GA_bot),
        ("GA-C",       GP_L,   goal_y, GP_R,   GA_bot),
        ("GA-R",       GP_R,   goal_y, GA_R,   GA_bot),

        # 4-6: PA body — Front | PA centre | Back. Front & Back are the FULL
        # 16.5m-deep flanks (from goal line to PA bottom, NOT just the half
        # below the 6-yd line).
        ("Front",      PA_L,   goal_y, GA_L,   PA_bot),   # left PA flank
        ("PA",         GA_L,   GA_bot, GA_R,   PA_bot),   # central PA (below 6-yd)
        ("Back",       GA_R,   goal_y, PA_R,   PA_bot),   # right PA flank

        # 7-8: Wide wings — outside the PA, full PA depth
        ("Wide_L",     L_side, goal_y, PA_L,   PA_bot),
        ("Wide_R",     PA_R,   goal_y, R_side, PA_bot),

        # 9-11: Close-to-PA row (between PA bottom and attacking-third line)
        ("Close-L",    L_side, PA_bot, PA_L,   closer_y),
        ("Close-C",    PA_L,   PA_bot, PA_R,   closer_y),
        ("Close-R",    PA_R,   PA_bot, R_side, closer_y),

        # 12-14: Middle-third row
        ("Mid-L",      L_side, closer_y, PA_L,   mid_y),
        ("Mid-C",      PA_L,   closer_y, PA_R,   mid_y),
        ("Mid-R",      PA_R,   closer_y, R_side, mid_y),
    ]


def _build_full_zo_pixel_zones():
    """Pixel zones for the Full_field_zo_zones.png background. Same logical
    layout as full_pp, anchored to full_zo's painted lines (image 1227 x 1835)."""
    L_side   =  64   # left sideline (detected at corner)
    PA_L     = 278   # PA left (detected)
    GA_L     = 449   # 6-yard box left (detected)
    # GA-L / GA-C / GA-R dividers split the 6-yd box into 3 EQUAL columns
    # (matches the convention used in full_pp where the dividers are painted).
    GP_L     = 551   # left divider — 1/3 of 6-yd box width
    GP_R     = 652   # right divider — 2/3 of 6-yd box width
    GA_R     = 754   # 6-yard box right (detected)
    PA_R     = 927   # PA right (detected)
    R_side   =1154   # right sideline

    goal_y   =  67   # TRUE backline
    GA_bot   = 164   # 6-yard line (measured)
    PA_bot   = 358   # PA bottom
    closer_y = 644
    mid_y    = 930

    return [
        ("GA-L",       GA_L,   goal_y, GP_L,   GA_bot),
        ("GA-C",       GP_L,   goal_y, GP_R,   GA_bot),
        ("GA-R",       GP_R,   goal_y, GA_R,   GA_bot),

        ("Front",      PA_L,   goal_y, GA_L,   PA_bot),
        ("PA",         GA_L,   GA_bot, GA_R,   PA_bot),
        ("Back",       GA_R,   goal_y, PA_R,   PA_bot),

        ("Wide_L",     L_side, goal_y, PA_L,   PA_bot),
        ("Wide_R",     PA_R,   goal_y, R_side, PA_bot),

        ("Close-L",    L_side, PA_bot, PA_L,   closer_y),
        ("Close-C",    PA_L,   PA_bot, PA_R,   closer_y),
        ("Close-R",    PA_R,   PA_bot, R_side, closer_y),

        ("Mid-L",      L_side, closer_y, PA_L,   mid_y),
        ("Mid-C",      PA_L,   closer_y, PA_R,   mid_y),
        ("Mid-R",      PA_R,   closer_y, R_side, mid_y),
    ]


def _build_pb_pixel_zones():
    """Pixel zones for penalty_box_entries_zones.png (half-pitch, h=987).
    Same logical layout as full_pp's attacking half; dividers at 1/3 and 2/3
    of the 6-yd box width (matching the painted dividers in this image)."""
    L_side   =  66
    PA_L     = 290
    GA_L     = 468
    GP_L     = 576   # painted divider at 1/3
    GP_R     = 683   # painted divider at 2/3
    GA_R     = 790
    PA_R     = 972
    R_side   =1211

    goal_y   =  72   # TRUE backline (6-yd-box verticals start here)
    GA_bot   = 173   # 6-yard line (= 6-yd box bottom, detected band 172-177)
    PA_bot   = 378   # PA bottom

    return [
        ("GA-L",   GA_L,   goal_y, GP_L,   GA_bot),
        ("GA-C",   GP_L,   goal_y, GP_R,   GA_bot),
        ("GA-R",   GP_R,   goal_y, GA_R,   GA_bot),
        ("Front",  PA_L,   goal_y, GA_L,   PA_bot),
        ("PA",     GA_L,   GA_bot, GA_R,   PA_bot),
        ("Back",   GA_R,   goal_y, PA_R,   PA_bot),
        ("Wide_L", L_side, goal_y, PA_L,   PA_bot),
        ("Wide_R", PA_R,   goal_y, R_side, PA_bot),
    ]


def _build_cz_pixel_zones():
    """Pixel zones for cross_zones.png (half-pitch, h=982). Same logical layout
    as pb_entries, plus the Close-L/C/R row between PA bottom and the painted
    close-to-PA line (this image shows that area too)."""
    L_side   =  66
    PA_L     = 290
    GA_L     = 468
    GP_L     = 576
    GP_R     = 683
    GA_R     = 790
    PA_R     = 972
    R_side   =1211

    goal_y   =  72
    GA_bot   = 167
    PA_bot   = 370
    closer_y = 671

    return [
        ("GA-L",    GA_L,   goal_y, GP_L,   GA_bot),
        ("GA-C",    GP_L,   goal_y, GP_R,   GA_bot),
        ("GA-R",    GP_R,   goal_y, GA_R,   GA_bot),
        ("Front",   PA_L,   goal_y, GA_L,   PA_bot),
        ("PA",      GA_L,   GA_bot, GA_R,   PA_bot),
        ("Back",    GA_R,   goal_y, PA_R,   PA_bot),
        ("Wide_L",  L_side, goal_y, PA_L,   PA_bot),
        ("Wide_R",  PA_R,   goal_y, R_side, PA_bot),
        ("Close-L", L_side, PA_bot, PA_L,   closer_y),
        ("Close-C", PA_L,   PA_bot, PA_R,   closer_y),
        ("Close-R", PA_R,   PA_bot, R_side, closer_y),
    ]


_FULL_PP_PIXEL_ZONES = _build_full_pp_pixel_zones()
_FULL_ZO_PIXEL_ZONES = _build_full_zo_pixel_zones()
_PB_PIXEL_ZONES      = _build_pb_pixel_zones()
_CZ_PIXEL_ZONES      = _build_cz_pixel_zones()


def _pixel_zone_for(px, py, pixel_zones):
    """Return the name of the first pixel zone that contains (px, py),
    or None."""
    for name, x0, y0, x1, y1 in pixel_zones:
        if x0 <= px <= x1 and y0 <= py <= y1:
            return name
    return None


def _overlay_zone_counters(fig, events, image_key, pixel_zones,
                            point_attr="start"):
    """Tally events into pixel zones (by `point_attr` = 'start' or 'end') and
    add ONE black-text counter annotation per zone showing the tally. The pitch
    background stays unaltered — no fill, no border, just a small white-ish
    label box with the count in black. Returns the dict {zone_name: count}."""
    counts = {name: 0 for name, *_ in pixel_zones}
    for e in events:
        if point_attr == "end":
            x_m, y_m = e.end_x, e.end_y
        else:
            x_m, y_m = e.start_x, e.start_y
        mp = _metric_to_pixel(x_m, y_m, image_key)
        if mp is None:
            continue
        zn = _pixel_zone_for(mp[0], mp[1], pixel_zones)
        if zn:
            counts[zn] += 1
    for name, px0, py0, px1, py1 in pixel_zones:
        cx, cy = (px0 + px1) / 2, (py0 + py1) / 2
        fig.add_annotation(
            x=cx, y=cy, text=f"<b>{counts[name]}</b>",
            showarrow=False,
            font=dict(color="black", size=14, family="Arial Black"),
            bgcolor="rgba(255,255,255,0.78)",
        )
    return counts


def _pitch_image_size(image_key):
    cfg = _PITCH_IMAGES.get(image_key)
    return cfg["size"] if cfg else None


def _pitch_image_path(image_key):
    cfg = _PITCH_IMAGES.get(image_key)
    return cfg["path"] if cfg else None


def _plotly_pitch_image(image_key, fig_height=None):
    """Build an empty Plotly figure with the named Zone-Figures pitch image
    as the background, axes set to pixel coordinates and locked (no pan/zoom).
    Returns (fig, (iw, ih)) — the size is needed by callers to set ranges."""
    cfg = _PITCH_IMAGES[image_key]
    iw, ih = cfg["size"]
    if cfg.get("rotate_cw_deg"):
        img = _cached_rotated_image(str(cfg["path"]), int(cfg["rotate_cw_deg"]))
    else:
        img = _cached_image(str(cfg["path"]))
    fig = go.Figure()
    fig.add_layout_image(dict(
        source=img, xref="x", yref="y",
        x=0, y=0, sizex=iw, sizey=ih,
        sizing="stretch", layer="below",
    ))
    fig.update_layout(
        xaxis=dict(range=[0, iw], visible=False, fixedrange=True),
        yaxis=dict(range=[ih, 0], visible=False, fixedrange=True,
                    scaleanchor="x"),
        height=fig_height if fig_height is not None else (
            int(ih * 0.65) if ih > 1000 else int(ih * 0.85)
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        plot_bgcolor="white", paper_bgcolor="white",
        showlegend=False, dragmode=False,
    )
    return fig, (iw, ih)


def _metric_to_def_pixel(x_m, y_m, side):
    """Map a tracking position (attacking-right normalised metres) to pixel
    coordinates on the schematic no_names_left / no_names_right image.

    Mapping is piecewise-affine, computed per zone (the images are stylised,
    not a true top-down view, so a single linear transform is impossible).
    Falls back to the nearest defined zone when the point lies outside every
    zone, so every player renders somewhere reasonable on the schematic."""
    pos = "top_left" if side == "L" else "top_right"
    zones_metric = _build_corner_zones()[pos]
    zone_pixels = _DEF_L_ZONES if side == "L" else _DEF_R_ZONES

    zone_name = _assign_zone(x_m, y_m, zones_metric)
    if zone_name is None or zone_name not in zone_pixels:
        # Fallback: choose the closest defined zone by point-to-rectangle
        # distance in metric space, then clamp the point into that zone.
        zone_name = _nearest_zone(x_m, y_m, zones_metric, zone_pixels)
        if zone_name is None:
            return None
    m_rect = zones_metric[zone_name]
    p_rect = zone_pixels[zone_name]
    mxs = [p[0] for p in m_rect]; mys = [p[1] for p in m_rect]
    pxs = [p[0] for p in p_rect]; pys = [p[1] for p in p_rect]
    mx0, mx1 = min(mxs), max(mxs); my0, my1 = min(mys), max(mys)
    px0, px1 = min(pxs), max(pxs); py0, py1 = min(pys), max(pys)
    # Clamp to zone so out-of-zone points land at the nearest edge
    cx = max(mx0, min(mx1, x_m))
    cy = max(my0, min(my1, y_m))
    fy = (cy - my0) / (my1 - my0) if my1 > my0 else 0.5
    fx = (cx - mx0) / (mx1 - mx0) if mx1 > mx0 else 0.5
    px = px0 + fy * (px1 - px0)
    py = py0 + (1 - fx) * (py1 - py0)  # higher metric_x → top of image
    return px, py


def _nearest_zone(x_m, y_m, zones_metric, zone_pixels):
    """Pick the zone whose metric bounding box is closest to (x_m, y_m).
    Only considers zones that also exist in zone_pixels."""
    best_name, best_d = None, float("inf")
    for name, rect in zones_metric.items():
        if name not in zone_pixels:
            continue
        mxs = [p[0] for p in rect]; mys = [p[1] for p in rect]
        mx0, mx1 = min(mxs), max(mxs); my0, my1 = min(mys), max(mys)
        dx = max(mx0 - x_m, 0, x_m - mx1)
        dy = max(my0 - y_m, 0, y_m - my1)
        d = (dx * dx + dy * dy) ** 0.5
        if d < best_d:
            best_d, best_name = d, name
    return best_name


def _corner_side_from_event(corner_event):
    """Determine which TV side the corner was taken from.

    In SciSports per-team normalised coords (team attacks +x), the y-axis is
    flipped relative to the TV view: y=+34 is the TV-LEFT touchline, y=-34 is
    TV-RIGHT.  `_get_corner_position` calls y=-34 "top_left" and y=+34
    "top_right", so the L/R assignment is the OPPOSITE of the position name.
    """
    pos = _get_corner_position(corner_event.start_x, corner_event.start_y)
    if pos is None:
        pos = "top_left" if corner_event.start_y < 0 else "top_right"
    # "top_right" (y=+34) = TV-LEFT corner; "top_left" (y=-34) = TV-RIGHT.
    return "L" if "right" in pos else "R"


# ================================================================
# VIDEO PLAYER
# ================================================================

def show_video_for_event(match, event, key_suffix=""):
    available = {k: v for k, v in match.cameras.items() if v != "PLACEHOLDER"}
    if not available:
        st.warning("No video files configured for this match.")
        return

    camera_names = list(available.keys())
    radio_key = f"camera_select_{key_suffix}" if key_suffix else "camera_select"
    if len(camera_names) > 1:
        cam = st.radio("Camera", camera_names, horizontal=True, key=radio_key)
    else:
        cam = camera_names[0]

    # If the clip is already on disk (or in the in-memory cache), skip the
    # spinner — extraction is instant. Only show the spinner for genuinely
    # new clips.
    from video_utils import get_clip_path as _get_clip_path
    cached_path = _get_clip_path(
        available[cam],
        max(0, event.video_time_sec - 5.0),
        event.video_time_sec + 12.0,
    )
    needs_extract = not cached_path.exists()
    try:
        if needs_extract:
            with st.spinner(f"Extracting clip ({cam})..."):
                clip = _cached_extract_clip(available[cam], event.video_time_sec)
        else:
            clip = _cached_extract_clip(available[cam], event.video_time_sec)
        st.video(clip)
    except Exception as e:
        st.error(f"Clip failed: {e}")


# ================================================================
# VIZ: CORNERS (side images with clickable zones)
# ================================================================

def _render_corner_side(side_classified, side, img_path, zone_pixels, zone_centers,
                        all_events, key_prefix):
    """Render one side's corner image with zone counts, and list clips per zone."""
    st.markdown(f"**{'Left' if side == 'L' else 'Right'} Corners ({len(side_classified)})**")
    if not side_classified:
        st.caption("No corners from this side.")
        return

    img = plt.imread(str(img_path))
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.imshow(img)

    zone_counts = Counter(c[2] for c in side_classified if c[2])
    max_count = max(zone_counts.values()) if zone_counts else 1

    for zone_name, poly in zone_pixels.items():
        cnt = zone_counts.get(zone_name, 0)
        if cnt == 0:
            continue
        xs = [p[0] for p in poly] + [poly[0][0]]
        ys = [p[1] for p in poly] + [poly[0][1]]
        alpha = 0.25 + 0.55 * (cnt / max_count)
        ax.fill(xs, ys, color="#e74c3c", alpha=alpha, edgecolor="white", linewidth=1.5)
        cx, cy = zone_centers.get(zone_name, (np.mean(xs), np.mean(ys)))
        ax.text(cx, cy, str(cnt), color="white", fontsize=12, fontweight="bold",
                ha="center", va="center",
                bbox=dict(boxstyle="circle,pad=0.25", fc="#2c3e50", ec="white"))

    ax.axis("off")
    ax.set_xlim(0, img.shape[1])
    ax.set_ylim(img.shape[0], 0)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Clickable list per zone (most common first)
    st.markdown("*Click a zone below to expand clips*")
    for zone_name, cnt in zone_counts.most_common():
        with st.expander(f"{zone_name.replace('_', ' ')} ({cnt})"):
            zone_events = [c[0] for c in side_classified if c[2] == zone_name]
            _event_buttons(zone_events, all_events, f"{key_prefix}_{zone_name}")


def viz_def_corner_event(events, team, match):
    """Sidebar dispatch for the Defending Corners event type."""
    _viz_defending_corners(events, team, match)


def viz_att_corner_event(events, team, match):
    """Sidebar dispatch for the Attacking Corners event type. Renders tabs
    for the different attacking-corner visualisations."""
    if team == BOTH_LABEL:
        st.info("Pick one team in the sidebar to analyse their attacking corners.")
        return
    own_corners = [e for e in match.events
                    if e.event_type == "corner" and e.team == team]
    if not own_corners:
        st.caption(f"No attacking corners by {team} in this match.")
        return

    tabs = st.tabs([
        "Delivery placement", "Shot conversion per zone", "Per-side breakdown",
    ])
    with tabs[0]:
        st.markdown("**Where their corners are placed (per zone, per side)**")
        _att_corner_placement(own_corners, events, key_prefix="att_corner_place")
    with tabs[1]:
        st.markdown("**Shot conversion per zone**")
        _att_corner_shot_rate(own_corners, events, match,
                                key_prefix="att_corner_shot")
    with tabs[2]:
        st.markdown("**Per-side clip lists**")
        _att_corner_legacy_lists(own_corners, events)


def _classify_own_corner(corner_event):
    """Side + zone for an attacking corner (corner taker is the analysed team).
    Uses image-zone labels: GA1-3 / CA1-3 / Edge_Zone / Front_Zone /
    Back_Zone / Short_Corner_Zone."""
    return _classify_def_corner(corner_event)


def _att_corner_zone_panel(side, side_corners, key_prefix, value_fn, label_fn,
                             caption=""):
    """Render one side's attacking-corner image with red-shaded zones and
    custom text per zone (driven by value_fn / label_fn)."""
    title = "Left-side corners" if side == "L" else "Right-side corners"
    st.markdown(f"_{title}_ — {len(side_corners)} corner(s)")
    if not side_corners:
        st.caption("None from this side.")
        return

    img_path = _LEFT_CORNER_IMG if side == "L" else _RIGHT_CORNER_IMG
    polys = _ATT_L_ZONES if side == "L" else _ATT_R_ZONES
    centers = _ATT_L_CENTERS if side == "L" else _ATT_R_CENTERS

    events_by_zone = {}
    for ev in side_corners:
        _, zone = _classify_own_corner(ev)
        if zone:
            events_by_zone.setdefault(zone, []).append(ev)
    counts = {z: len(v) for z, v in events_by_zone.items()}

    fig = go.Figure()
    try:
        img = _cached_image(str(img_path))
        iw, ih = img.size
    except Exception:
        st.warning(f"Image not found: {img_path}")
        return
    fig.add_layout_image(
        dict(source=img, xref="x", yref="y",
             x=0, y=0, sizex=iw, sizey=ih,
             sizing="stretch", layer="below"),
    )

    # Colour each populated zone (intensity scales with the value)
    vals = {z: value_fn(z, events_by_zone[z], counts[z]) for z in counts}
    max_val = max(vals.values()) if vals else 1.0
    for zone, poly in polys.items():
        if counts.get(zone, 0) == 0:
            continue
        v = vals.get(zone, 0.0)
        alpha = 0.25 + 0.55 * (v / max_val if max_val > 0 else 0.0)
        fig.add_shape(
            type="path", path=_polygon_path(poly),
            fillcolor=f"rgba(231,76,60,{alpha:.2f})",
            line=dict(color="rgba(0,0,0,0)", width=0),
            layer="above",
        )

    xs, ys, txts, cds, hovers = [], [], [], [], []
    for zone, (cx, cy) in centers.items():
        if counts.get(zone, 0) == 0:
            continue
        xs.append(cx); ys.append(cy); cds.append(zone)
        txts.append(label_fn(zone, events_by_zone[zone], counts[zone]))
        hovers.append(f"<b>{zone}</b><br>{counts[zone]} corner(s)")
    if xs:
        # Large transparent hit-zone scatter (clickable) — Plotly's selection
        # hit-test only fires on markers that have meaningful size; size=1
        # transparent markers don't register clicks.
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="markers",
            marker=dict(size=55, color="rgba(0,0,0,0)",
                         line=dict(color="rgba(0,0,0,0)", width=0)),
            customdata=cds, hovertext=hovers, hoverinfo="text",
            showlegend=False, name=f"hit_{side}",
        ))
        # Text labels on top
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="text",
            text=txts, textposition="middle center",
            textfont=dict(color="black", size=13, family="Arial Black"),
            hoverinfo="skip", showlegend=False,
        ))
    display_w = 750
    display_h = int(ih * display_w / iw) + 30
    fig.update_layout(
        xaxis=dict(range=[0, iw], visible=False, fixedrange=True),
        yaxis=dict(range=[ih, 0], visible=False, fixedrange=True),
        height=display_h, margin=dict(l=0, r=0, t=0, b=0),
        plot_bgcolor="white", paper_bgcolor="white",
        showlegend=False, dragmode=False,
    )
    result = st.plotly_chart(
        fig, use_container_width=True, key=f"{key_prefix}_{side}_chart",
        on_select="rerun", selection_mode="points",
        config={"displayModeBar": False, "scrollZoom": False,
                "doubleClick": False},
    )
    if result:
        sel = result.get("selection") if isinstance(result, dict) else None
        if sel and sel.get("points"):
            pt = sel["points"][0]
            zone = pt.get("customdata")
            if zone:
                sig = f"{side}_{zone}"
                ck = f"__consumed_{key_prefix}_{side}"
                if st.session_state.get(ck) != sig:
                    st.session_state[ck] = sig
                    group = events_by_zone.get(zone, [])
                    if group:
                        _apply_filter_from_click(
                            group,
                            filter_key=f"{key_prefix}:{side}:{zone}",
                            label=f"{'Left' if side == 'L' else 'Right'} att corner → {zone}",
                        )
    if caption:
        st.caption(caption)


def _att_corner_placement(own_corners, nav_events, key_prefix):
    """% of corners placed in each zone, split L vs R, click-to-filter clips."""
    left_corners, right_corners = [], []
    for ev in own_corners:
        side, _ = _classify_own_corner(ev)
        (left_corners if side == "L" else right_corners).append(ev)

    def val(zone, evts, n):
        # Value = share of corners placed in this zone (for colour scaling)
        return n
    def lbl(zone, evts, n):
        return f"{n}"

    _att_corner_zone_panel("L", left_corners, key_prefix, val, lbl,
                             caption="Number of attacking corners delivered into each zone.")
    _att_corner_zone_panel("R", right_corners, key_prefix, val, lbl)


def _att_corner_shot_rate(own_corners, nav_events, match, key_prefix):
    """Shot conversion: shots / total per zone, with shot rate %. Sequence-based."""
    shot_seqs = {e.sequence_id for e in match.events
                  if e.event_type in ("shot", "shot_on_target", "goal", "big_chance")
                  and e.sequence_id >= 0}

    left_corners, right_corners = [], []
    for ev in own_corners:
        side, _ = _classify_own_corner(ev)
        (left_corners if side == "L" else right_corners).append(ev)

    def n_shots(evts):
        return sum(1 for ev in evts if getattr(ev, "sequence_id", -1) in shot_seqs)
    def val(zone, evts, n):
        return (n_shots(evts) / n) if n else 0.0
    def lbl(zone, evts, n):
        s = n_shots(evts)
        return f"{s}/{n} · {(s/n)*100:.0f}%" if n else "—"

    _att_corner_zone_panel("L", left_corners, key_prefix, val, lbl,
                             caption="`shots / total · %` per zone — counts every shot in the "
                                      "same sequence as the corner.")
    _att_corner_zone_panel("R", right_corners, key_prefix, val, lbl)


def _att_corner_legacy_lists(own_corners, nav_events):
    """Per-side clip lists (legacy compact view)."""
    zones_by_pos = _build_corner_zones()
    classified = []
    for e in own_corners:
        pos = _get_corner_position(e.start_x, e.start_y)
        if pos is None:
            pos = "top_left" if e.start_y < 0 else "top_right"
        side = "L" if "left" in pos else "R"
        zones = zones_by_pos[pos]
        if e.sub_type == "CORNER_SHORT":
            zone = "Short_Corner_Zone"
        else:
            zone = _assign_zone(e.end_x, e.end_y, zones)
        classified.append((e, side, zone))
    left = [c for c in classified if c[1] == "L"]
    right = [c for c in classified if c[1] == "R"]
    col1, col2 = st.columns(2)
    with col1:
        _render_corner_side(left, "L", _LEFT_CORNER_IMG, _ATT_L_ZONES,
                             _ATT_L_CENTERS, nav_events, key_prefix="acorner_L")
    with col2:
        _render_corner_side(right, "R", _RIGHT_CORNER_IMG, _ATT_R_ZONES,
                             _ATT_R_CENTERS, nav_events, key_prefix="acorner_R")


# ================================================================
# DEFENDING CORNERS (model-driven)
# ================================================================

_PA3_BG = DATA_DIR / "PA (3).png"

# Zone -> approximate (x, y) center in SciSports metres on the defended half
# (attacking-right convention from the model: defended goal at +52.5).
# Used to colour and label the PA(3).png background. Coordinates are
# approximate visual centers chosen to match the example image.
_CORNER_ZONE_CENTERS_M = {
    "GA1": (49.0,  6.0),  "GA2": (49.0,  0.0),  "GA3": (49.0, -6.0),
    "CA1": (43.5,  6.0),  "CA2": (43.5,  0.0),  "CA3": (43.5, -6.0),
    "PA1": (38.5,  6.0),  "PA2": (38.5,  0.0),  "PA3": (38.5, -6.0),
    "PE1": (33.5,  6.0),  "PE2": (33.5,  0.0),  "PE3": (33.5, -6.0),
    "PC1": (29.0,  6.0),  "PC2": (29.0,  0.0),  "PC3": (29.0, -6.0),
    "FRONT": (45.0, 18.0),
    "EDGE":  (32.0, 18.0),
    "Back Zone": (32.0, -18.0),
    "Short corner": (47.0, 25.0),
}


def _team_defending_corners(match, team):
    """Return all match corner events where `team` was DEFENDING (corner taker
    is the opponent). Includes both synced and unsynced corners (clips need
    them all). `team` of BOTH means both teams' defending corners."""
    out = []
    for e in match.events:
        if e.event_type != "corner":
            continue
        if team == BOTH_LABEL:
            out.append(e)
        elif e.team != team:  # opponent is the taker -> we are defending
            out.append(e)
    return sorted(out, key=lambda x: x.game_time_ms)


def _viz_defending_corners(events, team, match):
    """Inside col_viz portion: caption + warnings + Vis 1.

    Vis 2-7 are rendered AFTER the col_video/col_viz block at full width via
    `_render_defending_corners_extras` (called from main() so they can use the
    full screen width below the clip and Vis 1)."""
    if team == BOTH_LABEL:
        st.info("Pick one team in the sidebar to analyse their defending corners.")
        return

    defending_team = team
    def_corners = _team_defending_corners(match, defending_team)
    if not def_corners:
        st.caption(f"No defending corners for {defending_team} in this match.")
        return

    st.caption(
        f"{defending_team} defended **{len(def_corners)} corner(s)** in this match. "
        "Unsynced corners (event/tracking misalignment) are used only for clip "
        "playback and the Delivery Zones map; all other visualisations work on "
        "synced corners only."
    )

    events_path = _events_path_for_match(match)
    positions_path = _positions_path_for_match(match)
    analysis = None
    analysis_err = None
    if events_path and positions_path:
        with st.spinner("Running corner role models..."):
            analysis, analysis_err = _cached_corner_analysis(
                str(events_path), str(positions_path), defending_team,
            )
    else:
        analysis_err = "Events or tracking JSON not found for this match."

    rows = (analysis or {}).get("rows", []) if analysis else []

    if analysis_err:
        st.warning(
            f"Role predictions are disabled: {analysis_err}\n\n"
            "Delivery zones and clips still work. To enable role analysis, place "
            "`defender_role_rf.joblib`, `attacker_role_rf.joblib`, "
            "`feature_columns.json` alongside `app.py`."
        )

    # ---- Vis 1: Corners that turned into a shot ----
    st.markdown("### Corners that turned into a shot")
    _vis1_delivery_zones(def_corners, rows, events, key_prefix="def_corner_zones",
                          match=match)

    # Mark that we have model rows so main() renders Vis 2-7 below the columns.
    st.session_state["__corner_extras_ready"] = bool(rows)


def _render_defending_corners_extras(events, team, match):
    """Full-width Vis 2-7, rendered by main() AFTER the col_video/col_viz block.
    Uses the same cached analysis (free re-call thanks to @st.cache_data)."""
    if team == BOTH_LABEL:
        return
    if not st.session_state.pop("__corner_extras_ready", False):
        return

    events_path = _events_path_for_match(match)
    positions_path = _positions_path_for_match(match)
    if not events_path or not positions_path:
        return
    analysis, _ = _cached_corner_analysis(
        str(events_path), str(positions_path), team,
    )
    if not analysis:
        return
    rows = analysis.get("rows", [])
    aggregates = analysis.get("aggregates", {})
    if not rows:
        return

    st.markdown("---")
    st.markdown("### Defending corner analysis — role-based visualisations")

    r1c1, r1c2 = st.columns(2)
    with r1c1:
        st.markdown("**Vis 2 — Player roles map**")
        _vis2_role_map(rows, match, team, nav_events=events)
    with r1c2:
        st.markdown("**Vis 3 — Player roles table**")
        _vis3_role_table(rows, match, team)

    r2c1, r2c2 = st.columns(2)
    with r2c1:
        st.markdown("**Vis 4 — Avg role counts (by short option)**")
        _vis5_avg_role_counts(aggregates)
    with r2c2:
        st.markdown("**Vis 5 — Zones usually zonally marked**")
        _vis6_zonal_zones(aggregates)

    r3c1, _ = st.columns(2)
    with r3c1:
        st.markdown("**Vis 6 — Attackers exceed markers**")
        _vis7_attackers_exceed(aggregates)

    # ---- Vis 7: Magnet board (full-width, bottom of page) ----
    st.markdown("---")
    _vis7_magnet_board(rows, match, team)


# ----------------------------------------------------------------
# Vis 1: Delivery zones using left_side_corner.png + right_side_corner.png
# ----------------------------------------------------------------

def _classify_def_corner(corner_event):
    """Return (side, image_zone_name) using the same scheme as the legacy
    Attacking-Corners view (Front_Zone / Back_Zone / Short_Corner_Zone /
    GA1-3 / CA1-3 / Edge_Zone).

    Side convention MUST match `_corner_side_from_event` (used by Vis 2):
    in SciSports per-team coords y=+34 is the TV-LEFT touchline, so
    `_get_corner_position` returns "top_right" for the TV-LEFT corner. The
    L/R label is therefore the OPPOSITE of the position-name direction."""
    pos = _get_corner_position(corner_event.start_x, corner_event.start_y)
    if pos is None:
        pos = "top_left" if corner_event.start_y < 0 else "top_right"
    side = "L" if "right" in pos else "R"  # match Vis 2
    if getattr(corner_event, "sub_type", "") == "CORNER_SHORT":
        return side, "Short_Corner_Zone"
    zones = _build_corner_zones()[pos]
    return side, _assign_zone(corner_event.end_x, corner_event.end_y, zones)


def _polygon_path(poly):
    """SVG path string for a closed polygon, used by Plotly add_shape(type='path')."""
    pts = " ".join(f"L {x},{y}" for x, y in poly[1:])
    return f"M {poly[0][0]},{poly[0][1]} {pts} Z"


def _vis1_side_panel(side, side_corners, all_events, model_rows, key_prefix, match):
    """One side's clickable zone heatmap over the L or R corner image."""
    title = "From the left" if side == "L" else "From the right"
    st.markdown(f"**{title}** — {len(side_corners)} corner(s)")
    if not side_corners:
        st.caption("None.")
        return

    img_path = _LEFT_CORNER_IMG if side == "L" else _RIGHT_CORNER_IMG
    polys = _ATT_L_ZONES if side == "L" else _ATT_R_ZONES
    centers = _ATT_L_CENTERS if side == "L" else _ATT_R_CENTERS

    # Bucket events by image zone label
    events_by_zone = {}
    for ev in side_corners:
        _, zone = _classify_def_corner(ev)
        if zone:
            events_by_zone.setdefault(zone, []).append(ev)

    # Shot detection by sequence id (works for synced AND unsynced corners)
    shot_sequence_ids = {e.sequence_id for e in match.events
                          if e.event_type in ("shot", "shot_on_target", "goal", "big_chance")
                          and e.sequence_id >= 0}

    shot_count_by_zone = {}
    for zone, evts in events_by_zone.items():
        n_shots = sum(1 for ev in evts
                       if getattr(ev, "sequence_id", -1) in shot_sequence_ids)
        if n_shots:
            shot_count_by_zone[zone] = n_shots

    # Total per-zone count
    total_by_zone = {z: len(v) for z, v in events_by_zone.items()}
    shot_pct = {z: (shot_count_by_zone.get(z, 0) / total_by_zone[z])
                 for z in total_by_zone}

    # Render image with Plotly
    fig = go.Figure()
    try:
        img = _cached_image(str(img_path))
        iw, ih = img.size
    except Exception:
        st.warning(f"Image not found: {img_path}")
        return
    fig.add_layout_image(
        dict(source=img, xref="x", yref="y",
             x=0, y=0, sizex=iw, sizey=ih,
             sizing="stretch", layer="below"),
    )

    # Colour each populated zone with red intensity ~ shot rate
    max_rate = max(shot_pct.values()) if shot_pct else 1.0
    for zone, poly in polys.items():
        n = total_by_zone.get(zone, 0)
        if n == 0:
            continue
        rate = shot_pct.get(zone, 0.0)
        # Alpha: at least 0.20 so zones with 0% shots are still visible
        alpha = 0.25 + 0.55 * (rate / max_rate if max_rate > 0 else 0.0)
        fig.add_shape(
            type="path", path=_polygon_path(poly),
            fillcolor=f"rgba(231,76,60,{alpha:.2f})",
            line=dict(color="rgba(0,0,0,0)", width=0),
            layer="above",
        )

    # Scatter at zone centers for labels + clicks
    xs, ys, txts, cds, hovers = [], [], [], [], []
    for zone, (cx, cy) in centers.items():
        n = total_by_zone.get(zone, 0)
        if n == 0:
            continue
        shots = shot_count_by_zone.get(zone, 0)
        xs.append(cx); ys.append(cy); cds.append(zone)
        # Vis 1 label: "shots/total" only — no percentage, per user spec
        txts.append(f"{shots}/{n}")
        hovers.append(
            f"<b>{zone}</b><br>{n} corner{'s' if n != 1 else ''}"
            f"<br>{shots} shot{'s' if shots != 1 else ''}"
        )
    if xs:
        # Hit-zone scatter (clickable, transparent)
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="markers",
            marker=dict(size=55, color="rgba(0,0,0,0)",
                         line=dict(color="rgba(0,0,0,0)", width=0)),
            customdata=cds, hovertext=hovers, hoverinfo="text",
            showlegend=False, name=f"hit_{side}",
            cliponaxis=False,
        ))
        # Text labels on top
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="text",
            text=txts, textposition="middle center",
            textfont=dict(color="black", size=14, family="Arial Black"),
            hoverinfo="skip", showlegend=False, cliponaxis=False,
        ))

    display_w = 750
    display_h = int(ih * display_w / iw) + 30
    fig.update_layout(
        xaxis=dict(range=[0, iw], visible=False, fixedrange=True),
        yaxis=dict(range=[ih, 0], visible=False, fixedrange=True),
        height=display_h, margin=dict(l=0, r=0, t=0, b=0),
        plot_bgcolor="white", paper_bgcolor="white",
        showlegend=False, dragmode=False,
    )
    result = st.plotly_chart(
        fig, use_container_width=True, key=f"{key_prefix}_{side}_chart",
        on_select="rerun", selection_mode="points",
        config={"displayModeBar": False, "scrollZoom": False,
                "doubleClick": False},
    )

    if result:
        sel = result.get("selection") if isinstance(result, dict) else None
        if sel and sel.get("points"):
            pt = sel["points"][0]
            zone = pt.get("customdata")
            if zone:
                sig = f"{side}_{zone}"
                ck = f"__consumed_{key_prefix}_{side}"
                if st.session_state.get(ck) != sig:
                    st.session_state[ck] = sig
                    group = events_by_zone.get(zone, [])
                    if group:
                        _apply_filter_from_click(
                            group,
                            filter_key=f"{key_prefix}:{side}:{zone}",
                            label=f"{'Left' if side == 'L' else 'Right'} corner → {zone}",
                        )


def _vis1_delivery_zones(def_corners, model_rows, nav_events, key_prefix, match):
    """Split corners into L/R sides; render each side over its dedicated image."""
    left_corners, right_corners = [], []
    for ev in def_corners:
        side, _ = _classify_def_corner(ev)
        (left_corners if side == "L" else right_corners).append(ev)

    _vis1_side_panel("L", left_corners, nav_events, model_rows, key_prefix, match)
    _vis1_side_panel("R", right_corners, nav_events, model_rows, key_prefix, match)
    pass  # caption removed per user request


# ----------------------------------------------------------------
# (Old Vis 2 — "Average defensive setup" — was removed in favour of the
# Player roles map; it lived here.)


# ----------------------------------------------------------------
# Vis 3 + 4: Per-corner role map and role table
# ----------------------------------------------------------------

def _vis34_picker_key(defending_team):
    """Session-state key for the shared corner picker (Vis 3 + Vis 4 + clip jump)."""
    return f"__vis34_pick_{defending_team}"


def _resolve_vis34_pick(rows, defending_team):
    """Render the shared corner selectbox at the top of Vis 3's panel.
    Returns the picked row. Also triggers a clip jump when the user changes it.
    Called by both Vis 3 and Vis 4 so they stay in sync; only Vis 3 renders the
    selectbox (Vis 4 just reads the same session key)."""
    key = _vis34_picker_key(defending_team)
    if not rows:
        return None
    if st.session_state.get(key, 0) >= len(rows):
        st.session_state[key] = 0
    return rows[st.session_state.get(key, 0)]


def _vis2_role_map(rows, match, defending_team, nav_events):
    if not rows:
        st.caption("No synced corners.")
        return

    key = _vis34_picker_key(defending_team)
    def _label_for(r):
        ev = _find_event_for_corner_row(r, nav_events)
        clock = ev.game_time_display if ev is not None else r.get("match_clock", "??:??")
        # Determine side from the KICKER'S start position, not the model's
        # end-y-based corner_side (which can disagree with which flag they stood at).
        kicker_side = (_corner_side_from_event(ev)
                       if ev is not None else r.get("corner_side", "?"))
        return f"{clock}  ({kicker_side})  —  zone {r['delivery']['zone']}"

    labels = [_label_for(r) for r in rows]
    pick = st.selectbox(
        "Pick a corner",
        range(len(rows)),
        format_func=lambda i: labels[i],
        key=key,
    )
    r = rows[pick]

    # Explicit "Watch clip" button — replaces the implicit auto-jump that
    # used to happen on selectbox change.
    if st.button("▶ Watch clip", key=f"{key}_watch",
                  use_container_width=False):
        match_ev = _find_event_for_corner_row(r, nav_events)
        if match_ev is not None:
            _jump_to_event(match_ev, nav_events)
        else:
            st.warning("Couldn't locate the matching event for this corner.")

    # Choose the new Zone-Figures half-pitch by the corner-taker's start side
    match_ev = _find_event_for_corner_row(r, nav_events)
    if match_ev is not None:
        side = _corner_side_from_event(match_ev)
    else:
        side = r.get("corner_side", "L")
    image_key = "sg_left_nozones" if side == "L" else "sg_right_nozones"

    fig, (iw, ih) = _plotly_pitch_image(image_key, fig_height=420)

    # Opponents (attackers) — gray dots, no jersey number
    xa, ya = [], []
    for a in r["attackers"]:
        p = a.get("position_at_kick") or a.get("position_at_setup")
        if not p:
            continue
        mapped = _metric_to_pixel(p[0], p[1], image_key)
        if mapped is None:
            continue
        xa.append(mapped[0]); ya.append(mapped[1])
    if xa:
        fig.add_trace(go.Scatter(
            x=xa, y=ya, mode="markers",
            marker=dict(size=16, color="#7f8c8d", opacity=0.85,
                         line=dict(color="white", width=1.5)),
            name="Opponent", hoverinfo="skip",
        ))

    role_color = {"MAN": "#e74c3c", "ZONAL": "#27ae60",
                   "SHORT": "#f39c12", "COUNTER": "#9b59b6"}
    role_groups = {}
    for d in r["defenders"]:
        p = d.get("position_at_kick") or d.get("position_at_setup")
        if not p:
            continue
        mapped = _metric_to_pixel(p[0], p[1], image_key)
        if mapped is None:
            continue
        role_groups.setdefault(d["role"], []).append(
            (mapped[0], mapped[1], d["jersey"], d.get("player_name", ""))
        )
    for role, items in role_groups.items():
        color = role_color.get(role, "#34495e")
        fig.add_trace(go.Scatter(
            x=[i[0] for i in items], y=[i[1] for i in items],
            mode="markers+text",
            marker=dict(size=24, color=color,
                         line=dict(color="white", width=2)),
            text=[str(i[2]) for i in items], textposition="middle center",
            textfont=dict(color="white", size=11),
            hovertext=[f"{i[3]} (#{i[2]}) — {role}" for i in items],
            hoverinfo="text", name=role,
        ))

    gk = r.get("goalkeeper")
    if gk:
        gp = gk.get("position_at_kick") or gk.get("position_at_setup")
        if gp:
            mapped = _metric_to_pixel(gp[0], gp[1], image_key)
            if mapped is not None:
                fig.add_trace(go.Scatter(
                    x=[mapped[0]], y=[mapped[1]],
                    mode="markers+text",
                    marker=dict(size=26, color="#000000",
                                 line=dict(color="white", width=2)),
                    text=[str(gk.get("jersey", ""))],
                    textposition="middle center",
                    textfont=dict(color="white", size=11),
                    hovertext=[f"GK {gk.get('player_name','')} (#{gk.get('jersey','')})"],
                    hoverinfo="text", name="Goalkeeper",
                ))

    end_x, end_y = r["delivery"].get("end_x"), r["delivery"].get("end_y")
    if end_x is not None and end_y is not None:
        mapped = _metric_to_pixel(end_x, end_y, image_key)
        if mapped is not None:
            fig.add_trace(go.Scatter(
                x=[mapped[0]], y=[mapped[1]], mode="markers",
                marker=dict(size=18, color="#f1c40f", symbol="x",
                             line=dict(color="black", width=2)),
                name="Delivery", hoverinfo="skip",
            ))

    fig.update_layout(
        showlegend=True,
        legend=dict(orientation="h", y=-0.05, bgcolor="rgba(0,0,0,0.05)"),
    )
    st.plotly_chart(fig, use_container_width=True, key="vis2_role_map",
                     config={"displayModeBar": False, "scrollZoom": False,
                             "doubleClick": False})


def _vis3_role_table(rows, match, defending_team):
    if not rows:
        st.caption("No synced corners.")
        return
    r = _resolve_vis34_pick(rows, defending_team)
    if r is None:
        return
    # Prefer the matching Event's clock (correct formatting); fall back to model
    # output for the (rare) case where no event match is found.
    nav_events_for_label = []  # unused here — caption just shows whatever we have
    st.caption(f"Showing: {r.get('match_clock', '??:??')} ({r['corner_side']})")
    import pandas as pd
    table_rows = []
    att_by_jersey = {a["jersey"]: a.get("player_name", "") for a in r["attackers"]}
    for d in r["defenders"]:
        marks = d.get("marks_jersey")
        marks_str = ""
        if d["role"] == "MAN" and marks is not None:
            mk_name = att_by_jersey.get(marks, "")
            marks_str = f"#{marks} {mk_name}".strip()
        table_rows.append({
            "Jersey": d["jersey"],
            "Player": d.get("player_name", "") or f"#{d['jersey']}",
            "Role": d["role"],
            "Marks": marks_str,
        })
    df = pd.DataFrame(table_rows).sort_values(["Role", "Jersey"]).reset_index(drop=True)
    st.dataframe(df, use_container_width=True, hide_index=True)
    _vis_role_feedback(r, defending_team)


def _find_event_for_corner_row(corner_row, events):
    """Find the Event object in `events` whose game_time_ms matches the
    corner row's kick_time_ms (within 250 ms tolerance)."""
    target = corner_row.get("kick_time_ms")
    if target is None:
        return None
    best, best_d = None, 1_000_000
    for ev in events:
        d = abs((ev.game_time_ms or 0) - target)
        if d < best_d:
            best, best_d = ev, d
    return best if best_d <= 250 else None


def _vis_role_feedback(corner_result, defending_team):
    """A compact 'one option' feedback affordance that lets the coach confirm
    or correct the auto-assigned roles for this corner. Writes to
    LABELING_SHEET_PATH."""
    with st.expander("✎ Submit role feedback (helps improve the model)"):
        choice = st.radio(
            "Are all roles correct?",
            ["All roles correct", "There is a mistake"],
            horizontal=True, key=f"fb_choice_{corner_result['corner_id']}",
        )
        analyser, err = _get_corner_analyser()
        if err or analyser is None:
            st.caption(f"Feedback disabled: {err}")
            return
        if choice == "All roles correct":
            if st.button("Confirm and save",
                          key=f"fb_confirm_{corner_result['corner_id']}"):
                try:
                    analyser.confirm_corner(corner_result, LABELING_SHEET_PATH)
                    st.success("Saved to labelling sheet.")
                except Exception as e:
                    st.error(f"Save failed: {e}")
        else:
            st.caption("Edit any defender role below, then click Save.")
            overrides = {}
            for d in corner_result["defenders"]:
                new_role = st.selectbox(
                    f"#{d['jersey']} {d.get('player_name','')}",
                    ["MAN", "ZONAL", "SHORT", "COUNTER"],
                    index=["MAN", "ZONAL", "SHORT", "COUNTER"].index(
                        d["role"] if d["role"] in ("MAN","ZONAL","SHORT","COUNTER") else "ZONAL"),
                    key=f"fb_role_{corner_result['corner_id']}_{d['jersey']}",
                )
                if new_role != d["role"]:
                    overrides[d["jersey"]] = new_role
            if st.button("Save corrections",
                          key=f"fb_save_{corner_result['corner_id']}"):
                try:
                    analyser.submit_role_corrections(
                        corner_result, role_overrides=overrides,
                        labeling_sheet_path=LABELING_SHEET_PATH,
                    )
                    st.success(f"Saved {len(overrides)} correction(s).")
                except Exception as e:
                    st.error(f"Save failed: {e}")


# ----------------------------------------------------------------
# Vis 5: Average role counts split by has_short
# ----------------------------------------------------------------

def _vis5_avg_role_counts(aggregates):
    v5 = aggregates.get("vis5_role_counts", {})
    if not v5:
        st.caption("No data.")
        return
    # Flat (no nested columns — this widget runs inside an outer st.columns call)
    for key, title in [("with_short", "Opponent HAS short player"),
                         ("without_short", "Opponent does NOT have short player")]:
        block = v5.get(key, {})
        st.markdown(f"_{title}_")
        if not block:
            st.caption("—")
            continue
        line = (
            f"<span style='color:#27ae60'>{block.get('zonal',0):.1f} Zonal</span>"
            f" &nbsp;|&nbsp; <span style='color:#e74c3c'>{block.get('man',0):.1f} Man</span>"
            f" &nbsp;|&nbsp; <span style='color:#f39c12'>{block.get('short',0):.1f} Short</span>"
            f" &nbsp;|&nbsp; <span style='color:#9b59b6'>{block.get('counter',0):.1f} Counter</span>"
        )
        st.markdown(f"<div style='font-size: 1.0rem;'>{line}</div>",
                     unsafe_allow_html=True)


# ----------------------------------------------------------------
# Vis 6: Zones usually zonally marked
# ----------------------------------------------------------------

def _vis6_zonal_zones(aggregates):
    v6 = aggregates.get("vis6_zonal_zones", {})
    if not v6:
        st.caption("No data.")
        return
    # Merge PA1-3 + PE1-3 into a single "Edge" bucket (taking the MAX, since
    # the value is a per-corner percentage and a defender in any of those
    # subzones still counts as "covering the edge").
    merged = {}
    edge_keys = {"PA1", "PA2", "PA3", "PE1", "PE2", "PE3"}
    for z, v in v6.items():
        if z in edge_keys:
            merged["Edge"] = max(merged.get("Edge", 0.0), v)
        else:
            merged[z] = v
    fig = go.Figure()
    items = sorted(merged.items(), key=lambda kv: -kv[1])
    fig.add_trace(go.Bar(
        x=[v * 100 for _, v in items],
        y=[z for z, _ in items],
        orientation="h",
        marker=dict(color="#27ae60"),
        text=[f"{v*100:.0f}%" for _, v in items], textposition="outside",
    ))
    fig.update_layout(
        height=max(220, 24 * len(items) + 80),
        margin=dict(l=10, r=40, t=10, b=30),
        xaxis=dict(title="% of corners with a ZONAL defender in this zone",
                    range=[0, 110]),
        yaxis=dict(autorange="reversed"),
        plot_bgcolor="white", showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True, key="vis6_bar",
                     config={"displayModeBar": False})


# ----------------------------------------------------------------
# Vis 7: % corners where attackers exceed markers
# ----------------------------------------------------------------

def _vis7_attackers_exceed(aggregates):
    pct = aggregates.get("vis7_attackers_exceed_markers_pct")
    if pct is None:
        st.caption("No data.")
        return
    st.metric("Corners where attackers in attacking roles outnumber man-markers",
               f"{pct * 100:.0f}%")
    st.caption("Attacking roles = TARGET / DECOY / BLOCK_GK / BLOCK_DEF. "
                "Excludes SECOND_BALL and STATIC.")


# ----------------------------------------------------------------
# Vis 7 (magnet board): full-width planned-attack -> predicted defence
# ----------------------------------------------------------------

# Zone centres in the attacking-right metric frame (mirrors magnet_board)
_MB_ZONE_CENTRES = {
    "GA1": (49.75, +6.10), "GA2": (49.75, 0.0), "GA3": (49.75, -6.10),
    "PA1": (44.25, +6.10), "PA2": (44.25, 0.0), "PA3": (44.25, -6.10),
    "PE1": (38.75, +6.10), "PE2": (38.75, 0.0), "PE3": (38.75, -6.10),
    "PC1": (34.20, +6.10), "PC2": (34.20, 0.0), "PC3": (34.20, -6.10),
}


@_fragment
def _vis7_magnet_board(rows, match, defending_team):
    """Full-width magnet board: user sets up an attacking corner, module
    predicts the opponent's likely defensive setup from historical corners.

    Layout:
      * Pitch (full width, uses no_names_left/right.png — same as Vis 2)
      * Per-attacker editor (player picker from team roster, role, sliders)
      * Textual explanations / open zones / per-player rationale
    """
    try:
        from magnet_board import MagnetBoard, EXAMPLE_PLANNED_ATTACK
    except Exception as e:
        st.warning(f"Magnet board module not available: {e}")
        return
    if not rows:
        st.info("Magnet board needs at least one synced corner.")
        return
    try:
        board = MagnetBoard(rows)
    except Exception as e:
        st.info(f"Magnet board could not be built: {e}")
        return

    st.markdown("### Vis 7 — Magnet Board")
    st.caption(
        f"Plan your attacking corner. The pitch shows your attackers "
        f"(blue) and the opponent ({defending_team})'s most likely "
        f"defensive response (coloured by role)."
    )

    # ---- Available team rosters (for the FCDB-style player picker) ----
    teams_in_match = [t for t in (match.home_team, match.away_team) if t]
    default_attacking_team = next(
        (t for t in teams_in_match if t != defending_team), teams_in_match[0]
    )

    roster_by_team = {}
    for name, info in (match.players or {}).items():
        team_name = info.get("team", "")
        shirt = info.get("shirt", 0)
        if not team_name or not shirt:
            continue
        roster_by_team.setdefault(team_name, []).append((int(shirt), name))
    for t in roster_by_team:
        roster_by_team[t].sort(key=lambda kv: kv[0])

    pa_key = f"mb_planned_attack_{defending_team}"
    if pa_key not in st.session_state:
        st.session_state[pa_key] = {
            "corner_side":     EXAMPLE_PLANNED_ATTACK["corner_side"],
            "delivery_zone":   EXAMPLE_PLANNED_ATTACK["delivery_zone"],
            "attacking_team":  default_attacking_team,
            "attackers":       [dict(a) for a in EXAMPLE_PLANNED_ATTACK["attackers"]],
        }
    pa = st.session_state[pa_key]
    pa.setdefault("attacking_team", default_attacking_team)

    # ---- Top row controls: corner side · delivery zone · team · reset ----
    tc1, tc2, tc3, tc4 = st.columns([1, 1, 1.4, 1])
    with tc1:
        pa["corner_side"] = st.selectbox(
            "Corner side", ["L", "R"],
            index=["L", "R"].index(pa.get("corner_side", "R")),
            key=f"mb_side_{defending_team}",
        )
    with tc2:
        zones = ["—", "GA1", "GA2", "GA3", "PA1", "PA2", "PA3",
                  "PE1", "PE2", "PE3", "FRONT", "EDGE"]
        cur = pa.get("delivery_zone") or "—"
        picked = st.selectbox(
            "Delivery zone", zones,
            index=zones.index(cur) if cur in zones else 0,
            key=f"mb_zone_{defending_team}",
        )
        pa["delivery_zone"] = None if picked == "—" else picked
    with tc3:
        team_options = teams_in_match or [pa["attacking_team"]]
        pa["attacking_team"] = st.selectbox(
            "Attackers from team",
            team_options,
            index=team_options.index(pa["attacking_team"])
                if pa["attacking_team"] in team_options else 0,
            key=f"mb_team_{defending_team}",
            help="Pick the team whose roster fills the attacker dropdowns.",
        )
    with tc4:
        st.write("")  # vertical alignment
        if st.button("Reset attackers", key=f"mb_reset_{defending_team}"):
            st.session_state[pa_key] = {
                "corner_side":    EXAMPLE_PLANNED_ATTACK["corner_side"],
                "delivery_zone":  EXAMPLE_PLANNED_ATTACK["delivery_zone"],
                "attacking_team": default_attacking_team,
                "attackers":      [dict(a) for a in EXAMPLE_PLANNED_ATTACK["attackers"]],
            }
            st.rerun()

    # ---- Pick which attacker to edit / add or remove attackers ----
    roster = roster_by_team.get(pa["attacking_team"], [])
    # Allow planning up to 10 attackers (full outfield minus GK).
    MAX_ATTACKERS = 10
    n_att = len(pa["attackers"])

    nc1, nc2, nc3 = st.columns([1, 1, 2.5])
    with nc1:
        if st.button(f"+ Add attacker (currently {n_att}/{MAX_ATTACKERS})",
                       key=f"mb_add_{defending_team}",
                       disabled=n_att >= MAX_ATTACKERS):
            pa["attackers"].append({
                "jersey": (max((a.get("jersey", 0) for a in pa["attackers"]),
                                default=0) + 1) or n_att + 1,
                "player_name": "",
                "role_intent": "TARGET",
                "start_pos": [40.0, 0.0],
                "end_pos":   [47.0, 0.0],
            })
            st.rerun()
    with nc2:
        if st.button("− Remove last", key=f"mb_remove_{defending_team}",
                       disabled=n_att <= 1):
            pa["attackers"].pop()
            st.rerun()

    n_att = len(pa["attackers"])
    att_idx = st.selectbox(
        "Edit attacker (this one gets a yellow halo on the pitch)",
        range(n_att),
        format_func=lambda i: (
            f"Slot {i + 1}  —  #{pa['attackers'][i].get('jersey','?')} "
            f"({pa['attackers'][i].get('role_intent','?')})"
        ),
        key=f"mb_att_pick_{defending_team}",
    )
    att_idx = min(att_idx, n_att - 1)
    att = pa["attackers"][att_idx]

    # ---- Per-attacker controls (4 columns: player / role / start / end) ----
    pc1, pc2, pc3, pc4 = st.columns([1.5, 1, 1.2, 1.2])
    with pc1:
        if roster:
            options = ["— custom —"] + [f"{s}. {n}" for s, n in roster]
            cur_jersey = att.get("jersey")
            cur_idx = 0
            for i, (s, _) in enumerate(roster, start=1):
                if s == cur_jersey:
                    cur_idx = i
                    break
            picked = st.selectbox(
                f"Player ({pa['attacking_team']})",
                options, index=cur_idx,
                key=f"mb_player_{defending_team}_{att_idx}",
            )
            if picked != "— custom —":
                shirt_picked = int(picked.split(".")[0])
                att["jersey"] = shirt_picked
                att["player_name"] = picked.split(". ", 1)[1] if ". " in picked else ""
            else:
                att["jersey"] = st.number_input(
                    "Custom jersey", min_value=1, max_value=99,
                    value=int(att.get("jersey", att_idx + 1)),
                    key=f"mb_j_{defending_team}_{att_idx}",
                )
                att["player_name"] = ""
        else:
            att["jersey"] = st.number_input(
                "Jersey", min_value=1, max_value=99,
                value=int(att.get("jersey", att_idx + 1)),
                key=f"mb_j_{defending_team}_{att_idx}",
            )
    with pc2:
        role_options = ["TARGET", "DECOY", "STATIC", "SECOND_BALL",
                         "BLOCK_GK", "BLOCK_DEF"]
        att["role_intent"] = st.selectbox(
            "Role intent", role_options,
            index=role_options.index(att.get("role_intent", "TARGET"))
                if att.get("role_intent", "TARGET") in role_options else 0,
            key=f"mb_role_{defending_team}_{att_idx}",
        )
    with pc3:
        st.markdown("_Start position_")
        att["start_pos"] = [
            st.slider("Start x (m to goal-line)", 0.0, 52.5,
                       float(att.get("start_pos", [40.0, 0.0])[0]),
                       step=0.5,
                       key=f"mb_sx_{defending_team}_{att_idx}"),
            st.slider("Start y (left ↔ right)", -34.0, 34.0,
                       float(att.get("start_pos", [40.0, 0.0])[1]),
                       step=0.5,
                       key=f"mb_sy_{defending_team}_{att_idx}"),
        ]
    with pc4:
        st.markdown("_End position (at delivery)_")
        att["end_pos"] = [
            st.slider("End x", 0.0, 52.5,
                       float(att.get("end_pos", [47.0, 0.0])[0]),
                       step=0.5,
                       key=f"mb_ex_{defending_team}_{att_idx}"),
            st.slider("End y", -34.0, 34.0,
                       float(att.get("end_pos", [47.0, 0.0])[1]),
                       step=0.5,
                       key=f"mb_ey_{defending_team}_{att_idx}"),
        ]

    # ---- Predict ----
    try:
        prediction = board.predict_defensive_setup(pa)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return
    shape = prediction["shape"]

    # ---- Shape tile ----
    conf_color = {"high": "#27ae60", "medium": "#f39c12",
                   "low": "#e74c3c"}.get(shape["confidence"], "#7f8c8d")
    st.markdown(
        f"<div style='padding:10px 14px; background:rgba(0,0,0,0.04); "
        f"border-left:5px solid {conf_color}; border-radius:4px; "
        f"margin:8px 0;'>"
        f"<b>Expected setup:</b> {shape['n_zonal']} zonal · "
        f"{shape['n_man']} man · {shape['n_short']} short · "
        f"{shape['n_counter']} counter "
        f"&nbsp;&nbsp;|&nbsp;&nbsp; "
        f"<b>Confidence:</b> {shape['confidence']} "
        f"&nbsp;&nbsp;|&nbsp;&nbsp; "
        f"<b>Based on:</b> {shape['sample_size']} corners"
        f"</div>",
        unsafe_allow_html=True,
    )
    if shape["confidence"] == "low":
        st.caption("⚠️ Limited historical data — prediction is "
                    "indicative only.")

    # ---- Pitch render (same background as Vis 2: clean half-pitch, no zones) ----
    side = pa["corner_side"]
    image_key = "sg_left_nozones" if side == "L" else "sg_right_nozones"
    fig, (iw, ih) = _plotly_pitch_image(image_key, fig_height=540)

    # Open zones — translucent yellow circles (mapped via metric->pixel)
    for z in prediction.get("open_zones", []):
        zname = z["zone"]
        if zname in _MB_ZONE_CENTRES:
            zx, zy = _MB_ZONE_CENTRES[zname]
            mapped = _metric_to_pixel(zx, zy, image_key)
            if mapped is not None:
                fig.add_shape(
                    type="circle",
                    x0=mapped[0] - 35, y0=mapped[1] - 35,
                    x1=mapped[0] + 35, y1=mapped[1] + 35,
                    fillcolor="rgba(241,196,15,0.35)",
                    line=dict(color="rgba(241,196,15,0.9)", width=2, dash="dot"),
                    layer="above",
                )
                fig.add_annotation(
                    x=mapped[0], y=mapped[1] - 45, text=zname, showarrow=False,
                    font=dict(color="black", size=11, family="Arial Black"),
                )

    # MAN lines: defender -> assigned attacker (pixel coords)
    att_jersey_to_endpx = {}
    for a in pa["attackers"]:
        ep = a.get("end_pos")
        if ep:
            mapped = _metric_to_pixel(ep[0], ep[1], image_key)
            if mapped is not None:
                att_jersey_to_endpx[a.get("jersey")] = mapped
    for d in prediction["defenders"]:
        if d["predicted_role"] != "MAN":
            continue
        mks = d.get("marks_jersey")
        tgt = att_jersey_to_endpx.get(mks)
        if not tgt:
            continue
        d_mapped = _metric_to_pixel(d["predicted_position"][0],
                                      d["predicted_position"][1], image_key)
        if d_mapped is None:
            continue
        fig.add_shape(
            type="line",
            x0=d_mapped[0], y0=d_mapped[1], x1=tgt[0], y1=tgt[1],
            line=dict(color="#e74c3c", width=2, dash="dash"),
        )

    # Planned attackers — bright blue, jersey + last name
    ax, ay, atxt, ahov, ahalox, ahaloy = [], [], [], [], [], []
    for i, a in enumerate(pa["attackers"]):
        ep = a.get("end_pos") or a.get("start_pos")
        if not ep:
            continue
        mapped = _metric_to_pixel(ep[0], ep[1], image_key)
        if mapped is None:
            continue
        ax.append(mapped[0]); ay.append(mapped[1])
        last_name = (a.get("player_name", "") or "").strip().split()[-1] \
            if a.get("player_name") else ""
        label = f"{a.get('jersey','')}"
        atxt.append(label)
        ahov.append(
            f"<b>#{a.get('jersey','')} {a.get('player_name','')}</b><br>"
            f"{a.get('role_intent','')}<br>"
            f"end: ({ep[0]:.1f}, {ep[1]:.1f})"
        )
        if i == att_idx:
            ahalox.append(mapped[0]); ahaloy.append(mapped[1])

    # Yellow halo for the currently-selected attacker (renders under the dot)
    if ahalox:
        fig.add_trace(go.Scatter(
            x=ahalox, y=ahaloy, mode="markers",
            marker=dict(size=46, color="rgba(241,196,15,0.55)",
                         line=dict(color="#f1c40f", width=3)),
            hoverinfo="skip", showlegend=False,
        ))
    if ax:
        fig.add_trace(go.Scatter(
            x=ax, y=ay, mode="markers+text",
            marker=dict(size=32, color="#2980b9",
                         line=dict(color="white", width=3)),
            text=atxt, textposition="middle center",
            textfont=dict(color="white", size=13, family="Arial Black"),
            hovertext=ahov, hoverinfo="text",
            name="Planned attackers",
        ))
        # Add last-name labels just below the dots
        for x, y, a in zip(ax, ay, pa["attackers"]):
            ln = (a.get("player_name", "") or "").strip().split()[-1] \
                if a.get("player_name") else ""
            if ln:
                fig.add_annotation(
                    x=x, y=y + 22, text=ln, showarrow=False,
                    font=dict(color="#1f3a93", size=10, family="Arial Black"),
                    bgcolor="rgba(255,255,255,0.7)",
                )

    # Predicted defenders by role
    role_colors = {"MAN": "#e74c3c", "ZONAL": "#27ae60",
                    "SHORT": "#f39c12", "COUNTER": "#9b59b6"}
    role_groups = {}
    for d in prediction["defenders"]:
        role_groups.setdefault(d["predicted_role"], []).append(d)
    for role, defs in role_groups.items():
        color = role_colors.get(role, "#34495e")
        dxs, dys, dtxt, dhov = [], [], [], []
        for d in defs:
            mp = _metric_to_pixel(d["predicted_position"][0],
                                    d["predicted_position"][1], image_key)
            if mp is None:
                continue
            dxs.append(mp[0]); dys.append(mp[1])
            dtxt.append(str(d["jersey"]))
            dhov.append(d.get("explanation", ""))
        if dxs:
            fig.add_trace(go.Scatter(
                x=dxs, y=dys, mode="markers+text",
                marker=dict(size=26, color=color,
                             line=dict(color="white", width=2)),
                text=dtxt, textposition="middle center",
                textfont=dict(color="white", size=12),
                hovertext=dhov, hoverinfo="text", name=role,
            ))

    fig.update_layout(
        showlegend=True,
        legend=dict(orientation="h", y=-0.04, bgcolor="rgba(0,0,0,0.05)"),
    )
    st.plotly_chart(fig, use_container_width=True,
                     key=f"mb_pitch_{defending_team}",
                     config={"displayModeBar": False,
                              "scrollZoom": False, "doubleClick": False})

    # ---- Textual rationale ----
    st.markdown(f"**Shape rationale:** {shape.get('explanation', '')}")
    if prediction.get("open_zones"):
        st.markdown("**Open zones (yellow on pitch):**")
        for z in prediction["open_zones"]:
            st.caption(f"• **{z['zone']}** — {z['explanation']}")
    with st.expander("Why these defenders? (per-player explanation)"):
        for d in prediction["defenders"]:
            line = (f"- **#{d['jersey']} {d.get('player_name','')}** "
                    f"— **{d['predicted_role']}**")
            if d.get("marks_jersey") is not None:
                line += f" (marks #{d['marks_jersey']})"
            st.markdown(line)
            st.caption(d.get("explanation", ""))


# ================================================================
# VIZ: GOAL KICKS
# ================================================================

def _gk_distance(e):
    dx = e.end_x - e.start_x
    dy = e.end_y - e.start_y
    return (dx*dx + dy*dy) ** 0.5


def _gk_is_short(e) -> bool:
    """Short GK = sub_type is CORNER_SHORT-equivalent (short pass) or travel < 32m."""
    return _gk_distance(e) < 32


def _gk_phase(e) -> str:
    return "Build-up" if _gk_is_short(e) else "Long Ball"


def viz_goal_kicks(events, team, match):
    team_events = list(events) if team == BOTH_LABEL else [e for e in events if e.team == team]
    if not team_events:
        st.info(f"No goal kicks by {team}.")
        return

    tabs = st.tabs(["Goal Kicks (Execution)", "GK Sequences (Aftermath)"])

    with tabs[0]:
        _render_gk_execution(team_events, events, team, match)
    with tabs[1]:
        _render_gk_sequences(team_events, events, team, match)


def _render_gk_execution(team_events, nav_events, team, match):
    # ---- Vertical zonal placement pitch ----
    st.markdown("**Zonal Placement (vertical pitch)**")
    kind = st.radio("Show", ["Short", "Long"], horizontal=True, key="gk_zonal_kind")
    st.caption("Arrows: GK → destination zone. Green=successful, red=unsuccessful. Click an arrow endpoint.")
    subset = [e for e in team_events if (_gk_is_short(e) if kind == "Short"
                                          else not _gk_is_short(e))]
    _render_gk_vertical_zones(subset, nav_events, key=f"gk_vert_{kind}")

    st.markdown("---")
    # ---- Phase volume bars ----
    st.markdown("**Goal Kicks by Phase Type**")
    _render_gk_phase_bar(team_events, nav_events, match, key_prefix="gk_phase")

    st.markdown("---")
    # ---- Receiver distribution ----
    st.markdown("**Receiver Distribution**")
    col_s, col_l = st.columns(2)
    with col_s:
        st.markdown("*Short GK receivers*")
        _render_gk_receivers([e for e in team_events if _gk_is_short(e)],
                             nav_events, match, key="gk_recv_short")
    with col_l:
        st.markdown("*Long GK receivers*")
        _render_gk_receivers([e for e in team_events if not _gk_is_short(e)],
                             nav_events, match, key="gk_recv_long")

    # Average positions are rendered in the left column (under the video) so
    # the user does not need to scroll past every other chart to reach them.


def _render_gk_vertical_zones(gks, nav_events, key):
    """Goal-kick destination zones with arrows + counts. Uses the painted
    pixel zones of `full_field_zones_ProgrPasses-CC-More.png` so the zone
    rectangles line up with the white lines exactly. Classification is by
    PIXEL position of the GK end point."""
    if not gks:
        st.caption("No goal kicks in this category.")
        return
    fig, _ = _plotly_pitch_image("full_pp", fig_height=640)
    pixel_zones = _FULL_PP_PIXEL_ZONES

    zone_hits = {name: [] for name, *_ in pixel_zones}
    end_px = []
    for e in gks:
        ep = _metric_to_pixel(e.end_x, e.end_y, "full_pp")
        end_px.append(ep)
        if ep is None:
            continue
        z = _pixel_zone_for(ep[0], ep[1], pixel_zones)
        if z is not None:
            zone_hits[z].append(e)
    max_cnt = max((len(v) for v in zone_hits.values()), default=1) or 1

    zone_centres_px = {}
    for name, px0, py0, px1, py1 in pixel_zones:
        cnt = len(zone_hits.get(name, []))
        alpha = 0.15 + 0.5 * (cnt / max_cnt) if cnt else 0.05
        fig.add_shape(type="rect", x0=px0, y0=py0, x1=px1, y1=py1,
                      fillcolor=f"rgba(255,215,0,{alpha:.2f})",
                      line=dict(color="rgba(0,0,0,0.4)", width=1),
                      layer="above")
        cx, cy = (px0 + px1) / 2, (py0 + py1) / 2
        zone_centres_px[name] = (cx, cy)
        if cnt:
            fig.add_annotation(
                x=cx, y=cy, text=f"<b>{name}</b><br>{cnt}",
                showarrow=False,
                font=dict(color="black", size=12, family="Arial Black"),
                bgcolor="rgba(255,255,255,0.75)",
            )

    xs, ys, cds, colors, hovers = [], [], [], [], []
    for i, e in enumerate(gks):
        s_px = _metric_to_pixel(e.start_x, e.start_y, "full_pp")
        e_px = _metric_to_pixel(e.end_x, e.end_y, "full_pp")
        if s_px is None or e_px is None:
            continue
        sx, sy = s_px; ex, ey = e_px
        color = "#27ae60" if e.result == "SUCCESSFUL" else "#e74c3c"
        fig.add_annotation(
            x=ex, y=ey, ax=sx, ay=sy,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=3, arrowsize=1.2, arrowwidth=2,
            arrowcolor=color, text="", opacity=0.85,
        )
        xs.append(ex); ys.append(ey); cds.append(i); colors.append(color)
        hovers.append(f"{e.game_time_display} - {_pname(e)} \u2192 {_rname(e) or '?'}<br>{e.result}")
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers",
        marker=dict(size=12, color=colors, line=dict(color="white", width=1.5)),
        customdata=cds, hovertext=hovers, hoverinfo="text",
        showlegend=False, name="arrow_ends",
    ))
    if zone_centres_px:
        z_xs = [c[0] for c in zone_centres_px.values()]
        z_ys = [c[1] for c in zone_centres_px.values()]
        z_cds = [f"zone:{name}" for name in zone_centres_px]
        fig.add_trace(go.Scatter(
            x=z_xs, y=z_ys, mode="markers",
            marker=dict(size=80, color="rgba(0,0,0,0)",
                         line=dict(color="rgba(0,0,0,0)", width=0)),
            customdata=z_cds, hoverinfo="skip", showlegend=False,
            name="zone_hits",
        ))
    result = st.plotly_chart(fig, use_container_width=True, key=key,
                             on_select="rerun", selection_mode="points",
                             config={"displayModeBar": False,
                                      "scrollZoom": False, "doubleClick": False})
    if result:
        sel = result.get("selection") if isinstance(result, dict) else None
        if sel and sel.get("points"):
            pt = sel["points"][0]
            cd = pt.get("customdata")
            if isinstance(cd, list):
                cd = cd[0]
            if isinstance(cd, str) and cd.startswith("zone:"):
                zone_name = cd.split(":", 1)[1]
                sig = f"zone_{zone_name}"
                ck = f"__consumed_{key}"
                if st.session_state.get(ck) != sig:
                    st.session_state[ck] = sig
                    group = zone_hits.get(zone_name, [])
                    if group:
                        _apply_filter_from_click(
                            group,
                            filter_key=f"{key}:{zone_name}",
                            label=f"GK -> {zone_name}",
                        )
            elif isinstance(cd, int) and 0 <= cd < len(gks):
                sig = f"arrow_{cd}"
                ck = f"__consumed_{key}"
                if st.session_state.get(ck) != sig:
                    st.session_state[ck] = sig
                    _jump_to_event(gks[cd], nav_events)


def _render_gk_phase_bar(team_events, nav_events, match, key_prefix):
    phases = {"Build-up": [], "Long Ball": []}
    for e in team_events:
        phases[_gk_phase(e)].append(e)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=list(phases.keys()),
        y=[len(v) for v in phases.values()],
        marker=dict(color=["#3498db", "#e67e22"]),
        customdata=list(phases.keys()),
        text=[len(v) for v in phases.values()], textposition="outside",
        hovertemplate="<b>%{x}</b><br>%{y} goal kicks<extra></extra>",
    ))
    fig.update_layout(
        height=260, margin=dict(l=10, r=10, t=10, b=30),
        yaxis=dict(title="Goal Kicks"),
        plot_bgcolor="white", showlegend=False,
    )
    result = st.plotly_chart(fig, use_container_width=True, key=key_prefix,
                             on_select="rerun", selection_mode="points")
    st.caption("Click a bar to cycle through that phase's clips.")
    if result:
        sel = result.get("selection") if isinstance(result, dict) else None
        if sel and sel.get("points"):
            pt = sel["points"][0]
            phase = pt.get("customdata")
            if isinstance(phase, list):
                phase = phase[0]
            if phase and phases[phase]:
                sig = str(phase)
                ck = f"__consumed_{key_prefix}"
                if st.session_state.get(ck) != sig:
                    st.session_state[ck] = sig
                    group = phases[phase]
                    _apply_filter_from_click(
                        group,
                        filter_key=f"{key_prefix}:{phase}",
                        label=f"Phase: {phase}",
                    )


def _render_gk_receivers(gks, nav_events, match, key):
    if not gks:
        st.caption("None.")
        return
    # Include "No Receiver" for GKs without valid receiver
    bucket = {}
    for e in gks:
        r = e.receiver if (e.receiver and e.receiver != "NOT_APPLICABLE") else "No Receiver / Out"
        bucket.setdefault(r, []).append(e)
    ordered = sorted(bucket.items(), key=lambda kv: -len(kv[1]))
    labels = [_match_player_label(match, p) if p != "No Receiver / Out" else p
              for p, _ in ordered]
    counts = [len(v) for _, v in ordered]
    custom = [p for p, _ in ordered]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=counts, y=labels, orientation="h",
        marker=dict(color="#3498db"),
        customdata=custom,
        hovertemplate="<b>%{y}</b><br>%{x} GKs<extra></extra>",
        text=counts, textposition="outside",
    ))
    fig.update_layout(
        height=max(220, 26 * len(labels) + 60),
        margin=dict(l=10, r=30, t=10, b=30),
        xaxis=dict(title="Goal Kicks"),
        yaxis=dict(autorange="reversed"),
        plot_bgcolor="white",
        showlegend=False,
    )
    result = st.plotly_chart(fig, use_container_width=True, key=key,
                             on_select="rerun", selection_mode="points")
    if result:
        sel = result.get("selection") if isinstance(result, dict) else None
        if sel and sel.get("points"):
            pt = sel["points"][0]
            target = pt.get("customdata")
            if isinstance(target, list):
                target = target[0]
            if target:
                sig = str(target)
                ck = f"__consumed_{key}"
                if st.session_state.get(ck) != sig:
                    st.session_state[ck] = sig
                    group = bucket.get(target, [])
                    if group:
                        _apply_filter_from_click(
                            group,
                            filter_key=f"{key}:{target}",
                            label=f"Receiver: {target}",
                        )


def _detect_lineups_for_gks(gks, frames, frames_t, side):
    """Group goal kicks by the 11-player set on the pitch (for the given side)
    detected from tracking. Returns list of (frozenset_pids, [gk_events],
    {pid: shirt}), sorted by number of goal kicks descending."""
    import bisect
    lineup_groups = {}
    lineup_pid_shirt = {}
    for gk in gks:
        t0 = gk.game_time_ms
        t1 = t0 + 3000
        lo = bisect.bisect_left(frames_t, t0)
        hi = bisect.bisect_right(frames_t, t1)
        window = frames[lo:hi]
        if not window:
            continue
        frame_count = {}
        pid_to_shirt = {}
        for f in window:
            for pl in f.get(side, []):
                pid = pl.get("p")
                if pid is None:
                    continue
                frame_count[pid] = frame_count.get(pid, 0) + 1
                pid_to_shirt[pid] = pl.get("s", 0)
        if not frame_count:
            continue
        # Pick the 11 players with the most frame appearances in this window
        top11 = sorted(frame_count, key=lambda p: -frame_count[p])[:11]
        key = frozenset(top11)
        if key not in lineup_groups:
            lineup_groups[key] = []
            lineup_pid_shirt[key] = {pid: pid_to_shirt.get(pid, 0) for pid in top11}
        lineup_groups[key].append(gk)
    return sorted(
        [(k, v, lineup_pid_shirt[k]) for k, v in lineup_groups.items()],
        key=lambda x: -len(x[1]),
    )


def _avg_positions_for_gks_list(gks, frames, frames_t):
    """Compute average (x, y, shirt) per player across 3-second windows after
    each GK event. Returns {'h': {pid: (x, y, shirt)}, 'a': {pid: (...)}}."""
    import bisect
    if not gks:
        return None
    buckets = {"h": {}, "a": {}}
    for e in gks:
        t0 = e.game_time_ms
        t1 = t0 + 3000
        lo = bisect.bisect_left(frames_t, t0)
        hi = bisect.bisect_right(frames_t, t1)
        for f in frames[lo:hi]:
            for side in ("h", "a"):
                for pl in f.get(side, []):
                    pid = pl.get("p")
                    if pid is None:
                        continue
                    d = buckets[side].setdefault(pid, {"xs": [], "ys": [],
                                                        "s": pl.get("s", 0)})
                    d["xs"].append(pl.get("x", 0))
                    d["ys"].append(pl.get("y", 0))
    out = {"h": {}, "a": {}}
    for side in ("h", "a"):
        for pid, d in buckets[side].items():
            if d["xs"]:
                out[side][pid] = (sum(d["xs"]) / len(d["xs"]),
                                   sum(d["ys"]) / len(d["ys"]),
                                   d["s"])
    return out


def _shirt_to_lastname_map(match):
    """{shirt: 'Last name'} reverse lookup from match.players. We split on
    whitespace and take the last token as a compact display label."""
    out = {}
    for name, info in (match.players or {}).items():
        shirt = info.get("shirt", 0)
        if not shirt:
            continue
        last = name.strip().split()[-1] if name.strip() else ""
        out[shirt] = last or name
    return out


@_fragment
def _render_lineup_avg_widget(lineups, match, frames, frames_t, key_prefix):
    """Average-positions pitch on top, then lineup selector + roster below.
    Wrapped in @st.fragment so changing the lineup radio only reruns this
    widget instead of the whole page."""
    if not lineups:
        st.caption("Not enough tracking data to detect lineups.")
        return

    shirt_to_name = _shirt_to_lastname_map(match)

    option_labels = [
        f"Option {i + 1}  —  {len(gks)} goal kick{'s' if len(gks) != 1 else ''}"
        for i, (_, gks, _) in enumerate(lineups)
    ]

    # Read the picker value WITHOUT rendering the widget yet (so we know which
    # lineup to draw the pitch for). The widget itself is rendered below.
    selected_idx = st.session_state.get(f"{key_prefix}_radio", 0)
    if selected_idx >= len(lineups):
        selected_idx = 0
    _, selected_gks, pid_shirt = lineups[selected_idx]

    # 1. Pitch (top)
    avg = _avg_positions_for_gks_list(selected_gks, frames, frames_t)
    _draw_avg_positions_pitch(avg, match, key=f"{key_prefix}_pitch")

    # 2. Options (radio) + roster (below pitch)
    st.radio(
        "Lineup",
        range(len(lineups)),
        format_func=lambda i: option_labels[i],
        key=f"{key_prefix}_radio",
        label_visibility="collapsed",
    )

    sorted_players = sorted(pid_shirt.items(), key=lambda kv: kv[1] or 99)
    roster_lines = []
    for pid, shirt in sorted_players:
        last = shirt_to_name.get(shirt, "")
        label = f"{shirt}. {last}" if last else f"#{shirt}"
        roster_lines.append(f"|&nbsp;&nbsp;{label}")
    st.markdown(
        "<div style='font-family: monospace; font-size: 0.82rem; line-height: 1.45; "
        "padding: 4px 8px; background: rgba(0,0,0,0.03); border-radius: 4px;'>"
        + "<br>".join(roster_lines) + "</div>",
        unsafe_allow_html=True,
    )


def _render_gk_avg_positions(team_events, team, match):
    pos_path = _positions_path_for_match(match)
    if not pos_path:
        st.caption("No tracking data available for this match.")
        return

    with st.spinner("Loading tracking positions..."):
        frames = _cached_positions(str(pos_path))
    if not frames:
        st.caption("Tracking data empty.")
        return

    frames_t = [f["t"] for f in frames]

    # Determine which teams to render (1 team = 2 charts; BOTH = 4 charts)
    if team == BOTH_LABEL:
        teams_to_show = [(match.home_team, "h"), (match.away_team, "a")]
    else:
        side_key = "h" if team == match.home_team else "a"
        teams_to_show = [(team, side_key)]

    for t_name, t_side in teams_to_show:
        t_events = [e for e in team_events if e.team == t_name]
        short_gks = [e for e in t_events if _gk_is_short(e)]
        long_gks = [e for e in t_events if not _gk_is_short(e)]

        if len(teams_to_show) > 1:
            st.markdown(f"#### {t_name}")

        s_lineups = _detect_lineups_for_gks(short_gks, frames, frames_t, t_side)
        l_lineups = _detect_lineups_for_gks(long_gks, frames, frames_t, t_side)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"*Short GK (n={len(short_gks)})*")
            _render_lineup_avg_widget(
                s_lineups, match, frames, frames_t,
                key_prefix=f"gk_short_{t_name.replace(' ', '_')}",
            )
        with col2:
            st.markdown(f"*Long GK (n={len(long_gks)})*")
            _render_lineup_avg_widget(
                l_lineups, match, frames, frames_t,
                key_prefix=f"gk_long_{t_name.replace(' ', '_')}",
            )


def _draw_avg_positions_pitch(avg, match, key):
    if not avg:
        st.caption("No data.")
        return
    fig, _ = _plotly_pitch_image("full_zo", fig_height=560)
    for side, color, label in [("h", "#e74c3c", match.home_team),
                                ("a", "#3498db", match.away_team)]:
        xs, ys, txt = [], [], []
        for pid, (x, y, s) in avg[side].items():
            mapped = _metric_to_pixel(x, y, "full_zo")
            if mapped is None:
                continue
            xs.append(mapped[0]); ys.append(mapped[1]); txt.append(str(s))
        if not xs:
            continue
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="markers+text",
            marker=dict(size=22, color=color, line=dict(color="white", width=1.5)),
            text=txt, textposition="middle center",
            textfont=dict(color="white", size=10),
            name=label, hoverinfo="text",
        ))
    fig.update_layout(showlegend=True,
                      legend=dict(orientation="h", y=-0.02,
                                   bgcolor="rgba(0,0,0,0.3)",
                                   font=dict(color="white")))
    st.plotly_chart(fig, use_container_width=True, key=key)


def _render_gk_sequences(team_events, nav_events, team, match):
    short_gks = [e for e in team_events if _gk_is_short(e)]
    if not short_gks:
        st.info("No short goal kicks to analyze.")
        return

    seq_ids = {e.sequence_id for e in short_gks if e.sequence_id >= 0}
    seq_events = [e for e in match.events
                  if e.sequence_id in seq_ids and e.team == (short_gks[0].team
                                                              if team != BOTH_LABEL else e.team)]
    if team != BOTH_LABEL:
        seq_events = [e for e in seq_events if e.team == team]

    # --- Player involvement chart ---
    st.markdown("**Player involvement in short-GK sequences**")
    pc = Counter(e.player for e in seq_events
                  if e.player and e.player != "NOT_APPLICABLE")
    if not pc:
        st.caption("No player involvement data.")
    else:
        players_sorted = [p for p, _ in pc.most_common()]
        labels = [_match_player_label(match, p) for p in players_sorted]
        counts = [pc[p] for p in players_sorted]
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=counts, y=labels, orientation="h",
            marker=dict(color="#9b59b6"),
            customdata=players_sorted,
            hovertemplate="<b>%{y}</b><br>%{x} events<extra></extra>",
            text=counts, textposition="outside",
        ))
        fig.update_layout(
            height=max(260, 26 * len(labels) + 60),
            margin=dict(l=10, r=30, t=10, b=30),
            xaxis=dict(title="Events in sequence"),
            yaxis=dict(autorange="reversed"),
            plot_bgcolor="white", showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True, key="gk_seq_players")

    # --- Sequence outcome summary ---
    st.markdown("---")
    st.markdown("**Sequence Outcomes**")
    st.caption("How each short-GK sequence ended.")
    outcomes = []
    for gk in short_gks:
        seq = [e for e in match.events if e.sequence_id == gk.sequence_id]
        seq.sort(key=lambda x: x.game_time_ms)
        # Determine outcome
        shot_in = any(e.event_type in ("shot", "shot_on_target", "goal", "big_chance")
                      for e in seq)
        lost = any(e.team != gk.team for e in seq)
        if any(e.event_type == "goal" for e in seq):
            cat = "Goal"
        elif shot_in:
            cat = "Shot"
        elif lost:
            cat = "Lost possession"
        else:
            cat = "Retained / Other"
        outcomes.append((gk, cat))

    cat_counts = Counter(c for _, c in outcomes)
    cat_order = ["Goal", "Shot", "Retained / Other", "Lost possession"]
    cat_colors = {"Goal": "#27ae60", "Shot": "#f39c12",
                  "Retained / Other": "#3498db", "Lost possession": "#e74c3c"}
    fig = go.Figure()
    for cat in cat_order:
        cnt = cat_counts.get(cat, 0)
        if cnt == 0:
            continue
        fig.add_trace(go.Bar(
            x=[cnt], y=["Outcome"], orientation="h", name=cat,
            marker=dict(color=cat_colors[cat]),
            customdata=[cat],
            text=[str(cnt)], textposition="inside",
            textfont=dict(color="white", size=11),
            hovertemplate=f"<b>{cat}</b>: %{{x}}<extra></extra>",
        ))
    fig.update_layout(
        barmode="stack", height=160,
        margin=dict(l=10, r=10, t=10, b=30),
        xaxis=dict(title="Sequences"),
        yaxis=dict(visible=False),
        plot_bgcolor="white",
        legend=dict(orientation="h", y=-0.4),
    )
    result = st.plotly_chart(fig, use_container_width=True, key="gk_seq_outcome",
                             on_select="rerun", selection_mode="points")
    if result:
        sel = result.get("selection") if isinstance(result, dict) else None
        if sel and sel.get("points"):
            pt = sel["points"][0]
            cat = pt.get("customdata")
            if isinstance(cat, list):
                cat = cat[0]
            if cat:
                matching = [gk for gk, c in outcomes if c == cat]
                if matching:
                    sig = str(cat)
                    ck = "__consumed_gk_seq_outcome"
                    if st.session_state.get(ck) != sig:
                        st.session_state[ck] = sig
                        _apply_filter_from_click(
                            matching,
                            filter_key=f"gk_seq_outcome:{cat}",
                            label=f"Sequence outcome: {cat}",
                        )


# ================================================================
# VIZ: FREE KICKS (threat map with Boot/Arrow icons)
# ================================================================

def viz_free_kicks(events, team, match):
    team_events = list(events) if team == BOTH_LABEL else [e for e in events if e.team == team]
    if not team_events:
        st.info(f"No free kicks by {team}.")
        return

    fig, (iw, ih) = _plotly_pitch_image("full_zo", fig_height=620)

    shot_xs, shot_ys, shot_cd, shot_hover = [], [], [], []
    cross_xs, cross_ys, cross_cd, cross_hover = [], [], [], []
    pass_xs, pass_ys, pass_cd, pass_hover = [], [], [], []

    for i, e in enumerate(team_events):
        mp = _metric_to_pixel(e.start_x, e.start_y, "full_zo")
        if mp is None:
            continue
        dx, dy = mp
        hover = f"{e.game_time_display} - {_pname(e)}<br>{e.sub_type} ({e.result})"
        if e.sub_type == "SHOT_FREE_KICK":
            shot_xs.append(dx); shot_ys.append(dy); shot_cd.append(i); shot_hover.append(hover)
        elif "CROSS" in e.sub_type:
            cross_xs.append(dx); cross_ys.append(dy); cross_cd.append(i); cross_hover.append(hover)
        else:
            pass_xs.append(dx); pass_ys.append(dy); pass_cd.append(i); pass_hover.append(hover)

    if shot_xs:
        fig.add_trace(go.Scatter(
            x=shot_xs, y=shot_ys, mode="markers",
            marker=dict(size=22, color="#f39c12", symbol="star",
                        line=dict(color="white", width=2)),
            customdata=shot_cd, hovertext=shot_hover, hoverinfo="text",
            name="FK Shot",
        ))
    if cross_xs:
        fig.add_trace(go.Scatter(
            x=cross_xs, y=cross_ys, mode="markers",
            marker=dict(size=20, color="#3498db", symbol="diamond",
                        line=dict(color="white", width=2)),
            customdata=cross_cd, hovertext=cross_hover, hoverinfo="text",
            name="FK Cross",
        ))
    if pass_xs:
        fig.add_trace(go.Scatter(
            x=pass_xs, y=pass_ys, mode="markers",
            marker=dict(size=14, color="#95a5a6", symbol="circle",
                        line=dict(color="white", width=1)),
            customdata=pass_cd, hovertext=pass_hover, hoverinfo="text",
            name="FK Pass",
        ))
    fig.add_annotation(
        x=iw // 2, y=20,
        text="<b>\u2191 ATTACKING \u2191</b>", showarrow=False,
        font=dict(color="white", size=12, family="Arial Black"),
        bgcolor="rgba(0,0,0,0.4)", xanchor="center",
    )
    fig.update_layout(showlegend=True,
                       legend=dict(bgcolor="rgba(0,0,0,0.3)",
                                    font=dict(color="white")))
    title_tag = "Both" if team == BOTH_LABEL else team
    st.markdown(f"**Free Kick Threat Map ({title_tag})** \u2014 stars=shots, diamonds=crosses")
    st.caption("All free kicks shown as if both teams attack upward.")
    result = st.plotly_chart(fig, use_container_width=True, key="fk_map",
                             on_select="rerun", selection_mode="points")
    idx_map = {i: e for i, e in enumerate(team_events)}
    _handle_plotly_click(result, "fk_map", idx_map, events)

    # Takers
    st.markdown("---")
    st.markdown("**Takers**")
    for player, cnt in Counter(e.player for e in team_events).most_common():
        shirt = match.players.get(player, {}).get("shirt", 0) if match.players else 0
        name_label = f"{shirt}. {player}" if shirt else player
        with st.expander(f"{name_label} ({cnt})"):
            _event_buttons([e for e in team_events if e.player == player],
                            events, f"fk_taker_{player}")

    # Top 5 biggest chances from free kicks (xG for direct shots, xA for shot-creating passes)
    st.markdown("---")
    st.markdown("**Top 5 Biggest Chances From Free Kicks**")
    _render_fk_top_chances(team_events, team, match, events, key_prefix="fk_xga")


# ================================================================
# VIZ: CROSSES (origin heatmap on wings with clickable markers)
# ================================================================

def viz_crosses(events, team, match):
    team_events = list(events) if team == BOTH_LABEL else [e for e in events if e.team == team]
    if not team_events:
        st.info(f"No crosses by {team}.")
        return

    tabs = st.tabs(["Cross Origins", "Left / Right Zonal", "Target Zones", "Chances Created"])

    title_tag = "Both" if team == BOTH_LABEL else team

    with tabs[0]:
        # Cross Origins: use Opp_half_no_zones.png (attacking half, goal at top).
        st.caption("Goal at top \u2b06  \u00b7  green = successful cross  \u00b7  red = unsuccessful")
        fig, (iw, ih) = _plotly_pitch_image("opp_half", fig_height=580)
        succ_x, succ_y, succ_cd, succ_hover = [], [], [], []
        fail_x, fail_y, fail_cd, fail_hover = [], [], [], []
        arrow_pairs = []
        for i, e in enumerate(team_events):
            s_px = _metric_to_pixel(e.start_x, e.start_y, "opp_half")
            e_px = _metric_to_pixel(e.end_x, e.end_y, "opp_half")
            if s_px is None:
                continue
            hover = f"{e.game_time_display} - {_pname(e)}<br>{e.result}"
            if e.result == "SUCCESSFUL":
                succ_x.append(s_px[0]); succ_y.append(s_px[1])
                succ_cd.append(i); succ_hover.append(hover)
            else:
                fail_x.append(s_px[0]); fail_y.append(s_px[1])
                fail_cd.append(i); fail_hover.append(hover)
            if e_px is not None:
                arrow_pairs.append((s_px, e_px, e.result))
        for (sx, sy), (ex, ey), res in arrow_pairs:
            fig.add_shape(type="line", x0=sx, y0=sy, x1=ex, y1=ey,
                          line=dict(color="rgba(255,255,255,0.25)", width=1))
        if succ_x:
            fig.add_trace(go.Scatter(
                x=succ_x, y=succ_y, mode="markers",
                marker=dict(size=14, color="#27ae60", line=dict(color="white", width=2)),
                customdata=succ_cd, hovertext=succ_hover, hoverinfo="text",
                name="Successful",
            ))
        if fail_x:
            fig.add_trace(go.Scatter(
                x=fail_x, y=fail_y, mode="markers",
                marker=dict(size=12, color="#e74c3c", line=dict(color="white", width=1.5)),
                customdata=fail_cd, hovertext=fail_hover, hoverinfo="text",
                name="Unsuccessful",
            ))
        fig.update_layout(showlegend=True,
                           legend=dict(bgcolor="rgba(0,0,0,0.3)",
                                        font=dict(color="white")))
        result = st.plotly_chart(fig, use_container_width=True, key="cross_map",
                                 on_select="rerun", selection_mode="points",
                                 config={"displayModeBar": False, "scrollZoom": False,
                                          "doubleClick": False})
        idx_map = {i: e for i, e in enumerate(team_events)}
        _handle_plotly_click(result, "cross_map", idx_map, events)

    with tabs[1]:
        st.markdown("**Left & Right Side Crosses (zonal)**")
        st.caption("Arrows: origin \u2192 destination. Green = successful, red = unsuccessful. "
                    "Click an arrow endpoint.")
        col_l, col_r = st.columns(2)
        with col_l:
            _render_cross_side(team_events, events, side="L", key="cross_side_L")
        with col_r:
            _render_cross_side(team_events, events, side="R", key="cross_side_R")

    with tabs[2]:
        st.markdown("**Target Zones (per player) \u2014 split by crossing side**")
        col_tl, col_tr = st.columns(2)
        with col_tl:
            st.markdown("*Left-side crosses*")
            _render_cross_target_zones(
                [e for e in team_events if e.start_y > 3], events, team, match,
                key_prefix="cross_targets_L",
            )
        with col_tr:
            st.markdown("*Right-side crosses*")
            _render_cross_target_zones(
                [e for e in team_events if e.start_y < -3], events, team, match,
                key_prefix="cross_targets_R",
            )

    with tabs[3]:
        st.markdown("**Top 5 Biggest Chances From Crosses (xA)**")
        _render_top_chances_from_passes(team_events, team, match, events,
                                         key_prefix="cross_xa", score_fn=_xa_score,
                                         score_label="xA")


def _render_cross_side(team_events, nav_events, side, key):
    """Render one flank (Left y>3 or Right y<-3) on a vertical pitch with
    destination zones inside the box."""
    side_label = "Left Side" if side == "L" else "Right Side"
    flank = [e for e in team_events if (e.start_y > 3 if side == "L" else e.start_y < -3)]
    st.markdown(f"**{side_label} ({len(flank)})**")
    if not flank:
        st.caption("No crosses from this side.")
        return

    fig, _ = _plotly_pitch_image("cross_zones", fig_height=520)

    dest_counts = Counter()
    dest_events = {}
    for e in flank:
        z = _att_zone_of(e.end_x, e.end_y)
        if z:
            dest_counts[z] += 1
            dest_events.setdefault(z, []).append(e)

    zones = _build_att_half_zones()
    max_cnt = max(dest_counts.values()) if dest_counts else 1
    zone_centres_px = {}
    for name, (x0, x1, y0, y1) in zones.items():
        cnt = dest_counts.get(name, 0)
        if cnt == 0:
            continue
        alpha = 0.18 + 0.50 * (cnt / max_cnt)
        p00 = _metric_to_pixel(x0, y0, "cross_zones")
        p11 = _metric_to_pixel(x1, y1, "cross_zones")
        if p00 is None or p11 is None:
            continue
        vx0, vx1 = sorted((p00[0], p11[0]))
        vy0, vy1 = sorted((p00[1], p11[1]))
        fig.add_shape(type="rect", x0=vx0, y0=vy0, x1=vx1, y1=vy1,
                      fillcolor=f"rgba(255,215,0,{alpha:.2f})",
                      line=dict(color="rgba(0,0,0,0.4)", width=1),
                      layer="above")
        cx, cy = (vx0+vx1)/2, (vy0+vy1)/2
        zone_centres_px[name] = (cx, cy)
        fig.add_annotation(x=cx, y=cy, text=f"<b>{cnt}</b>",
                           showarrow=False,
                           font=dict(color="black", size=12, family="Arial Black"),
                           bgcolor="rgba(255,255,255,0.7)")

    origin_bands = {
        "Deep":   (17.5, 30),
        "Mid":    (30,   44),
        "Byline": (44,   52.5),
    }
    y0_f, y1_f = (10, 34) if side == "L" else (-34, -10)
    for band, (x0, x1) in origin_bands.items():
        cnt = sum(1 for e in flank if x0 <= e.start_x <= x1
                  and y0_f <= e.start_y <= y1_f)
        if cnt == 0:
            continue
        p00 = _metric_to_pixel(x0, y0_f, "cross_zones")
        p11 = _metric_to_pixel(x1, y1_f, "cross_zones")
        if p00 is None or p11 is None:
            continue
        vx0, vx1 = sorted((p00[0], p11[0]))
        vy0, vy1 = sorted((p00[1], p11[1]))
        fig.add_shape(type="rect", x0=vx0, y0=vy0, x1=vx1, y1=vy1,
                      fillcolor="rgba(52,152,219,0.18)",
                      line=dict(color="rgba(0,0,0,0.4)", width=1),
                      layer="above")
        fig.add_annotation(x=(vx0+vx1)/2, y=(vy0+vy1)/2,
                            text=f"{band}<br><b>{cnt}</b>",
                            showarrow=False,
                            font=dict(color="black", size=11),
                            bgcolor="rgba(255,255,255,0.7)")

    xs, ys, cds, hovers, colors = [], [], [], [], []
    sxs, sys, s_colors = [], [], []
    for i, e in enumerate(flank):
        color = "#27ae60" if e.result == "SUCCESSFUL" else "#e74c3c"
        s_px = _metric_to_pixel(e.start_x, e.start_y, "cross_zones")
        e_px = _metric_to_pixel(e.end_x, e.end_y, "cross_zones")
        if s_px is None or e_px is None:
            continue
        sx, sy = s_px; ex, ey = e_px
        fig.add_annotation(
            x=ex, y=ey, ax=sx, ay=sy,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=3, arrowsize=1.2, arrowwidth=2,
            arrowcolor=color, text="", opacity=0.8,
        )
        xs.append(ex); ys.append(ey); cds.append(i); colors.append(color)
        hovers.append(f"{e.game_time_display} - {_pname(e)}<br>{e.result}")
        sxs.append(sx); sys.append(sy); s_colors.append(color)

    # Start-of-cross dots (origin) — open circles in the cross outcome colour.
    if sxs:
        fig.add_trace(go.Scatter(
            x=sxs, y=sys, mode="markers",
            marker=dict(size=10, color="rgba(0,0,0,0)",
                         line=dict(color=s_colors, width=2)),
            hoverinfo="skip", showlegend=False, name="cross_starts",
        ))
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers",
        marker=dict(size=12, color=colors, line=dict(color="white", width=1.5)),
        customdata=cds, hovertext=hovers, hoverinfo="text", name="arrow_ends",
        showlegend=False,
    ))
    # Zone hit-zones (transparent) for click-to-filter
    if zone_centres_px:
        z_xs = [c[0] for c in zone_centres_px.values()]
        z_ys = [c[1] for c in zone_centres_px.values()]
        z_cds = [f"zone:{n}" for n in zone_centres_px]
        fig.add_trace(go.Scatter(
            x=z_xs, y=z_ys, mode="markers",
            marker=dict(size=60, color="rgba(0,0,0,0)",
                         line=dict(color="rgba(0,0,0,0)", width=0)),
            customdata=z_cds, hoverinfo="skip", showlegend=False,
            name="zone_hits",
        ))
    result = st.plotly_chart(fig, use_container_width=True, key=key,
                             on_select="rerun", selection_mode="points",
                             config={"displayModeBar": False,
                                      "scrollZoom": False, "doubleClick": False})
    if result:
        sel = result.get("selection") if isinstance(result, dict) else None
        if sel and sel.get("points"):
            pt = sel["points"][0]
            cd = pt.get("customdata")
            if isinstance(cd, list):
                cd = cd[0]
            if isinstance(cd, str) and cd.startswith("zone:"):
                zone_name = cd.split(":", 1)[1]
                sig = f"zone_{zone_name}"
                ck = f"__consumed_{key}"
                if st.session_state.get(ck) != sig:
                    st.session_state[ck] = sig
                    group = dest_events.get(zone_name, [])
                    if group:
                        _apply_filter_from_click(
                            group,
                            filter_key=f"{key}:{zone_name}",
                            label=f"Cross -> {zone_name}",
                        )
            elif isinstance(cd, int) and 0 <= cd < len(flank):
                sig = f"arrow_{cd}"
                ck = f"__consumed_{key}"
                if st.session_state.get(ck) != sig:
                    st.session_state[ck] = sig
                    _jump_to_event(flank[cd], nav_events)


def _render_cross_target_zones(team_events, nav_events, team, match, key_prefix):
    """Horizontal stacked bar: rows = players, stacks = destination zones."""
    if not team_events:
        st.caption("No crosses from this side.")
        return

    def _target_zone(e):
        z = _att_zone_of(e.end_x, e.end_y)
        if z is None:
            return "Outside box"
        rename = {
            "6yd L": "Near post 6yd", "6yd C": "6yd center", "6yd R": "Far post 6yd",
            "Box L": "Near penalty", "Box CL": "Near center",
            "Box CR": "Far center", "Box R": "Far penalty",
            "Edge L": "Edge L", "Edge C": "Edge C", "Edge R": "Edge R",
        }
        return rename.get(z, z)

    # Group per team when Both
    if team == BOTH_LABEL:
        groups = [(match.home_team, [e for e in team_events if e.team == match.home_team]),
                  (match.away_team, [e for e in team_events if e.team == match.away_team])]
    else:
        groups = [(team, team_events)]

    zone_order = ["Near post 6yd", "6yd center", "Far post 6yd",
                  "Near penalty", "Near center", "Far center", "Far penalty",
                  "Edge L", "Edge C", "Edge R", "Outside box"]
    # Consistent zone colour palette (matches order)
    zone_colors = ["#1abc9c", "#27ae60", "#16a085",
                   "#3498db", "#9b59b6", "#e67e22", "#e74c3c",
                   "#f1c40f", "#f39c12", "#d35400", "#95a5a6"]
    zone_color = dict(zip(zone_order, zone_colors))

    for team_name, crosses in groups:
        if not crosses:
            continue
        # (player, zone) -> [events]
        grid = {}
        for e in crosses:
            z = _target_zone(e)
            grid.setdefault((e.player, z), []).append(e)

        # Sort players by total crosses
        players_total = Counter(e.player for e in crosses)
        players_sorted = [p for p, _ in players_total.most_common()]
        player_labels = [_match_player_label(match, p) for p in players_sorted]

        fig = go.Figure()
        for z in zone_order:
            xs = [len(grid.get((p, z), [])) for p in players_sorted]
            if sum(xs) == 0:
                continue
            fig.add_trace(go.Bar(
                y=player_labels, x=xs, orientation="h",
                name=z,
                marker=dict(color=zone_color[z]),
                customdata=[[p, z] for p in players_sorted],
                hovertemplate="<b>%{y}</b><br>" + z + ": %{x}<extra></extra>",
            ))
        fig.update_layout(
            barmode="stack", height=max(220, 26 * len(player_labels) + 100),
            margin=dict(l=10, r=10, t=10, b=30),
            xaxis=dict(title=""),  # blank — legend values run below the axis
            yaxis=dict(autorange="reversed"),
            plot_bgcolor="white",
            legend=dict(orientation="h", y=-0.2, font=dict(size=9)),
        )
        if team == BOTH_LABEL:
            st.markdown(f"*{team_name}*")
        chart_key = f"{key_prefix}_{team_name}"
        result = st.plotly_chart(fig, use_container_width=True, key=chart_key,
                                 on_select="rerun", selection_mode="points")
        if result:
            sel = result.get("selection") if isinstance(result, dict) else None
            if sel and sel.get("points"):
                pt = sel["points"][0]
                cd = pt.get("customdata")
                if isinstance(cd, list) and len(cd) == 2:
                    pname, zone = cd
                    sig = f"{pname}_{zone}"
                    ck = f"__consumed_{chart_key}"
                    if st.session_state.get(ck) != sig:
                        st.session_state[ck] = sig
                        group = grid.get((pname, zone), [])
                        if group:
                            _apply_filter_from_click(
                                group,
                                filter_key=f"{chart_key}:{pname}:{zone}",
                                label=f"{pname} \u2192 {zone}",
                            )


def _xa_score(pass_event, shot):
    """xA = xG of the shot created by this pass/cross."""
    return shot.xg


def _xga_score(pass_event, shot):
    """Combined xG+xA: weighs free-kick shots (own xG) and shot-creating passes equally."""
    return shot.xg


def _find_next_shot_in_sequence(pass_event, match):
    """Return the next SHOT in the same sequence by the same team after this pass/cross."""
    if pass_event.sequence_id < 0:
        return None
    cand = [e for e in match.events
            if e.event_type == "shot"
            and e.sequence_id == pass_event.sequence_id
            and e.team == pass_event.team
            and e.game_time_ms >= pass_event.game_time_ms]
    cand.sort(key=lambda x: x.game_time_ms)
    return cand[0] if cand else None


def _render_fk_top_chances(fk_events, team, match, nav_events, key_prefix):
    """Top 5 free kicks ranked by xG (own shot) or xA (assist to a shot).
    Bars are labelled with the metric: e.g. "7. Gudelj 0.40 xG" or "10. Steijn 0.32 xA"."""
    if team == BOTH_LABEL:
        groups = [(match.home_team, [e for e in fk_events if e.team == match.home_team]),
                  (match.away_team, [e for e in fk_events if e.team == match.away_team])]
    else:
        groups = [(team, [e for e in fk_events if e.team == team])]

    for team_name, evts in groups:
        rows = []
        for e in evts:
            # Direct shot from a free kick
            if e.sub_type == "SHOT_FREE_KICK" and e.xg > 0:
                rows.append((e, e.xg, "xG", None))
                continue
            # Free kick that created a shot (xA = shot's xG)
            shot = _find_next_shot_in_sequence(e, match)
            if shot is not None and shot.xg > 0:
                rows.append((e, shot.xg, "xA", shot))
        rows.sort(key=lambda r: -r[1])
        top = rows[:5]

        st.markdown(f"*{team_name}*")
        if not top:
            st.caption("No shot-creating free kicks.")
            continue

        labels, values, colors, cds = [], [], [], []
        for i, (e, score, metric, shot) in enumerate(top):
            if metric == "xA":
                tgt = _rname(e) or (_pname(shot) if shot else "?")
                label = f"{_pname(e)} → {tgt} ({score:.2f} {metric})"
                colors.append("#3498db")
            else:
                label = f"{_pname(e)} ({score:.2f} {metric})"
                colors.append("#f39c12")
            labels.append(label)
            values.append(score)
            cds.append(i)

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=values, y=labels, orientation="h",
            marker=dict(color=colors),
            customdata=cds,
            hovertemplate="<b>%{y}</b><br>Score: %{x:.3f}<extra></extra>",
        ))
        fig.update_layout(
            height=240, margin=dict(l=10, r=10, t=10, b=30),
            xaxis=dict(title="xG / xA"),
            yaxis=dict(autorange="reversed"),
            plot_bgcolor="white",
            showlegend=False,
        )
        chart_key = f"{key_prefix}_{team_name}"
        result = st.plotly_chart(fig, use_container_width=True, key=chart_key,
                                 on_select="rerun", selection_mode="points")
        idx_map = {i: top[i][0] for i in range(len(top))}
        _handle_plotly_click(result, chart_key, idx_map, nav_events)


def _render_top_chances_from_passes(pass_events, team, match, nav_events,
                                     key_prefix, score_fn, score_label):
    """Render a per-team horizontal bar chart of the top-5 chances created.
    `pass_events` is the set of passes/crosses/free kicks to analyze.
    Each bar is clickable via plotly and jumps to that pass's clip."""
    # Group by team
    if team == BOTH_LABEL:
        groups = [(match.home_team, [e for e in pass_events if e.team == match.home_team]),
                  (match.away_team, [e for e in pass_events if e.team == match.away_team])]
    else:
        groups = [(team, [e for e in pass_events if e.team == team])]

    for team_name, evts in groups:
        rows = []
        for e in evts:
            shot = _find_next_shot_in_sequence(e, match)
            if shot is None or shot.xg <= 0:
                continue
            score = score_fn(e, shot)
            rows.append((e, shot, score))
        rows.sort(key=lambda r: -r[2])
        top = rows[:5]

        st.markdown(f"*{team_name}*")
        if not top:
            st.caption(f"No shot-creating {key_prefix.split('_')[0]}s.")
            continue

        labels = []
        values = []
        cds = []
        for i, (p, shot, score) in enumerate(top):
            passer = _pname(p)
            target = _rname(p) or _pname(shot)
            labels.append(f"{passer} → {target} ({p.game_time_display})")
            values.append(score)
            cds.append(i)

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=values, y=labels, orientation="h",
            marker=dict(color="#e67e22"),
            customdata=cds,
            hovertemplate=f"<b>%{{y}}</b><br>{score_label}: %{{x:.3f}}<extra></extra>",
        ))
        fig.update_layout(
            height=240, margin=dict(l=10, r=10, t=10, b=30),
            xaxis=dict(title=score_label),
            yaxis=dict(autorange="reversed"),
            plot_bgcolor="white",
            showlegend=False,
        )
        chart_key = f"{key_prefix}_{team_name}"
        result = st.plotly_chart(fig, use_container_width=True, key=chart_key,
                                 on_select="rerun", selection_mode="points")
        idx_map = {i: top[i][0] for i in range(len(top))}
        _handle_plotly_click(result, chart_key, idx_map, nav_events)


# ================================================================
# VIZ: GOALS (assist-to-goal vector + shot map)
# ================================================================

def viz_goals(events, team, match):
    # Split events into two groups. In Both mode, split by home vs away teams.
    if team == BOTH_LABEL:
        group_a = [e for e in events if e.team == match.home_team]
        group_b = [e for e in events if e.team == match.away_team]
        label_a, label_b = match.home_team, match.away_team
        title = f"Goals Map ({match.home_team} vs {match.away_team})"
    else:
        group_a = [e for e in events if e.team == team]
        group_b = [e for e in events if e.team != team]
        label_a, label_b = team, "Opponent"
        title = f"Goals Map ({team} vs Opponent)"

    st.markdown(f"**{title}**")

    fig = _plotly_pitch_vertical(fig_height=540)

    def find_assist(goal):
        cand = [x for x in match.events
                if x.sequence_id == goal.sequence_id
                and x.team == goal.team
                and x.game_time_ms < goal.game_time_ms
                and x.event_type in ("cross", "free_kick", "goal_kick", "corner", "key_pass")]
        cand.sort(key=lambda x: x.game_time_ms)
        return cand[-1] if cand else None

    trace_groups = []

    def add_goal_trace(evts, color, name):
        xs, ys, cds, hovers = [], [], [], []
        for i, e in enumerate(evts):
            dx, dy = _v(e.start_x, e.start_y)
            xs.append(dx); ys.append(dy); cds.append(i)
            hovers.append(f"{e.game_time_display} - {_pname(e)}<br>xG: {e.xg:.2f}")
            assist = find_assist(e)
            if assist:
                ax_, ay_ = _v(assist.start_x, assist.start_y)
                fig.add_annotation(
                    x=dx, y=dy, ax=ax_, ay=ay_,
                    xref="x", yref="y", axref="x", ayref="y",
                    arrowhead=2, arrowsize=1.5, arrowwidth=2.5,
                    arrowcolor=color, showarrow=True, text="",
                )
        if xs:
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="markers",
                marker=dict(size=22, color=color, symbol="star",
                            line=dict(color="white", width=2)),
                customdata=cds, hovertext=hovers, hoverinfo="text", name=name,
            ))
            trace_groups.append(evts)

    add_goal_trace(group_a, "#27ae60", label_a)
    add_goal_trace(group_b, "#f1c40f" if team == BOTH_LABEL else "#95a5a6", label_b)
    fig.update_layout(showlegend=True,
                      legend=dict(bgcolor="rgba(0,0,0,0.3)", font=dict(color="white")))

    result = st.plotly_chart(fig, use_container_width=True, key="goals_map",
                             on_select="rerun", selection_mode="points")
    if result:
        sel = result.get("selection") if isinstance(result, dict) else None
        if sel:
            pts = sel.get("points", [])
            if pts:
                pt = pts[0]
                trace_idx = pt.get("curve_number", 0)
                pidx = pt.get("point_index", 0)
                click_sig = f"{trace_idx}_{pidx}"
                if st.session_state.get("__consumed_goals_map") != click_sig:
                    st.session_state["__consumed_goals_map"] = click_sig
                    group = trace_groups[trace_idx] if trace_idx < len(trace_groups) else []
                    if pidx < len(group):
                        _jump_to_event(group[pidx], events)

    # List view
    st.markdown("---")
    st.markdown("**Goal Clips**")
    for e in group_a + group_b:
        assist = find_assist(e)
        label = f"{e.game_time_display} - {e.team} - {_pname(e)} (xG: {e.xg:.2f})"
        if assist:
            label += f"  [assist: {_pname(assist)}]"
        _jump_button(label, e, events, key=f"goal_{e.game_time_ms}_{e.team}")


# ================================================================
# VIZ: CLASSIC SHOT MAP (with half filter, clickable)
# ================================================================

@_fragment
@_fragment
def viz_shots_unified(events, team, match):
    """Unified Shots view: replaces the old Goals / Shots-on-Target / Shots-Total
    event types. Tabs:
      1. Shot locations — shots on the cross_zones half-pitch with zone counters
      2. Outcomes & Phases (stacked bar)
      3. Per Player (vertical bar)

    A top-level radio filters to All / On Target / Goals — these reuse the
    same `events` list (which already contains every shot)."""
    if not events:
        st.info("No shots in this match.")
        return

    filt_choice = st.radio(
        "Filter",
        ["All shots", "On target", "Goals only"],
        horizontal=True, key="shots_unified_filter",
    )

    def _shot_filter(es):
        if filt_choice == "Goals only":
            return [e for e in es if e.result == "SUCCESSFUL"]
        if filt_choice == "On target":
            return [e for e in es
                     if getattr(e, "shot_type", "") == "ON_TARGET"
                     or e.result == "SUCCESSFUL"]
        return es

    if team == BOTH_LABEL:
        group_a = _shot_filter([e for e in events if e.team == match.home_team])
        group_b = _shot_filter([e for e in events if e.team == match.away_team])
        label_a, label_b = match.home_team, match.away_team
    else:
        group_a = _shot_filter([e for e in events if e.team == team])
        group_b = _shot_filter([e for e in events if e.team != team])
        label_a, label_b = team, "Opponent"

    half_filter = st.radio("Half", ["All", "1st Half", "2nd Half"],
                             horizontal=True, key="half_shots_unified")

    def _half(es):
        if half_filter == "1st Half":
            return [e for e in es if e.game_time_ms < 45 * 60 * 1000]
        if half_filter == "2nd Half":
            return [e for e in es if e.game_time_ms >= 45 * 60 * 1000]
        return es

    a_f = _half(group_a); b_f = _half(group_b)
    all_shown = a_f + b_f

    tab_loc, tab_phase, tab_player = st.tabs(
        ["Shot Locations", "Outcomes & Phases", "Per Player"]
    )

    # ---- Tab 1: shot locations on cross_zones image ----
    with tab_loc:
        st.markdown(f"**Shot locations — {label_a} ({len(a_f)}) vs {label_b} ({len(b_f)})**")
        st.caption("Dot = shot origin (size = xG). Green = Goal. Click a shot to play the clip.")
        fig, _ = _plotly_pitch_image("cross_zones", fig_height=620)
        _overlay_zone_counters(fig, a_f + b_f, "cross_zones",
                                _CZ_PIXEL_ZONES, point_attr="start")
        trace_groups2 = []
        for evts, name, color_team in [(b_f, label_b, "#e67e22"),
                                          (a_f, label_a, "#3498db")]:
            xs, ys, sizes, cds, hovers, fill_colors, line_colors = \
                [], [], [], [], [], [], []
            for i, e in enumerate(evts):
                mp = _metric_to_pixel(e.start_x, e.start_y, "cross_zones")
                if mp is None:
                    continue
                xs.append(mp[0]); ys.append(mp[1])
                sizes.append(max(12, e.xg * 90))
                cds.append(i)
                outcome = _shot_outcome(e)
                # Goals get green; all other shots keep the team colour.
                if outcome == "Goal":
                    fill_colors.append("#27ae60")
                    line_colors.append("#ffffff")
                else:
                    fill_colors.append(color_team)
                    line_colors.append("rgba(255,255,255,0.7)")
                hovers.append(
                    f"{e.game_time_display} - {_pname(e)}<br>"
                    f"{outcome} · xG {e.xg:.2f}"
                )
            if xs:
                fig.add_trace(go.Scatter(
                    x=xs, y=ys, mode="markers",
                    marker=dict(size=sizes, color=fill_colors, symbol="circle",
                                 line=dict(color=line_colors, width=2), opacity=0.9),
                    customdata=cds, hovertext=hovers, hoverinfo="text",
                    name=name,
                ))
                trace_groups2.append(evts)
        result = st.plotly_chart(fig, use_container_width=True,
                                   key="shots_loc_map",
                                   on_select="rerun", selection_mode="points",
                                   config={"displayModeBar": False,
                                            "scrollZoom": False,
                                            "doubleClick": False})
        if result:
            sel = result.get("selection") if isinstance(result, dict) else None
            if sel and sel.get("points"):
                pt = sel["points"][0]
                trace_idx = pt.get("curve_number", 0)
                pidx = pt.get("point_index", 0)
                sig = f"loc_{trace_idx}_{pidx}"
                ck = "__consumed_shots_loc_map"
                if st.session_state.get(ck) != sig:
                    st.session_state[ck] = sig
                    group = trace_groups2[trace_idx] if trace_idx < len(trace_groups2) else []
                    if pidx < len(group):
                        _jump_to_event(group[pidx], events)
        cols = st.columns(4)
        cols[0].metric(f"{label_a} Shots", len(a_f))
        cols[1].metric(f"{label_a} xG", f"{sum(e.xg for e in a_f):.2f}")
        cols[2].metric(f"{label_b} Shots", len(b_f))
        cols[3].metric(f"{label_b} xG", f"{sum(e.xg for e in b_f):.2f}")

    with tab_phase:
        st.markdown("**Shot Outcomes & Phases**")
        _render_shot_phase_bar(all_shown, events,
                                key_prefix="shot_phase_unified", match=match)

    with tab_player:
        st.markdown("**Shots Per Player**")
        _render_shots_per_player(all_shown, events, team, match,
                                  key_prefix="shots_per_player_unified")


def viz_shots(events, team, match, title="Shots"):
    if team == BOTH_LABEL:
        group_a = [e for e in events if e.team == match.home_team]
        group_b = [e for e in events if e.team == match.away_team]
        label_a, label_b = match.home_team, match.away_team
    else:
        group_a = [e for e in events if e.team == team]
        group_b = [e for e in events if e.team != team]
        label_a, label_b = team, "Opponent"

    half_filter = st.radio("Half", ["All", "1st Half", "2nd Half"],
                           horizontal=True, key=f"half_{title}")

    def filt(es):
        if half_filter == "1st Half":
            return [e for e in es if e.game_time_ms < 45 * 60 * 1000]
        if half_filter == "2nd Half":
            return [e for e in es if e.game_time_ms >= 45 * 60 * 1000]
        return es

    a_f = filt(group_a)
    b_f = filt(group_b)
    all_shown = a_f + b_f

    tab_map, tab_phase, tab_player = st.tabs(
        ["Shot Map", "Outcomes & Phases", "Per Player"]
    )

    with tab_map:
        st.markdown(f"**{title} Map - {label_a} ({len(a_f)}) vs {label_b} ({len(b_f)})**")
        st.caption("Click a shot to watch the clip. Dots are colored by phase.")

        # Attacking-half pitch (horizontal, static, fills the full frame)
        fig = _plotly_pitch_attacking_half(fig_height=520)

        # Zone counts across both teams
        zone_counts = Counter()
        for e in all_shown:
            z = _att_zone_of(e.start_x, e.start_y)
            if z:
                zone_counts[z] += 1
        _draw_att_zone_counts(fig, zone_counts)

        trace_groups = []

        def add_trace(evts, name):
            xs, ys, sizes, cds, hovers, colors, symbols, line_colors = (
                [], [], [], [], [], [], [], []
            )
            for i, e in enumerate(evts):
                x, y = _normalize_pos(e.start_x, e.start_y)
                xs.append(x); ys.append(y)
                sizes.append(max(12, e.xg * 90))
                cds.append(i)
                phase = _shot_phase(e, match)
                outcome = _shot_outcome(e)
                colors.append(SHOT_PHASE_COLORS.get(phase, "#7f8c8d"))
                symbols.append("star" if outcome == "Goal" else "circle")
                line_colors.append("#ffffff" if outcome == "Goal" else "rgba(255,255,255,0.7)")
                hovers.append(
                    f"{e.game_time_display} - {_pname(e)}<br>"
                    f"{phase} · {outcome} · xG {e.xg:.2f}"
                )
            if xs:
                fig.add_trace(go.Scatter(
                    x=xs, y=ys, mode="markers",
                    marker=dict(size=sizes, color=colors, symbol=symbols,
                                line=dict(color=line_colors, width=2),
                                opacity=0.92),
                    customdata=cds, hovertext=hovers, hoverinfo="text", name=name,
                ))
                trace_groups.append(evts)

        add_trace(b_f, label_b)
        add_trace(a_f, label_a)

        for phase, col in SHOT_PHASE_COLORS.items():
            fig.add_trace(go.Scatter(
                x=[None], y=[None], mode="markers",
                marker=dict(size=12, color=col, line=dict(color="white", width=1.5)),
                name=phase, showlegend=True,
            ))

        fig.update_layout(showlegend=True,
                          legend=dict(bgcolor="rgba(0,0,0,0.4)",
                                      font=dict(color="white", size=11),
                                      orientation="h", y=-0.02))

        result = st.plotly_chart(fig, use_container_width=True,
                                   key=f"shot_map_{title}",
                                   on_select="rerun", selection_mode="points",
                                   config={"displayModeBar": False,
                                           "scrollZoom": False,
                                           "doubleClick": False,
                                           "staticPlot": False})
        if result:
            sel = result.get("selection") if isinstance(result, dict) else None
            if sel:
                pts = sel.get("points", [])
                if pts:
                    pt = pts[0]
                    trace_idx = pt.get("curve_number", 0)
                    pidx = pt.get("point_index", 0)
                    sig = f"{title}_{trace_idx}_{pidx}"
                    ck = f"__consumed_shot_map_{title}"
                    if st.session_state.get(ck) != sig:
                        st.session_state[ck] = sig
                        group = trace_groups[trace_idx] if trace_idx < len(trace_groups) else []
                        if pidx < len(group):
                            _jump_to_event(group[pidx], events)

        cols = st.columns(4)
        cols[0].metric(f"{label_a} Shots", len(a_f))
        cols[1].metric(f"{label_a} xG", f"{sum(e.xg for e in a_f):.2f}")
        cols[2].metric(f"{label_b} Shots", len(b_f))
        cols[3].metric(f"{label_b} xG", f"{sum(e.xg for e in b_f):.2f}")

    with tab_phase:
        st.markdown("**Shot Outcomes & Phases**")
        _render_shot_phase_bar(all_shown, events,
                                key_prefix=f"shot_phase_{title}", match=match)

    with tab_player:
        st.markdown("**Shots Per Player**")
        _render_shots_per_player(all_shown, events, team, match,
                                  key_prefix=f"shots_per_player_{title}")


def _render_shot_phase_bar(shots, nav_events, key_prefix, match):
    """Horizontal stacked bar: rows = phase groups, stacks = outcome colors."""
    if not shots:
        st.caption("No shots to chart.")
        return

    phase_order = ["In Play", "Set Piece Phase", "Direct Set Piece"]
    outcome_order = ["Goal", "Save", "Blocked", "Hit Woodwork", "Miss"]

    # Map "Open Play" phase -> "In Play" row label for display
    def _row_for(phase):
        return "In Play" if phase == "Open Play" else phase

    # Build counts: (row, outcome) -> list[shot]
    grid = {(r, o): [] for r in phase_order for o in outcome_order}
    for s in shots:
        row = _row_for(_shot_phase(s, match))
        out = _shot_outcome(s)
        grid[(row, out)].append(s)

    fig = go.Figure()
    for outcome in outcome_order:
        xs = [len(grid[(r, outcome)]) for r in phase_order]
        # customdata = row label so click handler can find the shots
        fig.add_trace(go.Bar(
            y=phase_order, x=xs, orientation="h", name=outcome,
            marker=dict(color=SHOT_OUTCOME_COLORS[outcome]),
            customdata=[[r, outcome] for r in phase_order],
            hovertemplate="<b>%{y} · " + outcome + "</b><br>%{x} shots<extra></extra>",
            text=[str(v) if v else "" for v in xs], textposition="inside",
            textfont=dict(color="white", size=11),
        ))

    fig.update_layout(
        barmode="stack", height=240, margin=dict(l=10, r=10, t=10, b=30),
        xaxis=dict(title="Shots"),
        yaxis=dict(autorange="reversed"),
        plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(orientation="h", y=-0.3),
    )
    result = st.plotly_chart(fig, use_container_width=True, key=key_prefix,
                             on_select="rerun", selection_mode="points")
    if result:
        sel = result.get("selection") if isinstance(result, dict) else None
        if sel and sel.get("points"):
            pt = sel["points"][0]
            cd = pt.get("customdata")
            if isinstance(cd, list) and len(cd) == 2:
                row, outcome = cd
                sig = f"{row}_{outcome}"
                ck = f"__consumed_{key_prefix}"
                if st.session_state.get(ck) != sig:
                    st.session_state[ck] = sig
                    group = grid.get((row, outcome), [])
                    if group:
                        _apply_filter_from_click(
                            group,
                            filter_key=f"{key_prefix}:{row}:{outcome}",
                            label=f"{row} / {outcome}",
                        )


def _render_shots_per_player(shots, nav_events, team, match, key_prefix):
    """Vertical bar: one bar per player, click to cycle through their shots."""
    if not shots:
        st.caption("No shots.")
        return

    # If Both: split into home/away groups; else one group.
    if team == BOTH_LABEL:
        groups = [(match.home_team, [s for s in shots if s.team == match.home_team]),
                  (match.away_team, [s for s in shots if s.team == match.away_team])]
    else:
        groups = [(team, [s for s in shots if s.team == team])]

    for team_name, team_shots in groups:
        if not team_shots:
            continue
        pc = Counter(s.player for s in team_shots)
        players_sorted = [p for p, _ in pc.most_common()]
        labels = [_match_player_label(match, p) for p in players_sorted]
        counts = [pc[p] for p in players_sorted]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=labels, y=counts,
            marker=dict(color="#3498db"),
            customdata=players_sorted,
            hovertemplate="<b>%{x}</b><br>%{y} shots<extra></extra>",
            text=counts, textposition="outside",
        ))
        fig.update_layout(
            height=max(260, 30 * len(labels) + 80),
            margin=dict(l=10, r=10, t=10, b=80),
            xaxis=dict(title="", tickangle=-40),
            yaxis=dict(title="Shots"),
            plot_bgcolor="white",
            showlegend=False,
        )
        st.markdown(f"*{team_name}*")
        chart_key = f"{key_prefix}_{team_name}"
        result = st.plotly_chart(fig, use_container_width=True, key=chart_key,
                                 on_select="rerun", selection_mode="points")
        if result:
            sel = result.get("selection") if isinstance(result, dict) else None
            if sel and sel.get("points"):
                pt = sel["points"][0]
                player_name = pt.get("customdata")
                if isinstance(player_name, list):
                    player_name = player_name[0]
                if player_name:
                    sig = str(player_name)
                    ck = f"__consumed_{chart_key}"
                    if st.session_state.get(ck) != sig:
                        st.session_state[ck] = sig
                        pshots = [s for s in team_shots if s.player == player_name]
                        if pshots:
                            _apply_filter_from_click(
                                pshots,
                                filter_key=f"{chart_key}:{player_name}",
                                label=f"Shots: {player_name}",
                            )


# ================================================================
# VIZ: BIG CHANCES (match momentum timeline)
# ================================================================

@_fragment
def viz_big_chances(events, team, match):
    if team == BOTH_LABEL:
        team_events = [e for e in events if e.team == match.home_team]
        opp_events = [e for e in events if e.team == match.away_team]
        label_a, label_b = match.home_team, match.away_team
    else:
        team_events = [e for e in events if e.team == team]
        opp_events = [e for e in events if e.team != team]
        label_a, label_b = team, "Opponent"

    all_shots = [e for e in match.events if e.event_type == "shot"]
    if team == BOTH_LABEL:
        shots_a = [s for s in all_shots if s.team == match.home_team]
        shots_b = [s for s in all_shots if s.team == match.away_team]
    else:
        shots_a = [s for s in all_shots if s.team == team]
        shots_b = [s for s in all_shots if s.team != team]

    tab_xg, tab_timeline, tab_players = st.tabs(
        ["Cumulative xG", "Big Chances Timeline", "Player involvement"]
    )

    def _xg_steps(shots):
        shots = sorted(shots, key=lambda s: s.game_time_ms)
        xs, ys, hovers, cds = [0], [0.0], ["0' - 0.00 xG"], [-1]
        total = 0.0
        for i, s in enumerate(shots):
            minute = s.game_time_ms / 60000
            xs.append(minute); ys.append(total); hovers.append(""); cds.append(-1)
            total += s.xg
            xs.append(minute); ys.append(total)
            hovers.append(f"{s.game_time_display} - {_pname(s)} (+{s.xg:.2f} xG → {total:.2f})")
            cds.append(i)
        xs.append(95); ys.append(total); hovers.append(""); cds.append(-1)
        return xs, ys, hovers, cds, shots

    xs_a, ys_a, hov_a, cd_a, shots_a = _xg_steps(shots_a)
    xs_b, ys_b, hov_b, cd_b, shots_b = _xg_steps(shots_b)

    with tab_xg:
        st.markdown(f"**Cumulative xG ({label_a} vs {label_b})**")
        xg_fig = go.Figure()
        xg_fig.add_trace(go.Scatter(
            x=xs_a, y=ys_a, mode="lines+markers", name=f"{label_a} ({ys_a[-1]:.2f})",
            line=dict(color="#e74c3c", width=3, shape="hv"),
            marker=dict(size=[10 if c >= 0 else 0 for c in cd_a], color="#e74c3c"),
            customdata=cd_a, hovertext=hov_a, hoverinfo="text",
        ))
        xg_fig.add_trace(go.Scatter(
            x=xs_b, y=ys_b, mode="lines+markers",
            name=f"{label_b} ({ys_b[-1]:.2f})",
            line=dict(color="#f1c40f" if team == BOTH_LABEL else "#7f8c8d",
                       width=3, shape="hv"),
            marker=dict(size=[10 if c >= 0 else 0 for c in cd_b],
                        color="#f1c40f" if team == BOTH_LABEL else "#7f8c8d"),
            customdata=cd_b, hovertext=hov_b, hoverinfo="text",
        ))
        xg_fig.add_shape(type="line", x0=45, x1=45, y0=0,
                         y1=max(ys_a[-1], ys_b[-1]) + 0.3,
                         line=dict(color="#7f8c8d", width=1, dash="dash"))
        xg_fig.update_layout(
            xaxis=dict(range=[0, 95], title="Minute",
                       tickmode="array", tickvals=[0, 15, 30, 45, 60, 75, 90]),
            yaxis=dict(title="Cumulative xG"),
            height=300,
            margin=dict(l=40, r=10, t=10, b=40),
            plot_bgcolor="white", showlegend=True,
        )
        xg_result = st.plotly_chart(xg_fig, use_container_width=True,
                                      key="xg_cumulative",
                                      on_select="rerun",
                                      selection_mode="points")
        if xg_result:
            sel = xg_result.get("selection") if isinstance(xg_result, dict) else None
            if sel and sel.get("points"):
                pt = sel["points"][0]
                cd = pt.get("customdata")
                if isinstance(cd, list):
                    cd = cd[0]
                trace_idx = pt.get("curve_number", 0)
                sig = f"xg_{trace_idx}_{cd}"
                if cd is not None and cd >= 0 and st.session_state.get("__consumed_xg_cum") != sig:
                    st.session_state["__consumed_xg_cum"] = sig
                    src = shots_a if trace_idx == 0 else shots_b
                    if cd < len(src):
                        _jump_to_event(src[cd], events)

    with tab_timeline:
        st.markdown(f"**Big Chances Timeline ({label_a} vs {label_b})**")
        fig = go.Figure()
        fig.add_shape(type="line", x0=0, y0=0, x1=95, y1=0,
                      line=dict(color="#2c3e50", width=3))
        fig.add_shape(type="line", x0=45, y0=-1, x1=45, y1=1,
                      line=dict(color="#7f8c8d", width=2, dash="dash"))
        fig.add_annotation(x=45, y=1.3, text="HT", showarrow=False,
                           font=dict(color="#7f8c8d", size=10))
        trace_groups = []

        def add_chances(evts, side_y, color, name):
            xs, ys, cds, hovers, sizes = [], [], [], [], []
            for i, e in enumerate(evts):
                xs.append(e.game_time_ms / 60000)
                ys.append(side_y)
                cds.append(i)
                sizes.append(max(14, e.xg * 80))
                hovers.append(f"{e.game_time_display} - {_pname(e)}<br>xG: {e.xg:.3f}")
            if xs:
                fig.add_trace(go.Scatter(
                    x=xs, y=ys, mode="markers",
                    marker=dict(size=sizes, color=color, symbol="star",
                                line=dict(color="white", width=2)),
                    customdata=cds, hovertext=hovers, hoverinfo="text",
                    name=name,
                ))
                trace_groups.append(evts)

        add_chances(team_events, 0.5, "#e74c3c", label_a)
        add_chances(opp_events, -0.5,
                     "#f1c40f" if team == BOTH_LABEL else "#95a5a6", label_b)

        fig.update_layout(
            xaxis=dict(range=[0, 95], title="Minute",
                       tickmode="array", tickvals=[0, 15, 30, 45, 60, 75, 90]),
            yaxis=dict(range=[-2, 2], visible=False),
            height=260,
            margin=dict(l=30, r=10, t=20, b=40),
            showlegend=True, plot_bgcolor="white",
        )
        result = st.plotly_chart(fig, use_container_width=True,
                                   key="bc_timeline",
                                   on_select="rerun", selection_mode="points")
        if result:
            sel = result.get("selection") if isinstance(result, dict) else None
            if sel and sel.get("points"):
                pt = sel["points"][0]
                trace_idx = pt.get("curve_number", 0)
                pidx = pt.get("point_index", 0)
                sig = f"bc_{trace_idx}_{pidx}"
                if st.session_state.get("__consumed_bc_timeline") != sig:
                    st.session_state["__consumed_bc_timeline"] = sig
                    group = trace_groups[trace_idx] if trace_idx < len(trace_groups) else []
                    if pidx < len(group):
                        _jump_to_event(group[pidx], events)

    with tab_players:
        # Players involved in big chances — one match only, so we show per-match totals.
        st.markdown("**Players involved in big chances**")
        all_bc = team_events + opp_events
        if not all_bc:
            st.caption("No big chances.")
        else:
            # Count per player (any team)
            pc_bc = Counter(e.player for e in all_bc)
            ps_bc = [p for p, _ in pc_bc.most_common()]
            ls_bc = [_match_player_label(match, p) for p in ps_bc]
            vs_bc = [pc_bc[p] for p in ps_bc]
            fig_bc = go.Figure()
            fig_bc.add_trace(go.Bar(
                x=vs_bc, y=ls_bc, orientation="h",
                marker=dict(color="#e74c3c"),
                customdata=ps_bc,
                text=vs_bc, textposition="outside",
                hovertemplate="<b>%{y}</b><br>%{x} big chance(s)<extra></extra>",
            ))
            fig_bc.update_layout(
                height=max(240, 28 * len(ls_bc) + 60),
                xaxis=dict(title="Big chances"),
                yaxis=dict(autorange="reversed"),
                margin=dict(l=10, r=40, t=10, b=30),
                plot_bgcolor="white", showlegend=False,
            )
            bc_pr = st.plotly_chart(fig_bc, use_container_width=True,
                                      key="bc_player_bar",
                                      on_select="rerun", selection_mode="points")
            if bc_pr:
                sel = bc_pr.get("selection") if isinstance(bc_pr, dict) else None
                if sel and sel.get("points"):
                    pt = sel["points"][0]
                    pn_bc = pt.get("customdata")
                    if isinstance(pn_bc, list): pn_bc = pn_bc[0]
                    if pn_bc:
                        sig = str(pn_bc)
                        ck = "__consumed_bc_player_bar"
                        if st.session_state.get(ck) != sig:
                            st.session_state[ck] = sig
                            plist_bc = [e for e in all_bc if e.player == pn_bc]
                            if plist_bc:
                                _apply_filter_from_click(
                                    plist_bc,
                                    filter_key=f"bc_player:{pn_bc}",
                                    label=f"Big chances: {pn_bc}",
                                )


# ================================================================
# VIZ: BALL RECOVERIES (pitch thirds bar chart)
# ================================================================

@_fragment
def viz_recoveries(events, team, match):
    team_events = list(events) if team == BOTH_LABEL else [e for e in events if e.team == team]
    if not team_events:
        st.info(f"No ball recoveries by {team}.")
        return

    def third(e):
        x, _ = _normalize_pos(e.start_x, e.start_y)
        if x < -17.5:
            return "Defensive Third"
        if x < 17.5:
            return "Middle Third"
        return "Attacking Third"

    classified = [(e, third(e)) for e in team_events]
    counts = Counter(c[1] for c in classified)
    order = ["Defensive Third", "Middle Third", "Attacking Third"]
    teams_in = sorted({e.team for e in team_events})
    team_color_map = {t: c for t, c in zip(teams_in, ["#16a085", "#e67e22"])}

    tab_map, tab_player, tab_clips = st.tabs(
        ["Map & Thirds", "Per Player", "Clips by Third"]
    )

    with tab_map:
        fig = go.Figure()
        colors = {"Defensive Third": "#e74c3c",
                  "Middle Third": "#f39c12",
                  "Attacking Third": "#27ae60"}
        for zone in order:
            fig.add_trace(go.Bar(
                x=[zone], y=[counts.get(zone, 0)],
                marker_color=colors[zone],
                customdata=[zone],
                hovertemplate=f"<b>{zone}</b><br>%{{y}} recoveries<extra></extra>",
                showlegend=False,
            ))
        fig.update_layout(height=320, xaxis=dict(title=""),
                           yaxis=dict(title="Recoveries"),
                           margin=dict(l=40, r=10, t=30, b=40),
                           plot_bgcolor="white")
        st.markdown(f"**Recoveries by Pitch Third ({team})**")
        st.plotly_chart(fig, use_container_width=True, key="rec_thirds")

        # Clean pitch background (Full_field_zo_zones.png — no baked-in zone labels)
        pfig, (iw, ih) = _plotly_pitch_image("full_zo", fig_height=560)
        pfig.add_annotation(
            x=iw // 2, y=ih - 12,
            text="⬆ Attacking direction ⬆", showarrow=False,
            font=dict(color="white", size=11, family="Arial Black"),
            bgcolor="rgba(0,0,0,0.45)", xanchor="center",
        )
        # Draw the 14 pixel zones with counts and zone labels
        zone_counts_px = {}
        for zone_name, *_ in _FULL_ZO_PIXEL_ZONES:
            zone_counts_px[zone_name] = 0
        for e in team_events:
            mp = _metric_to_pixel(e.start_x, e.start_y, "full_zo")
            if mp is None:
                continue
            zn = _pixel_zone_for(mp[0], mp[1], _FULL_ZO_PIXEL_ZONES)
            if zn:
                zone_counts_px[zn] = zone_counts_px.get(zn, 0) + 1
        max_cnt2 = max(zone_counts_px.values()) if zone_counts_px else 1
        for zone_name, px0, py0, px1, py1 in _FULL_ZO_PIXEL_ZONES:
            cnt2 = zone_counts_px.get(zone_name, 0)
            alpha2 = 0.12 + 0.50 * (cnt2 / max_cnt2) if max_cnt2 > 0 else 0.08
            pfig.add_shape(type="rect", x0=px0, y0=py0, x1=px1, y1=py1,
                            fillcolor=f"rgba(255,255,255,{alpha2:.2f})",
                            line=dict(color="rgba(255,255,255,0.5)", width=1),
                            layer="above")
            cx, cy = (px0 + px1) / 2, (py0 + py1) / 2
            pfig.add_annotation(
                x=cx, y=cy,
                text=f"<b>{zone_name}</b><br>{cnt2}",
                showarrow=False,
                font=dict(color="black", size=10, family="Arial Black"),
                bgcolor="rgba(255,255,255,0.75)",
            )

        xs, ys = [], []
        for e in team_events:
            mp = _metric_to_pixel(e.start_x, e.start_y, "full_zo")
            if mp is None:
                continue
            xs.append(mp[0]); ys.append(mp[1])
        cds = list(range(len(team_events)))
        hovers = [f"{e.game_time_display} - {e.team} - {_pname(e)}" for e in team_events]
        dot_colors = [team_color_map.get(e.team, "#16a085") for e in team_events]
        pfig.add_trace(go.Scatter(
            x=xs, y=ys, mode="markers",
            marker=dict(size=12, color=dot_colors,
                         line=dict(color="white", width=1.5)),
            customdata=cds, hovertext=hovers, hoverinfo="text", name="Recovery",
        ))
        result = st.plotly_chart(pfig, use_container_width=True,
                                   key="rec_pitch", on_select="rerun",
                                   selection_mode="points",
                                   config={"displayModeBar": False,
                                            "scrollZoom": False,
                                            "doubleClick": False})
        idx_map = {i: e for i, e in enumerate(team_events)}
        _handle_plotly_click(result, "rec_pitch", idx_map, events)

    with tab_player:
        st.markdown("**Recoveries Per Player**")
        if team == BOTH_LABEL:
            groups = [(match.home_team, [e for e in team_events if e.team == match.home_team]),
                      (match.away_team, [e for e in team_events if e.team == match.away_team])]
        else:
            groups = [(team, team_events)]
        for team_name, evts in groups:
            st.markdown(f"*{team_name}*")
            pc = Counter(e.player for e in evts)
            if not pc:
                st.caption("No recoveries.")
                continue
            players_sorted = [p for p, _ in pc.most_common()]
            labels = [_match_player_label(match, p) for p in players_sorted]
            bar_counts = [pc[p] for p in players_sorted]
            bar_fig = go.Figure()
            bar_fig.add_trace(go.Bar(
                x=bar_counts, y=labels, orientation="h",
                marker=dict(color=team_color_map.get(team_name, "#16a085")),
                hovertemplate="<b>%{y}</b>: %{x} recoveries<extra></extra>",
            ))
            bar_fig.update_layout(
                height=max(180, 28 * len(labels) + 60),
                xaxis=dict(title="Recoveries"),
                yaxis=dict(autorange="reversed"),
                margin=dict(l=10, r=10, t=10, b=30),
                plot_bgcolor="white", showlegend=False,
            )
            st.plotly_chart(bar_fig, use_container_width=True,
                              key=f"rec_player_bar_{team_name}")

    with tab_clips:
        for zone in order:
            zone_events = [c[0] for c in classified if c[1] == zone]
            with st.expander(f"{zone} ({len(zone_events)})"):
                for e in zone_events:
                    _jump_button(f"{e.game_time_display} - {e.team} - {_pname(e)}",
                                  e, events,
                                  key=f"rec_{zone}_{e.game_time_ms}_{e.player}")


# ================================================================
# VIZ: INTERCEPTIONS (action dots by player with dropdown)
# ================================================================

@_fragment
def viz_interceptions(events, team, match):
    team_events = list(events) if team == BOTH_LABEL else [e for e in events if e.team == team]
    if not team_events:
        st.info(f"No interceptions by {team}.")
        return

    players = sorted(set(e.player for e in team_events),
                     key=lambda p: -sum(1 for e in team_events if e.player == p))
    player_counts = Counter(e.player for e in team_events)
    team_color_map = {t: c for t, c in zip(sorted({e.team for e in team_events}),
                                             ["#2980b9", "#e67e22"])}

    tab_map, tab_player, tab_clips = st.tabs(
        ["Pitch Map", "Per Player", "Clip List"]
    )

    with tab_map:
        # Map display label -> player_name so we can filter by canonical name
        display_to_player = {}
        options = ["All Players"]
        for p in players:
            lbl = f"{_match_player_label(match, p)} ({player_counts[p]})"
            options.append(lbl)
            display_to_player[lbl] = p
        selected = st.selectbox("Player filter", options, key="int_player")
        if selected == "All Players":
            filt_events = team_events
        else:
            player_name = display_to_player.get(selected, "")
            filt_events = [e for e in team_events if e.player == player_name]

        st.markdown(f"**Interception Locations ({len(filt_events)})**")
        pfig, _ = _plotly_pitch_image("full_zo", fig_height=560)
        # Zone counters: tally interceptions by start position
        _overlay_zone_counters(pfig, filt_events, "full_zo", _FULL_ZO_PIXEL_ZONES,
                                point_attr="start")
        xs, ys = [], []
        for e in filt_events:
            mp = _metric_to_pixel(e.start_x, e.start_y, "full_zo")
            if mp is None:
                continue
            xs.append(mp[0]); ys.append(mp[1])
        cds = list(range(len(filt_events)))
        hovers = [f"{e.game_time_display} - {e.team} - {_pname(e)}" for e in filt_events]
        dot_colors = [team_color_map.get(e.team, "#2980b9") for e in filt_events]
        pfig.add_trace(go.Scatter(
            x=xs, y=ys, mode="markers",
            marker=dict(size=14, color=dot_colors,
                         line=dict(color="white", width=1.5)),
            customdata=cds, hovertext=hovers, hoverinfo="text",
            name="Interception",
        ))
        result = st.plotly_chart(pfig, use_container_width=True,
                                   key="int_pitch", on_select="rerun",
                                   selection_mode="points")
        idx_map = {i: e for i, e in enumerate(filt_events)}
        _handle_plotly_click(result, "int_pitch", idx_map, events)

    with tab_player:
        st.markdown("**Interceptions Per Player**")
        if team == BOTH_LABEL:
            groups = [(match.home_team, [e for e in team_events if e.team == match.home_team]),
                      (match.away_team, [e for e in team_events if e.team == match.away_team])]
        else:
            groups = [(team, team_events)]
        for team_name, evts in groups:
            st.markdown(f"*{team_name}*")
            pc = Counter(e.player for e in evts)
            if not pc:
                st.caption("None.")
                continue
            players_sorted = [p for p, _ in pc.most_common()]
            labels = [_match_player_label(match, p) for p in players_sorted]
            bar_counts = [pc[p] for p in players_sorted]
            bar_fig = go.Figure()
            bar_fig.add_trace(go.Bar(
                x=bar_counts, y=labels, orientation="h",
                marker=dict(color=team_color_map.get(team_name, "#2980b9")),
                hovertemplate="<b>%{y}</b>: %{x} interceptions<extra></extra>",
            ))
            bar_fig.update_layout(
                height=max(180, 28 * len(labels) + 60),
                xaxis=dict(title="Interceptions"),
                yaxis=dict(autorange="reversed"),
                margin=dict(l=10, r=10, t=10, b=30),
                plot_bgcolor="white", showlegend=False,
            )
            st.plotly_chart(bar_fig, use_container_width=True,
                              key=f"int_player_bar_{team_name}")

    with tab_clips:
        st.markdown("**Clips**")
        for e in team_events:
            _jump_button(f"{e.game_time_display} - {e.team} - {_pname(e)}",
                          e, events,
                          key=f"int_clip_{e.game_time_ms}_{e.player}")


# ================================================================
# VIZ: KEY PASSES (pitch with clickable arrows showing pass trajectory)
# ================================================================

def viz_key_passes(events, team, match):
    if team == BOTH_LABEL:
        team_events = list(events)
        title_suffix = f"{match.home_team} vs {match.away_team}"
    else:
        team_events = [e for e in events if e.team == team]
        title_suffix = team

    if not team_events:
        st.info(f"No key passes for {title_suffix}.")
        return

    kp_tab_map, kp_tab_bar = st.tabs(["Key Pass Map", "Most key passes"])

    team_colors = {}
    if team == BOTH_LABEL:
        team_colors[match.home_team] = "#3498db"
        team_colors[match.away_team] = "#e67e22"
    else:
        team_colors[team] = "#3498db"
    with kp_tab_map:
        st.caption("Click an arrow endpoint to watch the clip.  Goal at top.")
        fig, _ = _plotly_pitch_image("full_zo", fig_height=600)
        xs, ys, cds, hovers, colors = [], [], [], [], []
        for i, e in enumerate(team_events):
            s_px = _metric_to_pixel(e.start_x, e.start_y, "full_zo")
            e_px = _metric_to_pixel(e.end_x, e.end_y, "full_zo")
            if s_px is None or e_px is None:
                continue
            sx, sy = s_px; ex, ey = e_px
            color = team_colors.get(e.team, "#3498db")
            fig.add_annotation(
                x=ex, y=ey, ax=sx, ay=sy,
                xref="x", yref="y", axref="x", ayref="y",
                arrowhead=3, arrowsize=1.3, arrowwidth=2.2,
                arrowcolor=color, showarrow=True, text="", opacity=0.85,
            )
            xs.append(ex); ys.append(ey); cds.append(i); colors.append(color)
            hovers.append(
                f"{e.game_time_display} - {e.team}<br>{_pname(e)} "
                f"→ {_rname(e) or '?'}<br>{e.sub_type} ({e.result})"
            )
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="markers",
            marker=dict(size=14, color=colors, line=dict(color="white", width=2)),
            customdata=cds, hovertext=hovers, hoverinfo="text", name="Key Pass",
        ))
        result = st.plotly_chart(fig, use_container_width=True, key="kp_map",
                                 on_select="rerun", selection_mode="points",
                                 config={"displayModeBar": False, "scrollZoom": False,
                                          "doubleClick": False})
        idx_map = {i: e for i, e in enumerate(team_events)}
        _handle_plotly_click(result, "kp_map", idx_map, events)

    with kp_tab_bar:
        st.markdown("**Most key passes** — click a bar to filter clips to that player")
        passer_counts = Counter(e.player for e in team_events)
        ps_kp = [p for p, _ in passer_counts.most_common()]
        ls_kp = [_match_player_label(match, p) for p in ps_kp]
        vs_kp = [passer_counts[p] for p in ps_kp]
        bf_kp = go.Figure()
        bf_kp.add_trace(go.Bar(
            x=vs_kp, y=ls_kp, orientation="h",
            marker=dict(color="#3498db"),
            customdata=ps_kp,
            text=vs_kp, textposition="outside",
            hovertemplate="<b>%{y}</b><br>%{x} key passes<extra></extra>",
        ))
        bf_kp.update_layout(
            height=max(260, 28 * len(ls_kp) + 60),
            xaxis=dict(title="Key passes"),
            yaxis=dict(autorange="reversed"),
            margin=dict(l=10, r=40, t=10, b=30),
            plot_bgcolor="white", showlegend=False,
        )
        kp_br = st.plotly_chart(bf_kp, use_container_width=True, key="kp_player_bar",
                                  on_select="rerun", selection_mode="points")
        if kp_br:
            sel = kp_br.get("selection") if isinstance(kp_br, dict) else None
            if sel and sel.get("points"):
                pt = sel["points"][0]
                pn = pt.get("customdata")
                if isinstance(pn, list): pn = pn[0]
                if pn:
                    sig = str(pn)
                    ck = "__consumed_kp_bar"
                    if st.session_state.get(ck) != sig:
                        st.session_state[ck] = sig
                        plist = [e for e in team_events if e.player == pn]
                        if plist:
                            _apply_filter_from_click(
                                plist, filter_key=f"kp_player:{pn}",
                                label=f"Key passes by {pn}",
                            )

# ================================================================
# VIZ: FINAL 3rd (passes into, receptions, penalty-box entries)
# ================================================================

def _in_penalty_box(x, y):
    """True if (x, y) is inside the attacking-half penalty box."""
    return x >= 36 and abs(y) <= 20.16


def viz_final_third(events, team, match):
    all_passes = [e for e in match.events if e.event_type == "pass"]
    all_carries = [e for e in match.events if e.event_type == "carry"]
    all_crosses = [e for e in match.events if e.event_type in ("cross", "corner", "free_kick")]

    def _team_filter(lst):
        if team == BOTH_LABEL:
            return lst
        return [e for e in lst if e.team == team]

    tabs = st.tabs(["Passes Into F3", "Receptions in F3", "Penalty-Box Entries"])

    with tabs[0]:
        passes_into = [e for e in _team_filter(all_passes)
                       if e.end_third == 3 and e.start_third != 3]
        st.markdown("**Passes into Final 3rd (stacked: successful vs unsuccessful)**")
        _render_stacked_player_bar(
            passes_into, events, team, match,
            key_prefix="f3_passes",
            success_pred=lambda e: e.result == "SUCCESSFUL",
        )

    with tabs[1]:
        st.markdown("**Receptions in Final 3rd (successful only)**")
        receptions = [e for e in _team_filter(all_passes)
                      if e.result == "SUCCESSFUL" and e.end_third == 3
                      and e.receiver and e.receiver != "NOT_APPLICABLE"]
        _render_reception_bar(receptions, events, team, match, key_prefix="f3_recv")

    with tabs[2]:
        st.markdown("**Penalty Box Entries (passes + carries)**")
        st.caption("Vectors show each entry. Click an arrow endpoint or a player bar to play.")
        entries = []
        for e in (_team_filter(all_passes) + _team_filter(all_carries)
                   + _team_filter(all_crosses)):
            if _in_penalty_box(e.end_x, e.end_y) and not _in_penalty_box(e.start_x, e.start_y):
                entries.append(e)
        _render_box_entries(entries, events, team, match, key_prefix="f3_box")


def _render_stacked_player_bar(passes, nav_events, team, match,
                                key_prefix, success_pred):
    """Horizontal stacked bar per player: green (pass succeeded), red (failed).
    Clicking green plays successes, red plays failures — cycles through."""
    if not passes:
        st.caption("No passes to show.")
        return

    if team == BOTH_LABEL:
        groups = [(match.home_team, [e for e in passes if e.team == match.home_team]),
                  (match.away_team, [e for e in passes if e.team == match.away_team])]
    else:
        groups = [(team, passes)]

    for team_name, evts in groups:
        if not evts:
            continue
        # Tally per player
        players = {}
        for e in evts:
            d = players.setdefault(e.player, {"succ": [], "fail": []})
            d["succ" if success_pred(e) else "fail"].append(e)
        ordered = sorted(players.items(),
                          key=lambda kv: -(len(kv[1]["succ"]) + len(kv[1]["fail"])))
        labels = [_match_player_label(match, p) for p, _ in ordered]
        succ_counts = [len(d["succ"]) for _, d in ordered]
        fail_counts = [len(d["fail"]) for _, d in ordered]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=labels, x=succ_counts, orientation="h", name="Successful",
            marker=dict(color="#27ae60"),
            customdata=[[p, "succ"] for p, _ in ordered],
            hovertemplate="<b>%{y}</b><br>%{x} successful<extra></extra>",
        ))
        fig.add_trace(go.Bar(
            y=labels, x=fail_counts, orientation="h", name="Unsuccessful",
            marker=dict(color="#e74c3c"),
            customdata=[[p, "fail"] for p, _ in ordered],
            hovertemplate="<b>%{y}</b><br>%{x} unsuccessful<extra></extra>",
        ))
        fig.update_layout(
            barmode="stack", height=max(240, 28 * len(labels) + 60),
            margin=dict(l=10, r=10, t=10, b=40),
            xaxis=dict(title="Passes"),
            yaxis=dict(autorange="reversed"),
            plot_bgcolor="white",
            legend=dict(orientation="h", y=-0.12),
        )
        st.markdown(f"*{team_name}*")
        chart_key = f"{key_prefix}_{team_name}"
        result = st.plotly_chart(fig, use_container_width=True, key=chart_key,
                                 on_select="rerun", selection_mode="points")
        if result:
            sel = result.get("selection") if isinstance(result, dict) else None
            if sel and sel.get("points"):
                pt = sel["points"][0]
                cd = pt.get("customdata")
                if isinstance(cd, list) and len(cd) == 2:
                    pname, kind = cd
                    sig = f"{pname}_{kind}"
                    ck = f"__consumed_{chart_key}"
                    if st.session_state.get(ck) != sig:
                        st.session_state[ck] = sig
                        target = players[pname][kind]
                        if target:
                            _apply_filter_from_click(
                                target,
                                filter_key=f"{chart_key}:{pname}:{kind}",
                                label=f"{pname} / {kind}",
                            )


def _render_reception_bar(receptions, nav_events, team, match, key_prefix):
    if not receptions:
        st.caption("No receptions.")
        return

    if team == BOTH_LABEL:
        groups = [(match.home_team, [e for e in receptions if e.team == match.home_team]),
                  (match.away_team, [e for e in receptions if e.team == match.away_team])]
    else:
        groups = [(team, receptions)]

    for team_name, evts in groups:
        if not evts:
            continue
        pc = Counter(e.receiver for e in evts)
        players_sorted = [p for p, _ in pc.most_common()]
        labels = [_match_player_label(match, p) for p in players_sorted]
        counts = [pc[p] for p in players_sorted]
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=labels, y=counts,
            marker=dict(color="#1abc9c"),
            customdata=players_sorted,
            hovertemplate="<b>%{x}</b><br>%{y} receptions<extra></extra>",
            text=counts, textposition="outside",
        ))
        fig.update_layout(
            height=max(260, 30 * len(labels) + 80),
            margin=dict(l=10, r=10, t=10, b=80),
            xaxis=dict(title="", tickangle=-40),
            yaxis=dict(title="Receptions"),
            plot_bgcolor="white",
            showlegend=False,
        )
        st.markdown(f"*{team_name}*")
        chart_key = f"{key_prefix}_{team_name}"
        result = st.plotly_chart(fig, use_container_width=True, key=chart_key,
                                 on_select="rerun", selection_mode="points")
        if result:
            sel = result.get("selection") if isinstance(result, dict) else None
            if sel and sel.get("points"):
                pt = sel["points"][0]
                pname = pt.get("customdata")
                if isinstance(pname, list):
                    pname = pname[0]
                if pname:
                    sig = str(pname)
                    ck = f"__consumed_{chart_key}"
                    if st.session_state.get(ck) != sig:
                        st.session_state[ck] = sig
                        recs = [e for e in evts if e.receiver == pname]
                        if recs:
                            _apply_filter_from_click(
                                recs,
                                filter_key=f"{chart_key}:{pname}",
                                label=f"Receptions: {pname}",
                            )


def _render_box_entries(entries, nav_events, team, match, key_prefix):
    if not entries:
        st.caption("No penalty box entries.")
        return

    col_map, col_bar = st.columns([1.2, 1])

    with col_map:
        fig, _ = _plotly_pitch_image("pb_entries", fig_height=560)
        # Zone counters: tally entries by their END point (where they landed)
        _overlay_zone_counters(fig, entries, "pb_entries", _PB_PIXEL_ZONES,
                                point_attr="end")
        xs, ys, cds, hovers, colors = [], [], [], [], []
        for i, e in enumerate(entries):
            color = "#27ae60" if e.result == "SUCCESSFUL" else "#e74c3c"
            kind = "Carry" if e.event_type == "carry" else "Pass"
            s_px = _metric_to_pixel(e.start_x, e.start_y, "pb_entries")
            e_px = _metric_to_pixel(e.end_x, e.end_y, "pb_entries")
            if s_px is None or e_px is None:
                continue
            sx, sy = s_px; ex, ey = e_px
            fig.add_annotation(
                x=ex, y=ey, ax=sx, ay=sy,
                xref="x", yref="y", axref="x", ayref="y",
                showarrow=True, arrowhead=3, arrowsize=1.2, arrowwidth=2,
                arrowcolor=color, text="", opacity=0.85,
            )
            xs.append(ex); ys.append(ey); cds.append(i); colors.append(color)
            hovers.append(f"{e.game_time_display} - {_pname(e)} ({kind})")
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="markers",
            marker=dict(size=12, color=colors, line=dict(color="white", width=1.5)),
            customdata=cds, hovertext=hovers, hoverinfo="text",
        ))
        result = st.plotly_chart(fig, use_container_width=True,
                                  key=f"{key_prefix}_map",
                                  on_select="rerun", selection_mode="points",
                                  config={"displayModeBar": False,
                                           "scrollZoom": False,
                                           "doubleClick": False})
        idx_map = {i: e for i, e in enumerate(entries)}
        _handle_plotly_click(result, f"{key_prefix}_map", idx_map, nav_events)

    with col_bar:
        pc = Counter(e.player for e in entries)
        players_sorted = [p for p, _ in pc.most_common()]
        labels = [_match_player_label(match, p) for p in players_sorted]
        counts = [pc[p] for p in players_sorted]
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=counts, y=labels, orientation="h",
            marker=dict(color="#e67e22"),
            customdata=players_sorted,
            hovertemplate="<b>%{y}</b><br>%{x} entries<extra></extra>",
            text=counts, textposition="outside",
        ))
        fig.update_layout(
            height=max(260, 28 * len(labels) + 60),
            margin=dict(l=10, r=40, t=10, b=30),
            xaxis=dict(title="Entries"),
            yaxis=dict(autorange="reversed"),
            plot_bgcolor="white",
            showlegend=False,
        )
        chart_key = f"{key_prefix}_bar"
        result = st.plotly_chart(fig, use_container_width=True, key=chart_key,
                                 on_select="rerun", selection_mode="points")
        if result:
            sel = result.get("selection") if isinstance(result, dict) else None
            if sel and sel.get("points"):
                pt = sel["points"][0]
                pname = pt.get("customdata")
                if isinstance(pname, list):
                    pname = pname[0]
                if pname:
                    sig = str(pname)
                    ck = f"__consumed_{chart_key}"
                    if st.session_state.get(ck) != sig:
                        st.session_state[ck] = sig
                        plist = [e for e in entries if e.player == pname]
                        if plist:
                            _apply_filter_from_click(
                                plist,
                                filter_key=f"{chart_key}:{pname}",
                                label=f"Box entries: {pname}",
                            )


# ================================================================
# VIZ: BUILD-UP (mid-3rd zonal, def→att, progressive passes)
# ================================================================

def viz_build_up(events, team, match):
    all_passes = [e for e in match.events if e.event_type == "pass"]

    def _tf(lst):
        if team == BOTH_LABEL:
            return lst
        return [e for e in lst if e.team == team]

    tabs = st.tabs(["Progression into Attacking \u2153",
                     "Progressive Passes",
                     "Own \u2153 \u2192 Middle \u2153"])

    with tabs[0]:
        st.markdown("**Passes Landing in the Attacking Third (by zone)**")
        st.caption("Zones match the reference image \u2014 all on the opponent half. "
                    "Green = successful, red = unsuccessful.")
        into_att = [e for e in _tf(all_passes) if e.end_third == 3]
        _render_pass_zone_map_vertical(into_att, events, key="build_into_att",
                                        dest_zone_fn=_att_zone_of,
                                        zone_rects=_build_att_half_zones())

    with tabs[1]:
        st.markdown("**Progressive Passes** (forward progress \u2265 10 m)")
        prog = [e for e in _tf(all_passes) if e.goal_progression >= 10]
        _render_progressive_passes(prog, events, team, match, key_prefix="build_prog")

    with tabs[2]:
        st.markdown("**Own Third \u2192 Middle Third**")
        st.caption("All passes that originate in the defensive third and finish "
                    "in the middle third.")
        into_mid = [e for e in _tf(all_passes)
                    if e.start_third == 1 and e.end_third == 2]
        _render_progression_arrows_vertical(into_mid, events, key="build_own2mid")


def _mid_third_zones():
    """Return middle-third zone rects: 3 columns × 4 rows."""
    # Middle third spans roughly x in (-17.5, 17.5), y in (-34, 34)
    zones = {}
    x_bands = [(-17.5, -5.83), (-5.83, 5.83), (5.83, 17.5)]
    y_bands = [(20.16, 34), (0, 20.16), (-20.16, 0), (-34, -20.16)]
    xlabels = ["Def", "Mid", "Att"]
    ylabels = ["Wing L", "Half L", "Half R", "Wing R"]
    for yi, (y0, y1) in enumerate(y_bands):
        for xi, (x0, x1) in enumerate(x_bands):
            zones[f"{ylabels[yi]}-{xlabels[xi]}"] = (x0, x1, y0, y1)
    return zones


def _mid_third_zone_of(x, y):
    for name, (x0, x1, y0, y1) in _mid_third_zones().items():
        if x0 <= x <= x1 and y0 <= y <= y1:
            return name
    return None


def _render_pass_zone_map(passes, nav_events, key, dest_zone_fn, zone_rects, zoom):
    """Draw zones + arrows for a set of passes, with clickable arrow endpoints."""
    if not passes:
        st.caption("No passes in this category.")
        return
    fig = _plotly_pitch(fig_height=460, xrange=(zoom[0], zoom[1]),
                        yrange=(zoom[2], zoom[3]), outline=False)

    # Zone counts (success vs fail)
    zone_stats = {}  # name -> {"succ": n, "fail": n, "list": [e...]}
    for e in passes:
        z = dest_zone_fn(e.end_x, e.end_y)
        if z is None:
            continue
        stat = zone_stats.setdefault(z, {"succ": 0, "fail": 0, "list": []})
        if e.result == "SUCCESSFUL":
            stat["succ"] += 1
        else:
            stat["fail"] += 1
        stat["list"].append(e)

    max_total = max((s["succ"]+s["fail"]) for s in zone_stats.values()) if zone_stats else 1
    for name, (x0, x1, y0, y1) in zone_rects.items():
        stat = zone_stats.get(name)
        if not stat:
            continue
        total = stat["succ"] + stat["fail"]
        alpha = 0.18 + 0.45 * (total / max_total)
        fig.add_shape(type="rect", x0=x0, y0=y0, x1=x1, y1=y1,
                      fillcolor=f"rgba(255,215,0,{alpha:.2f})",
                      line=dict(color="rgba(255,255,255,0.4)", width=1),
                      layer="below")
        fig.add_annotation(
            x=(x0+x1)/2, y=(y0+y1)/2,
            text=f"<b>{total}</b><br><span style='font-size:10px'>"
                 f"<span style='color:#27ae60'>{stat['succ']}</span>/"
                 f"<span style='color:#e74c3c'>{stat['fail']}</span></span>",
            showarrow=False, font=dict(color="white", size=13),
        )

    # Arrows
    xs, ys, cds, colors, hovers = [], [], [], [], []
    for i, e in enumerate(passes):
        color = "#27ae60" if e.result == "SUCCESSFUL" else "#e74c3c"
        fig.add_annotation(
            x=e.end_x, y=e.end_y, ax=e.start_x, ay=e.start_y,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.4,
            arrowcolor=color, text="", opacity=0.55,
        )
        xs.append(e.end_x); ys.append(e.end_y); cds.append(i); colors.append(color)
        hovers.append(f"{e.game_time_display} - {_pname(e)} \u2192 {_rname(e) or '?'}<br>{e.result}")
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers",
        marker=dict(size=9, color=colors, line=dict(color="white", width=1)),
        customdata=cds, hovertext=hovers, hoverinfo="text",
        showlegend=False,
    ))
    result = st.plotly_chart(fig, use_container_width=True, key=key,
                             on_select="rerun", selection_mode="points")
    idx_map = {i: e for i, e in enumerate(passes)}
    _handle_plotly_click(result, key, idx_map, nav_events)


def _render_pass_zone_map_vertical(passes, nav_events, key, dest_zone_fn=None, zone_rects=None):
    """Full-pitch image version of the zonal pass map. Zones are defined in
    PIXEL coords matching the painted white lines in the image. Passes are
    classified by projecting their end positions to pixels and checking which
    painted zone contains the pixel \u2014 guarantees dots line up with zones.

    The legacy ``dest_zone_fn`` / ``zone_rects`` parameters are kept for
    backwards compatibility but ignored; the pixel zones are the source of
    truth now."""
    if not passes:
        st.caption("No passes in this category.")
        return
    fig, _ = _plotly_pitch_image("full_pp", fig_height=640)
    pixel_zones = _FULL_PP_PIXEL_ZONES

    # Classify each pass by PIXEL position of its end point. This guarantees
    # the count under each painted zone matches the dots visible inside it.
    zone_stats = {}
    pass_end_px = []  # parallel to `passes`; (px, py) or None
    for e in passes:
        ep = _metric_to_pixel(e.end_x, e.end_y, "full_pp")
        pass_end_px.append(ep)
        if ep is None:
            continue
        z = _pixel_zone_for(ep[0], ep[1], pixel_zones)
        if z is None:
            continue
        stat = zone_stats.setdefault(z, {"succ": 0, "fail": 0, "list": []})
        if e.result == "SUCCESSFUL":
            stat["succ"] += 1
        else:
            stat["fail"] += 1
        stat["list"].append(e)

    max_total = max((s["succ"]+s["fail"]) for s in zone_stats.values()) if zone_stats else 1
    zone_centres_px = {}
    for name, px0, py0, px1, py1 in pixel_zones:
        stat = zone_stats.get(name)
        total = stat["succ"] + stat["fail"] if stat else 0
        alpha = 0.18 + 0.45 * (total / max_total) if total else 0.04
        # Draw every zone outline so the user can see all 14 painted zones,
        # even empty ones.
        fig.add_shape(type="rect", x0=px0, y0=py0, x1=px1, y1=py1,
                      fillcolor=f"rgba(255,215,0,{alpha:.2f})",
                      line=dict(color="rgba(0,0,0,0.45)", width=1),
                      layer="above")
        cx, cy = (px0 + px1) / 2, (py0 + py1) / 2
        zone_centres_px[name] = (cx, cy)
        if total:
            fig.add_annotation(
                x=cx, y=cy,
                text=f"<b>{total}</b><br><span style='font-size:10px'>"
                     f"<span style='color:#27ae60'>{stat['succ']}</span>/"
                     f"<span style='color:#e74c3c'>{stat['fail']}</span></span>",
                showarrow=False,
                font=dict(color="black", size=12, family="Arial Black"),
                bgcolor="rgba(255,255,255,0.75)",
            )

    xs, ys, cds, colors, hovers = [], [], [], [], []
    for i, e in enumerate(passes):
        color = "#27ae60" if e.result == "SUCCESSFUL" else "#e74c3c"
        s_px = _metric_to_pixel(e.start_x, e.start_y, "full_pp")
        e_px = pass_end_px[i]
        if s_px is None or e_px is None:
            continue
        sx, sy = s_px; ex, ey = e_px
        fig.add_annotation(
            x=ex, y=ey, ax=sx, ay=sy,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.4,
            arrowcolor=color, text="", opacity=0.55,
        )
        xs.append(ex); ys.append(ey); cds.append(i); colors.append(color)
        hovers.append(f"{e.game_time_display} - {_pname(e)} \u2192 {_rname(e) or '?'}<br>{e.result}")
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers",
        marker=dict(size=10, color=colors, line=dict(color="white", width=1)),
        customdata=cds, hovertext=hovers, hoverinfo="text",
        showlegend=False, name="arrow_ends",
    ))
    # Transparent hit-zones for zone-click filtering
    if zone_centres_px:
        z_xs = [c[0] for c in zone_centres_px.values()]
        z_ys = [c[1] for c in zone_centres_px.values()]
        z_cds = [f"zone:{name}" for name in zone_centres_px]
        fig.add_trace(go.Scatter(
            x=z_xs, y=z_ys, mode="markers",
            marker=dict(size=70, color="rgba(0,0,0,0)",
                         line=dict(color="rgba(0,0,0,0)", width=0)),
            customdata=z_cds, hoverinfo="skip", showlegend=False,
            name="zone_hits",
        ))
    result = st.plotly_chart(fig, use_container_width=True, key=key,
                             on_select="rerun", selection_mode="points",
                             config={"displayModeBar": False,
                                      "scrollZoom": False, "doubleClick": False})
    if result:
        sel = result.get("selection") if isinstance(result, dict) else None
        if sel and sel.get("points"):
            pt = sel["points"][0]
            cd = pt.get("customdata")
            if isinstance(cd, list):
                cd = cd[0]
            if isinstance(cd, str) and cd.startswith("zone:"):
                zone_name = cd.split(":", 1)[1]
                sig = f"zone_{zone_name}"
                ck = f"__consumed_{key}"
                if st.session_state.get(ck) != sig:
                    st.session_state[ck] = sig
                    group = zone_stats.get(zone_name, {}).get("list", [])
                    if group:
                        _apply_filter_from_click(
                            group,
                            filter_key=f"{key}:{zone_name}",
                            label=f"Pass \u2192 {zone_name}",
                        )
            elif isinstance(cd, int) and 0 <= cd < len(passes):
                sig = f"arrow_{cd}"
                ck = f"__consumed_{key}"
                if st.session_state.get(ck) != sig:
                    st.session_state[ck] = sig
                    _jump_to_event(passes[cd], nav_events)


def _render_progression_arrows_vertical(passes, nav_events, key):
    """Plain arrow-map on the full-pitch image. Used for own->mid view."""
    if not passes:
        st.caption("No passes in this category.")
        return
    fig, _ = _plotly_pitch_image("full_pp", fig_height=600)
    xs, ys, cds, colors, hovers = [], [], [], [], []
    for i, e in enumerate(passes):
        color = "#27ae60" if e.result == "SUCCESSFUL" else "#e74c3c"
        s_px = _metric_to_pixel(e.start_x, e.start_y, "full_pp")
        e_px = _metric_to_pixel(e.end_x, e.end_y, "full_pp")
        if s_px is None or e_px is None:
            continue
        sx, sy = s_px; ex, ey = e_px
        fig.add_annotation(
            x=ex, y=ey, ax=sx, ay=sy,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.4,
            arrowcolor=color, text="", opacity=0.55,
        )
        xs.append(ex); ys.append(ey); cds.append(i); colors.append(color)
        hovers.append(f"{e.game_time_display} - {_pname(e)}<br>{e.result}")
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers",
        marker=dict(size=10, color=colors, line=dict(color="white", width=1)),
        customdata=cds, hovertext=hovers, hoverinfo="text",
        showlegend=False,
    ))
    result = st.plotly_chart(fig, use_container_width=True, key=key,
                             on_select="rerun", selection_mode="points",
                             config={"displayModeBar": False,
                                      "scrollZoom": False, "doubleClick": False})
    idx_map = {i: e for i, e in enumerate(passes)}
    _handle_plotly_click(result, key, idx_map, nav_events)


def _render_progressive_passes(passes, nav_events, team, match, key_prefix):
    if not passes:
        st.caption("No progressive passes.")
        return

    col_bar, col_map = st.columns([1, 1.2])

    with col_bar:
        if team == BOTH_LABEL:
            groups = [(match.home_team, [e for e in passes if e.team == match.home_team]),
                      (match.away_team, [e for e in passes if e.team == match.away_team])]
        else:
            groups = [(team, passes)]

        for team_name, evts in groups:
            if not evts:
                continue
            pc = Counter(e.player for e in evts)
            players_sorted = [p for p, _ in pc.most_common()]
            labels = [_match_player_label(match, p) for p in players_sorted]
            counts = [pc[p] for p in players_sorted]
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=counts, y=labels, orientation="h",
                marker=dict(color="#16a085"),
                customdata=players_sorted,
                hovertemplate="<b>%{y}</b><br>%{x} progressive passes<extra></extra>",
                text=counts, textposition="outside",
            ))
            fig.update_layout(
                height=max(240, 28 * len(labels) + 60),
                margin=dict(l=10, r=30, t=10, b=30),
                xaxis=dict(title="Prog. passes"),
                yaxis=dict(autorange="reversed"),
                plot_bgcolor="white",
                showlegend=False,
            )
            st.markdown(f"*{team_name}*")
            chart_key = f"{key_prefix}_bar_{team_name}"
            result = st.plotly_chart(fig, use_container_width=True, key=chart_key,
                                     on_select="rerun", selection_mode="points")
            if result:
                sel = result.get("selection") if isinstance(result, dict) else None
                if sel and sel.get("points"):
                    pt = sel["points"][0]
                    pname = pt.get("customdata")
                    if isinstance(pname, list):
                        pname = pname[0]
                    if pname:
                        sig = str(pname)
                        ck = f"__consumed_{chart_key}"
                        if st.session_state.get(ck) != sig:
                            st.session_state[ck] = sig
                            plist = [e for e in evts if e.player == pname]
                            if plist:
                                _apply_filter_from_click(
                                    plist,
                                    filter_key=f"{chart_key}:{pname}",
                                    label=f"{pname}",
                                )

    with col_map:
        fig, _ = _plotly_pitch_image("full_pp", fig_height=560)
        xs, ys, cds, hovers, colors = [], [], [], [], []
        for i, e in enumerate(passes):
            color = "#27ae60" if e.result == "SUCCESSFUL" else "#e74c3c"
            s_px = _metric_to_pixel(e.start_x, e.start_y, "full_pp")
            e_px = _metric_to_pixel(e.end_x, e.end_y, "full_pp")
            if s_px is None or e_px is None:
                continue
            sx, sy = s_px; ex, ey = e_px
            fig.add_annotation(
                x=ex, y=ey, ax=sx, ay=sy,
                xref="x", yref="y", axref="x", ayref="y",
                showarrow=True, arrowhead=3, arrowsize=1, arrowwidth=1.4,
                arrowcolor=color, text="", opacity=0.6,
            )
            xs.append(ex); ys.append(ey); cds.append(i); colors.append(color)
            hovers.append(
                f"{e.game_time_display} - {_pname(e)}<br>"
                f"+{e.goal_progression:.1f} m progression"
            )
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="markers",
            marker=dict(size=10, color=colors, line=dict(color="white", width=1)),
            customdata=cds, hovertext=hovers, hoverinfo="text",
            showlegend=False,
        ))
        result = st.plotly_chart(fig, use_container_width=True,
                                  key=f"{key_prefix}_map",
                                  on_select="rerun", selection_mode="points",
                                  config={"displayModeBar": False,
                                           "scrollZoom": False,
                                           "doubleClick": False})
        idx_map = {i: e for i, e in enumerate(passes)}
        _handle_plotly_click(result, f"{key_prefix}_map", idx_map, nav_events)


# ================================================================
# VIZ: OFFENSIVE TRANSITIONS (recovery \u2192 what happens next)
# ================================================================

@st.cache_resource(show_spinner=False)
def _cached_regain_events(_match, cache_key):
    return [e for e in _match.events if e.event_type in ("recovery", "interception")]


def _regain_events(match):
    """All recoveries/interceptions (ball regains). Cached per match."""
    return _cached_regain_events(match, match.name)


def _third_of_length(x):
    """Classify length-x (SciSports +x = attacking direction) into named third."""
    if x < -17.5:
        return "Own Third"
    if x < 17.5:
        return "Middle Third"
    return "Opponent Third"


def _third_loss_name(x):
    """For defensive transitions: where we lost the ball. Att=our attacking 3rd."""
    if x < -17.5:
        return "Def 3rd"
    if x < 17.5:
        return "Middle 3rd"
    return "Att 3rd"


def _sequence_after(regain, match):
    """Team-possession chain starting at `regain` (inclusive). Stops when
    sequenceId changes or the ball changes team."""
    seq = [e for e in match.events
           if e.sequence_id == regain.sequence_id
           and e.team == regain.team
           and e.game_time_ms >= regain.game_time_ms]
    seq.sort(key=lambda e: e.game_time_ms)
    return seq


def _creates_scoring_opp(chain, match):
    """True if the team's sequence ends with a shot/big-chance/goal."""
    shot_types = {"shot", "shot_on_target", "goal", "big_chance"}
    return any(e.event_type in shot_types for e in chain)


def _first_pass_direction(chain):
    """Return 'Forward' / 'Square' / 'Backward' for the first pass in the chain,
    or None if the player didn't pass."""
    for e in chain[1:]:  # skip the regain itself
        if e.event_type not in ("pass", "cross", "key_pass", "free_kick", "corner"):
            continue
        dx = e.end_x - e.start_x  # +x = forward (team attacks +x)
        dy = abs(e.end_y - e.start_y)
        if abs(dx) < 3 and dy < 10:
            return "Square"
        if dx > 3:
            return "Forward"
        if dx < -3:
            return "Backward"
        return "Square"
    return None


def viz_off_transitions(events, team, match):
    regains = _regain_events(match)
    if team != BOTH_LABEL:
        regains = [e for e in regains if e.team == team]
    if not regains:
        st.info("No recoveries / interceptions logged.")
        return

    # Pre-compute sequence chains & outcomes
    records = []  # list of (regain, chain, opp_created, first_dir)
    for r in regains:
        chain = _sequence_after(r, match)
        records.append({
            "regain": r,
            "chain": chain,
            "created": _creates_scoring_opp(chain, match),
            "first_dir": _first_pass_direction(chain),
        })

    tabs = st.tabs(["Start-3rd \u00d7 Scoring Opp",
                     "First Pass Direction",
                     "Regain \u2192 Threat Map"])

    with tabs[0]:
        _render_transitions_third_bar(records, events, key_prefix="off_tr_third",
                                       third_fn=lambda r: _third_of_length(r.start_x),
                                       yes_label="Scoring Opp: Yes",
                                       no_label="Scoring Opp: No",
                                       outcome_fn=lambda rec: rec["created"])

    with tabs[1]:
        _render_first_pass_direction(records, events, team, match,
                                      key_prefix="off_tr_dir")

    with tabs[2]:
        _render_regain_threat_map(records, events, key="off_tr_map")


def _render_transitions_third_bar(records, nav_events, key_prefix, third_fn,
                                    yes_label, no_label, outcome_fn):
    """Grouped bar: thirds on x, Yes/No on grouped bars, clickable."""
    thirds = ["Own Third", "Middle Third", "Opponent Third"]
    # For defensive: thirds passed as "Def 3rd", "Middle 3rd", "Att 3rd"
    present = set()
    grid = {}  # (third, "yes"|"no") -> list of regain events
    for rec in records:
        t = third_fn(rec["regain"])
        present.add(t)
        out = "yes" if outcome_fn(rec) else "no"
        grid.setdefault((t, out), []).append(rec["regain"])

    # Derive ordering from whatever the third_fn produces
    ordered_thirds = [t for t in thirds if t in present] or sorted(present)
    if not ordered_thirds:
        # Perhaps the third_fn returns names like "Def 3rd" etc.
        ordered_thirds = ["Def 3rd", "Middle 3rd", "Att 3rd"]
        ordered_thirds = [t for t in ordered_thirds if t in present] or sorted(present)

    yes_counts = [len(grid.get((t, "yes"), [])) for t in ordered_thirds]
    no_counts  = [len(grid.get((t, "no"),  [])) for t in ordered_thirds]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=ordered_thirds, y=yes_counts, name=yes_label,
        marker=dict(color="#27ae60"),
        customdata=[[t, "yes"] for t in ordered_thirds],
        text=yes_counts, textposition="outside",
    ))
    fig.add_trace(go.Bar(
        x=ordered_thirds, y=no_counts, name=no_label,
        marker=dict(color="#e74c3c"),
        customdata=[[t, "no"] for t in ordered_thirds],
        text=no_counts, textposition="outside",
    ))
    fig.update_layout(
        barmode="group", height=320,
        margin=dict(l=10, r=10, t=10, b=40),
        xaxis=dict(title="Start 3rd"),
        yaxis=dict(title="Transitions"),
        plot_bgcolor="white",
        legend=dict(orientation="h", y=-0.2),
    )
    result = st.plotly_chart(fig, use_container_width=True, key=key_prefix,
                             on_select="rerun", selection_mode="points")
    if result:
        sel = result.get("selection") if isinstance(result, dict) else None
        if sel and sel.get("points"):
            pt = sel["points"][0]
            cd = pt.get("customdata")
            if isinstance(cd, list) and len(cd) == 2:
                third, out = cd
                group = grid.get((third, out), [])
                if group:
                    sig = f"{third}_{out}"
                    ck = f"__consumed_{key_prefix}"
                    if st.session_state.get(ck) != sig:
                        st.session_state[ck] = sig
                        out_label = "Scoring Opp" if out == "yes" else "No Scoring Opp"
                        _apply_filter_from_click(
                            group,
                            filter_key=f"{key_prefix}:{third}:{out}",
                            label=f"{third} \u2192 {out_label}",
                        )


def _render_first_pass_direction(records, nav_events, team, match, key_prefix):
    """100% stacked horizontal bar: one row per player, stacks = direction share."""
    # Team split
    if team == BOTH_LABEL:
        groups = [(match.home_team, [r for r in records if r["regain"].team == match.home_team]),
                  (match.away_team, [r for r in records if r["regain"].team == match.away_team])]
    else:
        groups = [(team, [r for r in records if r["regain"].team == team])]

    dir_order = ["Forward", "Square", "Backward"]
    dir_color = {"Forward": "#27ae60", "Square": "#f39c12", "Backward": "#e74c3c"}

    for team_name, recs in groups:
        by_player = {}
        for r in recs:
            d = r["first_dir"]
            if d is None:
                continue
            p = r["regain"].player
            by_player.setdefault(p, {"Forward": [], "Square": [], "Backward": []})[d].append(r["regain"])
        if not by_player:
            st.caption(f"{team_name}: no transitions with a first pass.")
            continue

        # Sort by total passes desc
        sorted_players = sorted(
            by_player.keys(),
            key=lambda p: -sum(len(by_player[p][d]) for d in dir_order),
        )
        labels = [_match_player_label(match, p) for p in sorted_players]
        # Percentages
        fig = go.Figure()
        for d in dir_order:
            pct, cd = [], []
            for p in sorted_players:
                total = sum(len(by_player[p][dd]) for dd in dir_order)
                val = len(by_player[p][d])
                pct.append(100 * val / total if total else 0)
                cd.append([p, d])
            fig.add_trace(go.Bar(
                y=labels, x=pct, orientation="h", name=d,
                marker=dict(color=dir_color[d]),
                customdata=cd,
                hovertemplate="<b>%{y}</b><br>" + d + ": %{x:.0f}%<extra></extra>",
                text=[f"{p:.0f}%" if p >= 8 else "" for p in pct],
                textposition="inside",
                textfont=dict(color="white", size=10),
            ))
        fig.update_layout(
            barmode="stack", height=max(220, 28 * len(labels) + 80),
            margin=dict(l=10, r=10, t=10, b=30),
            xaxis=dict(title="% of first passes", range=[0, 100]),
            yaxis=dict(autorange="reversed"),
            plot_bgcolor="white",
            legend=dict(orientation="h", y=-0.2),
        )
        st.markdown(f"*{team_name}*")
        chart_key = f"{key_prefix}_{team_name}"
        result = st.plotly_chart(fig, use_container_width=True, key=chart_key,
                                  on_select="rerun", selection_mode="points")
        if result:
            sel = result.get("selection") if isinstance(result, dict) else None
            if sel and sel.get("points"):
                pt = sel["points"][0]
                cd = pt.get("customdata")
                if isinstance(cd, list) and len(cd) == 2:
                    pname, direction = cd
                    sig = f"{pname}_{direction}"
                    ck = f"__consumed_{chart_key}"
                    if st.session_state.get(ck) != sig:
                        st.session_state[ck] = sig
                        group = by_player.get(pname, {}).get(direction, [])
                        if group:
                            _apply_filter_from_click(
                                group,
                                filter_key=f"{chart_key}:{pname}:{direction}",
                                label=f"{pname} \u2014 {direction}",
                            )


def _render_regain_threat_map(records, nav_events, key):
    """Full-pitch image: dot at regain, arrow to chain end, color by elapsed."""
    if not records:
        st.caption("No transitions to map.")
        return
    fig, _ = _plotly_pitch_image("full_pp", fig_height=620)

    def _elapsed_sec(chain):
        if len(chain) < 2:
            return 0.0
        return max(0.0, (chain[-1].game_time_ms - chain[0].game_time_ms) / 1000.0)

    xs, ys, cds, colors, hovers = [], [], [], [], []
    for i, rec in enumerate(records):
        r = rec["regain"]
        chain = rec["chain"]
        end_ev = chain[-1] if chain else r
        elapsed = _elapsed_sec(chain)
        if elapsed < 10:
            color = "#e74c3c"
        elif elapsed < 25:
            color = "#f39c12"
        else:
            color = "#3498db"
        s_px = _metric_to_pixel(r.start_x, r.start_y, "full_pp")
        e_px = _metric_to_pixel(end_ev.end_x or end_ev.start_x,
                                  end_ev.end_y or end_ev.start_y, "full_pp")
        if s_px is None or e_px is None:
            continue
        sx, sy = s_px; ex, ey = e_px
        fig.add_annotation(
            x=ex, y=ey, ax=sx, ay=sy,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=3, arrowsize=1.1, arrowwidth=2,
            arrowcolor=color, text="", opacity=0.7,
        )
        xs.append(sx); ys.append(sy); cds.append(i); colors.append(color)
        shot_label = "\u2605 shot" if rec["created"] else "no shot"
        hovers.append(
            f"{r.game_time_display} - {_pname(r)}<br>"
            f"{elapsed:.1f}s \u2014 {shot_label}"
        )
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers",
        marker=dict(size=10, color=colors, line=dict(color="white", width=1)),
        customdata=cds, hovertext=hovers, hoverinfo="text",
    ))
    result = st.plotly_chart(fig, use_container_width=True, key=key,
                             on_select="rerun", selection_mode="points")
    idx_map = {i: rec["regain"] for i, rec in enumerate(records)}
    _handle_plotly_click(result, key, idx_map, nav_events)
    st.caption("Red = fast break (<10s), orange = medium (<25s), blue = slow build-up.")


# ================================================================
# VIZ: DEFENSIVE TRANSITIONS (ball loss \u2192 response)
# ================================================================

@st.cache_resource(show_spinner=False)
def _cached_all_losses(_match, cache_key):
    """O(n) loss detection — was O(n^2) via repeated list.index(). Cached
    per match (cache_key = match.name) so we don't redo this on every rerun."""
    ev_sorted = sorted(_match.events, key=lambda e: (e.game_time_ms, 0))
    losses = []
    for i, e in enumerate(ev_sorted):
        if e.event_type not in ("recovery", "interception"):
            continue
        for j in range(i - 1, -1, -1):
            prev = ev_sorted[j]
            if prev.team and prev.team != e.team:
                losses.append({
                    "team_lost": prev.team,
                    "loss_time_ms": prev.game_time_ms,
                    "loss_x": prev.end_x or prev.start_x,
                    "loss_y": prev.end_y or prev.start_y,
                    "regain_event": e,
                    "loss_event": prev,
                })
                break
    return losses


def _loss_events(match, team_filter=None):
    """Detect where the ball was lost. Cached per match; team_filter is
    applied on read so different team selections share the cached compute."""
    losses = _cached_all_losses(match, match.name)
    if team_filter:
        losses = [l for l in losses if l["team_lost"] == team_filter]
    return losses


def _opp_scoring_opp(loss, match):
    """Did the opponent create a shot within the same sequence after this loss?"""
    r = loss["regain_event"]
    shot_types = {"shot", "shot_on_target", "goal", "big_chance"}
    return any(
        e.event_type in shot_types and e.team == r.team
        and e.sequence_id == r.sequence_id
        and e.game_time_ms >= r.game_time_ms
        for e in match.events
    )


def _counter_press_active(loss, match, radius=8.0, within_sec=5.0):
    """True if the losing team made a defensive action (tackle/interception/recovery)
    within `radius` metres of the loss point and within `within_sec` seconds."""
    t0 = loss["loss_time_ms"]
    t1 = t0 + int(within_sec * 1000)
    lx, ly = loss["loss_x"], loss["loss_y"]
    team = loss["team_lost"]
    for e in match.events:
        if e.team != team:
            continue
        if not (t0 <= e.game_time_ms <= t1):
            continue
        if e.event_type not in ("recovery", "interception"):
            continue
        dx = e.start_x - lx
        dy = e.start_y - ly
        if (dx*dx + dy*dy) ** 0.5 <= radius:
            return True
    return False


def viz_def_transitions(events, team, match):
    losses = _loss_events(match, team_filter=None if team == BOTH_LABEL else team)
    if not losses:
        st.info("No ball losses detected.")
        return

    # Pre-classify
    for l in losses:
        l["opp_created"] = _opp_scoring_opp(l, match)
        l["counter_pressed"] = _counter_press_active(l, match)

    tabs = st.tabs(["Start-3rd \u00d7 Opp Scoring Opp",
                     "Counter-Press Map",
                     "Dual-Media Viewer"])

    # Fake "event-like" records so the shared handler works: use the regain_event
    # as the clickable (it carries a video timestamp near the loss).
    records = [{"regain": l["regain_event"], "chain": [],
                 "created": l["opp_created"],
                 "first_dir": None,
                 "_loss": l} for l in losses]

    with tabs[0]:
        # Classify loss by OUR attacking third (remember per-team events are already
        # normalized so +x = attacking). loss_x uses losing team's frame.
        _render_transitions_third_bar(
            records, events, key_prefix="def_tr_third",
            third_fn=lambda r_ev: _third_loss_name(
                next(l["loss_x"] for l in losses if l["regain_event"] is r_ev)),
            yes_label="Opponent Scoring Opp: Yes",
            no_label="Opponent Scoring Opp: No",
            outcome_fn=lambda rec: rec["created"],
        )

    with tabs[1]:
        _render_counter_press_map(losses, events, key="def_tr_map")

    with tabs[2]:
        _render_dual_media_viewer(losses, events, match, key="def_tr_dual")


def _render_counter_press_map(losses, nav_events, key):
    """Full-pitch image: dot at each loss. Green=counter-pressed, red=dropped off.
    8m radius circle (computed in pixel space) around each dot."""
    if not losses:
        st.caption("No losses to map.")
        return
    fig, _ = _plotly_pitch_image("full_pp", fig_height=620)
    cfg = _PITCH_IMAGES["full_pp"]
    r_px = 8.0 * cfg["scale_v"]  # 8m converted to pixels (vertical scale)

    xs, ys, cds, colors, hovers = [], [], [], [], []
    for i, l in enumerate(losses):
        mp = _metric_to_pixel(l["loss_x"], l["loss_y"], "full_pp")
        if mp is None:
            continue
        dx, dy = mp
        color = "#27ae60" if l["counter_pressed"] else "#e74c3c"
        fig.add_shape(type="circle",
                       x0=dx - r_px, y0=dy - r_px,
                       x1=dx + r_px, y1=dy + r_px,
                       line=dict(color=color, width=1, dash="dot"),
                       fillcolor=f"rgba({'39,174,96' if l['counter_pressed'] else '231,76,60'},0.08)",
                       layer="below")
        xs.append(dx); ys.append(dy); cds.append(i); colors.append(color)
        hovers.append(
            f"{l['regain_event'].game_time_display} - lost by {l['team_lost']}<br>"
            f"{'Counter-pressed' if l['counter_pressed'] else 'Dropped off'}"
        )
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers",
        marker=dict(size=12, color=colors, line=dict(color="white", width=1.5)),
        customdata=cds, hovertext=hovers, hoverinfo="text",
    ))
    result = st.plotly_chart(fig, use_container_width=True, key=key,
                             on_select="rerun", selection_mode="points")
    idx_map = {i: l["regain_event"] for i, l in enumerate(losses)}
    _handle_plotly_click(result, key, idx_map, nav_events)
    st.caption("Green = active counter-press within 5s in an 8m radius. "
                "Red = dropped off.")


@_fragment
def _render_dual_media_viewer(losses, events, match, key):
    """Animated 2D tracking video synced to the event currently shown in the
    main clip player. Plotly's native play/pause/scrub controls drive the
    animation so the user can run it side-by-side with the clip.
    Wrapped in @st.fragment so changing the colour pickers only reruns this
    widget — the rest of the page (and the model analysis) stays untouched."""
    if not losses:
        st.caption("No losses.")
        return
    pos_path = _positions_path_for_match(match)
    frames = _cached_positions(str(pos_path)) if pos_path else []
    if not pos_path:
        st.caption("No tracking data for this match.")
        return
    if not frames:
        st.caption("Tracking data empty.")
        return

    # Find the loss dict that matches the event currently selected in the
    # main clip player. The selectbox there indexes into the events list.
    sel_idx = st.session_state.get("selected_event_idx", 0)
    sel_event = events[sel_idx] if 0 <= sel_idx < len(events) else None
    nav_loss = None
    if sel_event is not None:
        sel_t = sel_event.game_time_ms
        best, best_d = None, 10_000_000
        for l in losses:
            ev = l.get("loss_event")
            if ev is None:
                continue
            d = abs((ev.game_time_ms or 0) - (sel_t or 0))
            if d < best_d:
                best, best_d = l, d
        if best is not None and best_d <= 500:
            nav_loss = best
    if nav_loss is None:
        nav_loss = losses[0]

    cc1, cc2, cc3 = st.columns([1, 1, 2])
    with cc1:
        home_color = st.color_picker(
            f"{match.home_team} colour", value="#e74c3c",
            key=f"{key}_color_home",
        )
    with cc2:
        away_color = st.color_picker(
            f"{match.away_team} colour", value="#3498db",
            key=f"{key}_color_away",
        )
    with cc3:
        st.markdown(f"_Synced to:_ **{nav_loss['regain_event'].game_time_display} "
                    f"\u2014 lost by {nav_loss['team_lost']}**")

    # Window MATCHES the extract_clip window (pad_before=5s, pad_after=12s)
    # so the 2D animation starts at the same wall-clock moment as the clip.
    import bisect
    frame_ts = [f.get("t", 0) for f in frames]
    t0 = max(0, nav_loss["loss_time_ms"] - 5000)
    t1 = nav_loss["loss_time_ms"] + 12000
    step_ms = 200  # 5 fps animation
    sample_times = list(range(int(t0), int(t1) + 1, step_ms))
    samples = []
    for ts in sample_times:
        fi = bisect.bisect_left(frame_ts, ts)
        fi = min(fi, len(frames) - 1)
        samples.append(frames[fi])

    _render_2d_animation(
        samples, sample_times, nav_loss["loss_time_ms"],
        home_color=home_color, away_color=away_color,
        step_ms=step_ms, key=f"{key}_anim",
    )
    _render_sync_bar(step_ms=step_ms)
    st.caption(
        "Press **\u25b6 Play clip + 2D** to start both at the same time, "
        "**\u23f8 Pause both** to pause together, or **\u27f2 Reset** to rewind both "
        "to the start of the clip. Use Next/Prev in the clip player to "
        "switch to a different event."
    )


def _render_sync_bar(step_ms):
    """Inject a small HTML control that plays / pauses / resets BOTH the video
    element AND the Plotly animation in one click.

    Streamlit's HTML injection (st.html / components.html) does NOT execute
    <script> tags inserted via innerHTML (browser security rule). What DOES
    work is inline ``onclick`` attributes \u2014 those run in the parent page's JS
    context and can reach ``document.querySelector('video')`` and the Plotly
    chart directly. To avoid duplicating code in every button, each button
    inlines a tiny ``(function(){...})()`` that does the action."""
    # The JS body, shared by all three buttons. Uses double quotes inside so we
    # can safely place it inside an onclick="..." attribute by HTML-escaping
    # only ", &, <, and >.
    js_find = (
        "var v=document.querySelector('video');"
        "var anim=null;"
        "var ps=document.querySelectorAll('.js-plotly-plot');"
        "for(var i=0;i<ps.length;i++){"
        "  var p=ps[i];"
        "  if(p.layout&&p.layout.sliders&&p.layout.sliders.length){anim=p;break;}"
        "}"
    )
    play_js = js_find + (
        f"if(v)v.play().catch(function(){{}});"
        f"if(anim&&window.Plotly)"
        f"window.Plotly.animate(anim,[null],"
        f"{{frame:{{duration:{step_ms},redraw:true}},"
        f"fromcurrent:true,transition:{{duration:0}}}});"
    )
    pause_js = js_find + (
        "if(v)v.pause();"
        "if(anim&&window.Plotly)"
        "window.Plotly.animate(anim,[[null]],"
        "{frame:{duration:0,redraw:false},mode:'immediate'});"
    )
    reset_js = js_find + (
        "if(v){v.pause();v.currentTime=0;}"
        "if(anim&&window.Plotly&&anim._transitionData"
        "&&anim._transitionData._frames"
        "&&anim._transitionData._frames.length){"
        "window.Plotly.animate(anim,[anim._transitionData._frames[0].name],"
        "{frame:{duration:0,redraw:true},mode:'immediate'});"
        "}"
    )

    def _attr_escape(s: str) -> str:
        # Minimal HTML-attribute escaping: only & < > " need escaping. We use
        # single quotes everywhere in the JS so " never appears.
        return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    sync_html = f"""
<style>
  .sync-bar {{
    display: flex; gap: 8px; margin: 6px 0 4px 0; align-items: center;
    flex-wrap: wrap;
  }}
  .sync-bar button {{
    padding: 8px 18px; border: 0; border-radius: 4px; cursor: pointer;
    font-weight: 600; color: white; font-size: 14px;
  }}
  .sync-bar button:hover {{ opacity: 0.85; }}
  .sync-bar .btn-play  {{ background: #27ae60; }}
  .sync-bar .btn-pause {{ background: #7f8c8d; }}
  .sync-bar .btn-reset {{ background: #3498db; }}
</style>
<div class="sync-bar">
  <button class="btn-play"  onclick="{_attr_escape(play_js)}">\u25b6 Play clip + 2D</button>
  <button class="btn-pause" onclick="{_attr_escape(pause_js)}">\u23f8 Pause both</button>
  <button class="btn-reset" onclick="{_attr_escape(reset_js)}">\u27f2 Reset</button>
</div>
"""
    # st.html (1.32+) injects directly into the DOM (no iframe) so onclick
    # handlers run in the parent context. components.html (the only other
    # built-in HTML injector) is sandboxed and would be cross-origin to the
    # video / Plotly chart, so it cannot reach them.
    if hasattr(st, "html"):
        st.html(sync_html)
    else:
        st.markdown(sync_html, unsafe_allow_html=True)


def _build_pitch_layout_shapes():
    """Static pitch outline shapes for the horizontal half-pitch view."""
    line = "#444444"; bg = "#ffffff"
    shapes = [
        dict(type="rect", x0=-PITCH_X, y0=-PITCH_Y, x1=PITCH_X, y1=PITCH_Y,
              fillcolor=bg, line=dict(color=line, width=2), layer="below"),
        dict(type="line", x0=0, y0=-PITCH_Y, x1=0, y1=PITCH_Y,
              line=dict(color=line, width=1)),
        dict(type="circle", x0=-9.15, y0=-9.15, x1=9.15, y1=9.15,
              line=dict(color=line, width=1)),
    ]
    for side in (-1, 1):
        shapes.append(dict(type="rect",
                            x0=side * PITCH_X, y0=-20.16,
                            x1=side * (PITCH_X - 16.5), y1=20.16,
                            line=dict(color=line, width=1)))
        shapes.append(dict(type="rect",
                            x0=side * PITCH_X, y0=-9.16,
                            x1=side * (PITCH_X - 5.5), y1=9.16,
                            line=dict(color=line, width=1)))
    return shapes


def _frame_traces(frame, home_color, away_color, image_key="full_zo_h"):
    """Build the three Scatter traces (home, away, ball) for a tracking frame,
    projected into the named image's pixel space (default: horizontal full-zo).
    Always returns three traces so Plotly animation can replace them by index."""
    def _proj(pl):
        return _metric_to_pixel(pl.get("x", 0), pl.get("y", 0), image_key)

    home_xs, home_ys, home_txt = [], [], []
    for pl in frame.get("h", []):
        mp = _proj(pl)
        if mp is None:
            continue
        home_xs.append(mp[0]); home_ys.append(mp[1])
        home_txt.append(str(pl.get("s", "")))
    away_xs, away_ys, away_txt = [], [], []
    for pl in frame.get("a", []):
        mp = _proj(pl)
        if mp is None:
            continue
        away_xs.append(mp[0]); away_ys.append(mp[1])
        away_txt.append(str(pl.get("s", "")))
    b = frame.get("b") or {}
    bx, by = [], []
    if b and "x" in b:
        mp = _metric_to_pixel(b["x"], b.get("y", 0), image_key)
        if mp is not None:
            bx = [mp[0]]; by = [mp[1]]
    return [
        go.Scatter(x=home_xs, y=home_ys, mode="markers+text",
                    marker=dict(size=22, color=home_color,
                                 line=dict(color="white", width=2)),
                    text=home_txt, textposition="middle center",
                    textfont=dict(color="white", size=10),
                    hoverinfo="skip", showlegend=False),
        go.Scatter(x=away_xs, y=away_ys, mode="markers+text",
                    marker=dict(size=22, color=away_color,
                                 line=dict(color="white", width=2)),
                    text=away_txt, textposition="middle center",
                    textfont=dict(color="white", size=10),
                    hoverinfo="skip", showlegend=False),
        go.Scatter(x=bx, y=by, mode="markers",
                    marker=dict(size=10, color="#f1c40f", symbol="circle",
                                 line=dict(color="#000000", width=1)),
                    hoverinfo="skip", showlegend=False),
    ]


def _render_2d_animation(samples, sample_times, focal_ms,
                           home_color, away_color, step_ms, key):
    """Plotly animated figure on the HORIZONTAL Full_field_zo_zones.png. The
    figure opens on the snapshot nearest to ``focal_ms`` (the loss moment)
    and the time window matches the clip extraction window for same-moment
    sync."""
    if not samples:
        return

    image_key = "full_zo_h"
    cfg = _PITCH_IMAGES[image_key]
    iw, ih = cfg["size"]
    img = _cached_rotated_image(str(cfg["path"]), int(cfg["rotate_cw_deg"]))

    init_idx = min(
        range(len(sample_times)),
        key=lambda i: abs(sample_times[i] - focal_ms),
    )
    init_traces = _frame_traces(samples[init_idx], home_color, away_color, image_key)
    plotly_frames = [
        go.Frame(name=str(ts),
                  data=_frame_traces(fr, home_color, away_color, image_key))
        for fr, ts in zip(samples, sample_times)
    ]

    play_button = dict(
        label="▶ Play", method="animate",
        args=[None, {"frame": {"duration": step_ms, "redraw": True},
                      "fromcurrent": True,
                      "transition": {"duration": 0}}],
    )
    pause_button = dict(
        label="⏸ Pause", method="animate",
        args=[[None], {"frame": {"duration": 0, "redraw": False},
                        "mode": "immediate",
                        "transition": {"duration": 0}}],
    )
    slider_steps = []
    for fr, ts in zip(plotly_frames, sample_times):
        rel = (ts - focal_ms) / 1000.0
        slider_steps.append(dict(
            method="animate",
            args=[[fr.name],
                  {"frame": {"duration": step_ms, "redraw": True},
                    "mode": "immediate",
                    "transition": {"duration": 0}}],
            label=f"{rel:+.1f}s",
        ))

    fig = go.Figure(
        data=init_traces,
        frames=plotly_frames,
        layout=go.Layout(
            xaxis=dict(range=[0, iw], visible=False, fixedrange=True),
            yaxis=dict(range=[ih, 0], visible=False, fixedrange=True,
                        scaleanchor="x"),
            images=[dict(source=img, xref="x", yref="y",
                          x=0, y=0, sizex=iw, sizey=ih,
                          sizing="stretch", layer="below")],
            plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
            margin=dict(l=0, r=0, t=10, b=90), height=520,
            showlegend=False, dragmode=False,
            updatemenus=[dict(
                type="buttons",
                buttons=[play_button, pause_button],
                direction="left",
                x=0.02, y=-0.05,
                xanchor="left", yanchor="top",
                pad=dict(t=10, r=10), bgcolor="#f7f7f7",
            )],
            sliders=[dict(
                steps=slider_steps,
                active=init_idx,
                x=0.16, y=-0.06,
                len=0.80,
                pad=dict(t=10, b=5),
                currentvalue=dict(prefix="Δt = ", font=dict(size=12)),
                transition=dict(duration=0),
            )],
        ),
    )
    st.plotly_chart(fig, use_container_width=True, key=key,
                     config={"displayModeBar": False, "scrollZoom": False,
                             "doubleClick": False})


def _draw_animated_tracking_pitch_LEGACY_UNUSED(samples, sample_times, focal_ms,
                                                  home_color, away_color, key):
    """Kept for compatibility — replaced by _render_2d_animation. Not called
    anywhere; safe to delete after a clean run."""
    fig = go.Figure()
    if not samples:
        return
    n = len(samples)
    max_dt = max(abs(t - focal_ms) for t in sample_times) or 1
    for i, (frame, ts) in enumerate(zip(samples, sample_times)):
        # Opacity: 0.15 at the edges \u2192 1.0 at the focal moment
        dt_norm = abs(ts - focal_ms) / max_dt
        opacity = 0.15 + 0.85 * (1.0 - dt_norm)
        size_focal = 18 if abs(ts - focal_ms) < 250 else 9
        is_focal = abs(ts - focal_ms) < 250

        for side_key, color in (("h", home_color), ("a", away_color)):
            xs, ys, txt = [], [], []
            for pl in frame.get(side_key, []):
                xs.append(pl.get("x", 0)); ys.append(pl.get("y", 0))
                txt.append(str(pl.get("s", "")) if is_focal else "")
            if not xs:
                continue
            fig.add_trace(go.Scatter(
                x=xs, y=ys,
                mode="markers+text" if is_focal else "markers",
                marker=dict(size=size_focal, color=color, opacity=opacity,
                             line=dict(color="white", width=1 if is_focal else 0)),
                text=txt, textposition="middle center",
                textfont=dict(color="white", size=10),
                hoverinfo="skip", showlegend=False,
            ))
        # Ball trail
        b = frame.get("b") or {}
        if b and "x" in b:
            fig.add_trace(go.Scatter(
                x=[b["x"]], y=[b["y"]], mode="markers",
                marker=dict(size=8 if is_focal else 4,
                             color="#f1c40f", opacity=opacity,
                             symbol="circle",
                             line=dict(color="#000000",
                                       width=1 if is_focal else 0)),
                hoverinfo="skip", showlegend=False,
            ))

    fig.update_layout(
        xaxis=dict(range=[-PITCH_X, PITCH_X], visible=False, fixedrange=True),
        yaxis=dict(range=[-PITCH_Y, PITCH_Y], visible=False, fixedrange=True,
                    scaleanchor="x"),
        plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
        margin=dict(l=0, r=0, t=0, b=0), height=420, showlegend=False,
        dragmode=False,
    )
    st.plotly_chart(fig, use_container_width=True, key=key,
                     config={"displayModeBar": False, "scrollZoom": False,
                             "doubleClick": False})


def _draw_bw_tracking_pitch(frame, key):
    """Black & white 2D pitch with the home/away dots and ball at this frame."""
    # Black & white colours: lines white, home = filled black, away = outlined black
    fig = go.Figure()
    line = "#000000"
    bg = "#ffffff"
    fig.add_shape(type="rect", x0=-PITCH_Y, y0=-PITCH_X, x1=PITCH_Y, y1=PITCH_X,
                   fillcolor=bg, line=dict(color=line, width=2), layer="below")
    fig.add_shape(type="line", x0=-PITCH_Y, y0=0, x1=PITCH_Y, y1=0,
                   line=dict(color=line, width=1))
    fig.add_shape(type="circle", x0=-9.15, y0=-9.15, x1=9.15, y1=9.15,
                   line=dict(color=line, width=1))
    for side in (-1, 1):
        fig.add_shape(type="rect", x0=-20.16, y0=side*52.5,
                       x1=20.16, y1=side*(52.5-16.5),
                       line=dict(color=line, width=1))
        fig.add_shape(type="rect", x0=-9.16, y0=side*52.5,
                       x1=9.16, y1=side*(52.5-5.5),
                       line=dict(color=line, width=1))
    # Dots (transform to vertical display coords)
    for side_key, symbol, fill in (("h", "circle", "#000000"),
                                     ("a", "circle-open", "#000000")):
        xs, ys, txt = [], [], []
        for pl in frame.get(side_key, []):
            dx, dy = _v(pl.get("x", 0), pl.get("y", 0))
            xs.append(dx); ys.append(dy); txt.append(str(pl.get("s", "")))
        if xs:
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="markers+text",
                marker=dict(size=18, color=fill, symbol=symbol,
                             line=dict(color="#000000", width=1.5)),
                text=txt, textposition="middle center",
                textfont=dict(color="#ffffff" if side_key == "h" else "#000000", size=9),
                hoverinfo="text", showlegend=False,
            ))
    # Ball
    b = frame.get("b") or {}
    if b:
        bx, by = _v(b.get("x", 0), b.get("y", 0))
        fig.add_trace(go.Scatter(
            x=[bx], y=[by], mode="markers",
            marker=dict(size=10, color="#ffff00", symbol="circle",
                         line=dict(color="#000000", width=1.5)),
            hoverinfo="skip", showlegend=False,
        ))
    fig.update_layout(
        xaxis=dict(range=[-PITCH_Y, PITCH_Y], visible=False),
        yaxis=dict(range=[-PITCH_X, PITCH_X], visible=False, scaleanchor="x"),
        plot_bgcolor=bg, paper_bgcolor=bg,
        margin=dict(l=0, r=0, t=0, b=0), height=520, showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True, key=key)


# ================================================================
# TIMELINE (0-90 min clip navigator)
# ================================================================

def _render_event_timeline(events, current_idx, key="timeline"):
    """Draw a 0-90 minute strip with a tick per clip. Click a tick to jump.
    The currently-selected event is highlighted."""
    if not events:
        return
    xs, colors, sizes, cds, hovers = [], [], [], [], []
    for i, e in enumerate(events):
        minute = (e.game_time_ms or 0) / 60000.0
        xs.append(minute)
        is_current = (i == current_idx)
        colors.append("#ffcc00" if is_current else ("#1e88e5" if e.team == events[0].team else "#e53935"))
        sizes.append(14 if is_current else 9)
        cds.append(i)
        hovers.append(f"{e.game_time_display} \u2014 {e.team} \u2014 {_pname(e)}")

    fig = go.Figure()
    # Baseline
    fig.add_shape(type="line", x0=0, x1=90, y0=0, y1=0,
                   line=dict(color="#888", width=2))
    # Minute tick marks every 15 min
    for m in (0, 15, 30, 45, 60, 75, 90):
        fig.add_shape(type="line", x0=m, x1=m, y0=-0.3, y1=0.3,
                       line=dict(color="#bbb", width=1))
        fig.add_annotation(x=m, y=-0.8, text=f"{m}'", showarrow=False,
                            font=dict(size=10, color="#555"))
    # Half-time marker
    fig.add_shape(type="line", x0=45, x1=45, y0=-1.2, y1=1.2,
                   line=dict(color="#aaa", width=1, dash="dot"))
    # Event dots
    fig.add_trace(go.Scatter(
        x=xs, y=[0] * len(xs),
        mode="markers",
        marker=dict(size=sizes, color=colors,
                    line=dict(color="white", width=1)),
        customdata=cds,
        hovertext=hovers, hoverinfo="text",
    ))
    fig.update_layout(
        height=90,
        margin=dict(l=10, r=10, t=5, b=10),
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(range=[-2, 92], showgrid=False, zeroline=False,
                    showticklabels=False, fixedrange=True),
        yaxis=dict(range=[-1.5, 1.5], showgrid=False, zeroline=False,
                    showticklabels=False, fixedrange=True),
        showlegend=False,
    )
    result = st.plotly_chart(fig, use_container_width=True, key=key,
                             on_select="rerun", selection_mode="points")
    # Click handling: jump to the clicked event
    try:
        sel = (result.get("selection") or {}).get("points") or []
    except Exception:
        sel = []
    if sel:
        cd = sel[0].get("customdata")
        target = cd[0] if isinstance(cd, (list, tuple)) else cd
        if isinstance(target, int) and target != current_idx:
            # Guard: only act once per click (plotly re-delivers on reruns)
            consumed_key = f"__consumed_{key}"
            last = st.session_state.get(consumed_key)
            signature = (target, sel[0].get("point_index"))
            if last != signature:
                st.session_state[consumed_key] = signature
                st.session_state["pending_event_idx"] = target
                st.rerun()


# ================================================================
# VIZ MAP
# ================================================================

VIZ_MAP = {
    "def_corner": viz_def_corner_event,
    "att_corner": viz_att_corner_event,
    "goal_kick": viz_goal_kicks,
    "free_kick": viz_free_kicks,
    "cross": viz_crosses,
    "key_pass": viz_key_passes,
    "shots_all": viz_shots_unified,
    "big_chance": viz_big_chances,
    "recovery": viz_recoveries,
    "interception": viz_interceptions,
    "build_up": viz_build_up,
    "final_3rd": viz_final_third,
    "off_transition": viz_off_transitions,
    "def_transition": viz_def_transitions,
}


def _events_for_view(match, event_type):
    """Return events relevant to the sidebar event selector for this view.

    For literal event types (corner, shot, ...), returns events of that type.
    For aggregate views (build_up, final_3rd), returns the primary event list
    driving the view (used by the sidebar dropdown / video player)."""
    if event_type == "final_3rd":
        # Passes entering the final third, sorted by time
        return sorted(
            [e for e in match.events
             if e.event_type == "pass" and e.end_third == 3 and e.start_third != 3],
            key=lambda e: e.game_time_ms,
        )
    if event_type == "build_up":
        # All progressive-ish passes in the own half + middle (start_third <= 2)
        return sorted(
            [e for e in match.events
             if e.event_type == "pass" and e.start_third in (1, 2)],
            key=lambda e: e.game_time_ms,
        )
    if event_type == "off_transition":
        # Ball regains (recovery / interception) as the driver events
        return sorted(_regain_events(match), key=lambda e: e.game_time_ms)
    if event_type == "def_transition":
        # Events marking where we lost the ball (the losing team's last action)
        loss_dicts = _loss_events(match)
        return sorted(
            [l["loss_event"] for l in loss_dicts if l.get("loss_event")],
            key=lambda e: e.game_time_ms,
        )
    if event_type in ("def_corner", "att_corner"):
        # Both surface every corner — clip-side filtering is intentional so
        # users can flick between attacking and defending clips.
        return get_events_by_type(match, "corner")
    if event_type == "shots_all":
        # Unified Shots view: every shot (incl. on target / goals / big chances),
        # sorted by time so the clip selector orders them naturally.
        return sorted(
            [e for e in match.events if e.event_type == "shot"],
            key=lambda e: e.game_time_ms,
        )
    return get_events_by_type(match, event_type)


# ================================================================
# MAIN APP
# ================================================================

def main():
    st.set_page_config(page_title="D&V Team Analysis", layout="wide")
    # Surface any OneDrive bootstrap error now that set_page_config has run.
    for _err in _BOOTSTRAP_ERROR:
        st.error(f"OneDrive sync failed: {_err}")

    # Blue sidebar styling
    st.markdown("""
    <style>
    [data-testid="stSidebar"] {
        background-color: #1e3a8a;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] .stRadio label,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: white !important;
    }
    [data-testid="stSidebar"] hr {
        border-color: rgba(255, 255, 255, 0.3) !important;
    }
    /* Compact main layout so the dashboard fits on one screen */
    .block-container {
        padding-top: 0.8rem !important;
        padding-bottom: 0.4rem !important;
        max-width: 100% !important;
    }
    h1, h2, h3 { margin-top: 0.2rem !important; margin-bottom: 0.4rem !important; }
    .stMarkdown p { margin-bottom: 0.25rem !important; }
    [data-testid="stMetricValue"] { font-size: 1.05rem !important; }
    [data-testid="stMetricLabel"] { font-size: 0.7rem !important; }
    div[data-baseweb="select"] > div { min-height: 30px !important; }
    .stButton > button { padding: 0.15rem 0.6rem !important; min-height: 30px !important; }
    .stTabs [data-baseweb="tab-list"] button { padding: 0.25rem 0.6rem !important; }
    hr { margin: 0.3rem 0 !important; }
    </style>
    """, unsafe_allow_html=True)

    st.title("D&V Team Analysis")

    matches = discover_matches(DATA_DIR)
    if not matches:
        st.error("No match files found.")
        return

    # --- CRITICAL: Consume pending_event_idx BEFORE any widgets render ---
    if "pending_event_idx" in st.session_state:
        pending = st.session_state.pop("pending_event_idx")
        st.session_state.pop("event_selector", None)
        st.session_state["selected_event_idx"] = pending
        st.session_state["__initial_event_idx"] = pending

    # --- Sidebar ---
    with st.sidebar:
        st.header("Match")
        match_names = [f"{m.home_team} vs {m.away_team}" if m.home_team else m.name for m in matches]
        selected_match_idx = st.selectbox("Select match", range(len(matches)),
                                           format_func=lambda i: match_names[i])
        match = matches[selected_match_idx]

        if not match.cameras:
            st.warning("No video URLs configured. Add them to videos.json.")

        st.divider()
        st.header("Event Type")
        event_type_label = st.radio("Select event", list(EVENT_TYPES.keys()))
        event_type = EVENT_TYPES[event_type_label]

        st.divider()
        st.header("Team")
        teams = [t for t in [match.home_team, match.away_team] if t]
        team_options = teams + [BOTH_LABEL]
        selected_team = st.radio("Show analysis for", team_options, horizontal=True)

    all_events = _events_for_view(match, event_type)
    # Filter the event dropdown by team selection; "Both" keeps everything.
    # Special-case: for "corner" we always show clips for BOTH teams so the
    # user can browse defensive AND attacking corner clips while analysing.
    if selected_team == BOTH_LABEL or event_type in ("def_corner", "att_corner"):
        events = all_events
    else:
        events = [e for e in all_events if e.team == selected_team]

    # Detect event_type/team/match changes and reset event_selector widget state.
    # Also clear any chart-click filter (it is scoped to the previous context).
    current_ctx = (selected_match_idx, event_type, selected_team)
    if st.session_state.get("__last_ctx") != current_ctx:
        st.session_state.pop("event_selector", None)
        st.session_state["selected_event_idx"] = 0
        st.session_state["__last_ctx"] = current_ctx
        _clear_event_filter()

    # Apply click-driven chart filter, if active
    filter_sigs = st.session_state.get("event_filter_sigs")
    if filter_sigs:
        filtered = [e for e in events if (e.game_time_ms, e.player) in filter_sigs]
        if filtered:
            events = filtered
        else:
            _clear_event_filter()

    # Initial selected index
    if "selected_event_idx" not in st.session_state:
        st.session_state["selected_event_idx"] = 0
    if st.session_state["selected_event_idx"] >= len(events):
        st.session_state["selected_event_idx"] = 0
    # Guard stale event_selector state (e.g. after events list shrinks)
    if "event_selector" in st.session_state and (
        not isinstance(st.session_state["event_selector"], int)
        or st.session_state["event_selector"] >= len(events)
    ):
        st.session_state.pop("event_selector", None)

    # --- Main area ---
    st.subheader(f"{event_type_label} - {match.home_team} vs {match.away_team}")
    if selected_team == BOTH_LABEL:
        st.caption(f"{len(events)} events (both teams)")
    else:
        st.caption(f"{len(events)} events by {selected_team}")

    col_video, col_viz = st.columns([1, 1])

    with col_video:
        direct_event = st.session_state.pop("direct_play_event", None)

        # Chart-click filter chip (click a bar/segment to drill down; click again to clear)
        flabel = st.session_state.get("event_filter_label")
        if flabel:
            chip_c1, chip_c2 = st.columns([4, 1])
            with chip_c1:
                st.info(f"\U0001F3AF Filtered: {flabel} \u2014 {len(events)} clip(s)")
            with chip_c2:
                if st.button("Clear", key="clear_event_filter", use_container_width=True):
                    _clear_event_filter()
                    st.session_state.pop("event_selector", None)
                    st.session_state["selected_event_idx"] = 0
                    st.rerun()

        if not events and not direct_event:
            st.info(f"No {event_type_label.lower()} events in this match.")
        else:
            if events:
                event_labels = [f"{e.game_time_display} - {e.team} - {_pname(e)}" for e in events]
                # Derive initial index (either from pending consumption or current state)
                init_idx = st.session_state.get("__initial_event_idx",
                                                  st.session_state["selected_event_idx"])
                st.session_state.pop("__initial_event_idx", None)
                if init_idx >= len(events):
                    init_idx = 0
                selected_event_idx = st.selectbox(
                    "Select event to watch",
                    range(len(events)),
                    format_func=lambda i: event_labels[i],
                    index=init_idx,
                    key="event_selector",
                )
                st.session_state["selected_event_idx"] = selected_event_idx

            if direct_event:
                event = direct_event
                st.info(f"Playing: {event.game_time_display} - {event.event_type} - {event.team} - {_pname(event)}")
            else:
                event = events[selected_event_idx]

            show_video_for_event(match, event)

            # --- Timeline (0-90 min) with clickable ticks ---
            if events:
                _render_event_timeline(events, selected_event_idx, key="main_timeline")

            # --- Next button + keyboard nav hint ---
            nav_cols = st.columns([1, 1, 3])
            with nav_cols[0]:
                if st.button("\u23ED Next", key="btn_next_event", use_container_width=True):
                    nxt = (selected_event_idx + 1) % len(events) if events else 0
                    st.session_state["pending_event_idx"] = nxt
                    st.rerun()
            with nav_cols[1]:
                if st.button("\u23EE Prev", key="btn_prev_event", use_container_width=True):
                    prv = (selected_event_idx - 1) % len(events) if events else 0
                    st.session_state["pending_event_idx"] = prv
                    st.rerun()
            with nav_cols[2]:
                st.caption("Tip: Tab/Shift+Tab to focus Next, then Enter to advance.")

            # Details (compact)
            st.markdown(f"**{event.game_time_display}** | {event.team} | {_pname(event)}")
            detail_cols = st.columns(4)
            with detail_cols[0]:
                st.metric("Result", event.result.replace("_", " ").title())
            with detail_cols[1]:
                if event.xg > 0:
                    st.metric("xG", f"{event.xg:.3f}")
            with detail_cols[2]:
                if event.receiver and event.receiver != "NOT_APPLICABLE":
                    st.metric("Receiver", _rname(event) or event.receiver)
            with detail_cols[3]:
                if event.body_part and event.body_part != "NOT_APPLICABLE":
                    st.metric("Body Part", event.body_part.replace("_", " ").title())

            # --- Goal-kick average positions panel (left column, under clip) ---
            if event_type == "goal_kick" and events:
                st.markdown("---")
                st.markdown("**Average Positions (from tracking data)**")
                team_events_for_avg = (
                    list(all_events) if selected_team == BOTH_LABEL
                    else [e for e in all_events if e.team == selected_team]
                )
                _render_gk_avg_positions(team_events_for_avg, selected_team, match)

    with col_viz:
        st.markdown("**Analysis**")
        viz_fn = VIZ_MAP.get(event_type)
        if viz_fn:
            viz_fn(events, selected_team, match)

    # --- Full-width extras (rendered outside the col_video/col_viz pair) ---
    if event_type == "def_corner":
        _render_defending_corners_extras(events, selected_team, match)


if __name__ == "__main__":
    main()
