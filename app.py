"""FCDB Match Tracker - Streamlit app for match event analysis with video clips."""

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


def _positions_path_for_match(match):
    """Derive SciSportsPositions JSON path from the match prefix."""
    for f in DATA_DIR.glob(f"{match.name}*SciSportsPositions*.json"):
        return f
    return None

DATA_DIR = Path(__file__).parent

EVENT_TYPES = {
    "Corners": "corner",
    "Goal Kicks": "goal_kick",
    "Free Kicks": "free_kick",
    "Crosses": "cross",
    "Key Passes": "key_pass",
    "Build-Up": "build_up",
    "Final 3rd": "final_3rd",
    "Offensive Transitions": "off_transition",
    "Defensive Transitions": "def_transition",
    "Goals": "goal",
    "Shots on Target": "shot_on_target",
    "Shots Total": "shot",
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


# ================================================================
# VIDEO PLAYER
# ================================================================

def show_video_for_event(match, event):
    available = {k: v for k, v in match.cameras.items() if v != "PLACEHOLDER"}
    if not available:
        st.warning("No video files configured for this match.")
        return

    camera_names = list(available.keys())
    if len(camera_names) > 1:
        cam = st.radio("Camera", camera_names, horizontal=True, key="camera_select")
    else:
        cam = camera_names[0]

    with st.spinner(f"Loading clip ({cam})..."):
        try:
            clip = extract_clip(available[cam], event.video_time_sec)
            st.video(str(clip))
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


def viz_corners(events, team, match):
    team_events = list(events) if team == BOTH_LABEL else [e for e in events if e.team == team]
    zones_by_pos = _build_corner_zones()

    classified = []  # (event, side, zone)
    for e in team_events:
        pos = _get_corner_position(e.start_x, e.start_y)
        if pos is None:
            pos = "top_left" if e.start_y < 0 else "top_right"
        side = "L" if "left" in pos else "R"
        zones = zones_by_pos[pos]
        # CORNER_SHORT is always a short-corner pass regardless of end location
        if e.sub_type == "CORNER_SHORT":
            zone = "Short_Corner_Zone"
        else:
            zone = _assign_zone(e.end_x, e.end_y, zones)
        classified.append((e, side, zone))

    left = [c for c in classified if c[1] == "L"]
    right = [c for c in classified if c[1] == "R"]

    col1, col2 = st.columns(2)
    with col1:
        _render_corner_side(left, "L", _LEFT_CORNER_IMG, _ATT_L_ZONES, _ATT_L_CENTERS,
                             events, key_prefix="corner_L")
    with col2:
        _render_corner_side(right, "R", _RIGHT_CORNER_IMG, _ATT_R_ZONES, _ATT_R_CENTERS,
                             events, key_prefix="corner_R")

    # Shots from corners
    st.markdown("---")
    st.markdown("**Corners Leading to Shot**")
    all_shots = [e for e in match.events if e.event_type == "shot"]
    corner_seqs = {e.sequence_id for e in events if e.sequence_id >= 0}
    if team == BOTH_LABEL:
        shots_from_corners = [s for s in all_shots if s.sequence_id in corner_seqs]
    else:
        shots_from_corners = [s for s in all_shots if s.sequence_id in corner_seqs
                              and s.team == team]
    if shots_from_corners:
        shooter_counts = Counter(s.player for s in shots_from_corners)
        for player, count in shooter_counts.most_common():
            clips = [s for s in shots_from_corners if s.player == player]
            label = _match_player_label(match, player)
            with st.expander(f"{label} - {count} shot(s) from corners"):
                _event_buttons(clips, events, f"corner_shooter_{player}")
    else:
        st.caption("No shots from corners.")


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

    # ---- Average positions (from tracking) ----
    st.markdown("---")
    st.markdown("**Average Positions (from tracking data)**")
    _render_gk_avg_positions(team_events, team, match)


def _render_gk_vertical_zones(gks, nav_events, key):
    if not gks:
        st.caption("No goal kicks in this category.")
        return
    fig = _plotly_pitch_vertical(fig_height=600)

    # Split the pitch into destination zones. We orient the GK team attacking
    # UPWARD (toward +y in display). SciSports events are already per-team
    # normalized with team attacking +x; we map event.x -> display.y via _v.
    zones = {
        "Own L":          (-PITCH_Y, 0,       -PITCH_X,        -PITCH_X/2),    # near-GK left half
        "Own R":          (0,        PITCH_Y, -PITCH_X,        -PITCH_X/2),
        "Mid L Wing":     (-PITCH_Y, -11,     -PITCH_X/2,       PITCH_X/2),
        "Mid Central":    (-11,      11,      -PITCH_X/2,       PITCH_X/2),
        "Mid R Wing":     (11,       PITCH_Y, -PITCH_X/2,       PITCH_X/2),
        "Final L":        (-PITCH_Y, -11,      PITCH_X/2,       PITCH_X),
        "Final Central":  (-11,      11,       PITCH_X/2,       PITCH_X),
        "Final R":        (11,       PITCH_Y,  PITCH_X/2,       PITCH_X),
    }
    # Count per zone (using vertical display coords)
    zone_hits = {name: [] for name in zones}
    for e in gks:
        dx, dy = _v(e.end_x, e.end_y)
        for name, (x0, x1, y0, y1) in zones.items():
            if x0 <= dx <= x1 and y0 <= dy <= y1:
                zone_hits[name].append(e)
                break
    max_cnt = max((len(v) for v in zone_hits.values()), default=1) or 1
    for name, (x0, x1, y0, y1) in zones.items():
        cnt = len(zone_hits[name])
        if cnt == 0:
            continue
        alpha = 0.15 + 0.5 * (cnt / max_cnt)
        fig.add_shape(type="rect", x0=x0, y0=y0, x1=x1, y1=y1,
                      fillcolor=f"rgba(255,215,0,{alpha:.2f})",
                      line=dict(color="rgba(255,255,255,0.4)", width=1),
                      layer="below")
        fig.add_annotation(x=(x0+x1)/2, y=(y0+y1)/2,
                           text=f"<b>{name}</b><br>{cnt}",
                           showarrow=False, font=dict(color="white", size=12))

    # Arrows from GK start to end
    xs, ys, cds, colors, hovers = [], [], [], [], []
    for i, e in enumerate(gks):
        sx, sy = _v(e.start_x, e.start_y)
        ex, ey = _v(e.end_x, e.end_y)
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
        marker=dict(size=11, color=colors, line=dict(color="white", width=1.5)),
        customdata=cds, hovertext=hovers, hoverinfo="text",
    ))
    result = st.plotly_chart(fig, use_container_width=True, key=key,
                             on_select="rerun", selection_mode="points")
    idx_map = {i: e for i, e in enumerate(gks)}
    _handle_plotly_click(result, key, idx_map, nav_events)


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
                    idx_key = f"__cycle_{key_prefix}_{phase}"
                    i = st.session_state.get(idx_key, -1) + 1
                    if i >= len(group):
                        i = 0
                    st.session_state[idx_key] = i
                    _jump_to_event(group[i], nav_events)


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
                        idx_key = f"__cycle_{key}_{target}"
                        i = st.session_state.get(idx_key, -1) + 1
                        if i >= len(group):
                            i = 0
                        st.session_state[idx_key] = i
                        _jump_to_event(group[i], nav_events)


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

    # Group GKs
    short_gks = [e for e in team_events if _gk_is_short(e)]
    long_gks = [e for e in team_events if not _gk_is_short(e)]

    # Pre-index frames by t for fast window look-up (binary search)
    frames_t = [f["t"] for f in frames]
    import bisect

    def _avg_for_gks(gks):
        """Merge 3-second windows starting at each GK kick time."""
        if not gks:
            return None
        # Gather all positions in the union of windows per player
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
                if not d["xs"]:
                    continue
                out[side][pid] = (sum(d["xs"])/len(d["xs"]),
                                   sum(d["ys"])/len(d["ys"]),
                                   d["s"])
        return out

    short_avg = _avg_for_gks(short_gks)
    long_avg = _avg_for_gks(long_gks)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"*Short GK (n={len(short_gks)})*")
        _draw_avg_positions_pitch(short_avg, match, key="gk_avg_short")
    with col2:
        st.markdown(f"*Long GK (n={len(long_gks)})*")
        _draw_avg_positions_pitch(long_avg, match, key="gk_avg_long")


def _draw_avg_positions_pitch(avg, match, key):
    if not avg:
        st.caption("No data.")
        return
    fig = _plotly_pitch_vertical(fig_height=480)
    for side, color, label in [("h", "#e74c3c", match.home_team),
                                ("a", "#3498db", match.away_team)]:
        xs, ys, txt = [], [], []
        for pid, (x, y, s) in avg[side].items():
            dx, dy = _v(x, y)
            xs.append(dx); ys.append(dy); txt.append(str(s))
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
                        idx_key = f"__cycle_gk_seq_outcome_{cat}"
                        i = st.session_state.get(idx_key, -1) + 1
                        if i >= len(matching):
                            i = 0
                        st.session_state[idx_key] = i
                        _jump_to_event(matching[i], nav_events)


# ================================================================
# VIZ: FREE KICKS (threat map with Boot/Arrow icons)
# ================================================================

def viz_free_kicks(events, team, match):
    team_events = list(events) if team == BOTH_LABEL else [e for e in events if e.team == team]
    if not team_events:
        st.info(f"No free kicks by {team}.")
        return

    fig = _plotly_pitch_vertical(fig_height=540)

    shot_xs, shot_ys, shot_cd, shot_hover = [], [], [], []
    cross_xs, cross_ys, cross_cd, cross_hover = [], [], [], []
    pass_xs, pass_ys, pass_cd, pass_hover = [], [], [], []

    for i, e in enumerate(team_events):
        dx, dy = _v(e.start_x, e.start_y)
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
        x=0, y=PITCH_X - 2,
        text="<b>\u2191 ATTACKING \u2191</b>", showarrow=False,
        font=dict(color="white", size=12),
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
        fig = _plotly_pitch_vertical(fig_height=540)
        succ_x, succ_y, succ_cd, succ_hover = [], [], [], []
        fail_x, fail_y, fail_cd, fail_hover = [], [], [], []
        for i, e in enumerate(team_events):
            dx, dy = _v(e.start_x, e.start_y)
            hover = f"{e.game_time_display} - {_pname(e)}<br>{e.result}"
            if e.result == "SUCCESSFUL":
                succ_x.append(dx); succ_y.append(dy); succ_cd.append(i); succ_hover.append(hover)
            else:
                fail_x.append(dx); fail_y.append(dy); fail_cd.append(i); fail_hover.append(hover)
        for e in team_events:
            sx, sy = _v(e.start_x, e.start_y)
            ex, ey = _v(e.end_x, e.end_y)
            fig.add_shape(type="line", x0=sx, y0=sy, x1=ex, y1=ey,
                          line=dict(color="rgba(255,255,255,0.35)", width=1))
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
        st.markdown(f"**Cross Origins ({title_tag})** \u2014 click a dot to watch")
        result = st.plotly_chart(fig, use_container_width=True, key="cross_map",
                                 on_select="rerun", selection_mode="points")
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

    fig = _plotly_pitch_vertical(fig_height=520)

    # Destination zone counts (attacking-half zones, drawn on vertical pitch)
    dest_counts = Counter()
    for e in flank:
        z = _att_zone_of(e.end_x, e.end_y)
        if z:
            dest_counts[z] += 1

    zones = _build_att_half_zones()
    max_cnt = max(dest_counts.values()) if dest_counts else 1
    for name, (x0, x1, y0, y1) in zones.items():
        cnt = dest_counts.get(name, 0)
        if cnt == 0:
            continue
        alpha = 0.18 + 0.50 * (cnt / max_cnt)
        (dx0, dy0) = _v(x0, y0)
        (dx1, dy1) = _v(x1, y1)
        vx0, vx1 = sorted((dx0, dx1))
        vy0, vy1 = sorted((dy0, dy1))
        fig.add_shape(type="rect", x0=vx0, y0=vy0, x1=vx1, y1=vy1,
                      fillcolor=f"rgba(255,215,0,{alpha:.2f})",
                      line=dict(color="rgba(255,255,255,0.4)", width=1),
                      layer="below")
        fig.add_annotation(x=(vx0+vx1)/2, y=(vy0+vy1)/2, text=f"<b>{cnt}</b>",
                           showarrow=False, font=dict(color="white", size=12))

    # Origin bands on the flank (Deep / Mid / Byline along attacking length)
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
        (dx0, dy0) = _v(x0, y0_f)
        (dx1, dy1) = _v(x1, y1_f)
        vx0, vx1 = sorted((dx0, dx1))
        vy0, vy1 = sorted((dy0, dy1))
        fig.add_shape(type="rect", x0=vx0, y0=vy0, x1=vx1, y1=vy1,
                      fillcolor="rgba(52,152,219,0.18)",
                      line=dict(color="rgba(255,255,255,0.4)", width=1),
                      layer="below")
        fig.add_annotation(x=(vx0+vx1)/2, y=(vy0+vy1)/2,
                            text=f"{band}<br><b>{cnt}</b>",
                            showarrow=False, font=dict(color="white", size=11))

    # Arrows + clickable endpoint dots
    xs, ys, cds, hovers, colors = [], [], [], [], []
    for i, e in enumerate(flank):
        color = "#27ae60" if e.result == "SUCCESSFUL" else "#e74c3c"
        sx, sy = _v(e.start_x, e.start_y)
        ex, ey = _v(e.end_x, e.end_y)
        fig.add_annotation(
            x=ex, y=ey, ax=sx, ay=sy,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=3, arrowsize=1.2, arrowwidth=2,
            arrowcolor=color, text="", opacity=0.8,
        )
        xs.append(ex); ys.append(ey); cds.append(i); colors.append(color)
        hovers.append(f"{e.game_time_display} - {_pname(e)}<br>{e.result}")

    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers",
        marker=dict(size=11, color=colors, line=dict(color="white", width=1.5)),
        customdata=cds, hovertext=hovers, hoverinfo="text", name="",
        showlegend=False,
    ))
    result = st.plotly_chart(fig, use_container_width=True, key=key,
                             on_select="rerun", selection_mode="points")
    idx_map = {i: e for i, e in enumerate(flank)}
    _handle_plotly_click(result, key, idx_map, nav_events)


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
            xaxis=dict(title="Crosses"),
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
                            idx_key = f"__cycle_{chart_key}_{pname}_{zone}"
                            i = st.session_state.get(idx_key, -1) + 1
                            if i >= len(group):
                                i = 0
                            st.session_state[idx_key] = i
                            _jump_to_event(group[i], nav_events)


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

    st.markdown(f"**{title} Map - {label_a} ({len(a_f)}) vs {label_b} ({len(b_f)})**")
    st.caption("Click a shot to watch the clip. Dots are colored by phase.")

    # Zoomed attacking-third pitch with zone counts
    fig = _plotly_pitch(fig_height=520, xrange=(18, 55), yrange=(-34, 34), outline=False)

    # Zone counts across both teams
    zone_counts = Counter()
    for e in all_shown:
        z = _att_zone_of(e.start_x, e.start_y)
        if z:
            zone_counts[z] += 1
    _draw_att_zone_counts(fig, zone_counts)

    trace_groups = []

    def add_trace(evts, name):
        # Color by phase; size by xG; marker star if Goal
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

    # Legend via invisible scatter points
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

    result = st.plotly_chart(fig, use_container_width=True, key=f"shot_map_{title}",
                             on_select="rerun", selection_mode="points")
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

    # Metrics
    cols = st.columns(4)
    cols[0].metric(f"{label_a} Shots", len(a_f))
    cols[1].metric(f"{label_a} xG", f"{sum(e.xg for e in a_f):.2f}")
    cols[2].metric(f"{label_b} Shots", len(b_f))
    cols[3].metric(f"{label_b} xG", f"{sum(e.xg for e in b_f):.2f}")

    # ---- Shot Outcomes & Phases (stacked bar chart) ----
    st.markdown("---")
    st.markdown("**Shot Outcomes & Phases**")
    _render_shot_phase_bar(all_shown, events, key_prefix=f"shot_phase_{title}", match=match)

    # ---- Shots per player (vertical bar chart) ----
    st.markdown("---")
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
                        # Cycle: click once plays first; subsequent clicks advance.
                        idx_key = f"__cycle_{key_prefix}_{row}_{outcome}"
                        i = st.session_state.get(idx_key, -1) + 1
                        if i >= len(group):
                            i = 0
                        st.session_state[idx_key] = i
                        _jump_to_event(group[i], nav_events)


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
                            idx_key = f"__cycle_{chart_key}_{player_name}"
                            i = st.session_state.get(idx_key, -1) + 1
                            if i >= len(pshots):
                                i = 0
                            st.session_state[idx_key] = i
                            _jump_to_event(pshots[i], nav_events)


# ================================================================
# VIZ: BIG CHANCES (match momentum timeline)
# ================================================================

def viz_big_chances(events, team, match):
    if team == BOTH_LABEL:
        team_events = [e for e in events if e.team == match.home_team]
        opp_events = [e for e in events if e.team == match.away_team]
        label_a, label_b = match.home_team, match.away_team
    else:
        team_events = [e for e in events if e.team == team]
        opp_events = [e for e in events if e.team != team]
        label_a, label_b = team, "Opponent"

    # ---- Cumulative xG chart (all shots, not just big chances) ----
    st.markdown(f"**Cumulative xG ({label_a} vs {label_b})**")
    all_shots = [e for e in match.events if e.event_type == "shot"]
    if team == BOTH_LABEL:
        shots_a = [s for s in all_shots if s.team == match.home_team]
        shots_b = [s for s in all_shots if s.team == match.away_team]
    else:
        shots_a = [s for s in all_shots if s.team == team]
        shots_b = [s for s in all_shots if s.team != team]

    def _xg_steps(shots):
        shots = sorted(shots, key=lambda s: s.game_time_ms)
        xs, ys, hovers, cds = [0], [0.0], ["0' - 0.00 xG"], [-1]
        total = 0.0
        for i, s in enumerate(shots):
            minute = s.game_time_ms / 60000
            # Step: first carry previous total at this minute, then add xG
            xs.append(minute); ys.append(total); hovers.append(""); cds.append(-1)
            total += s.xg
            xs.append(minute); ys.append(total)
            hovers.append(f"{s.game_time_display} - {_pname(s)} (+{s.xg:.2f} xG → {total:.2f})")
            cds.append(i)
        # Extend to 95' for readability
        xs.append(95); ys.append(total); hovers.append(""); cds.append(-1)
        return xs, ys, hovers, cds, shots

    xs_a, ys_a, hov_a, cd_a, shots_a = _xg_steps(shots_a)
    xs_b, ys_b, hov_b, cd_b, shots_b = _xg_steps(shots_b)

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
        line=dict(color="#f1c40f" if team == BOTH_LABEL else "#7f8c8d", width=3, shape="hv"),
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
        plot_bgcolor="white",
        showlegend=True,
    )
    xg_result = st.plotly_chart(xg_fig, use_container_width=True, key="xg_cumulative",
                                 on_select="rerun", selection_mode="points")
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

    st.markdown("---")
    st.markdown(f"**Big Chances Timeline ({label_a} vs {label_b})**")

    fig = go.Figure()

    # Baseline timeline
    fig.add_shape(type="line", x0=0, y0=0, x1=95, y1=0,
                  line=dict(color="#2c3e50", width=3))
    # Halftime marker
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
                customdata=cds, hovertext=hovers, hoverinfo="text", name=name,
            ))
            trace_groups.append(evts)

    add_chances(team_events, 0.5, "#e74c3c", label_a)
    add_chances(opp_events, -0.5, "#f1c40f" if team == BOTH_LABEL else "#95a5a6", label_b)

    fig.update_layout(
        xaxis=dict(range=[0, 95], title="Minute",
                   tickmode="array", tickvals=[0, 15, 30, 45, 60, 75, 90]),
        yaxis=dict(range=[-2, 2], visible=False),
        height=260,
        margin=dict(l=30, r=10, t=20, b=40),
        showlegend=True,
        plot_bgcolor="white",
    )
    result = st.plotly_chart(fig, use_container_width=True, key="bc_timeline",
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

    # List
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**{label_a} Big Chances**")
        for e in sorted(team_events, key=lambda x: x.game_time_ms):
            _jump_button(f"{e.game_time_display} - {_pname(e)} (xG {e.xg:.2f})",
                         e, events, key=f"bc_team_{e.game_time_ms}")
    with col2:
        st.markdown(f"**{label_b} Big Chances**")
        for e in sorted(opp_events, key=lambda x: x.game_time_ms):
            _jump_button(f"{e.game_time_display} - {_pname(e)} (xG {e.xg:.2f})",
                         e, events, key=f"bc_opp_{e.game_time_ms}")


# ================================================================
# VIZ: BALL RECOVERIES (pitch thirds bar chart)
# ================================================================

def viz_recoveries(events, team, match):
    team_events = list(events) if team == BOTH_LABEL else [e for e in events if e.team == team]
    if not team_events:
        st.info(f"No ball recoveries by {team}.")
        return

    # Thirds: defensive (own half), middle, attacking
    # Normalize so team attacks toward +x. Thirds at -52.5..-17.5 / -17.5..+17.5 / +17.5..+52.5
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
    fig.update_layout(
        height=320,
        xaxis=dict(title=""),
        yaxis=dict(title="Recoveries"),
        margin=dict(l=40, r=10, t=30, b=40),
        plot_bgcolor="white",
    )
    st.markdown(f"**Recoveries by Pitch Third ({team})**")
    st.plotly_chart(fig, use_container_width=True, key="rec_thirds")

    # Visualize on vertical pitch
    pfig = _plotly_pitch_vertical(fig_height=520)
    # Draw thirds as horizontal bands (since pitch is now vertical: length = y-axis)
    third_bands = [("Defensive Third", (-52.5, -17.5)),
                    ("Middle Third",    (-17.5,  17.5)),
                    ("Attacking Third", ( 17.5,  52.5))]
    max_cnt = max(counts.values()) if counts else 1
    for zone, (x0_len, x1_len) in third_bands:
        cnt = counts.get(zone, 0)
        alpha = 0.2 + 0.55 * (cnt / max_cnt)
        # Transform: length coords become display-y, width is full (-34..34)
        pfig.add_shape(type="rect", x0=-34, y0=x0_len, x1=34, y1=x1_len,
                       fillcolor=f"rgba(255,255,255,{alpha*0.25})",
                       line=dict(color="white", width=1))
        pfig.add_annotation(x=28, y=(x0_len+x1_len)/2,
                             text=f"<b>{cnt}</b>", showarrow=False,
                             font=dict(color="white", size=16))
    # Scatter recoveries (color by team when Both) on vertical pitch
    teams_in = sorted({e.team for e in team_events})
    team_color_map = {t: c for t, c in zip(teams_in, ["#16a085", "#e67e22"])}
    xs, ys = [], []
    for e in team_events:
        dx, dy = _v(e.start_x, e.start_y)
        xs.append(dx); ys.append(dy)
    cds = list(range(len(team_events)))
    hovers = [f"{e.game_time_display} - {e.team} - {_pname(e)}" for e in team_events]
    colors = [team_color_map.get(e.team, "#16a085") for e in team_events]
    pfig.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers",
        marker=dict(size=10, color=colors, line=dict(color="white", width=1.5)),
        customdata=cds, hovertext=hovers, hoverinfo="text", name="Recovery",
    ))
    result = st.plotly_chart(pfig, use_container_width=True, key="rec_pitch",
                             on_select="rerun", selection_mode="points")
    idx_map = {i: e for i, e in enumerate(team_events)}
    _handle_plotly_click(result, "rec_pitch", idx_map, events)

    # ---- Per-team, per-player recovery bar chart ----
    st.markdown("---")
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
        counts = [pc[p] for p in players_sorted]
        bar_fig = go.Figure()
        bar_fig.add_trace(go.Bar(
            x=counts, y=labels, orientation="h",
            marker=dict(color=team_color_map.get(team_name, "#16a085")),
            hovertemplate="<b>%{y}</b>: %{x} recoveries<extra></extra>",
        ))
        bar_fig.update_layout(
            height=max(180, 28 * len(labels) + 60),
            xaxis=dict(title="Recoveries"),
            yaxis=dict(autorange="reversed"),
            margin=dict(l=10, r=10, t=10, b=30),
            plot_bgcolor="white",
            showlegend=False,
        )
        st.plotly_chart(bar_fig, use_container_width=True,
                        key=f"rec_player_bar_{team_name}")

    # Expandable list per third
    st.markdown("---")
    st.markdown("**Clips by Third**")
    for zone in order:
        zone_events = [c[0] for c in classified if c[1] == zone]
        with st.expander(f"{zone} ({len(zone_events)})"):
            for e in zone_events:
                _jump_button(f"{e.game_time_display} - {e.team} - {_pname(e)}", e, events,
                             key=f"rec_{zone}_{e.game_time_ms}_{e.player}")


# ================================================================
# VIZ: INTERCEPTIONS (action dots by player with dropdown)
# ================================================================

def viz_interceptions(events, team, match):
    team_events = list(events) if team == BOTH_LABEL else [e for e in events if e.team == team]
    if not team_events:
        st.info(f"No interceptions by {team}.")
        return

    players = sorted(set(e.player for e in team_events),
                     key=lambda p: -sum(1 for e in team_events if e.player == p))
    player_counts = Counter(e.player for e in team_events)

    # Map display label -> player_name so we can filter internally by canonical name
    display_to_player = {}
    options = ["All Players"]
    for p in players:
        label = f"{_match_player_label(match, p)} ({player_counts[p]})"
        options.append(label)
        display_to_player[label] = p
    selected = st.selectbox("Player filter", options, key="int_player")
    if selected == "All Players":
        filt_events = team_events
    else:
        player_name = display_to_player.get(selected, "")
        filt_events = [e for e in team_events if e.player == player_name]

    st.markdown(f"**Interception Locations ({len(filt_events)})**")

    pfig = _plotly_pitch_vertical(fig_height=520)
    team_color_map = {t: c for t, c in zip(sorted({e.team for e in filt_events}),
                                             ["#2980b9", "#e67e22"])}
    xs, ys = [], []
    for e in filt_events:
        dx, dy = _v(e.start_x, e.start_y)
        xs.append(dx); ys.append(dy)
    cds = list(range(len(filt_events)))
    hovers = [f"{e.game_time_display} - {e.team} - {_pname(e)}" for e in filt_events]
    colors = [team_color_map.get(e.team, "#2980b9") for e in filt_events]
    pfig.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers",
        marker=dict(size=12, color=colors, line=dict(color="white", width=1.5)),
        customdata=cds, hovertext=hovers, hoverinfo="text", name="Interception",
    ))
    result = st.plotly_chart(pfig, use_container_width=True, key="int_pitch",
                             on_select="rerun", selection_mode="points")
    idx_map = {i: e for i, e in enumerate(filt_events)}
    _handle_plotly_click(result, "int_pitch", idx_map, events)

    # Player bar chart (per team when Both)
    st.markdown("---")
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
        counts = [pc[p] for p in players_sorted]
        bar_fig = go.Figure()
        bar_fig.add_trace(go.Bar(
            x=counts, y=labels, orientation="h",
            marker=dict(color=team_color_map.get(team_name, "#2980b9")),
            hovertemplate="<b>%{y}</b>: %{x} interceptions<extra></extra>",
        ))
        bar_fig.update_layout(
            height=max(180, 28 * len(labels) + 60),
            xaxis=dict(title="Interceptions"),
            yaxis=dict(autorange="reversed"),
            margin=dict(l=10, r=10, t=10, b=30),
            plot_bgcolor="white",
            showlegend=False,
        )
        st.plotly_chart(bar_fig, use_container_width=True,
                        key=f"int_player_bar_{team_name}")

    # List
    st.markdown("**Clips**")
    for e in filt_events:
        _jump_button(f"{e.game_time_display} - {e.team} - {_pname(e)}", e, events,
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

    st.markdown(f"**Key Passes ({title_suffix})** - click an arrow endpoint to watch")
    st.caption("Arrows point from the pass origin to the shot-preparing delivery. Tap the endpoint dot.")

    fig = _plotly_pitch_vertical(fig_height=540)

    team_colors = {}
    if team == BOTH_LABEL:
        team_colors[match.home_team] = "#3498db"
        team_colors[match.away_team] = "#e67e22"
    else:
        team_colors[team] = "#3498db"

    xs, ys, cds, hovers, colors = [], [], [], [], []
    for i, e in enumerate(team_events):
        sx, sy = _v(e.start_x, e.start_y)
        ex, ey = _v(e.end_x, e.end_y)
        color = team_colors.get(e.team, "#3498db")
        fig.add_annotation(
            x=ex, y=ey, ax=sx, ay=sy,
            xref="x", yref="y", axref="x", ayref="y",
            arrowhead=3, arrowsize=1.3, arrowwidth=2.2,
            arrowcolor=color, showarrow=True, text="", opacity=0.85,
        )
        xs.append(ex); ys.append(ey); cds.append(i); colors.append(color)
        hovers.append(
            f"{e.game_time_display} - {e.team}<br>{_pname(e)} \u2192 {_rname(e) or '?'}<br>{e.sub_type} ({e.result})"
        )

    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers",
        marker=dict(size=14, color=colors, line=dict(color="white", width=2)),
        customdata=cds, hovertext=hovers, hoverinfo="text",
        name="Key Pass",
    ))
    fig.update_layout(showlegend=False)

    result = st.plotly_chart(fig, use_container_width=True, key="kp_map",
                             on_select="rerun", selection_mode="points")
    idx_map = {i: e for i, e in enumerate(team_events)}
    _handle_plotly_click(result, "kp_map", idx_map, events)

    # List breakdown by passer
    st.markdown("---")
    passer_counts = Counter(e.player for e in team_events)
    st.markdown("**Passers**")
    for player, cnt in passer_counts.most_common():
        label_p = _match_player_label(match, player)
        with st.expander(f"{label_p} ({cnt})"):
            for e in [x for x in team_events if x.player == player]:
                tgt = _rname(e) or "?"
                label = f"{e.game_time_display} - {e.team} → {tgt}"
                _jump_button(label, e, events, key=f"kp_{e.game_time_ms}_{e.player}")


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
                            idx_key = f"__cycle_{chart_key}_{pname}_{kind}"
                            i = st.session_state.get(idx_key, -1) + 1
                            if i >= len(target):
                                i = 0
                            st.session_state[idx_key] = i
                            _jump_to_event(target[i], nav_events)


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
                            idx_key = f"__cycle_{chart_key}_{pname}"
                            i = st.session_state.get(idx_key, -1) + 1
                            if i >= len(recs):
                                i = 0
                            st.session_state[idx_key] = i
                            _jump_to_event(recs[i], nav_events)


def _render_box_entries(entries, nav_events, team, match, key_prefix):
    if not entries:
        st.caption("No penalty box entries.")
        return

    col_map, col_bar = st.columns([1.2, 1])

    with col_map:
        fig = _plotly_pitch_vertical(fig_height=520)
        xs, ys, cds, hovers, colors = [], [], [], [], []
        for i, e in enumerate(entries):
            color = "#27ae60" if e.result == "SUCCESSFUL" else "#e74c3c"
            kind = "Carry" if e.event_type == "carry" else "Pass"
            sx, sy = _v(e.start_x, e.start_y)
            ex, ey = _v(e.end_x, e.end_y)
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
            marker=dict(size=11, color=colors, line=dict(color="white", width=1.5)),
            customdata=cds, hovertext=hovers, hoverinfo="text",
        ))
        fig.update_layout(showlegend=False)
        result = st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_map",
                                 on_select="rerun", selection_mode="points")
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
                            idx_key = f"__cycle_{chart_key}_{pname}"
                            i = st.session_state.get(idx_key, -1) + 1
                            if i >= len(plist):
                                i = 0
                            st.session_state[idx_key] = i
                            _jump_to_event(plist[i], nav_events)


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


def _render_pass_zone_map_vertical(passes, nav_events, key, dest_zone_fn, zone_rects):
    """Vertical pitch version of the zonal pass map. Zones are attacking-half
    rectangles (defined in SciSports coords) drawn on the vertical pitch via _v().
    The GK/goal sits at the TOP of the pitch (team attacks upward)."""
    if not passes:
        st.caption("No passes in this category.")
        return
    fig = _plotly_pitch_vertical(fig_height=560)

    # Zone counts (success vs fail)
    zone_stats = {}
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
    # Zone rects are in (x0, x1, y0, y1) SciSports coords — draw on vertical pitch
    # by transforming corners via _v(x, y) = (-y, x). Rectangle becomes another
    # rectangle because the transform is axis-swap + sign flip.
    for name, (x0, x1, y0, y1) in zone_rects.items():
        stat = zone_stats.get(name)
        if not stat:
            continue
        total = stat["succ"] + stat["fail"]
        alpha = 0.18 + 0.45 * (total / max_total)
        # Transform corners
        (dx0, dy0) = _v(x0, y0)
        (dx1, dy1) = _v(x1, y1)
        vx0, vx1 = sorted((dx0, dx1))
        vy0, vy1 = sorted((dy0, dy1))
        fig.add_shape(type="rect", x0=vx0, y0=vy0, x1=vx1, y1=vy1,
                      fillcolor=f"rgba(255,215,0,{alpha:.2f})",
                      line=dict(color="rgba(255,255,255,0.4)", width=1),
                      layer="below")
        fig.add_annotation(
            x=(vx0+vx1)/2, y=(vy0+vy1)/2,
            text=f"<b>{total}</b><br><span style='font-size:10px'>"
                 f"<span style='color:#27ae60'>{stat['succ']}</span>/"
                 f"<span style='color:#e74c3c'>{stat['fail']}</span></span>",
            showarrow=False, font=dict(color="white", size=12),
        )

    # Arrows from start -> end (transformed to vertical coords)
    xs, ys, cds, colors, hovers = [], [], [], [], []
    for i, e in enumerate(passes):
        color = "#27ae60" if e.result == "SUCCESSFUL" else "#e74c3c"
        sx, sy = _v(e.start_x, e.start_y)
        ex, ey = _v(e.end_x, e.end_y)
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
        marker=dict(size=9, color=colors, line=dict(color="white", width=1)),
        customdata=cds, hovertext=hovers, hoverinfo="text",
        showlegend=False,
    ))
    result = st.plotly_chart(fig, use_container_width=True, key=key,
                             on_select="rerun", selection_mode="points")
    idx_map = {i: e for i, e in enumerate(passes)}
    _handle_plotly_click(result, key, idx_map, nav_events)


def _render_progression_arrows_vertical(passes, nav_events, key):
    """Plain arrow-map on a vertical pitch (no zones). Used for own\u2192mid view."""
    if not passes:
        st.caption("No passes in this category.")
        return
    fig = _plotly_pitch_vertical(fig_height=520)
    xs, ys, cds, colors, hovers = [], [], [], [], []
    for i, e in enumerate(passes):
        color = "#27ae60" if e.result == "SUCCESSFUL" else "#e74c3c"
        sx, sy = _v(e.start_x, e.start_y)
        ex, ey = _v(e.end_x, e.end_y)
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
        marker=dict(size=9, color=colors, line=dict(color="white", width=1)),
        customdata=cds, hovertext=hovers, hoverinfo="text",
        showlegend=False,
    ))
    result = st.plotly_chart(fig, use_container_width=True, key=key,
                             on_select="rerun", selection_mode="points")
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
                                idx_key = f"__cycle_{chart_key}_{pname}"
                                i = st.session_state.get(idx_key, -1) + 1
                                if i >= len(plist):
                                    i = 0
                                st.session_state[idx_key] = i
                                _jump_to_event(plist[i], nav_events)

    with col_map:
        fig = _plotly_pitch_vertical(fig_height=520)
        xs, ys, cds, hovers, colors = [], [], [], [], []
        for i, e in enumerate(passes):
            color = "#27ae60" if e.result == "SUCCESSFUL" else "#e74c3c"
            sx, sy = _v(e.start_x, e.start_y)
            ex, ey = _v(e.end_x, e.end_y)
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
            marker=dict(size=8, color=colors, line=dict(color="white", width=1)),
            customdata=cds, hovertext=hovers, hoverinfo="text",
            showlegend=False,
        ))
        result = st.plotly_chart(fig, use_container_width=True,
                                  key=f"{key_prefix}_map",
                                  on_select="rerun", selection_mode="points")
        idx_map = {i: e for i, e in enumerate(passes)}
        _handle_plotly_click(result, f"{key_prefix}_map", idx_map, nav_events)


# ================================================================
# VIZ: OFFENSIVE TRANSITIONS (recovery \u2192 what happens next)
# ================================================================

def _regain_events(match):
    """All recoveries/interceptions (ball regains)."""
    return [e for e in match.events if e.event_type in ("recovery", "interception")]


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
                        idx_key = f"__cycle_{key_prefix}_{third}_{out}"
                        i = st.session_state.get(idx_key, -1) + 1
                        if i >= len(group):
                            i = 0
                        st.session_state[idx_key] = i
                        _jump_to_event(group[i], nav_events)


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
                            idx_key = f"__cycle_{chart_key}_{pname}_{direction}"
                            i = st.session_state.get(idx_key, -1) + 1
                            if i >= len(group):
                                i = 0
                            st.session_state[idx_key] = i
                            _jump_to_event(group[i], nav_events)


def _render_regain_threat_map(records, nav_events, key):
    """Vertical pitch: dot at regain, arrow to chain end, color by time elapsed."""
    if not records:
        st.caption("No transitions to map.")
        return
    fig = _plotly_pitch_vertical(fig_height=560)

    def _elapsed_sec(chain):
        if len(chain) < 2:
            return 0.0
        return max(0.0, (chain[-1].game_time_ms - chain[0].game_time_ms) / 1000.0)

    xs, ys, cds, colors, hovers = [], [], [], [], []
    for i, rec in enumerate(records):
        r = rec["regain"]
        chain = rec["chain"]
        end_ev = chain[-1] if chain else r
        # Color gradient: <10s red (fast break), <25s orange, else blue
        elapsed = _elapsed_sec(chain)
        if elapsed < 10:
            color = "#e74c3c"
        elif elapsed < 25:
            color = "#f39c12"
        else:
            color = "#3498db"
        sx, sy = _v(r.start_x, r.start_y)
        ex, ey = _v(end_ev.end_x or end_ev.start_x, end_ev.end_y or end_ev.start_y)
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

def _loss_events(match, team_filter=None):
    """Detect where we lost the ball. A loss = the last event of a sequence
    by this team, if the next event is by the opponent (via recovery/interception).
    We use the opponent's recovery location as the loss point (SciSports logs it
    from the recoverer's perspective)."""
    # Build a (team, game_time) index
    ev_sorted = sorted(match.events, key=lambda e: (e.game_time_ms, 0))
    # For each recovery by opponent X, find the most recent event by the other team
    losses = []
    for e in ev_sorted:
        if e.event_type not in ("recovery", "interception"):
            continue
        # Opponent team is the one that lost the ball
        for j in range(ev_sorted.index(e) - 1, -1, -1):
            prev = ev_sorted[j]
            if prev.team and prev.team != e.team:
                # prev represents the losing team's last action
                losses.append({
                    "team_lost": prev.team,
                    "loss_time_ms": prev.game_time_ms,
                    "loss_x": prev.end_x or prev.start_x,
                    "loss_y": prev.end_y or prev.start_y,
                    "regain_event": e,
                    "loss_event": prev,
                })
                break
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
        _render_dual_media_viewer(losses, match, key="def_tr_dual")


def _render_counter_press_map(losses, nav_events, key):
    """Vertical pitch: dot at each loss. Green=counter-pressed, red=dropped off.
    Small radius circle around each dot to indicate press zone."""
    if not losses:
        st.caption("No losses to map.")
        return
    fig = _plotly_pitch_vertical(fig_height=560)

    xs, ys, cds, colors, hovers = [], [], [], [], []
    for i, l in enumerate(losses):
        dx, dy = _v(l["loss_x"], l["loss_y"])
        color = "#27ae60" if l["counter_pressed"] else "#e74c3c"
        # Draw a small ring to indicate the 8m press radius (scaled to display)
        r_display = 8.0  # in metres, same scale on both axes
        fig.add_shape(type="circle",
                       x0=dx - r_display, y0=dy - r_display,
                       x1=dx + r_display, y1=dy + r_display,
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


def _render_dual_media_viewer(losses, match, key):
    """Side-by-side: mp4 clip + 2D black-and-white tracking canvas driven by a
    shared scrub slider. Not a true video-driven sync (Streamlit can't subscribe
    to a native video's timeupdate events), but the slider scrubs both."""
    st.caption("Select a loss event. The 2D canvas reads positions.json at the "
                "selected timestamp. The video alongside is cued to the clip.")
    if not losses:
        st.caption("No losses.")
        return

    labels = [f"{l['regain_event'].game_time_display} \u2014 {l['team_lost']}"
               for l in losses]
    idx = st.selectbox("Select loss event", range(len(losses)),
                        format_func=lambda i: labels[i], key=f"{key}_sel")
    loss = losses[idx]

    col_v, col_m = st.columns([1, 1])

    with col_v:
        st.markdown("**Match Video**")
        show_video_for_event(match, loss["regain_event"])

    with col_m:
        st.markdown("**2D Tracking (black & white)**")
        pos_path = _positions_path_for_match(match)
        if not pos_path:
            st.caption("No tracking data for this match.")
            return
        frames = _cached_positions(str(pos_path))
        if not frames:
            st.caption("Tracking data empty.")
            return
        # Scrub window: 5s before loss \u2192 10s after
        t0 = max(0, loss["loss_time_ms"] - 5000)
        t1 = loss["loss_time_ms"] + 10000
        import bisect
        frame_ts = [f.get("t", 0) for f in frames]
        slider_ms = st.slider(
            "Clip time (ms)",
            min_value=int(t0), max_value=int(t1),
            value=int(loss["loss_time_ms"]),
            step=100, key=f"{key}_slider",
        )
        fi = bisect.bisect_left(frame_ts, slider_ms)
        fi = min(fi, len(frames) - 1)
        frame = frames[fi] if frames else {}
        _draw_bw_tracking_pitch(frame, key=f"{key}_canvas")


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
    "corner": viz_corners,
    "goal_kick": viz_goal_kicks,
    "free_kick": viz_free_kicks,
    "cross": viz_crosses,
    "key_pass": viz_key_passes,
    "goal": viz_goals,
    "shot_on_target": lambda e, t, m: viz_shots(e, t, m, "Shots on Target"),
    "shot": lambda e, t, m: viz_shots(e, t, m, "All Shots"),
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
    return get_events_by_type(match, event_type)


# ================================================================
# MAIN APP
# ================================================================

def main():
    st.set_page_config(page_title="FCDB Match Tracker", layout="wide")

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

    st.title("FCDB Match Tracker")

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
    if selected_team == BOTH_LABEL:
        events = all_events
    else:
        events = [e for e in all_events if e.team == selected_team]

    # Detect event_type/team/match changes and reset event_selector widget state
    current_ctx = (selected_match_idx, event_type, selected_team)
    if st.session_state.get("__last_ctx") != current_ctx:
        st.session_state.pop("event_selector", None)
        st.session_state["selected_event_idx"] = 0
        st.session_state["__last_ctx"] = current_ctx

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

    with col_viz:
        st.markdown("**Analysis**")
        viz_fn = VIZ_MAP.get(event_type)
        if viz_fn:
            viz_fn(events, selected_team, match)


if __name__ == "__main__":
    main()
