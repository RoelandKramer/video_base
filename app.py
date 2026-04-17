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

from event_parser import discover_matches, get_events_by_type, Event, Match
from video_utils import extract_clip

DATA_DIR = Path(__file__).parent

EVENT_TYPES = {
    "Corners": "corner",
    "Goal Kicks": "goal_kick",
    "Free Kicks": "free_kick",
    "Crosses": "cross",
    "Key Passes": "key_pass",
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

def viz_goal_kicks(events, team, match):
    team_events = list(events) if team == BOTH_LABEL else [e for e in events if e.team == team]
    if not team_events:
        st.info(f"No goal kicks by {team}.")
        return

    # Classify each goal kick by end zone
    def classify(e):
        dx = e.end_x - e.start_x
        dy = e.end_y - e.start_y
        dist = (dx*dx + dy*dy) ** 0.5
        # Short = under 32m total travel; Long otherwise
        if dist < 32:
            return "Short Left" if e.end_y > 0 else "Short Right"
        # Long: classify by landing y
        if abs(e.end_y) < 12:
            return "Long Center"
        return "Long Left Wing" if e.end_y > 0 else "Long Right Wing"

    classified = [(e, classify(e)) for e in team_events]
    zone_counts = Counter(c[1] for c in classified)

    # Draw pitch and highlight zones
    fig = _plotly_pitch(fig_height=420)
    # Normalize start: GK always from the team's own goal. For pitch viz, put ball at left goal.
    # Use SciSports end positions directly, mirrored so team always attacks to the right.
    zone_shapes = {
        # (x0, x1, y0, y1) on pitch assuming team attacks left->right (GK from left side)
        "Short Left":       (-52.5, -30,  0,  34),
        "Short Right":      (-52.5, -30, -34,  0),
        "Long Left Wing":   (-30,    52.5, 12,  34),
        "Long Center":      (-30,    52.5, -12, 12),
        "Long Right Wing":  (-30,    52.5, -34, -12),
    }
    max_count = max(zone_counts.values()) if zone_counts else 1
    shape_x, shape_y, shape_cd, shape_hover = [], [], [], []
    for zone, (x0, x1, y0, y1) in zone_shapes.items():
        cnt = zone_counts.get(zone, 0)
        alpha = 0.15 + 0.65 * (cnt / max_count) if cnt else 0.1
        fig.add_shape(type="rect", x0=x0, y0=y0, x1=x1, y1=y1,
                      fillcolor=f"rgba(231, 76, 60, {alpha})",
                      line=dict(color="white", width=1.5))
        fig.add_annotation(x=(x0+x1)/2, y=(y0+y1)/2,
                           text=f"<b>{zone}</b><br>{cnt}",
                           showarrow=False,
                           font=dict(color="white", size=13))

    st.markdown(f"**Goal Kick Distribution ({team})**")
    st.plotly_chart(fig, use_container_width=True, key="gk_zones")

    st.markdown("**Clips by Zone**")
    for zone, cnt in zone_counts.most_common():
        zone_events = [c[0] for c in classified if c[1] == zone]
        with st.expander(f"{zone} ({cnt})"):
            _event_buttons(zone_events, events, f"gk_{zone.replace(' ', '_')}")

    # Receivers
    st.markdown("---")
    st.markdown("**Receivers**")
    recvs = Counter(e.receiver for e in team_events if e.receiver and e.receiver != "NOT_APPLICABLE")
    if recvs:
        fig2, ax = plt.subplots(figsize=(7, 3))
        players = [p for p, _ in recvs.most_common()]
        counts = [c for _, c in recvs.most_common()]
        ax.barh(range(len(players)), counts, color="#3498db")
        ax.set_yticks(range(len(players)))
        ax.set_yticklabels([_match_player_label(match, p) for p in players], fontsize=9)
        ax.set_xlabel("Times Received")
        ax.invert_yaxis()
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)


# ================================================================
# VIZ: FREE KICKS (threat map with Boot/Arrow icons)
# ================================================================

def viz_free_kicks(events, team, match):
    team_events = list(events) if team == BOTH_LABEL else [e for e in events if e.team == team]
    if not team_events:
        st.info(f"No free kicks by {team}.")
        return

    fig = _plotly_pitch(fig_height=420)

    # Normalize positions so team always attacks toward x > 0
    shot_xs, shot_ys, shot_cd, shot_hover = [], [], [], []
    cross_xs, cross_ys, cross_cd, cross_hover = [], [], [], []
    pass_xs, pass_ys, pass_cd, pass_hover = [], [], [], []

    for i, e in enumerate(team_events):
        x, y = _normalize_pos(e.start_x, e.start_y)
        hover = f"{e.game_time_display} - {_pname(e)}<br>{e.sub_type} ({e.result})"
        if e.sub_type == "SHOT_FREE_KICK":
            shot_xs.append(x); shot_ys.append(y); shot_cd.append(i); shot_hover.append(hover)
        elif "CROSS" in e.sub_type:
            cross_xs.append(x); cross_ys.append(y); cross_cd.append(i); cross_hover.append(hover)
        else:
            pass_xs.append(x); pass_ys.append(y); pass_cd.append(i); pass_hover.append(hover)

    if shot_xs:
        fig.add_trace(go.Scatter(
            x=shot_xs, y=shot_ys,
            mode="markers",
            marker=dict(size=22, color="#f39c12", symbol="star",
                        line=dict(color="white", width=2)),
            customdata=shot_cd,
            hovertext=shot_hover, hoverinfo="text",
            name="FK Shot (boot)",
        ))
    if cross_xs:
        fig.add_trace(go.Scatter(
            x=cross_xs, y=cross_ys,
            mode="markers",
            marker=dict(size=20, color="#3498db", symbol="arrow-right",
                        line=dict(color="white", width=2)),
            customdata=cross_cd,
            hovertext=cross_hover, hoverinfo="text",
            name="FK Cross",
        ))
    if pass_xs:
        fig.add_trace(go.Scatter(
            x=pass_xs, y=pass_ys,
            mode="markers",
            marker=dict(size=14, color="#95a5a6", symbol="circle",
                        line=dict(color="white", width=1)),
            customdata=pass_cd,
            hovertext=pass_hover, hoverinfo="text",
            name="FK Pass",
        ))
    fig.update_layout(showlegend=True, legend=dict(bgcolor="rgba(0,0,0,0.3)",
                                                    font=dict(color="white")))
    title_tag = "Both" if team == BOTH_LABEL else team
    st.markdown(f"**Free Kick Threat Map ({title_tag})** - stars=shots, arrows=crosses")
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

    fig = _plotly_pitch(fig_height=420)

    succ_x, succ_y, succ_cd, succ_hover = [], [], [], []
    fail_x, fail_y, fail_cd, fail_hover = [], [], [], []

    for i, e in enumerate(team_events):
        x, y = _normalize_pos(e.start_x, e.start_y)
        hover = f"{e.game_time_display} - {_pname(e)}<br>{e.result}"
        if e.result == "SUCCESSFUL":
            succ_x.append(x); succ_y.append(y); succ_cd.append(i); succ_hover.append(hover)
        else:
            fail_x.append(x); fail_y.append(y); fail_cd.append(i); fail_hover.append(hover)

    # Add arrows from start to end (short arrows for visual context)
    for e in team_events:
        sx, sy = _normalize_pos(e.start_x, e.start_y)
        ex, ey = _normalize_pos(e.end_x, e.end_y)
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
    fig.update_layout(showlegend=True, legend=dict(bgcolor="rgba(0,0,0,0.3)",
                                                    font=dict(color="white")))

    title_tag = "Both" if team == BOTH_LABEL else team
    st.markdown(f"**Cross Origins ({title_tag})** - click a dot to watch")
    result = st.plotly_chart(fig, use_container_width=True, key="cross_map",
                             on_select="rerun", selection_mode="points")
    idx_map = {i: e for i, e in enumerate(team_events)}
    _handle_plotly_click(result, "cross_map", idx_map, events)

    # ---- Top-5 biggest chances created from crosses (per team) ----
    st.markdown("---")
    st.markdown("**Top 5 Biggest Chances From Crosses (xA)**")
    _render_top_chances_from_passes(team_events, team, match, events,
                                     key_prefix="cross_xa", score_fn=_xa_score,
                                     score_label="xA")


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

    # Zoomed attacking-third pitch (box centered vertically, no outer frame)
    fig = _plotly_pitch(fig_height=520, xrange=(18, 55), yrange=(-30, 30), outline=False)

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
            x, y = _normalize_pos(e.start_x, e.start_y)
            xs.append(x); ys.append(y); cds.append(i)
            hovers.append(f"{e.game_time_display} - {_pname(e)}<br>xG: {e.xg:.2f}")
            assist = find_assist(e)
            if assist:
                ax_, ay_ = _normalize_pos(assist.start_x, assist.start_y)
                fig.add_annotation(
                    x=x, y=y, ax=ax_, ay=ay_,
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
        color_a, color_b = "#e74c3c", "#f1c40f"
    else:
        group_a = [e for e in events if e.team == team]
        group_b = [e for e in events if e.team != team]
        label_a, label_b = team, "Opponent"
        color_a, color_b = "#e74c3c", "#95a5a6"

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

    st.markdown(f"**{title} Map - {label_a} ({len(a_f)}) vs {label_b} ({len(b_f)})**")
    st.caption("Click a shot to watch the clip")

    # Zoomed attacking-third pitch (box centered vertically, no outer frame)
    fig = _plotly_pitch(fig_height=520, xrange=(18, 55), yrange=(-30, 30), outline=False)

    trace_groups = []

    def add_trace(evts, color, name, alpha=0.9):
        xs, ys, sizes, cds, hovers, symbols = [], [], [], [], [], []
        for i, e in enumerate(evts):
            x, y = _normalize_pos(e.start_x, e.start_y)
            xs.append(x); ys.append(y)
            sizes.append(max(10, e.xg * 80))
            cds.append(i)
            sym = "star" if e.result == "SUCCESSFUL" else "circle"
            symbols.append(sym)
            hovers.append(f"{e.game_time_display} - {_pname(e)}<br>xG: {e.xg:.3f}<br>{e.sub_type} ({e.result})")
        if xs:
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="markers",
                marker=dict(size=sizes, color=color,
                            symbol=symbols,
                            line=dict(color="white", width=1.5),
                            opacity=alpha),
                customdata=cds, hovertext=hovers, hoverinfo="text", name=name,
            ))
            trace_groups.append(evts)

    add_trace(b_f, color_b, label_b, alpha=0.75 if team == BOTH_LABEL else 0.5)
    add_trace(a_f, color_a, label_a, alpha=0.95)
    fig.update_layout(showlegend=True,
                      legend=dict(bgcolor="rgba(0,0,0,0.3)", font=dict(color="white")))

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

    # Visualize on pitch
    pfig = _plotly_pitch(fig_height=320)
    for zone, (x0, x1) in zip(order, [(-52.5, -17.5), (-17.5, 17.5), (17.5, 52.5)]):
        cnt = counts.get(zone, 0)
        alpha = 0.2 + 0.55 * (cnt / max(counts.values())) if counts else 0.2
        pfig.add_shape(type="rect", x0=x0, y0=-34, x1=x1, y1=34,
                       fillcolor=f"rgba(255,255,255,{alpha*0.3})",
                       line=dict(color="white", width=1))
        pfig.add_annotation(x=(x0+x1)/2, y=28, text=f"<b>{cnt}</b>",
                            showarrow=False, font=dict(color="white", size=16))
    # Scatter recoveries (color by team when Both)
    teams_in = sorted({e.team for e in team_events})
    team_color_map = {t: c for t, c in zip(teams_in, ["#16a085", "#e67e22"])}
    xs = [e.start_x for e in team_events]
    ys = [e.start_y for e in team_events]
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

    pfig = _plotly_pitch(fig_height=420)
    # Color by team when Both
    team_color_map = {t: c for t, c in zip(sorted({e.team for e in filt_events}),
                                             ["#2980b9", "#e67e22"])}
    xs = [e.start_x for e in filt_events]
    ys = [e.start_y for e in filt_events]
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

    fig = _plotly_pitch(fig_height=520, xrange=(-5, 55), yrange=(-34, 34), outline=False)

    # Colour by team so Both mode is readable
    team_colors = {}
    if team == BOTH_LABEL:
        team_colors[match.home_team] = "#3498db"
        team_colors[match.away_team] = "#e67e22"
    else:
        team_colors[team] = "#3498db"
        # Any non-selected team events would be opponents (shouldn't happen after filter)

    # Arrows from start -> end, plus clickable dot at the END (delivery point)
    xs, ys, cds, hovers, colors = [], [], [], [], []
    for i, e in enumerate(team_events):
        sx, sy = _normalize_pos(e.start_x, e.start_y)
        ex, ey = _normalize_pos(e.end_x, e.end_y)
        color = team_colors.get(e.team, "#3498db")
        fig.add_annotation(
            x=ex, y=ey, ax=sx, ay=sy,
            xref="x", yref="y", axref="x", ayref="y",
            arrowhead=3, arrowsize=1.3, arrowwidth=2.2,
            arrowcolor=color, showarrow=True, text="", opacity=0.85,
        )
        xs.append(ex); ys.append(ey); cds.append(i); colors.append(color)
        hovers.append(
            f"{e.game_time_display} - {e.team}<br>{_pname(e)} → {_rname(e) or '?'}<br>{e.sub_type} ({e.result})"
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
}


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

    all_events = get_events_by_type(match, event_type)
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

            # Details
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
