"""Streamlit app for viewing and analysing corner kick video clips from KKD matches."""

import streamlit as st
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Arc
from collections import defaultdict
from pathlib import Path
import json
import sys

from event_parser import discover_matches, Corner, Match
from video_utils import extract_clip

DATA_DIR = Path(__file__).parent

# Add research folder to path for corner_base imports
RESEARCH_DIR = Path(r"c:\Users\20203834\OneDrive\Master Data Science Entrepreneurship"
                    r"\FC Den Bosch - Thesis\Def. Corners Onderzoek")
sys.path.insert(0, str(RESEARCH_DIR))


# ================================================================
# PITCH DRAWING (adapted from corner_final_viz.py)
# ================================================================

def draw_corner_half(ax, goal_x=52.5, color="white", linewidth=1.5):
    sign = 1 if goal_x > 0 else -1
    rect = plt.Rectangle((-52.5, -34), 105, 68, linewidth=linewidth,
                          edgecolor=color, facecolor="#2d8a4e", zorder=0)
    ax.add_patch(rect)
    gx = sign * 52.5
    ax.plot([gx, gx - sign*16.5], [-20.16, -20.16], color=color, lw=linewidth, zorder=1)
    ax.plot([gx - sign*16.5, gx - sign*16.5], [-20.16, 20.16], color=color, lw=linewidth, zorder=1)
    ax.plot([gx, gx - sign*16.5], [20.16, 20.16], color=color, lw=linewidth, zorder=1)
    ax.plot([gx, gx - sign*5.5], [-9.16, -9.16], color=color, lw=linewidth, zorder=1)
    ax.plot([gx - sign*5.5, gx - sign*5.5], [-9.16, 9.16], color=color, lw=linewidth, zorder=1)
    ax.plot([gx, gx - sign*5.5], [9.16, 9.16], color=color, lw=linewidth, zorder=1)
    ax.plot([gx, gx + sign*1.5], [-3.66, -3.66], color=color, lw=linewidth+0.5, zorder=1)
    ax.plot([gx + sign*1.5, gx + sign*1.5], [-3.66, 3.66], color=color, lw=linewidth+0.5, zorder=1)
    ax.plot([gx, gx + sign*1.5], [3.66, 3.66], color=color, lw=linewidth+0.5, zorder=1)
    ax.plot(gx - sign*11, 0, "o", color=color, markersize=3, zorder=1)
    if sign > 0:
        ax.set_xlim(25, 56)
    else:
        ax.set_xlim(-56, -25)
    ax.set_ylim(-36, 36)
    ax.set_aspect("equal")
    ax.axis("off")


# ================================================================
# CORNER ROLE ANALYSIS (uses corner_base infrastructure)
# ================================================================

@st.cache_resource
def load_role_model():
    """Train the corner role detection model from ground-truth data."""
    try:
        from corner_base import (
            setup_data, build_precomps, build_ml_features,
            V6_FEAT_COLS, EXCL_MATCHES
        )
        from sklearn.ensemble import ExtraTreesClassifier

        gt, corner_meta, gt_fast = setup_data()
        precomp = build_precomps(corner_meta, win=5000)
        fdf = build_ml_features(gt_fast, precomp, EXCL_MATCHES)

        if fdf.empty:
            return None, None

        model = ExtraTreesClassifier(
            n_estimators=500, max_depth=6, min_samples_leaf=2, random_state=42
        )
        model.fit(fdf[V6_FEAT_COLS].values, fdf["gt_role"].values)
        return model, V6_FEAT_COLS
    except Exception as e:
        st.warning(f"Could not load role detection model: {e}")
        return None, None


def analyse_corner(match: Match, corner: Corner, model, feat_cols):
    """Run role detection on a single corner. Returns predictions dict and precomp."""
    try:
        from corner_base import (
            load_events, load_positions, get_gk_shirts,
            get_team_labels, precompute, MATCH_FILES, DATA_DIR as CB_DATA_DIR
        )
        from corner_final_viz import predict_corner_roles

        mid = match.match_id

        # Register this match's files if not already known
        if mid not in MATCH_FILES:
            ev_file = match.json_path
            pos_file = DATA_DIR / match.json_path.name.replace("Events", "Positions")
            if ev_file and pos_file.exists():
                MATCH_FILES[mid] = (ev_file.name, pos_file.name)
                # Also point DATA_DIR to our folder
                import corner_base
                corner_base.DATA_DIR = DATA_DIR
            else:
                return None, None

        # Determine defending team (opponent of corner taker)
        meta = load_events(mid)["metaData"]
        home = meta["homeTeamName"]
        away = meta["awayTeamName"]
        # Corner taker's team
        taker_team = corner.team
        if taker_team.lower() in home.lower() or home.lower() in taker_team.lower():
            def_label, att_label = "a", "h"
        else:
            def_label, att_label = "h", "a"

        gks = get_gk_shirts(mid)
        load_positions(mid)
        pc = precompute(mid, corner.game_time_ms, def_label, att_label, gks, 5000)
        preds = predict_corner_roles(pc, model, feat_cols)
        return preds, pc

    except Exception as e:
        st.warning(f"Role analysis failed: {e}")
        return None, None


def plot_corner_roles(pc, predictions, title=""):
    """Create a matplotlib figure showing defender roles on the pitch."""
    fig, ax = plt.subplots(figsize=(10, 8))

    ds_list = pc["def_shirts"]
    as_list = pc["att_shirts"]
    def_pos = pc["def_pos"]
    att_pos = pc["att_pos"]
    t1 = pc["kick_time"]

    # Detect goal end
    xs = [def_pos.get((ds, t1), (0, 0))[0] for ds in ds_list if (ds, t1) in def_pos]
    goal_x = 52.5 if (not xs or np.mean(xs) > 0) else -52.5
    draw_corner_half(ax, goal_x)

    # Attackers (red circles)
    for as_ in as_list:
        ap = att_pos.get((as_, t1))
        if ap:
            ax.plot(ap[0], ap[1], "o", color="#e74c3c", markersize=14, zorder=5,
                    markeredgecolor="white", markeredgewidth=1.5)
            ax.text(ap[0], ap[1], str(as_), ha="center", va="center",
                    fontsize=8, fontweight="bold", color="white", zorder=6)

    # Defenders (blue=marking, orange=zonal)
    for ds in ds_list:
        dp = def_pos.get((ds, t1))
        if not dp:
            continue
        pred = predictions.get(ds, ("zonal", None))
        role, target = pred

        if role == "marking":
            color = "#3498db"
            if target:
                ap = att_pos.get((target, t1))
                if ap:
                    ax.annotate("", xy=(ap[0], ap[1]), xytext=(dp[0], dp[1]),
                                arrowprops=dict(arrowstyle="->", color="#3498db",
                                                lw=2, ls="--"), zorder=3)
        else:
            color = "#f39c12"

        ax.plot(dp[0], dp[1], "s", color=color, markersize=14, zorder=5,
                markeredgecolor="white", markeredgewidth=1.5)
        ax.text(dp[0], dp[1], str(ds), ha="center", va="center",
                fontsize=8, fontweight="bold", color="white", zorder=6)

    # Ball
    if pc.get("ball_xy"):
        bxy = pc["ball_xy"]
        ax.plot(bxy[0], bxy[1], "o", color="white", markersize=10, zorder=7,
                markeredgecolor="black", markeredgewidth=1.5)

    # Legend
    marker_patch = mpatches.Patch(color="#3498db", label="Man-marking")
    zonal_patch = mpatches.Patch(color="#f39c12", label="Zonal")
    att_patch = mpatches.Patch(color="#e74c3c", label="Attacker")
    ax.legend(handles=[marker_patch, zonal_patch, att_patch],
              loc="upper left", fontsize=9, framealpha=0.9)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)

    plt.tight_layout()
    return fig


def plot_match_summary(all_predictions, match_name):
    """Create summary figures for all corners in a match."""
    figs = {}

    # Collect data
    corners_data = []
    player_roles = defaultdict(lambda: {"marking": 0, "zonal": 0})

    for corner_label, preds in all_predictions.items():
        n_mark = sum(1 for r, _ in preds.values() if r == "marking")
        n_zonal = sum(1 for r, _ in preds.values() if r == "zonal")
        corners_data.append({"label": corner_label, "n_marking": n_mark, "n_zonal": n_zonal})
        for jersey, (role, _) in preds.items():
            player_roles[jersey][role] += 1

    if not corners_data:
        return figs

    # --- Distribution of markers/zonal per corner ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    mark_counts = [c["n_marking"] for c in corners_data]
    zonal_counts = [c["n_zonal"] for c in corners_data]

    ax = axes[0]
    ax.bar(range(len(corners_data)), mark_counts, color="#3498db", edgecolor="white", alpha=0.85)
    ax.bar(range(len(corners_data)), zonal_counts, bottom=mark_counts,
           color="#f39c12", edgecolor="white", alpha=0.85)
    ax.set_xlabel("Corner", fontsize=10)
    ax.set_ylabel("Defenders", fontsize=10)
    ax.set_title("Defenders per Corner", fontsize=11, fontweight="bold")
    ax.set_xticks(range(len(corners_data)))
    ax.set_xticklabels([f"#{i+1}" for i in range(len(corners_data))], fontsize=8)
    ax.legend(["Man-marking", "Zonal"], fontsize=9)

    # Per-player breakdown
    ax = axes[1]
    players = sorted(player_roles.keys())
    if players:
        x = np.arange(len(players))
        w = 0.35
        mc = [player_roles[p]["marking"] for p in players]
        zc = [player_roles[p]["zonal"] for p in players]
        ax.bar(x - w/2, mc, w, label="Man-marking", color="#3498db", edgecolor="white")
        ax.bar(x + w/2, zc, w, label="Zonal", color="#f39c12", edgecolor="white")
        ax.set_xlabel("Jersey #", fontsize=10)
        ax.set_ylabel("Corners", fontsize=10)
        ax.set_title("Per-Player Role Frequency", fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([f"#{p}" for p in players], fontsize=8, rotation=45)
        ax.legend(fontsize=9)

    plt.tight_layout()
    figs["summary"] = fig
    return figs


# ================================================================
# MAIN APP
# ================================================================

def main():
    st.set_page_config(page_title="KKD Corner Viewer", layout="wide")
    st.title("KKD Corner Viewer")

    matches = discover_matches(DATA_DIR)
    if not matches:
        st.error("No match XML files found in the app directory.")
        return

    # Load role detection model
    model, feat_cols = load_role_model()

    # --- Sidebar ---
    with st.sidebar:
        st.header("Match Selection")
        match_names = [m.name for m in matches]
        selected_idx = st.selectbox(
            "Match", range(len(match_names)),
            format_func=lambda i: match_names[i],
        )
        match = matches[selected_idx]

        if not match.cameras:
            st.warning("No video URLs configured. Add them to videos.json.")

        st.divider()
        st.header("Corners")
        st.caption(f"{len(match.corners)} corners in this match")

        teams = sorted(set(c.team for c in match.corners))
        selected_team = st.radio("Filter by team", ["All"] + teams, horizontal=True)

        filtered_corners = match.corners
        if selected_team != "All":
            filtered_corners = [c for c in match.corners if c.team == selected_team]

        if not filtered_corners:
            st.info("No corners for this filter.")
            return

        corner_labels = [c.label for c in filtered_corners]
        selected_corner_idx = st.radio(
            "Select a corner", range(len(corner_labels)),
            format_func=lambda i: corner_labels[i],
        )

    corner = filtered_corners[selected_corner_idx]

    # --- Main area: two columns ---
    st.subheader(corner.label)

    if not match.cameras:
        st.error("No video URLs configured. Add this match to `videos.json`.")
        return

    # Camera tabs for video
    camera_names = list(match.cameras.keys())
    tabs = st.tabs(camera_names)

    for tab, cam_name in zip(tabs, camera_names):
        with tab:
            with st.spinner(f"Extracting clip ({cam_name})..."):
                try:
                    clip_path = extract_clip(
                        match.cameras[cam_name],
                        corner.video_start_sec,
                        corner.video_end_sec,
                    )
                    st.video(str(clip_path))
                except FileNotFoundError:
                    st.error("ffmpeg not found. Install with: `winget install Gyan.FFmpeg`")
                except Exception as e:
                    st.error(f"Failed to extract clip: {e}")

    # --- Corner Analysis ---
    st.divider()

    if model is None:
        st.info("Corner role detection model not available. "
                "Ensure the research data is in the expected location.")
        return

    col_pitch, col_stats = st.columns([1, 1])

    with col_pitch:
        st.markdown("**Defensive Setup**")
        preds, pc = analyse_corner(match, corner, model, feat_cols)
        if preds and pc:
            fig = plot_corner_roles(pc, preds, title=corner.label)
            st.pyplot(fig)
            plt.close(fig)

            n_mark = sum(1 for r, _ in preds.values() if r == "marking")
            n_zonal = sum(1 for r, _ in preds.values() if r == "zonal")

            st.markdown(f"**{n_mark}** man-markers, **{n_zonal}** zonal defenders")
        else:
            st.warning("Could not analyse this corner (missing position data).")

    with col_stats:
        st.markdown("**Role Details**")
        if preds:
            markers = []
            zonals = []
            for jersey, (role, target) in sorted(preds.items()):
                if role == "marking":
                    markers.append(f"#{jersey} marking #{target}" if target else f"#{jersey}")
                else:
                    zonals.append(f"#{jersey}")

            st.markdown("Man-markers:")
            for m in markers:
                st.markdown(f"- {m}")
            st.markdown("Zonal defenders:")
            for z in zonals:
                st.markdown(f"- {z}")

    # --- Match-level summary ---
    st.divider()
    st.subheader("Match Summary")

    with st.spinner("Analysing all corners..."):
        all_preds = {}
        for c in match.corners:
            p, _ = analyse_corner(match, c, model, feat_cols)
            if p:
                all_preds[c.label] = p

    if all_preds:
        summary_figs = plot_match_summary(all_preds, match.name)
        for fig_name, fig in summary_figs.items():
            st.pyplot(fig)
            plt.close(fig)

        # Clickable corner list grouped by configuration
        st.markdown("**Corners by defensive configuration:**")
        configs = defaultdict(list)
        for c in match.corners:
            p = all_preds.get(c.label)
            if p:
                n_mark = sum(1 for r, _ in p.values() if r == "marking")
                n_zonal = sum(1 for r, _ in p.values() if r == "zonal")
                configs[f"{n_mark} man-markers, {n_zonal} zonal"].append(c.label)

        for config, corner_list in sorted(configs.items()):
            st.markdown(f"**{config}:**")
            for cl in corner_list:
                st.markdown(f"- {cl}")


if __name__ == "__main__":
    main()
