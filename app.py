"""Streamlit app for viewing corner kick video clips from KKD matches."""

import streamlit as st
from pathlib import Path

from event_parser import discover_matches
from video_utils import extract_clip

DATA_DIR = Path(__file__).parent


def main():
    st.set_page_config(page_title="KKD Corner Viewer", layout="wide")
    st.title("KKD Corner Viewer")

    matches = discover_matches(DATA_DIR)

    if not matches:
        st.error("No match XML files found in the app directory.")
        return

    # --- Sidebar: match, corner, and camera selection ---
    with st.sidebar:
        st.header("Match Selection")

        match_names = [m.name for m in matches]
        selected_idx = st.selectbox(
            "Match",
            range(len(match_names)),
            format_func=lambda i: match_names[i],
        )
        match = matches[selected_idx]

        if not match.cameras:
            st.warning("No video URLs configured for this match. Add them to videos.json.")

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
            "Select a corner",
            range(len(corner_labels)),
            format_func=lambda i: corner_labels[i],
        )

    corner = filtered_corners[selected_corner_idx]

    # --- Main area ---
    st.subheader(corner.label)

    if not match.cameras:
        st.error("No video URLs configured. Add this match to `videos.json`.")
        return

    # Camera selector as tabs across the top
    camera_names = list(match.cameras.keys())
    tabs = st.tabs(camera_names)

    for tab, cam_name in zip(tabs, camera_names):
        with tab:
            col1, col2 = st.columns([3, 1])

            with col2:
                st.markdown("**Details**")
                st.markdown(f"- **Team:** {corner.team}")
                st.markdown(f"- **Player:** {corner.player}")
                st.markdown(f"- **Video time:** {corner.start_display}")
                st.markdown(f"- **Duration:** {corner.end_sec - corner.start_sec:.1f}s")

            with col1:
                with st.spinner(f"Extracting clip ({cam_name})..."):
                    try:
                        clip_path = extract_clip(
                            match.cameras[cam_name],
                            corner.start_sec,
                            corner.end_sec,
                        )
                        st.video(str(clip_path))
                    except FileNotFoundError:
                        st.error(
                            "ffmpeg not found. Install it with: `winget install Gyan.FFmpeg` "
                            "(local) or add `ffmpeg` to `packages.txt` (Streamlit Cloud)."
                        )
                    except Exception as e:
                        st.error(f"Failed to extract clip: {e}")


if __name__ == "__main__":
    main()
