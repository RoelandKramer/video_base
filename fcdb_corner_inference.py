"""
FCDB Corner Inference Module
============================

Drop-in module for the FCDB Video & Data Analyser. Loads the trained SQ1
(defender role) and SQ2 (attacker role) classifiers and produces a
structured per-corner output that is ready for the seven dashboard
visualisations.

Setup
-----
Place the three artefacts produced by ``train_production_models.py``
next to this file (or pass ``models_dir`` to ``CornerAnalyser``):

    defender_role_rf.joblib
    attacker_role_rf.joblib
    feature_columns.json

Inputs at inference time
------------------------
SciSports event JSON file path and SciSports position JSON file path,
both for the same match. The module pairs them by timestamp. Only
``synced`` corners are emitted (vis 2--7 require tracking data).

Public API
----------
    a = CornerAnalyser(models_dir="path/to/models")

    # 1. Analyse all corners in a match for a given defending team
    rows = a.analyse_match(events_path, positions_path,
                           defending_team="FC Den Bosch")

    # 2. Pre-computed aggregates for the 7 visualisations
    aggregates = a.compute_aggregates(rows)

    # 3. Lightweight feedback path (one small option on the visualisation)
    a.confirm_corner(corner_result, labeling_sheet_path)
    a.submit_role_corrections(corner_result,
                              role_overrides={jersey: "ZONAL", ...},
                              marks_overrides={jersey: 9, ...},  # optional
                              labeling_sheet_path)

The feedback methods are deliberately minimal: a coach reviewing a clip
either confirms the predicted roles (one click), or opens the list and
edits the role of any player whose assignment looks wrong. Both paths
append or update rows in the labelling sheet so that the thesis project
can retrain on the enlarged ground-truth set.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import joblib
from scipy.optimize import linear_sum_assignment

# ---------------------------------------------------------------------------
# Constants — must match what the thesis project uses
# ---------------------------------------------------------------------------
WINDOW_MS         = 3000     # pre-delivery window for tracking features (SQ1)
ATT_WINDOW_BEFORE = 3000     # attacker pre-delivery window  (SQ2)
ATT_WINDOW_AFTER  = 1500     # attacker post-delivery window (SQ2, can extend to ball arrival)
TICK_MS           = 100
GOAL_X            = 52.5

# Zone constants (match thesis Methods §4.4)
GA_X, PA_X, PE_X, PC_X = 47.0, 41.5, 36.0, 32.35
LANE_DIV = 9.16 - (18.32 / 3)

# Ball flight thresholds (metres of peak height)
FLIGHT_BANDS = [(0.5, "GROUND"), (3.0, "LOW"), (8.0, "MEDIUM"), (math.inf, "FLOATED")]


# ---------------------------------------------------------------------------
# Zone helpers
# ---------------------------------------------------------------------------
def zone_of(x: float, y: float, side: str = "R") -> str:
    """Classify an (x, y) position into a zone label (GA1..PC3, FRONT, EDGE,
    OTHER). The frame is assumed attacking-right normalised."""
    sign = 1 if side == "R" else -1
    ny   = y * sign
    in_central = abs(ny) <= 9.16
    ax = x if x > 0 else -x
    if not in_central:
        if ax > PE_X: return "FRONT" if ny > 0 else "EDGE"
        return "OUT"
    if ax > GA_X:   band = "GA"
    elif ax > PA_X: band = "PA"
    elif ax > PE_X: band = "PE"
    elif ax > PC_X: band = "PC"
    else:           return "OUT"
    if ny > LANE_DIV:    col = "1"
    elif ny > -LANE_DIV: col = "2"
    else:                col = "3"
    return f"{band}{col}"


def flight_band(peak_z: float) -> str:
    for thr, name in FLIGHT_BANDS:
        if peak_z <= thr:
            return name
    return "FLOATED"


def _normalise_to_attacking_right(x: float, y: float, attacking_dir: int) -> tuple[float, float]:
    """Mirror the frame so the attacked goal is always at +52.5."""
    return (-x, -y) if attacking_dir < 0 else (x, y)


# ===========================================================================
# DEFENDER FEATURE EXTRACTION (SQ1)
# ===========================================================================
def _compute_defender_features(frames: dict, tms: int, is_home: bool,
                                labeled_jerseys: Sequence[int]) -> dict[int, dict]:
    """Compute per-defender tracking features over the 3-second pre-delivery
    window. Mirrors train_defender_role.compute_tracking_features."""
    att_key = "h" if is_home else "a"
    def_key = "a" if is_home else "h"

    t0 = ((tms - WINDOW_MS) // TICK_MS) * TICK_MS
    t1 = ((tms + WINDOW_MS) // TICK_MS) * TICK_MS
    times = list(range(t0, t1 + TICK_MS, TICK_MS))

    att_traj, def_traj = {}, {}
    for t in times:
        fr = frames.get(t)
        if not fr: continue
        for p in fr.get(att_key, []):
            att_traj.setdefault(p["s"], []).append((t, p["x"], p["y"]))
        for p in fr.get(def_key, []):
            def_traj.setdefault(p["s"], []).append((t, p["x"], p["y"]))

    result = {}
    for d_jersey in labeled_jerseys:
        if d_jersey not in def_traj:
            continue
        pre_traj = [(t, x, y) for t, x, y in def_traj[d_jersey] if t <= tms]
        if len(pre_traj) < 5:
            continue
        d_dict = {t: (x, y) for t, x, y in pre_traj}

        min_dists, nn_frac_list, nn_jerseys = [], [], []
        per_pair_dists: dict[int, list[float]] = {}
        per_pair_close: dict[int, list[int]]   = {}

        for t, dx, dy in pre_traj:
            fr = frames.get(t)
            if not fr: continue
            att_at_t = [(p["s"], p["x"], p["y"]) for p in fr.get(att_key, [])]
            def_at_t = [(p["s"], p["x"], p["y"]) for p in fr.get(def_key, [])]
            if not att_at_t: continue
            dists = [math.hypot(ax - dx, ay - dy) for _, ax, ay in att_at_t]
            min_d = min(dists)
            nn_att_idx = dists.index(min_d)
            nn_att = att_at_t[nn_att_idx]
            min_dists.append(min_d)
            nn_jerseys.append(nn_att[0])
            if def_at_t:
                dd = [math.hypot(px-nn_att[1], py-nn_att[2]) for _, px, py in def_at_t]
                nn_frac_list.append(int(min(dd) >= min_d - 0.15))
            for a_s, ax, ay in att_at_t:
                d_to_a = math.hypot(ax - dx, ay - dy)
                per_pair_dists.setdefault(a_s, []).append(d_to_a)
                if def_at_t:
                    closest = min(math.hypot(px-ax, py-ay) for _, px, py in def_at_t)
                    per_pair_close.setdefault(a_s, []).append(int(d_to_a <= closest + 0.15))

        pair_summaries = []
        for a_s, dlist in per_pair_dists.items():
            if len(dlist) < 5: continue
            clist = per_pair_close.get(a_s, [])
            pair_summaries.append((a_s, float(np.mean(dlist)),
                                    float(np.min(dlist)),
                                    float(np.mean(clist)) if clist else 0.0))
        if pair_summaries:
            pair_summaries.sort(key=lambda t: t[1])
            best = pair_summaries[0]
            second = pair_summaries[1] if len(pair_summaries) > 1 else (None, 99.0, 99.0, 0.0)
            pair_best_mean, pair_best_min, pair_best_close = best[1], best[2], best[3]
            pair_gap_12 = second[1] - best[1]
        else:
            pair_best_mean = pair_best_min = 99.0
            pair_best_close = pair_gap_12 = 0.0

        nn_consist = (Counter(nn_jerseys).most_common(1)[0][1] / len(nn_jerseys)) if nn_jerseys else 0.0

        best_corr = 0.0
        if per_pair_dists:
            tgt = min(per_pair_dists, key=lambda k: np.mean(per_pair_dists[k]))
            if tgt in att_traj:
                a_dict = {t: (x, y) for t, x, y in att_traj[tgt] if t <= tms}
                common = sorted([t for t in d_dict if t in a_dict])
                if len(common) >= 10:
                    vdx = np.diff([d_dict[t][0] for t in common])
                    vdy = np.diff([d_dict[t][1] for t in common])
                    vax = np.diff([a_dict[t][0] for t in common])
                    vay = np.diff([a_dict[t][1] for t in common])
                    cx = float(np.corrcoef(vdx, vax)[0,1]) if vdx.std()>0.01 and vax.std()>0.01 else 0.0
                    cy = float(np.corrcoef(vdy, vay)[0,1]) if vdy.std()>0.01 and vay.std()>0.01 else 0.0
                    corr = (cx + cy) / 2
                    if not math.isnan(corr): best_corr = corr

        if len(min_dists) >= 5:
            ts_norm = np.linspace(0, 1, len(min_dists))
            dist_slope = float(np.polyfit(ts_norm, min_dists, 1)[0])
        else:
            dist_slope = 0.0

        result[d_jersey] = {
            "track_mean":     float(np.mean(min_dists)) if min_dists else 99.0,
            "track_min":      float(np.min(min_dists))  if min_dists else 99.0,
            "track_std":      float(np.std(min_dists))  if len(min_dists) > 1 else 0.0,
            "track_corr":     best_corr,
            "track_nn":       float(np.mean(nn_frac_list)) if nn_frac_list else 0.0,
            "track_consist":  nn_consist,
            "track_slope":    dist_slope,
            "pair_best_mean": pair_best_mean,
            "pair_best_min":  pair_best_min,
            "pair_best_close":pair_best_close,
            "pair_gap_12":    pair_gap_12,
        }
    return result


def _hungarian_marking(frames: dict, tms: int, is_home: bool,
                         man_jerseys: Sequence[int]) -> dict[int, int]:
    """Hungarian assignment of predicted MAN markers to specific attackers."""
    att_key = "h" if is_home else "a"
    def_key = "a" if is_home else "h"

    t0 = ((tms - WINDOW_MS) // TICK_MS) * TICK_MS
    times = list(range(t0, ((tms // TICK_MS) * TICK_MS) + TICK_MS, TICK_MS))

    pair_d, pair_c = {}, {}
    def_traj, att_traj = {}, {}

    for t in times:
        fr = frames.get(t)
        if not fr: continue
        defs_t = [(p["s"], p["x"], p["y"]) for p in fr.get(def_key, [])]
        atts_t = [(p["s"], p["x"], p["y"]) for p in fr.get(att_key, [])]
        for d_s, dx, dy in defs_t:
            def_traj.setdefault(d_s, []).append((t, dx, dy))
        for a_s, ax, ay in atts_t:
            att_traj.setdefault(a_s, []).append((t, ax, ay))
        for a_s, ax, ay in atts_t:
            if not defs_t: continue
            d_dists = [(d_s, math.hypot(dx-ax, dy-ay)) for d_s, dx, dy in defs_t]
            min_d = min(d for _, d in d_dists)
            for d_s, dx, dy in defs_t:
                d = math.hypot(dx-ax, dy-ay)
                pair_d.setdefault((d_s, a_s), []).append(d)
                pair_c.setdefault((d_s, a_s), []).append(int(d <= min_d + 0.15))

    def_jerseys = sorted(def_traj.keys())
    att_jerseys = sorted(att_traj.keys())
    if not def_jerseys or not att_jerseys:
        return {}

    cost = np.full((len(def_jerseys), len(att_jerseys)), 99.0)
    for i, d_s in enumerate(def_jerseys):
        d_dict = {t:(x,y) for t,x,y in def_traj[d_s]}
        for j, a_s in enumerate(att_jerseys):
            dlist = pair_d.get((d_s, a_s), [])
            clist = pair_c.get((d_s, a_s), [])
            if len(dlist) < 5: continue
            mean_dist = float(np.mean(dlist))
            close_frac = float(np.mean(clist))
            a_dict = {t:(x,y) for t,x,y in att_traj[a_s]}
            common = sorted([t for t in d_dict if t in a_dict])
            corr = 0.0
            if len(common) >= 10:
                vdx = np.diff([d_dict[t][0] for t in common])
                vdy = np.diff([d_dict[t][1] for t in common])
                vax = np.diff([a_dict[t][0] for t in common])
                vay = np.diff([a_dict[t][1] for t in common])
                cx = float(np.corrcoef(vdx, vax)[0,1]) if vdx.std()>0.01 and vax.std()>0.01 else 0.0
                cy = float(np.corrcoef(vdy, vay)[0,1]) if vdy.std()>0.01 and vay.std()>0.01 else 0.0
                c = (cx+cy)/2
                if not math.isnan(c): corr = c
            cost[i, j] = mean_dist - 2.0 * max(0.0, corr) - 1.5 * close_frac

    man_idx = [i for i, j in enumerate(def_jerseys) if j in set(man_jerseys)]
    if not man_idx:
        return {}
    row_ind, col_ind = linear_sum_assignment(cost[man_idx, :])
    return {def_jerseys[man_idx[r]]: att_jerseys[c] for r, c in zip(row_ind, col_ind)}


# ===========================================================================
# ATTACKER FEATURE EXTRACTION (SQ2)
# ===========================================================================
def _extract_attacker_features(frames: dict, gk_h: set, gk_a: set,
                                 corner: dict) -> list[dict]:
    """One feature dict per attacking outfield player. Mirrors
    extract_attacker_role_features.extract_corner_features."""
    tms         = int(corner["start_time_ms"])
    is_home     = bool(corner.get("is_home", True))
    side        = str(corner.get("corner_side", "R"))
    end_x       = corner.get("end_x")
    end_y       = corner.get("end_y")
    end_time_ms = corner.get("end_time_ms")

    att_key = "h" if is_home else "a"
    def_key = "a" if is_home else "h"
    gk_def_shirts = gk_a if is_home else gk_h

    t0 = ((tms - ATT_WINDOW_BEFORE) // TICK_MS) * TICK_MS
    t1 = ((tms + ATT_WINDOW_AFTER)  // TICK_MS) * TICK_MS
    if end_time_ms is not None and not pd.isna(end_time_ms):
        arr_tick = (int(end_time_ms) // TICK_MS) * TICK_MS
        arr_tick = max(arr_tick, (tms // TICK_MS) * TICK_MS)
        arr_tick = min(arr_tick, ((tms + 4000) // TICK_MS) * TICK_MS)
        if arr_tick > t1: t1 = arr_tick
    times = list(range(t0, t1 + TICK_MS, TICK_MS))

    kick_tick  = (tms // TICK_MS) * TICK_MS
    kick_frame = frames.get(kick_tick)
    attacking_dir = 1
    if kick_frame and kick_frame.get("b") and "x" in kick_frame["b"]:
        attacking_dir = 1 if kick_frame["b"]["x"] >= 0 else -1
    else:
        for off in (-100, 100, -200, 200, -300):
            ft = frames.get(kick_tick + off)
            if ft and ft.get("b") and "x" in ft["b"]:
                attacking_dir = 1 if ft["b"]["x"] >= 0 else -1
                break

    att_gk_shirts = gk_h if is_home else gk_a
    att_gk_on_attacking_half = set()
    if kick_frame:
        for p in kick_frame.get(att_key, []):
            if p["s"] in att_gk_shirts:
                xn, _ = _normalise_to_attacking_right(p["x"], p["y"], attacking_dir)
                if xn > 0:
                    att_gk_on_attacking_half.add(p["s"])

    att_traj, def_traj, gk_traj = {}, {}, []
    for t in times:
        fr = frames.get(t)
        if not fr: continue
        for p in fr.get(att_key, []):
            s = p["s"]
            if s in (gk_h | gk_a):
                if s in gk_def_shirts:
                    gk_traj.append((t, p["x"], p["y"])); continue
                if s in att_gk_on_attacking_half:
                    att_traj.setdefault(s, []).append((t, p["x"], p["y"]))
                continue
            att_traj.setdefault(s, []).append((t, p["x"], p["y"]))
        for p in fr.get(def_key, []):
            s = p["s"]
            if s in gk_def_shirts:
                gk_traj.append((t, p["x"], p["y"])); continue
            if s in (gk_h | gk_a): continue
            def_traj.setdefault(s, []).append((t, p["x"], p["y"]))

    if not att_traj:
        return []

    gk_by_t = {t:(x,y) for t,x,y in gk_traj}
    if not gk_by_t and def_traj:
        candidates = []
        for s, traj in def_traj.items():
            kp = next((p for p in traj if p[0]==kick_tick), None)
            if kp:
                candidates.append((s, kp, abs(kp[1])))
        if candidates:
            candidates.sort(key=lambda c: -c[2])
            for t,x,y in def_traj[candidates[0][0]]:
                gk_by_t[t] = (x,y)

    setup_tick = ((tms - ATT_WINDOW_BEFORE) // TICK_MS) * TICK_MS
    if end_time_ms is not None and not pd.isna(end_time_ms):
        end_tick = max((int(end_time_ms) // TICK_MS) * TICK_MS, kick_tick)
        end_tick = min(end_tick, ((tms + 4000) // TICK_MS) * TICK_MS)
    else:
        end_tick = ((tms + 1000) // TICK_MS) * TICK_MS

    ball_arrival = None
    if end_x is not None and not pd.isna(end_x) and end_y is not None and not pd.isna(end_y):
        ball_arrival = (float(end_x), float(end_y))

    rows = []
    for shirt, traj in att_traj.items():
        ptd = {t:(x,y) for t,x,y in traj}
        sp = ptd.get(setup_tick) or (traj[0][1], traj[0][2])
        kp = ptd.get(kick_tick)  or sp
        ep = ptd.get(end_tick)   or (traj[-1][1], traj[-1][2])

        sp_n = _normalise_to_attacking_right(*sp, attacking_dir)
        kp_n = _normalise_to_attacking_right(*kp, attacking_dir)
        ep_n = _normalise_to_attacking_right(*ep, attacking_dir)

        disp = math.hypot(ep_n[0]-sp_n[0], ep_n[1]-sp_n[1])
        mv = (ep_n[0]-sp_n[0], ep_n[1]-sp_n[1])

        if ball_arrival and disp > 0.5:
            bd = (ball_arrival[0]-sp_n[0], ball_arrival[1]-sp_n[1])
            bd_mag = math.hypot(*bd); mv_mag = disp
            alignment = ((mv[0]*bd[0] + mv[1]*bd[1]) / (bd_mag*mv_mag)
                          if bd_mag > 0.1 and mv_mag > 0.1 else 0.0)
        else:
            alignment = 0.0

        gk_dists = []
        for t in range(setup_tick, kick_tick + TICK_MS, TICK_MS):
            pp = ptd.get(t); gk = gk_by_t.get(t)
            if pp and gk:
                pp_n = _normalise_to_attacking_right(*pp, attacking_dir)
                gk_n = _normalise_to_attacking_right(*gk, attacking_dir)
                gk_dists.append(math.hypot(pp_n[0]-gk_n[0], pp_n[1]-gk_n[1]))
        dist_gk_mean    = float(np.mean(gk_dists)) if gk_dists else 99.0
        dist_gk_at_kick = gk_dists[-1]            if gk_dists else 99.0
        dist_gk_min     = float(np.min(gk_dists)) if gk_dists else 99.0

        def_dists = []
        dist_def_at_setup = 99.0; dist_def_at_kick = 99.0
        for t in range(setup_tick, end_tick + TICK_MS, TICK_MS):
            pp = ptd.get(t)
            if not pp: continue
            pp_n = _normalise_to_attacking_right(*pp, attacking_dir)
            best = 99.0
            for ds, dt in def_traj.items():
                dp = next((d for tt,*d in dt if tt==t), None)
                if dp:
                    dp_n = _normalise_to_attacking_right(*dp, attacking_dir)
                    d = math.hypot(pp_n[0]-dp_n[0], pp_n[1]-dp_n[1])
                    if d < best: best = d
            def_dists.append(best)
            if t == setup_tick: dist_def_at_setup = best
            if t == kick_tick:  dist_def_at_kick  = best
        dist_def_mean = float(np.mean(def_dists)) if def_dists else 99.0
        dist_def_min  = float(np.min(def_dists))  if def_dists else 99.0

        x_kick_n, y_kick_n = kp_n
        dist_goal_kick  = abs(GOAL_X - x_kick_n)
        is_in_box_kick  = int(x_kick_n > PE_X and abs(y_kick_n) < 20.16)
        is_at_edge_kick = int(PC_X < x_kick_n <= PE_X + 2 and abs(y_kick_n) < 25)

        if ball_arrival is not None:
            dist_ball_at_kick = math.hypot(ball_arrival[0]-kp_n[0], ball_arrival[1]-kp_n[1])
            dist_ball_at_end  = math.hypot(ball_arrival[0]-ep_n[0], ball_arrival[1]-ep_n[1])
            dist_ball_change  = dist_ball_at_end - dist_ball_at_kick
        else:
            dist_ball_at_kick = dist_ball_at_end = 99.0
            dist_ball_change  = 0.0

        rows.append({
            "match_id":     corner["match_id"],
            "start_time_ms":tms,
            "match_name":   corner.get("match_name",""),
            "jersey":       shirt,
            "is_home":      is_home,
            "corner_side":  side,
            "x_setup":      round(sp_n[0],2), "y_setup": round(sp_n[1],2),
            "x_kick":       round(kp_n[0],2), "y_kick":  round(kp_n[1],2),
            "x_end":        round(ep_n[0],2), "y_end":   round(ep_n[1],2),
            "displacement": round(disp, 3),
            "alignment_to_ball":   round(alignment, 3),
            "dist_to_gk_mean":     round(dist_gk_mean, 2),
            "dist_to_gk_at_kick":  round(dist_gk_at_kick, 2),
            "dist_to_gk_min":      round(dist_gk_min, 2),
            "dist_to_def_mean":    round(dist_def_mean, 2),
            "dist_to_def_min":     round(dist_def_min, 2),
            "dist_to_def_at_kick": round(dist_def_at_kick, 2),
            "dist_to_def_at_setup":round(dist_def_at_setup, 2),
            "dist_goal_kick":      round(dist_goal_kick, 2),
            "is_in_box_kick":      is_in_box_kick,
            "is_at_edge_kick":     is_at_edge_kick,
            "dist_ball_at_kick":   round(dist_ball_at_kick, 2),
            "dist_ball_at_end":    round(dist_ball_at_end, 2),
            "dist_ball_change":    round(dist_ball_change, 2),
            "ball_end_x":          end_x, "ball_end_y": end_y,
        })

    if rows and ball_arrival is not None:
        sorted_by_end = sorted(rows, key=lambda r: r["dist_ball_at_end"])
        rank2 = sorted_by_end[1]["dist_ball_at_end"] if len(sorted_by_end) > 1 else 99.0
        for rank, r in enumerate(sorted_by_end, start=1):
            r["rank_dist_ball_end"]  = rank
            r["is_closest_to_ball"]  = int(rank == 1)
            r["n_closer_to_ball"]    = rank - 1
            r["gap_to_closest"]      = round(r["dist_ball_at_end"] - sorted_by_end[0]["dist_ball_at_end"], 2)
            r["closest_lead"]        = round(rank2 - sorted_by_end[0]["dist_ball_at_end"], 2) if rank == 1 else 0.0
        sorted_by_kick = sorted(rows, key=lambda r: r["dist_ball_at_kick"])
        for rank, r in enumerate(sorted_by_kick, start=1):
            r["rank_dist_ball_kick"] = rank
    else:
        for r in rows:
            r["rank_dist_ball_end"]  = 99
            r["is_closest_to_ball"]  = 0
            r["n_closer_to_ball"]    = 99
            r["gap_to_closest"]      = 0.0
            r["closest_lead"]        = 0.0
            r["rank_dist_ball_kick"] = 99

    return rows


# ===========================================================================
# CornerAnalyser — the main public class
# ===========================================================================
class CornerAnalyser:
    """Loads the trained models once and exposes inference, aggregation,
    and a small feedback API for coach-confirmed corrections."""

    DEF_ROLES = ("MAN", "ZONAL", "SHORT", "COUNTER")
    ATT_ROLES = ("TARGET", "DECOY", "STATIC", "SECOND_BALL", "BLOCK_GK", "BLOCK_DEF")

    def __init__(self, models_dir: str | Path):
        models_dir = Path(models_dir)
        self.def_model = joblib.load(models_dir / "defender_role_rf.joblib")
        self.att_model = joblib.load(models_dir / "attacker_role_rf.joblib")
        with open(models_dir / "feature_columns.json") as f:
            cols = json.load(f)
        self.def_feat_cols = cols["defender"]
        self.att_feat_cols = cols["attacker"]
        self.models_dir = models_dir

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def analyse_match(self, events_path: str | Path,
                       positions_path: str | Path,
                       defending_team: str) -> list[dict]:
        """Return one dict per ``defending_team`` corner in the match. Only
        synced corners are emitted (vis 2--7 require tracking)."""
        with open(events_path, encoding="utf-8") as f:
            ev_root = json.load(f)
        with open(positions_path, encoding="utf-8") as f:
            pos_root = json.load(f)
        meta = ev_root["metaData"]
        frames = {fr["t"]: fr for fr in pos_root["data"]}

        home_name = meta["homeTeamName"]; away_name = meta["awayTeamName"]
        home_id   = meta["homeTeamId"];   away_id   = meta["awayTeamId"]

        player_name = {}
        gk_h, gk_a = set(), set()
        for p in ev_root.get("players", []):
            player_name[(p.get("teamId"), p.get("shirtNumber"))] = p.get("name", "")
            if "goalkeeper" in p.get("positionName","").lower():
                if p.get("teamId") == home_id: gk_h.add(p["shirtNumber"])
                elif p.get("teamId") == away_id: gk_a.add(p["shirtNumber"])

        results = []
        for ev in ev_root["data"]:
            if ev.get("subTypeName") not in ("CORNER_CROSSED", "CORNER_SHORT"):
                continue
            # Only synced corners (visualisations 2-7 require tracking)
            if not ev.get("synced", False):
                continue
            taker_team_id = ev.get("teamId")
            if taker_team_id == home_id:
                attacking_team = home_name; defending = away_name
                att_key = "h"; def_key = "a"; gk_def_shirts = gk_a; gk_att_shirts = gk_h
            elif taker_team_id == away_id:
                attacking_team = away_name; defending = home_name
                att_key = "a"; def_key = "h"; gk_def_shirts = gk_h; gk_att_shirts = gk_a
            else:
                continue
            if defending.lower() != defending_team.lower():
                continue

            tms       = int(ev["startTimeMs"])
            end_tms   = ev.get("endTimeMs")
            end_x     = ev.get("endPosXM"); end_y = ev.get("endPosYM")
            kick_tick = (tms // TICK_MS) * TICK_MS
            is_home_taker = (taker_team_id == home_id)

            # Attacking direction
            kick_frame = frames.get(kick_tick) or next(
                (frames.get(kick_tick + off) for off in (-100, 100, -200, 200, -300)
                 if frames.get(kick_tick + off)), None)
            attacking_dir = 1
            if kick_frame and kick_frame.get("b") and "x" in kick_frame["b"]:
                attacking_dir = 1 if kick_frame["b"]["x"] >= 0 else -1

            sample_ticks = {
                "t-3000": ((tms - 3000) // TICK_MS) * TICK_MS,
                "t-1000": ((tms - 1000) // TICK_MS) * TICK_MS,
                "t0":     kick_tick,
                "t+1000": ((tms + 1000) // TICK_MS) * TICK_MS,
            }

            def sample_positions(team_key, exclude_jerseys):
                out = {}
                for label, tk in sample_ticks.items():
                    fr = frames.get(tk)
                    if not fr: continue
                    for p in fr.get(team_key, []):
                        if p["s"] in exclude_jerseys: continue
                        xn, yn = _normalise_to_attacking_right(p["x"], p["y"], attacking_dir)
                        out.setdefault(p["s"], {})[label] = (round(xn, 2), round(yn, 2))
                return out

            def_positions = sample_positions(def_key, gk_def_shirts)
            att_positions = sample_positions(att_key, gk_att_shirts | {ev.get("playerId")})

            # Defending GK position(s) at the four sample ticks
            gk_positions = sample_positions(def_key, set())
            gk_positions = {j: p for j, p in gk_positions.items() if j in gk_def_shirts}

            # ---- Defender role prediction ----
            def_jerseys = list(def_positions.keys())
            def_feats = _compute_defender_features(frames, tms, is_home_taker, def_jerseys)
            def_role = {}
            for j in def_jerseys:
                if j not in def_feats:
                    def_role[j] = "ZONAL"; continue
                snap = def_positions[j].get("t0") or def_positions[j].get("t-1000") or (0, 0)
                feat = {
                    "x": snap[0], "y": snap[1],
                    "near_y": abs(snap[1]),
                    "dist_goal": GOAL_X - snap[0],
                    "dist_center_y": abs(snap[1]),
                    "disp": 0.0,
                    "min_dist_att": 0.0, "dist2_att": 0.0, "gap_att_12": 0.0,
                    "n_att_3m": 0, "n_att_5m": 0,
                    "rank_defs": 0, "gap_defs_12": 0.0, "is_mutual_snap": 0,
                    "n_def_2m": 0, "min_def_sep": 99.0,
                    "sz_depth": 0, "sz_col": 0, "ez_depth": 0, "ez_col": 0,
                    **def_feats[j],
                    "pair_rank_in_corner": 0, "pair_best_mean_rel": 0.0,
                    "track_mean_rank": 0,    "track_mean_rel":    0.0,
                }
                X = np.array([[feat.get(c, 0.0) for c in self.def_feat_cols]], dtype=float)
                def_role[j] = str(self.def_model.predict(X)[0])

            # Hungarian for predicted MAN markers
            man_jerseys = [j for j, r in def_role.items() if r == "MAN"]
            marking = _hungarian_marking(frames, tms, is_home_taker, man_jerseys)

            # ---- Attacker role prediction ----
            corner_meta = {
                "match_id":      meta.get("matchId"),
                "match_name":    f"{home_name} vs {away_name}",
                "start_time_ms": tms,
                "end_time_ms":   end_tms,
                "is_home":       is_home_taker,
                "corner_side":   "R" if (end_y is None or end_y >= 0) else "L",
                "end_x":         end_x,
                "end_y":         end_y,
            }
            att_feat_rows = _extract_attacker_features(frames, gk_h, gk_a, corner_meta)
            att_role = {}
            for row in att_feat_rows:
                jersey = int(row["jersey"])
                # Match the trained model's feature column order
                X = np.array([[row.get(c, 0.0) for c in self.att_feat_cols]], dtype=float)
                att_role[jersey] = str(self.att_model.predict(X)[0])

            # ---- Outcome ----
            shot, goal, xg_seq = self._outcome_from_sequence(ev_root["data"], ev.get("sequenceId"))
            ball_z = self._ball_peak_z(frames, tms)

            # ---- Assemble corner result ----
            corner = {
                "corner_id":      f"{meta.get('matchId')}:{tms}",
                "match_id":       meta.get("matchId"),
                "match_name":     f"{home_name} vs {away_name}",
                "match_date":     meta.get("dateMatch", ""),
                "kick_time_ms":   tms,
                "match_clock":    _ms_to_clock(tms, ev.get("partName")),
                "attacking_team": attacking_team,
                "defending_team": defending,
                "corner_side":    "R" if (end_y is None or end_y >= 0) else "L",
                "synced":         True,
                "delivery": {
                    "end_x":         end_x,
                    "end_y":         end_y,
                    "zone":          zone_of(end_x, end_y) if end_x is not None else "OTHER",
                    "ball_flight":   flight_band(ball_z) if ball_z is not None else "UNKNOWN",
                    "peak_height_m": ball_z,
                    "sub_type":      ev.get("subTypeName"),  # CROSSED or SHORT
                },
                "outcome": {"shot": shot, "goal": goal, "xg_seq": xg_seq},
                "defenders": [
                    {
                        "jersey":              j,
                        "player_name":         player_name.get(
                            (away_id if def_key == "a" else home_id, j), ""),
                        "role":                def_role.get(j, "ZONAL"),
                        "marks_jersey":        marking.get(j) if def_role.get(j) == "MAN" else None,
                        "position_at_setup":   def_positions[j].get("t-3000"),
                        "position_at_kick":    def_positions[j].get("t0"),
                        "position_at_end":     def_positions[j].get("t+1000"),
                    }
                    for j in def_jerseys
                ],
                "attackers": [
                    {
                        "jersey":            j,
                        "player_name":       player_name.get(
                            (home_id if att_key == "h" else away_id, j), ""),
                        "role":              att_role.get(j, "STATIC"),
                        "position_at_setup": att_positions[j].get("t-3000"),
                        "position_at_kick":  att_positions[j].get("t0"),
                        "position_at_end":   att_positions[j].get("t+1000"),
                    }
                    for j in att_positions.keys()
                ],
                "goalkeeper": (
                    {
                        "jersey":            next(iter(gk_positions)),
                        "player_name":       player_name.get(
                            (away_id if def_key == "a" else home_id,
                             next(iter(gk_positions))), ""),
                        "position_at_setup": next(iter(gk_positions.values())).get("t-3000"),
                        "position_at_kick":  next(iter(gk_positions.values())).get("t0"),
                        "position_at_end":   next(iter(gk_positions.values())).get("t+1000"),
                    }
                    if gk_positions else None
                ),
            }

            roles = list(def_role.values())
            n_zonal   = sum(r == "ZONAL"   for r in roles)
            n_man     = sum(r == "MAN"     for r in roles)
            n_short   = sum(r == "SHORT"   for r in roles)
            n_counter = sum(r == "COUNTER" for r in roles)
            n_attacking_roles = sum(att_role.get(j) in ("TARGET","DECOY","BLOCK_GK","BLOCK_DEF")
                                      for j in att_positions)
            corner["summary_stats"] = {
                "n_zonal":   n_zonal,  "n_man":     n_man,
                "n_short":   n_short,  "n_counter": n_counter,
                "has_short_player":       n_short > 0,
                "n_attacking_roles":      n_attacking_roles,
                "attackers_exceed_markers": n_attacking_roles > n_man,
            }
            results.append(corner)

        return results

    # ------------------------------------------------------------------
    # Aggregation (vis 1, 2, 5, 6, 7)
    # ------------------------------------------------------------------
    def compute_aggregates(self, rows: list[dict]) -> dict:
        if not rows:
            return {}

        # Vis 1: delivery zone × shot rate
        zone_stats = {}
        for r in rows:
            z = r["delivery"]["zone"]
            zone_stats.setdefault(z, {"n": 0, "shots": 0})
            zone_stats[z]["n"] += 1
            if r["outcome"]["shot"]: zone_stats[z]["shots"] += 1
        for z, s in zone_stats.items():
            s["shot_rate"] = s["shots"] / s["n"] if s["n"] else 0.0

        # Vis 2: average positions per time slice, split by has_short
        def _avg_per_slice(subset, team_key):
            slices = {"t-3000": [], "t-1000": [], "t0": [], "t+1000": []}
            slot_map = {"t-3000":"position_at_setup", "t-1000":"position_at_setup",
                         "t0":"position_at_kick", "t+1000":"position_at_end"}
            for r in subset:
                for p in r[team_key]:
                    for slc in slices:
                        pos = p.get(slot_map[slc])
                        if pos: slices[slc].append(pos)
            return {slc: list(np.mean(np.array(pts), axis=0)) if pts else None
                     for slc, pts in slices.items()}

        with_short    = [r for r in rows if r["summary_stats"]["has_short_player"]]
        without_short = [r for r in rows if not r["summary_stats"]["has_short_player"]]
        vis2 = {
            "with_short":    {"defenders": _avg_per_slice(with_short,    "defenders"),
                               "attackers": _avg_per_slice(with_short,    "attackers")},
            "without_short": {"defenders": _avg_per_slice(without_short, "defenders"),
                               "attackers": _avg_per_slice(without_short, "attackers")},
        }

        # Vis 5: average role counts split by has_short
        def _avg_counts(subset):
            if not subset: return {"zonal":0, "man":0, "short":0, "counter":0}
            keys = ["n_zonal","n_man","n_short","n_counter"]
            return {k.replace("n_",""): float(np.mean([r["summary_stats"][k] for r in subset]))
                     for k in keys}
        vis5 = {"with_short": _avg_counts(with_short),
                "without_short": _avg_counts(without_short)}

        # Vis 6: fraction of corners with a ZONAL defender in each zone
        zone_zonal_count = {}
        for r in rows:
            zones_with_zonal = set()
            for d in r["defenders"]:
                if d["role"] != "ZONAL": continue
                p = d.get("position_at_kick") or d.get("position_at_setup")
                if not p: continue
                zones_with_zonal.add(zone_of(p[0], p[1]))
            for z in zones_with_zonal:
                zone_zonal_count[z] = zone_zonal_count.get(z, 0) + 1
        vis6 = {z: c / len(rows) for z, c in zone_zonal_count.items()}

        # Vis 7
        vis7 = float(np.mean([r["summary_stats"]["attackers_exceed_markers"] for r in rows]))

        return {
            "vis1_zones": zone_stats,
            "vis2_average_positions": vis2,
            "vis5_role_counts": vis5,
            "vis6_zonal_zones": vis6,
            "vis7_attackers_exceed_markers_pct": vis7,
        }

    # ------------------------------------------------------------------
    # Feedback / labelling-sheet update (small option on the visualisation)
    # ------------------------------------------------------------------
    def confirm_corner(self, corner_result: dict, labeling_sheet_path: str | Path) -> None:
        """Coach picks "All roles correct" — write the auto-predicted roles
        as confirmed labels into the labelling sheet.

        Only synced corners are accepted (the visualisation only shows synced
        corners anyway). If the corner is already in the sheet it is replaced;
        otherwise it is appended.
        """
        self._write_corner_to_sheet(corner_result, role_overrides=None,
                                      marks_overrides=None,
                                      sheet_path=Path(labeling_sheet_path))

    def submit_role_corrections(self, corner_result: dict,
                                  role_overrides: dict[int, str],
                                  labeling_sheet_path: str | Path,
                                  marks_overrides: Optional[dict[int, int]] = None) -> None:
        """Coach picks "There is a mistake" and edits one or more roles.

        ``role_overrides`` is a mapping ``{jersey: corrected_role}`` for
        defenders. ``marks_overrides`` (optional) is ``{defender_jersey:
        attacker_jersey}`` for MAN markers whose marked attacker was wrong.
        Any jersey not in the override dict keeps its auto-predicted role.
        """
        self._write_corner_to_sheet(corner_result,
                                      role_overrides=role_overrides or {},
                                      marks_overrides=marks_overrides or {},
                                      sheet_path=Path(labeling_sheet_path))

    # ------------------------------------------------------------------
    def _write_corner_to_sheet(self, corner_result: dict,
                                 role_overrides: Optional[dict[int, str]],
                                 marks_overrides: Optional[dict[int, int]],
                                 sheet_path: Path) -> None:
        if not corner_result.get("synced", True):
            raise ValueError("Only synced corners can be added to the labelling sheet.")

        corner_id = corner_result["corner_id"]
        new_rows = []
        for d in corner_result["defenders"]:
            auto_role = d["role"]
            final_role = (role_overrides or {}).get(d["jersey"], auto_role)
            auto_marks = d.get("marks_jersey")
            final_marks = (marks_overrides or {}).get(d["jersey"], auto_marks)
            x, y = (d.get("position_at_kick") or d.get("position_at_setup") or (None, None))
            new_rows.append({
                "corner_id":     corner_id,
                "match_id":      corner_result["match_id"],
                "match_name":    corner_result["match_name"],
                "start_time_ms": corner_result["kick_time_ms"],
                "corner_side":   corner_result["corner_side"],
                "player_team":   "DEF",
                "jersey":        d["jersey"],
                "player_id":     None,
                "x":             x,
                "y":             y,
                "auto_role":     auto_role,
                "role":          final_role,
                "auto_marks":    auto_marks,
                "marks":         final_marks,
                "confirmed_by":  "DV_ANALYSER",
                "confirmed_at":  datetime.utcnow().isoformat(),
            })
        for a in corner_result["attackers"]:
            auto_role = a["role"]
            final_role = (role_overrides or {}).get(a["jersey"], auto_role)
            x, y = (a.get("position_at_kick") or a.get("position_at_setup") or (None, None))
            new_rows.append({
                "corner_id":     corner_id,
                "match_id":      corner_result["match_id"],
                "match_name":    corner_result["match_name"],
                "start_time_ms": corner_result["kick_time_ms"],
                "corner_side":   corner_result["corner_side"],
                "player_team":   "ATT",
                "jersey":        a["jersey"],
                "player_id":     None,
                "x":             x,
                "y":             y,
                "auto_role":     auto_role,
                "role":          final_role,
                "auto_marks":    None,
                "marks":         None,
                "confirmed_by":  "DV_ANALYSER",
                "confirmed_at":  datetime.utcnow().isoformat(),
            })
        new_df = pd.DataFrame(new_rows)

        if sheet_path.exists():
            existing = pd.read_csv(sheet_path)
            # Drop any existing rows for this corner_id, then append the new ones
            existing = existing[existing["corner_id"].astype(str) != str(corner_id)]
            out = pd.concat([existing, new_df], ignore_index=True, sort=False)
        else:
            out = new_df

        sheet_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(sheet_path, index=False)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _outcome_from_sequence(events: list[dict], sequence_id: int) -> tuple[bool, bool, float]:
        if sequence_id is None or sequence_id < 0:
            return False, False, 0.0
        shot = goal = False; xg = 0.0
        for e in events:
            if e.get("sequenceId") != sequence_id: continue
            base = e.get("baseTypeName", "")
            sub  = e.get("subTypeName", "") or ""
            if "SHOT" in base or sub in ("SHOT", "HEADER"):
                shot = True
                ev_xg = e.get("metrics", {}).get("xG", 0.0)
                if ev_xg > xg: xg = ev_xg
                if "GOAL" in sub.upper(): goal = True
        return shot, goal, float(xg)

    @staticmethod
    def _ball_peak_z(frames: dict, tms: int) -> Optional[float]:
        t0 = ((tms - 500) // TICK_MS) * TICK_MS
        t1 = ((tms + 2500) // TICK_MS) * TICK_MS
        zs = [fr["b"]["z"] for t in range(t0, t1 + TICK_MS, TICK_MS)
              for fr in [frames.get(t)] if fr and fr.get("b") and "z" in fr["b"]]
        return max(zs) if zs else None


def _ms_to_clock(tms: int, part_name: Optional[str]) -> str:
    s = tms // 1000
    m, s = divmod(s, 60)
    if part_name == "SECOND_HALF": m += 45
    return f"{m:02d}:{s:02d}"


# ---------------------------------------------------------------------------
# CLI for ad-hoc testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--events",     required=True)
    ap.add_argument("--positions",  required=True)
    ap.add_argument("--team",       required=True, help="Defending team name")
    ap.add_argument("--models",     default="output/models")
    ap.add_argument("--out",        default="-")
    args = ap.parse_args()

    analyser = CornerAnalyser(args.models)
    rows = analyser.analyse_match(args.events, args.positions, args.team)
    payload = {"corners": rows, "aggregates": analyser.compute_aggregates(rows)}

    if args.out == "-":
        print(json.dumps(payload, indent=2, default=str))
    else:
        Path(args.out).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        print(f"Wrote {len(rows)} corners and aggregates to {args.out}")
