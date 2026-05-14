"""
Magnet Board — Defensive Response Prediction
============================================

Companion module to ``fcdb_corner_inference.py``. Given a coach's
planned attacking corner (attacker start positions, attacker end
positions, optional roles, delivery target), this module predicts how
the chosen opponent is likely to organise themselves defensively,
based on the opponent's historical defending-corner data.

Predictions are descriptive (probabilistic), not deterministic. Every
predicted defender carries an explanation tied to the evidence
corners that support it, so the coach can click through to video.

Inputs
------
- ``opponent_corners``: list of analysed corners from
  ``CornerAnalyser.analyse_match(...)`` covering the opponent's
  defending corners across the season. Each corner already contains
  per-defender role and position, who marked whom, summary stats,
  delivery zone, and outcome.

Public API
----------
    from magnet_board import MagnetBoard

    board = MagnetBoard(opponent_corners)

    prediction = board.predict_defensive_setup(planned_attack)
    # prediction has the structure documented under "Output schema"
    # at the bottom of this file.

Usage in the analyser UI
------------------------
- The coach drags attacker dots onto the pitch (start + end positions).
- Optionally tags each attacker with a role intent (TARGET, DECOY, ...).
- Optionally sets a delivery target zone (GA1, GA2, ...).
- The UI calls ``board.predict_defensive_setup(planned_attack)``.
- The UI renders the predicted defender positions on the same pitch,
  colour-coded by role, with arrows from each MAN defender to the
  attacker they are expected to mark.
- Each predicted defender's explanation is shown as a tooltip and the
  ``evidence_corners`` list lets the coach pull up the past clips
  that support the prediction.

Output schema (one dict from predict_defensive_setup)
-----------------------------------------------------
    {
      "shape": {                          # the chosen defensive shape
        "n_zonal":   4,
        "n_man":     3,
        "n_short":   1,
        "n_counter": 1,
        "sample_size":   12,              # past corners used as evidence
        "confidence":    "high"/"med"/"low",   # based on sample size and homogeneity
        "explanation":   "..."
      },
      "defenders": [
        {
          "jersey":           5,
          "player_name":      "John Doe",
          "predicted_role":   "ZONAL",
          "predicted_position": [44.0, -3.5],
          "marks_jersey":     None,        # set only for MAN
          "explanation":      "Player #5 typically guards the near-post area (zone GA1). Observed as ZONAL in 9 of 12 of this opponent's defending corners.",
          "evidence_corners": ["match42:1234567", "match48:7654321", ...]
        }, ...
      ],
      "open_zones": [
        {
          "zone": "PA3",
          "explanation": "No defender historically covers this far-post zone; potentially exploitable."
        }, ...
      ]
    }

Planned attack input schema
---------------------------
    {
      "corner_side":     "L" or "R",
      "delivery_zone":   "GA2" | "GA1" | "PA3" | ... | None,
      "attackers": [
        {
          "jersey":     9,                   # optional, for tracking later
          "start_pos":  [40.0, -3.0],        # t = -3s, attacking-right normalised
          "end_pos":    [47.0, -2.0],        # t around delivery
          "role_intent": "TARGET"            # optional: TARGET / DECOY / STATIC / BLOCK_*
        }, ...
      ]
    }
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Optional, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment


# ---------------------------------------------------------------------------
# Zone helpers (must match fcdb_corner_inference.py)
# ---------------------------------------------------------------------------
GA_X, PA_X, PE_X, PC_X = 47.0, 41.5, 36.0, 32.35
LANE_DIV = 9.16 - (18.32 / 3)

_ZONE_CENTRES = {
    # (x, y) centres in the attacking-right normalised frame
    "GA1": (49.75, +6.10), "GA2": (49.75, 0.0), "GA3": (49.75, -6.10),
    "PA1": (44.25, +6.10), "PA2": (44.25, 0.0), "PA3": (44.25, -6.10),
    "PE1": (38.75, +6.10), "PE2": (38.75, 0.0), "PE3": (38.75, -6.10),
    "PC1": (34.20, +6.10), "PC2": (34.20, 0.0), "PC3": (34.20, -6.10),
}


def zone_of(x: float, y: float) -> str:
    in_central = abs(y) <= 9.16
    ax = x if x > 0 else -x
    if not in_central:
        if ax > PE_X: return "FRONT" if y > 0 else "EDGE"
        return "OUT"
    if ax > GA_X:   band = "GA"
    elif ax > PA_X: band = "PA"
    elif ax > PE_X: band = "PE"
    elif ax > PC_X: band = "PC"
    else:           return "OUT"
    if y > LANE_DIV:    col = "1"
    elif y > -LANE_DIV: col = "2"
    else:               col = "3"
    return f"{band}{col}"


# ---------------------------------------------------------------------------
# MagnetBoard
# ---------------------------------------------------------------------------
class MagnetBoard:
    """Predicts the opponent's likely defensive shape given a planned attack."""

    def __init__(self, opponent_corners: list[dict]):
        """
        Args:
            opponent_corners: list of corner result dicts (output of
                ``CornerAnalyser.analyse_match``) covering this opponent's
                defending corners across the season.
        """
        if not opponent_corners:
            raise ValueError("MagnetBoard needs at least one analysed corner.")
        self.corners = opponent_corners
        self._build_history_index()

    # ------------------------------------------------------------------
    # History indexing
    # ------------------------------------------------------------------
    def _build_history_index(self) -> None:
        """Pre-compute per-jersey aggregates from the opponent's corners."""
        # jersey -> list of (role, position_at_kick, marks_jersey, corner_id, delivery_zone)
        per_jersey: dict[int, list[dict]] = {}
        for c in self.corners:
            corner_id    = c["corner_id"]
            delivery_zone = c.get("delivery", {}).get("zone")
            for d in c["defenders"]:
                per_jersey.setdefault(d["jersey"], []).append({
                    "role":         d["role"],
                    "pos":          d.get("position_at_kick"),
                    "marks_jersey": d.get("marks_jersey"),
                    "corner_id":    corner_id,
                    "name":         d.get("player_name", ""),
                    "delivery_zone": delivery_zone,
                })
        self._per_jersey = per_jersey

        # Shape distribution across all corners
        shapes = []
        for c in self.corners:
            s = c.get("summary_stats", {})
            shapes.append((s.get("n_zonal",0), s.get("n_man",0),
                            s.get("n_short",0), s.get("n_counter",0),
                            s.get("has_short_player", False)))
        self._shapes = shapes

    # ------------------------------------------------------------------
    # Shape prediction
    # ------------------------------------------------------------------
    def _typical_shape(self, n_attackers_in_box: int,
                        filter_zone: Optional[str] = None) -> dict:
        """Choose the most likely (n_zonal, n_man, n_short, n_counter) for the
        opponent. Falls back to the global typical shape when zone-filtered
        sample is too small."""
        # Optionally filter to corners matching the delivery zone of the planned attack
        if filter_zone is not None:
            filtered_corners = [c for c in self.corners
                                  if c.get("delivery", {}).get("zone") == filter_zone]
            if len(filtered_corners) < 5:
                filtered_corners = self.corners  # fall back to all
                used_filter = False
                filter_note = (f" Sample too small for zone={filter_zone} "
                                f"(< 5 corners), using all corners instead.")
            else:
                used_filter = True
                filter_note = f" Filtered to corners delivered to {filter_zone}."
        else:
            filtered_corners = self.corners
            used_filter = False
            filter_note = ""

        # Mean role counts across the (possibly filtered) corners
        means = {k: float(np.mean([c["summary_stats"].get(k, 0)
                                     for c in filtered_corners]))
                  for k in ("n_zonal", "n_man", "n_short", "n_counter")}
        rounded = {k: int(round(v)) for k, v in means.items()}

        # Confidence: based on sample size and shape homogeneity
        n = len(filtered_corners)
        shape_counts = Counter((c["summary_stats"].get("n_zonal",0),
                                 c["summary_stats"].get("n_man",0),
                                 c["summary_stats"].get("n_short",0),
                                 c["summary_stats"].get("n_counter",0))
                                for c in filtered_corners)
        top_shape, top_n = shape_counts.most_common(1)[0]
        homogeneity = top_n / n

        if n >= 15 and homogeneity >= 0.5:
            confidence = "high"
        elif n >= 8:
            confidence = "medium"
        else:
            confidence = "low"

        return {
            "n_zonal":     rounded["n_zonal"],
            "n_man":       rounded["n_man"],
            "n_short":     rounded["n_short"],
            "n_counter":   rounded["n_counter"],
            "sample_size": n,
            "confidence":  confidence,
            "explanation": (
                f"Across {n} defending corners, this team averages "
                f"{means['n_zonal']:.1f} zonal, {means['n_man']:.1f} man, "
                f"{means['n_short']:.1f} short, {means['n_counter']:.1f} counter."
                + filter_note
            ),
        }

    # ------------------------------------------------------------------
    # Per-jersey ranking by role frequency
    # ------------------------------------------------------------------
    def _rank_by_role(self, role: str) -> list[tuple[int, int, float]]:
        """Return [(jersey, count, frequency_as_role), ...] sorted desc."""
        out = []
        for j, history in self._per_jersey.items():
            total = len(history)
            count = sum(1 for h in history if h["role"] == role)
            if count == 0: continue
            out.append((j, count, count / total))
        out.sort(key=lambda t: -t[1])
        return out

    # ------------------------------------------------------------------
    # Predicting positions for ZONAL defenders
    # ------------------------------------------------------------------
    def _zonal_position(self, jersey: int) -> Optional[tuple[float, float]]:
        """Mean position of this player when they were classified as ZONAL."""
        positions = [h["pos"] for h in self._per_jersey.get(jersey, [])
                      if h["role"] == "ZONAL" and h["pos"]]
        if not positions: return None
        arr = np.array(positions)
        return (round(float(arr[:, 0].mean()), 2),
                round(float(arr[:, 1].mean()), 2))

    def _short_position(self, jersey: int) -> Optional[tuple[float, float]]:
        positions = [h["pos"] for h in self._per_jersey.get(jersey, [])
                      if h["role"] == "SHORT" and h["pos"]]
        if not positions: return None
        arr = np.array(positions)
        return (round(float(arr[:, 0].mean()), 2),
                round(float(arr[:, 1].mean()), 2))

    def _counter_position(self, jersey: int) -> Optional[tuple[float, float]]:
        positions = [h["pos"] for h in self._per_jersey.get(jersey, [])
                      if h["role"] == "COUNTER" and h["pos"]]
        if not positions: return None
        arr = np.array(positions)
        return (round(float(arr[:, 0].mean()), 2),
                round(float(arr[:, 1].mean()), 2))

    # ------------------------------------------------------------------
    # Main prediction
    # ------------------------------------------------------------------
    def predict_defensive_setup(self, planned_attack: dict) -> dict:
        attackers = planned_attack.get("attackers", [])
        delivery_zone = planned_attack.get("delivery_zone")
        n_att_in_box = sum(
            1 for a in attackers
            if a.get("end_pos") and a["end_pos"][0] > PE_X  # roughly inside PA
        )
        if n_att_in_box == 0:
            n_att_in_box = len(attackers)

        # ---- Shape ----
        shape = self._typical_shape(n_att_in_box, filter_zone=delivery_zone)
        n_zonal, n_man = shape["n_zonal"], shape["n_man"]
        n_short, n_counter = shape["n_short"], shape["n_counter"]

        used_jerseys: set[int] = set()
        defenders_out = []

        # ---- ZONAL: pick most-frequent zonal jerseys ----
        zonal_candidates = self._rank_by_role("ZONAL")
        zonal_chosen = []
        for jersey, count, freq in zonal_candidates:
            if len(zonal_chosen) >= n_zonal: break
            pos = self._zonal_position(jersey)
            if pos is None: continue
            zonal_chosen.append((jersey, count, freq, pos))
            used_jerseys.add(jersey)

        for jersey, count, freq, pos in zonal_chosen:
            history = self._per_jersey[jersey]
            zonal_history = [h for h in history if h["role"] == "ZONAL"]
            n_zonal_corners = len(zonal_history)
            zone = zone_of(pos[0], pos[1])
            evidence = [h["corner_id"] for h in zonal_history[:6]]
            name = next((h["name"] for h in history if h["name"]), "")
            defenders_out.append({
                "jersey":             jersey,
                "player_name":        name,
                "predicted_role":     "ZONAL",
                "predicted_position": list(pos),
                "marks_jersey":       None,
                "explanation": (
                    f"Player #{jersey} typically guards zone {zone}. "
                    f"Observed as ZONAL in {n_zonal_corners} of "
                    f"{len(history)} of this opponent's defending corners "
                    f"({freq*100:.0f}%)."
                ),
                "evidence_corners":   evidence,
            })

        # ---- MAN: pick most-frequent man-markers, assign by Hungarian ----
        man_candidates = self._rank_by_role("MAN")
        man_jerseys = []
        for jersey, count, freq in man_candidates:
            if len(man_jerseys) >= n_man: break
            if jersey in used_jerseys: continue
            man_jerseys.append((jersey, count, freq))
            used_jerseys.add(jersey)

        # Rank planned attackers by "threat" (proximity to goal at end_pos)
        threat_sorted = sorted(
            enumerate(attackers),
            key=lambda ia: -(ia[1].get("end_pos") or [0, 0])[0]
        )
        threat_attackers = threat_sorted[:len(man_jerseys)]

        if man_jerseys and threat_attackers:
            # Build cost matrix: rows = man defenders' historical mean positions
            # when they man-marked, cols = planned attackers' end positions.
            cost = np.full((len(man_jerseys), len(threat_attackers)), 99.0)
            man_positions = []
            for i, (j, _, _) in enumerate(man_jerseys):
                hist_man = [h for h in self._per_jersey[j] if h["role"] == "MAN" and h["pos"]]
                if hist_man:
                    arr = np.array([h["pos"] for h in hist_man])
                    mean_pos = (float(arr[:, 0].mean()), float(arr[:, 1].mean()))
                else:
                    mean_pos = (40.0, 0.0)
                man_positions.append(mean_pos)
                for k, (_, a) in enumerate(threat_attackers):
                    end = a.get("end_pos") or (0.0, 0.0)
                    cost[i, k] = math.hypot(end[0] - mean_pos[0], end[1] - mean_pos[1])

            row_ind, col_ind = linear_sum_assignment(cost)
            assignment = {row_ind[i]: col_ind[i] for i in range(len(row_ind))}

            for i, (jersey, count, freq) in enumerate(man_jerseys):
                hist = self._per_jersey[jersey]
                hist_man = [h for h in hist if h["role"] == "MAN"]
                name = next((h["name"] for h in hist if h["name"]), "")
                if i in assignment:
                    att_idx, att = threat_attackers[assignment[i]]
                    end_pos = att.get("end_pos") or [0.0, 0.0]
                    # Predicted defender position = attacker's planned end pos
                    # nudged ~1m goalwards (typical man-marker offset).
                    pred_pos = [round(end_pos[0] + 1.0, 2), round(end_pos[1], 2)]
                    marks_jersey = att.get("jersey", att_idx)
                    explanation = (
                        f"Player #{jersey} typically picks up the most "
                        f"goal-side attacker. Observed as MAN-marker in "
                        f"{len(hist_man)} of {len(hist)} corners "
                        f"({freq*100:.0f}%). Predicted to track the "
                        f"attacker planned at "
                        f"({end_pos[0]:.1f}, {end_pos[1]:.1f})."
                    )
                else:
                    pred_pos = list(man_positions[i])
                    marks_jersey = None
                    explanation = (
                        f"Player #{jersey} is a frequent man-marker "
                        f"({freq*100:.0f}% of corners) but the planned "
                        f"attack has fewer attackers than the team's "
                        f"typical man-marker count."
                    )
                defenders_out.append({
                    "jersey":             jersey,
                    "player_name":        name,
                    "predicted_role":     "MAN",
                    "predicted_position": pred_pos,
                    "marks_jersey":       marks_jersey,
                    "explanation":        explanation,
                    "evidence_corners":   [h["corner_id"] for h in hist_man[:6]],
                })

        # ---- SHORT ----
        if n_short > 0:
            short_candidates = self._rank_by_role("SHORT")
            for jersey, count, freq in short_candidates[:n_short]:
                if jersey in used_jerseys: continue
                pos = self._short_position(jersey) or (40.0, 30.0)
                hist = self._per_jersey[jersey]
                hist_short = [h for h in hist if h["role"] == "SHORT"]
                name = next((h["name"] for h in hist if h["name"]), "")
                defenders_out.append({
                    "jersey":             jersey,
                    "player_name":        name,
                    "predicted_role":     "SHORT",
                    "predicted_position": list(pos),
                    "marks_jersey":       None,
                    "explanation": (
                        f"Player #{jersey} typically engages the short-corner "
                        f"option. Observed as SHORT in {len(hist_short)} of "
                        f"{len(hist)} corners ({freq*100:.0f}%)."
                    ),
                    "evidence_corners":   [h["corner_id"] for h in hist_short[:6]],
                })
                used_jerseys.add(jersey)

        # ---- COUNTER ----
        if n_counter > 0:
            counter_candidates = self._rank_by_role("COUNTER")
            for jersey, count, freq in counter_candidates[:n_counter]:
                if jersey in used_jerseys: continue
                pos = self._counter_position(jersey) or (10.0, 0.0)
                hist = self._per_jersey[jersey]
                hist_c = [h for h in hist if h["role"] == "COUNTER"]
                name = next((h["name"] for h in hist if h["name"]), "")
                defenders_out.append({
                    "jersey":             jersey,
                    "player_name":        name,
                    "predicted_role":     "COUNTER",
                    "predicted_position": list(pos),
                    "marks_jersey":       None,
                    "explanation": (
                        f"Player #{jersey} typically stays high for the "
                        f"counter. Observed as COUNTER in {len(hist_c)} of "
                        f"{len(hist)} corners ({freq*100:.0f}%)."
                    ),
                    "evidence_corners":   [h["corner_id"] for h in hist_c[:6]],
                })
                used_jerseys.add(jersey)

        # ---- Open zones ----
        open_zones = self._find_open_zones(defenders_out, attackers)

        return {
            "shape":      shape,
            "defenders":  defenders_out,
            "open_zones": open_zones,
        }

    # ------------------------------------------------------------------
    # Open-zone detection
    # ------------------------------------------------------------------
    def _find_open_zones(self, defenders: list[dict],
                          planned_attackers: list[dict],
                          threshold_m: float = 3.0) -> list[dict]:
        """Flag central zones where no predicted defender is within
        ``threshold_m`` of the zone centre."""
        open_zones = []
        for zone, centre in _ZONE_CENTRES.items():
            nearest = min(
                (math.hypot(d["predicted_position"][0] - centre[0],
                             d["predicted_position"][1] - centre[1])
                 for d in defenders), default=99.0)
            if nearest > threshold_m:
                # Only flag if at least one planned attacker actually ends
                # near this zone (otherwise it being open is irrelevant).
                attacker_here = any(
                    math.hypot((a.get("end_pos") or [0,0])[0] - centre[0],
                                (a.get("end_pos") or [0,0])[1] - centre[1]) < 4.0
                    for a in planned_attackers
                )
                if attacker_here:
                    open_zones.append({
                        "zone": zone,
                        "explanation": (
                            f"No defender historically covers zone {zone} "
                            f"in this opponent's setup; their nearest "
                            f"predicted defender is {nearest:.1f} m away. "
                            f"A planned attacker ends near this zone — "
                            f"potentially exploitable."
                        ),
                    })
        return open_zones


# ---------------------------------------------------------------------------
# Convenience for the UI: a minimal example planned-attack dict
# ---------------------------------------------------------------------------
EXAMPLE_PLANNED_ATTACK = {
    "corner_side":   "R",
    "delivery_zone": "GA2",
    "attackers": [
        {"jersey": 9,  "start_pos": [40, +3], "end_pos": [48, +1], "role_intent": "TARGET"},
        {"jersey": 10, "start_pos": [40,  0], "end_pos": [44, +5], "role_intent": "DECOY"},
        {"jersey": 11, "start_pos": [40, -3], "end_pos": [44, -5], "role_intent": "DECOY"},
        {"jersey": 4,  "start_pos": [38, +2], "end_pos": [49, -3], "role_intent": "TARGET"},
        {"jersey": 5,  "start_pos": [36, -2], "end_pos": [47,  0], "role_intent": "BLOCK_GK"},
    ]
}
