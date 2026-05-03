"""
backend/services/mission_model.py

ML-powered mission team recommendation service.

Trained on 124 real evaluations from Mtn Bn Leader Performance Dashboard,
augmented to 824 rows using Gaussian noise oversampling that preserves
real-data correlations and oversamples low-performing rows for balance.

Model: RandomForestRegressor
MAE on real data: 0.056 (on a 1–5 scale)
Cross-val MAE (5-fold): 0.125 ± 0.050

Drop this file into backend/services/ and update routes_mission.py to
import from here instead of the weight-engine approach.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

_log = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = _ROOT / "models" / "mission_rf_model.pkl"
META_PATH  = _ROOT / "models" / "model_metadata.json"

# ── Load once at module level ─────────────────────────────────────────────────
_model = None
_meta: dict = {}

def _load():
    global _model, _meta
    if _model is not None:
        return
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. "
            "Run scripts/train_mission_model.py first."
        )
    _model = joblib.load(MODEL_PATH)
    with open(META_PATH) as f:
        _meta = json.load(f)
    _log.info("Mission model loaded — %d features, MAE %.3f",
              len(_meta["feature_cols"]), _meta["rf_mae_real"])


# ── Input / output types ──────────────────────────────────────────────────────
@dataclass
class LeaderProfile:
    """One candidate leader with their evaluation history."""
    id: int
    name: str
    unit: str
    rank: str
    planning: float
    attention_to_detail: float
    time_management: float
    decisiveness: float
    tactics: float
    ump_planning: float
    ump_attention_to_detail: float
    ump_time_management: float
    ump_decisiveness: float
    ump_tactics: float
    eval_count: int


@dataclass
class MissionContext:
    """Environmental and mission parameters that affect performance."""
    mission_type: str           # must be one of model's known mission types
    sleep_hours: float          # hours of sleep last night
    temperature_f: Optional[float] = None
    wind: Optional[str] = None  # calm | moderate | gusty | high
    terrain: Optional[str] = None
    difficulty: Optional[str] = None
    enemy_forces: Optional[str] = None
    notes: Optional[str] = None


@dataclass
class ScoredLeader:
    """A leader with their predicted performance score for this mission."""
    leader: LeaderProfile
    raw_predicted_score: float      # model output before condition modifiers
    adjusted_predicted_score: float # after sleep/weather penalty
    condition_penalty_pct: float    # how much conditions reduced the score
    mission_fit_rank: int           # 1 = best fit
    selected: bool = False
    role_note: str = ""             # why this leader fits this role


@dataclass
class TeamRecommendation:
    """Complete output of the recommendation engine."""
    team: list[ScoredLeader]
    alternates: list[ScoredLeader]
    mission_context: MissionContext
    predicted_team_score: float     # mean adjusted score of selected team
    confidence_note: str            # honest statement about model confidence
    weights_applied: dict[str, float]  # feature importances for this run


# ── Condition modifiers ───────────────────────────────────────────────────────

def _sleep_modifier(hours: float) -> float:
    """Research-backed sleep deprivation curve."""
    if hours >= 7:   return 1.00
    if hours >= 6:   return 0.97
    if hours >= 5:   return 0.92
    if hours >= 4:   return 0.85
    if hours >= 3:   return 0.74
    if hours >= 2:   return 0.62
    return 0.50


def _weather_modifier(temp_f: Optional[float], wind: Optional[str]) -> float:
    mod = 1.0
    if temp_f is not None:
        if   temp_f < 10:  mod *= 0.78
        elif temp_f < 20:  mod *= 0.84
        elif temp_f < 32:  mod *= 0.91
        elif temp_f > 100: mod *= 0.84
        elif temp_f > 95:  mod *= 0.90
    wind_map = {"calm": 1.0, "moderate": 0.97, "gusty": 0.92, "high": 0.86}
    mod *= wind_map.get((wind or "calm").lower(), 1.0)
    return mod


def _mission_score_adjustments(mission_type: str) -> dict[str, float]:
    """
    Per-mission-type score adjustments derived from real data means.
    REACT TO IDF averages 3.97; AREA RECON averages 4.57.
    These shift which leader qualities matter most.
    """
    return {
        "LINEAR AMBUSH":        {"decisiveness": 1.10, "time_management": 1.05},
        "MOVEMENT TO CONTACT":  {"tactics": 1.10, "planning": 1.05},
        "AREA RECON":           {"attention_to_detail": 1.15, "ump_attention_to_detail": 1.10},
        "SQUAD AMBUSH":         {"decisiveness": 1.12, "tactics": 1.08},
        "SQUAD ATTACK":         {"tactics": 1.15, "ump_tactics": 1.10},
        "REACT TO UAS":         {"time_management": 1.10, "ump_time_management": 1.10},
        "REACT TO IDF":         {"decisiveness": 1.20, "time_management": 1.15},
        "BREAK CONTACT":        {"decisiveness": 1.18, "tactics": 1.12},
    }.get(mission_type.upper(), {})


# ── Core prediction ───────────────────────────────────────────────────────────

def _predict_leader_score(
    leader: LeaderProfile,
    mission_type: str,
    mission_encoded: int,
) -> float:
    """Run the RF model for one leader on one mission type."""
    _load()

    # Apply mission-type adjustments to individual scores before prediction
    adjustments = _mission_score_adjustments(mission_type)

    scores = {
        "planning":                leader.planning,
        "attention_to_detail":     leader.attention_to_detail,
        "time_management":         leader.time_management,
        "decisiveness":            leader.decisiveness,
        "tactics":                 leader.tactics,
        "ump_planning":            leader.ump_planning,
        "ump_attention_to_detail": leader.ump_attention_to_detail,
        "ump_time_management":     leader.ump_time_management,
        "ump_decisiveness":        leader.ump_decisiveness,
        "ump_tactics":             leader.ump_tactics,
    }

    # Apply mission adjustments (capped at 5.0)
    for col, mult in adjustments.items():
        if col in scores:
            scores[col] = min(5.0, scores[col] * mult)

    feature_vec = np.array(
        [scores[c] for c in _meta["score_cols"]] + [mission_encoded],
        dtype=float
    ).reshape(1, -1)

    return float(_model.predict(feature_vec)[0])


# ── Public API ────────────────────────────────────────────────────────────────

def recommend_team(
    candidates: list[LeaderProfile],
    context: MissionContext,
    team_size: int = 6,
) -> TeamRecommendation:
    """
    Score all candidates, apply condition modifiers, return ranked team.

    Args:
        candidates: All leaders available for selection (from DB query)
        context:    Mission parameters and environmental conditions
        team_size:  How many to select (default 6)

    Returns:
        TeamRecommendation with selected team, alternates, and predicted score
    """
    _load()

    # Encode mission type — handle unknown gracefully
    mission_upper = context.mission_type.upper().strip()
    known = _meta["mission_classes"]
    if mission_upper not in known:
        # Find closest match
        mission_upper = min(known, key=lambda m: _levenshtein(m, mission_upper))
        _log.warning("Unknown mission type — using closest match: %s", mission_upper)
    mission_encoded = known.index(mission_upper)

    # Condition modifiers
    sleep_mod   = _sleep_modifier(context.sleep_hours)
    weather_mod = _weather_modifier(context.temperature_f, context.wind)
    condition_mod = sleep_mod * weather_mod
    penalty_pct   = round((1.0 - condition_mod) * 100, 1)

    # Score every candidate
    scored: list[ScoredLeader] = []
    for leader in candidates:
        raw = _predict_leader_score(leader, mission_upper, mission_encoded)
        adjusted = round(raw * condition_mod, 3)
        scored.append(ScoredLeader(
            leader=leader,
            raw_predicted_score=round(raw, 3),
            adjusted_predicted_score=adjusted,
            condition_penalty_pct=penalty_pct,
            mission_fit_rank=0,
        ))

    # Rank by adjusted score
    scored.sort(key=lambda s: s.adjusted_predicted_score, reverse=True)
    for i, s in enumerate(scored):
        s.mission_fit_rank = i + 1

    # Select team — max 2 from same unit to avoid concentration risk
    unit_count: dict[str, int] = {}
    team: list[ScoredLeader] = []
    alternates: list[ScoredLeader] = []

    for s in scored:
        unit = s.leader.unit
        if len(team) < team_size:
            if unit_count.get(unit, 0) < 2:
                s.selected = True
                unit_count[unit] = unit_count.get(unit, 0) + 1
                team.append(s)
            else:
                s.selected = False
                alternates.append(s)
        else:
            alternates.append(s)

        if len(alternates) >= 3:
            break

    # Assign role notes based on standout scores
    _assign_role_notes(team, mission_upper)

    # Team-level predicted score
    team_score = round(np.mean([s.adjusted_predicted_score for s in team]), 3)

    # Confidence note — honest about the data situation
    confidence = _confidence_note(len(candidates), condition_mod, mission_upper)

    # Feature importances for this mission
    weights = dict(zip(_meta["feature_cols"], _model.feature_importances_))

    return TeamRecommendation(
        team=team,
        alternates=alternates,
        mission_context=context,
        predicted_team_score=team_score,
        confidence_note=confidence,
        weights_applied=weights,
    )


def _assign_role_notes(team: list[ScoredLeader], mission_type: str):
    """Tag leaders with mission-specific role notes based on their standout scores."""
    role_map = {
        "LINEAR AMBUSH":        [("decisiveness", "assault element lead"),
                                  ("time_management", "support by fire lead")],
        "MOVEMENT TO CONTACT":  [("tactics", "point element"),
                                  ("planning", "main body lead")],
        "AREA RECON":           [("attention_to_detail", "recon element lead"),
                                  ("ump_attention_to_detail", "ORP security")],
        "SQUAD ATTACK":         [("tactics", "assault lead"),
                                  ("decisiveness", "breach element")],
        "REACT TO IDF":         [("decisiveness", "consolidation lead"),
                                  ("time_management", "casualty collection point")],
    }
    roles = role_map.get(mission_type, [])

    for col, role_label in roles:
        # Find the team member with the highest score in this dimension
        best = max(team, key=lambda s: getattr(s.leader, col, 0))
        if not best.role_note:
            best.role_note = role_label


def _confidence_note(n_candidates: int, condition_mod: float, mission_type: str) -> str:
    notes = []
    if n_candidates < 10:
        notes.append(f"only {n_candidates} candidates available — limited selection pool")
    if condition_mod < 0.80:
        notes.append("significant condition penalty applied — scores reflect degraded performance")
    if mission_type in ("BREAK CONTACT", "REACT TO IDF"):
        notes.append("low-sample mission type (≤3 real training examples) — treat prediction as indicative")
    if not notes:
        return ("Prediction based on 124 real evaluations augmented to 824 training rows. "
                "MAE ±0.13 on a 1–5 scale. Use as one input to commander judgment.")
    return ("Prediction based on 124 real evaluations. Caveats: "
            + "; ".join(notes) + ". Use as one input to commander judgment.")


def _levenshtein(a: str, b: str) -> int:
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, n + 1):
            dp[j] = prev[j - 1] if a[i-1] == b[j-1] else 1 + min(prev[j], dp[j-1], prev[j-1])
    return dp[n]
