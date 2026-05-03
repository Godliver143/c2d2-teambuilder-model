"""
Mission environment inputs are not present in evaluations_full.csv, so they are not
learned by the GradientBoosting model. They are applied at inference time as
transparent multipliers on each soldier's predicted score (same pattern as
backend/app/services/mission_model.py for sleep/weather).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional

# Kept in sync with schemas.MissionContextInput (Literal unions / OpenAPI enums).
ALLOWED_WIND = frozenset({"calm", "moderate", "gusty", "high"})
ALLOWED_TERRAIN = frozenset(
    {"urban", "mountain", "jungle", "desert", "wooded", "forest", "open", "mixed"}
)
ALLOWED_DIFFICULTY = frozenset(
    {"low", "easy", "moderate", "medium", "high", "hard", "extreme"}
)
ALLOWED_ENEMY_FORCES = frozenset(
    {"light", "few", "moderate", "medium", "heavy", "numerous", "overwhelming"}
)


@dataclass
class MissionContext:
    sleep_hours: float = 7.0
    temperature_f: Optional[float] = None
    wind: Optional[str] = None
    terrain: Optional[str] = None
    difficulty: Optional[str] = None
    enemy_forces: Optional[str] = None
    notes: Optional[str] = None


def sleep_modifier(hours: float) -> float:
    if hours >= 7:
        return 1.00
    if hours >= 6:
        return 0.97
    if hours >= 5:
        return 0.92
    if hours >= 4:
        return 0.85
    if hours >= 3:
        return 0.74
    if hours >= 2:
        return 0.62
    return 0.50


def weather_modifier(temp_f: Optional[float], wind: Optional[str]) -> float:
    mod = 1.0
    if temp_f is not None:
        if temp_f < 10:
            mod *= 0.78
        elif temp_f < 20:
            mod *= 0.84
        elif temp_f < 32:
            mod *= 0.91
        elif temp_f > 100:
            mod *= 0.84
        elif temp_f > 95:
            mod *= 0.90
    w = (wind or "calm").lower().strip()
    if w not in ALLOWED_WIND:
        raise ValueError(
            f"Invalid wind {wind!r}. Allowed: {', '.join(sorted(ALLOWED_WIND))}"
        )
    wind_map = {"calm": 1.0, "moderate": 0.97, "gusty": 0.92, "high": 0.86}
    mod *= wind_map[w]
    return mod


def terrain_modifier(terrain: Optional[str]) -> float:
    if not terrain:
        return 1.0
    t = terrain.lower().strip()
    if t not in ALLOWED_TERRAIN:
        raise ValueError(
            f"Invalid terrain {terrain!r}. Allowed: {', '.join(sorted(ALLOWED_TERRAIN))}"
        )
    m = {
        "urban": 0.98,
        "mountain": 0.94,
        "jungle": 0.92,
        "desert": 0.96,
        "wooded": 1.0,
        "forest": 1.0,
        "open": 1.02,
        "mixed": 1.0,
    }
    return m[t]


def difficulty_modifier(difficulty: Optional[str]) -> float:
    if not difficulty:
        return 1.0
    d = difficulty.lower().strip()
    if d not in ALLOWED_DIFFICULTY:
        raise ValueError(
            f"Invalid difficulty {difficulty!r}. Allowed: {', '.join(sorted(ALLOWED_DIFFICULTY))}"
        )
    m = {
        "low": 1.02,
        "easy": 1.02,
        "moderate": 1.0,
        "medium": 1.0,
        "high": 0.94,
        "hard": 0.92,
        "extreme": 0.88,
    }
    return m[d]


def enemy_forces_modifier(enemy: Optional[str]) -> float:
    if not enemy:
        return 1.0
    e = enemy.lower().strip()
    if e not in ALLOWED_ENEMY_FORCES:
        raise ValueError(
            f"Invalid enemy_forces {enemy!r}. Allowed: {', '.join(sorted(ALLOWED_ENEMY_FORCES))}"
        )
    m = {
        "light": 1.02,
        "few": 1.02,
        "moderate": 1.0,
        "medium": 1.0,
        "heavy": 0.94,
        "numerous": 0.90,
        "overwhelming": 0.85,
    }
    return m[e]


def combined_condition_modifier(ctx: MissionContext) -> tuple[float, dict[str, float]]:
    s = sleep_modifier(ctx.sleep_hours)
    w = weather_modifier(ctx.temperature_f, ctx.wind)
    tr = terrain_modifier(ctx.terrain)
    df = difficulty_modifier(ctx.difficulty)
    en = enemy_forces_modifier(ctx.enemy_forces)
    combined = s * w * tr * df * en
    combined = max(0.45, min(1.0, combined))
    breakdown = {
        "sleep": float(round(s, 4)),
        "weather": float(round(w, 4)),
        "terrain": float(round(tr, 4)),
        "difficulty": float(round(df, 4)),
        "enemy_forces": float(round(en, 4)),
        "combined": float(round(combined, 4)),
    }
    breakdown["condition_penalty_pct"] = float(round((1.0 - combined) * 100, 1))
    return combined, breakdown


def context_public_dict(ctx: MissionContext) -> dict[str, Any]:
    """JSON-serialisable echo of what the caller sent (for API responses)."""
    return asdict(ctx)
