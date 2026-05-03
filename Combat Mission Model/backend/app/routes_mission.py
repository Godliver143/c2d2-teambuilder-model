"""
backend/app/routes_mission.py

ML-powered mission team recommendation endpoint.
Add to main.py:
    from backend.app.routes_mission import router as mission_router
    app.include_router(mission_router)
"""
from __future__ import annotations

import json
import logging
import os
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.orm import Session

from backend.app.auth_models import User
from backend.app.deps import get_current_user
from backend.app.auth_database import get_db
from backend.app.services.mission_model import (
    LeaderProfile,
    MissionContext,
    recommend_team,
)

_log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/mission", tags=["mission"])

SCORE_COLS = [
    "planning", "attention_to_detail", "time_management",
    "decisiveness", "tactics",
    "ump_planning", "ump_attention_to_detail", "ump_time_management",
    "ump_decisiveness", "ump_tactics",
]


# ── Request schema ────────────────────────────────────────────────────────────

class MissionRequest(BaseModel):
    mission_type: str = Field(..., description=(
        "One of: LINEAR AMBUSH, MOVEMENT TO CONTACT, AREA RECON, "
        "SQUAD AMBUSH, SQUAD ATTACK, REACT TO UAS, REACT TO IDF, BREAK CONTACT"
    ))
    sleep_hours: float = Field(..., ge=0, le=24)
    temperature_f: Optional[float] = Field(None, description="Ambient temperature in °F")
    wind: Optional[str] = Field(None, description="calm | moderate | gusty | high")
    terrain: Optional[str] = None
    difficulty: Optional[str] = None
    enemy_forces: Optional[str] = None
    team_size: int = Field(default=6, ge=2, le=12)
    notes: Optional[str] = None


# ── DB helpers ────────────────────────────────────────────────────────────────

def _load_candidates(db: Session, owner_ids: list[int]) -> list[LeaderProfile]:
    """
    Load all leaders with their average evaluation scores.
    Requires at least 1 graded evaluation to be included.
    """
    rows = db.execute(text("""
        SELECT
            l.id,
            l.rank || ' ' || l.name   AS full_name,
            l.name,
            l.rank,
            l.unit,
            ROUND(AVG(e.planning)::numeric, 3)                AS planning,
            ROUND(AVG(e.attention_to_detail)::numeric, 3)     AS attention_to_detail,
            ROUND(AVG(e.time_management)::numeric, 3)         AS time_management,
            ROUND(AVG(e.decisiveness)::numeric, 3)            AS decisiveness,
            ROUND(AVG(e.tactics)::numeric, 3)                 AS tactics,
            ROUND(AVG(e.ump_planning)::numeric, 3)            AS ump_planning,
            ROUND(AVG(e.ump_attention_to_detail)::numeric, 3) AS ump_attention_to_detail,
            ROUND(AVG(e.ump_time_management)::numeric, 3)     AS ump_time_management,
            ROUND(AVG(e.ump_decisiveness)::numeric, 3)        AS ump_decisiveness,
            ROUND(AVG(e.ump_tactics)::numeric, 3)             AS ump_tactics,
            COUNT(e.id)                                       AS eval_count
        FROM leaders l
        JOIN evaluations e ON e.leader_id = l.id
        WHERE l.owner_user_id = ANY(:oids)
        GROUP BY l.id, l.rank, l.name, l.unit
        HAVING COUNT(e.id) >= 1
        ORDER BY l.name
    """), {"oids": owner_ids}).fetchall()

    return [
        LeaderProfile(
            id=int(r.id),
            name=r.name,
            unit=r.unit,
            rank=r.rank,
            planning=float(r.planning or 3),
            attention_to_detail=float(r.attention_to_detail or 3),
            time_management=float(r.time_management or 3),
            decisiveness=float(r.decisiveness or 3),
            tactics=float(r.tactics or 3),
            ump_planning=float(r.ump_planning or 3),
            ump_attention_to_detail=float(r.ump_attention_to_detail or 3),
            ump_time_management=float(r.ump_time_management or 3),
            ump_decisiveness=float(r.ump_decisiveness or 3),
            ump_tactics=float(r.ump_tactics or 3),
            eval_count=int(r.eval_count),
        )
        for r in rows
    ]


def _serialize_result(result) -> dict:
    """Convert TeamRecommendation to JSON-serialisable dict."""
    def scored_to_dict(s):
        return {
            "id": s.leader.id,
            "name": s.leader.name,
            "rank": s.leader.rank,
            "unit": s.leader.unit,
            "eval_count": s.leader.eval_count,
            "raw_predicted_score": s.raw_predicted_score,
            "adjusted_predicted_score": s.adjusted_predicted_score,
            "condition_penalty_pct": s.condition_penalty_pct,
            "mission_fit_rank": s.mission_fit_rank,
            "role_note": s.role_note,
            "scores": {
                "planning":             s.leader.planning,
                "attention_to_detail":  s.leader.attention_to_detail,
                "time_management":      s.leader.time_management,
                "decisiveness":         s.leader.decisiveness,
                "tactics":              s.leader.tactics,
            },
        }
    return {
        "team": [scored_to_dict(s) for s in result.team],
        "alternates": [scored_to_dict(s) for s in result.alternates],
        "predicted_team_score": result.predicted_team_score,
        "confidence_note": result.confidence_note,
        "weights_applied": {k: round(v, 4) for k, v in result.weights_applied.items()},
        "mission_type": result.mission_context.mission_type,
        "condition_penalty_pct": result.team[0].condition_penalty_pct if result.team else 0,
    }


# ── Endpoint ──────────────────────────────────────────────────────────────────

@router.post("/recommend-team")
async def recommend_team_endpoint(
    body: MissionRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """
    Returns team recommendation + streaming Claude explanation.

    SSE format:
      data: {"type": "team", ...}      — structured recommendation (arrives first)
      data: {"type": "text", "text": "..."}  — Claude explanation chunks
      data: [DONE]
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()

    # Get owner scope (matches existing dashboard pattern)
    from data_helpers import _normalize_owner_ids
    owner_ids = _normalize_owner_ids(user.id)

    candidates = _load_candidates(db, list(owner_ids))
    if len(candidates) < body.team_size:
        raise HTTPException(
            status_code=422,
            detail=f"Need at least {body.team_size} evaluated leaders. Found {len(candidates)}.",
        )

    context = MissionContext(
        mission_type=body.mission_type,
        sleep_hours=body.sleep_hours,
        temperature_f=body.temperature_f,
        wind=body.wind,
        terrain=body.terrain,
        difficulty=body.difficulty,
        enemy_forces=body.enemy_forces,
        notes=body.notes,
    )

    result = recommend_team(candidates, context, team_size=body.team_size)
    payload = _serialize_result(result)

    async def generate():
        # 1. Emit structured data immediately so UI can render cards
        yield f"data: {json.dumps({'type': 'team', **payload})}\n\n"

        # 2. Stream Claude explanation
        if not api_key:
            yield f"data: {json.dumps({'type': 'text', 'text': 'ANTHROPIC_API_KEY not set — structured recommendation above is from the ML model.'})}\n\n"
            yield "data: [DONE]\n\n"
            return

        try:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)

            context_text = _build_claude_context(body, payload)

            with client.messages.stream(
                model="claude-sonnet-4-20250514",
                max_tokens=600,
                system="""You are a mission planning advisor for a military unit.
Given an ML-generated team recommendation and mission parameters, write a concise
briefing (3 short paragraphs max) covering:
1. Why this team fits this specific mission
2. Any risk factors — condition penalties, leaders with low scores in critical areas
3. The best alternate and why they weren't selected
Be direct. Use military language. Never say "I" or reference yourself.""",
                messages=[{"role": "user", "content": context_text}],
            ) as stream:
                for chunk in stream.text_stream:
                    yield f"data: {json.dumps({'type': 'text', 'text': chunk})}\n\n"

        except Exception as exc:
            _log.error("Claude stream error: %s", exc)
            yield f"data: {json.dumps({'type': 'text', 'text': f'Analysis unavailable: {exc}'})}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/mission-types")
def get_mission_types(user: User = Depends(get_current_user)):
    """Returns the mission types the model was trained on."""
    import json
    from pathlib import Path
    meta_path = Path(__file__).resolve().parent.parent.parent / "models" / "model_metadata.json"
    if not meta_path.exists():
        return {"mission_types": []}
    with open(meta_path) as f:
        meta = json.load(f)
    return {"mission_types": meta.get("mission_classes", [])}


def _build_claude_context(body: MissionRequest, payload: dict) -> str:
    lines = [
        f"Mission: {body.mission_type}",
        f"Sleep: {body.sleep_hours}h | Temp: {body.temperature_f or 'N/A'}°F | Wind: {body.wind or 'N/A'}",
        f"Condition penalty: {payload['condition_penalty_pct']}%",
        f"Predicted team score: {payload['predicted_team_score']}/5",
        "",
        "Recommended team (ML-ranked by adjusted predicted score):",
    ]
    for m in payload["team"]:
        role = f" [{m['role_note']}]" if m["role_note"] else ""
        lines.append(
            f"  {m['rank']} {m['name']} ({m['unit']}) "
            f"adj={m['adjusted_predicted_score']} raw={m['raw_predicted_score']}"
            f" — T:{m['scores']['tactics']} D:{m['scores']['decisiveness']}"
            f" P:{m['scores']['planning']}{role}"
        )
    lines += ["", "Top alternates:"]
    for a in payload["alternates"]:
        lines.append(f"  {a['name']} adj={a['adjusted_predicted_score']}")
    if body.notes:
        lines += ["", f"Commander notes: {body.notes}"]
    lines += ["", f"Model confidence: {payload['confidence_note']}"]
    return "\n".join(lines)
