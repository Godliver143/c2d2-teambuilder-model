"""
HTTP API for the mission-performance model — deployable ahead of the Athena AI layer.

Athena (later) can call this service for predictions, rankings, team options, and
read model registry metadata from ``GET /model/metadata``.

Endpoints:
  GET  /                                — service index (links for load balancers / humans)
  GET  /health                          — health + artifact paths
  GET  /model/metadata                  — training metadata JSON (registry / contracts)
  GET  /mission-types                   — list supported mission types
  GET  /mission-context/enums           — allowed mission_context enum strings
  POST /team/select                     — team options + pool breakdowns
  GET  /soldiers/rankings/{mission_type}— rank soldiers (model scores only; no context)
  POST /soldiers/rankings              — rank soldiers with optional mission_context
  GET  /soldiers                            — list all soldier profiles
  GET  /soldiers/{leader_identifier}    — get one soldier's profile
"""

import json
import os
import re
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse

from mission_context import (
    ALLOWED_DIFFICULTY,
    ALLOWED_ENEMY_FORCES,
    ALLOWED_TERRAIN,
    ALLOWED_WIND,
    MissionContext,
    combined_condition_modifier,
    context_public_dict,
)
from model import Athena, AVAILABLE_MISSION_TYPES, MODEL_METADATA_PATH, MODEL_PATH
from schemas import (
    HealthResponse,
    MissionContextInput,
    RankingsRequest,
    RankingsResponse,
    SoldierProfile,
    SoldiersListResponse,
    TeamSelectRequest,
    TeamSelectResponse,
)


def _resolved_artifact_path() -> str:
    return os.path.abspath(os.getenv("MODEL_PATH", str(MODEL_PATH)))


def _resolved_metadata_path() -> str:
    return os.path.abspath(os.getenv("MODEL_METADATA_PATH", str(MODEL_METADATA_PATH)))


def _sanitize_train_metrics(raw: dict | None) -> dict[str, float] | None:
    if not raw:
        return None
    out: dict[str, float] = {}
    for k, v in raw.items():
        try:
            out[str(k)] = float(v)
        except (TypeError, ValueError):
            continue
    return out or None


# ── Startup: train or load model ──────────────────────────────
athena: Athena | None = None


def _require_athena() -> Athena:
    if athena is None:
        raise HTTPException(status_code=503, detail="Model is not ready yet.")
    return athena


def _validate_mission_type(mission_type: str) -> None:
    if mission_type not in AVAILABLE_MISSION_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid mission_type '{mission_type}'. "
            f"Choose from: {AVAILABLE_MISSION_TYPES}",
        )


def _mission_context_from_input(
    body: MissionContextInput | None,
) -> MissionContext | None:
    if body is None:
        return None
    return MissionContext(
        sleep_hours=body.sleep_hours,
        temperature_f=body.temperature_f,
        wind=body.wind,
        terrain=body.terrain,
        difficulty=body.difficulty,
        enemy_forces=body.enemy_forces,
        notes=body.notes,
    )


def _normalize_cors_origins(raw: str) -> list[str]:
    """Split comma-/whitespace-separated values; dedupe; strip trailing slashes (no paths in origins)."""
    seen: dict[str, None] = {}
    out: list[str] = []
    for chunk in re.split(r"[,\s]+", raw.strip()):
        piece = chunk.strip().rstrip("/")
        if not piece or piece in seen:
            continue
        seen[piece] = None
        out.append(piece)
    return out


def _configure_cors(app: FastAPI) -> None:
    """
    Cross-origin handling for browsers and cross-site tools.

    ``CORS_ORIGINS``:
      - ``*`` — reflect ``Access-Control-Allow-Origin: *`` (cookies / credentialed
        ``fetch`` from browsers must NOT use ``credentials: \"include\"`` with this setting).
      - Comma-separated list (e.g. ``https://app.example.com``) — only those origins receive
        CORS replies; optionally enable cookies/credentials via ``CORS_ALLOW_CREDENTIALS=true``.

    Uses ``CORSMiddleware`` so ``OPTIONS`` preflight is answered with allow-methods / allow-headers / max_age.
    """
    raw = os.getenv("CORS_ORIGINS", "*").strip()
    if not raw:
        return
    origins = _normalize_cors_origins(raw)
    if "*" in origins and len(origins) > 1:
        raise ValueError(
            "CORS_ORIGINS: use '*' alone or explicit origins; do not combine '*' with other hosts."
        )
    if origins == ["*"]:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=False,
            allow_methods=["*"],
            allow_headers=["*"],
            max_age=int(os.getenv("CORS_PREFLIGHT_MAX_AGE", "86400")),
        )
        return

    if not origins:
        return

    cred_raw = os.getenv("CORS_ALLOW_CREDENTIALS", "true").strip().lower()
    allow_creds = cred_raw in ("1", "true", "yes", "on")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=allow_creds,
        allow_methods=["*"],
        allow_headers=["*"],
        max_age=int(os.getenv("CORS_PREFLIGHT_MAX_AGE", "86400")),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    global athena
    artifact = _resolved_artifact_path()
    if os.path.isfile(artifact):
        print("Loading pre-trained model...")
        athena = Athena.load(artifact)
    else:
        print("Training model from scratch...")
        athena = Athena()
        metrics = athena.train()
        athena.save(artifact)
        athena.save_metadata()
        print(f"Model trained. CV MAE: {metrics['cv_mae_mean']:.4f}")
    yield


app = FastAPI(
    title="Combat mission model — inference API",
    description=(
        "REST inference for the trained mission-performance model. Deploy with the joblib "
        "artifact (``MODEL_PATH``) and JSON metadata (``MODEL_METADATA_PATH``) from "
        "``python train.py``. Designed for downstream integration (e.g. Athena) that adds "
        "commander-facing reasoning; this service exposes scores, profiles, and team options."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

_configure_cors(app)


# ── Root / discovery ──────────────────────────────────────────
@app.get("/", tags=["System"])
def root():
    """Minimal index for probes and human discovery."""
    return {
        "service": "combat-mission-model",
        "version": "1.0.0",
        "health": "/health",
        "docs": "/docs",
        "openapi": "/openapi.json",
        "metadata": "/model/metadata",
        "team_selection_ui": "/ui/",
        "integration_note": (
            "Mission scoring and team construction; pair with Athena for narrative / coaching."
        ),
    }


# ── Health ────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    a = athena
    profiles = a.profiles if a is not None else None
    meta_path = _resolved_metadata_path()
    artifact_path = _resolved_artifact_path()
    return {
        "status": "ok",
        "model_trained": a is not None and a.scorer.feature_cols is not None,
        "soldiers_loaded": len(profiles) if profiles is not None else 0,
        "available_mission_types": list(AVAILABLE_MISSION_TYPES),
        "train_metrics": _sanitize_train_metrics(a.train_metrics if a else None),
        "artifact_path": artifact_path,
        "metadata_path": meta_path,
        "metadata_available": os.path.isfile(meta_path),
    }


@app.get("/model/metadata", tags=["System"])
def model_metadata():
    """
    Training / registry payload written by ``train.py`` (features, CV MAE, mission classes).
    Athena or other services can poll this for contract checks without loading joblib.
    """
    meta_path = Path(_resolved_metadata_path())
    if not meta_path.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"Metadata not found at {meta_path}. Run train.py or mount MODEL_METADATA_PATH.",
        )
    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=500, detail=f"Invalid JSON in metadata file: {e}"
        ) from e
    return JSONResponse(content=payload)


# ── Mission Types ─────────────────────────────────────────────
@app.get("/mission-types", tags=["Reference"])
def mission_types():
    return {"available_mission_types": list(AVAILABLE_MISSION_TYPES)}


@app.get("/mission-context/enums", tags=["Reference"])
def mission_context_enums():
    """Allowed values for ``mission_context`` string fields (matches OpenAPI enums)."""
    return {
        "wind": sorted(ALLOWED_WIND),
        "terrain": sorted(ALLOWED_TERRAIN),
        "difficulty": sorted(ALLOWED_DIFFICULTY),
        "enemy_forces": sorted(ALLOWED_ENEMY_FORCES),
    }


# ── Team Selection ────────────────────────────────────────────
def _team_rows_for_breakdown(team_df, breakdown: dict):
    roles = breakdown.get("strength_roles") or []
    role_by_id = {str(r["leader_identifier"]): r for r in roles}

    def _row_payload(row):
        rid = str(row["leader_identifier"])
        r = role_by_id.get(rid, {})
        return {
            "leader_identifier": rid,
            "leader_name": str(row["leader_name"]),
            "leader_rank": str(row["leader_rank"]),
            "leader_unit": str(row["leader_unit"]),
            "predicted_score": float(row["predicted_score"]),
            "raw_predicted_score": float(row["raw_predicted_score"]),
            "strength_role_key": r.get("role_key"),
            "strength_role_label": r.get("role_label"),
        }

    return [_row_payload(row) for _, row in team_df.iterrows()]


@app.post("/team/select", response_model=TeamSelectResponse, tags=["Team"])
def select_team(request: TeamSelectRequest):
    _validate_mission_type(request.mission_type)
    a = _require_athena()
    ctx = _mission_context_from_input(request.mission_context)
    result = a.select_team(
        request.mission_type,
        request.top_k,
        context=ctx,
        num_team_options=request.num_team_options,
    )

    team_df = result["team"]
    breakdown = result["breakdown"]

    return {
        "mission_type": result["mission_type"],
        "team_score": float(result["team_score"]),
        "team": _team_rows_for_breakdown(team_df, breakdown),
        "breakdown": breakdown,
        "team_options": [
            {
                "option_id": int(opt["option_id"]),
                "team_score": float(opt["team_score"]),
                "team": _team_rows_for_breakdown(opt["team"], opt["breakdown"]),
                "breakdown": opt["breakdown"],
                "slot_order": list(opt["slot_order"]),
            }
            for opt in result["team_options"]
        ],
        "num_team_options_requested": request.num_team_options,
        "num_team_options_returned": len(result["team_options"]),
        "candidate_task_breakdowns": result["candidate_task_breakdowns"],
        "mission_context": result.get("mission_context"),
        "condition_modifiers": result.get("condition_modifiers"),
    }


# ── Soldier Rankings ──────────────────────────────────────────
def _rankings_payload(mission_type: str, ctx: MissionContext | None) -> dict:
    ranked = _require_athena().rank_soldiers(mission_type, context=ctx)
    rows = [
        {
            "leader_identifier": str(row["leader_identifier"]),
            "leader_name": str(row["leader_name"]),
            "leader_rank": str(row["leader_rank"]),
            "leader_unit": str(row["leader_unit"]),
            "predicted_score": float(row["predicted_score"]),
            "raw_predicted_score": float(row["raw_predicted_score"]),
        }
        for _, row in ranked.iterrows()
    ]
    out: dict = {"mission_type": mission_type, "rankings": rows}
    if ctx is not None:
        _, cond = combined_condition_modifier(ctx)
        out["mission_context"] = context_public_dict(ctx)
        out["condition_modifiers"] = cond
    return out


@app.get("/soldiers/rankings/{mission_type}", response_model=RankingsResponse, tags=["Soldiers"])
def get_rankings(mission_type: str):
    """Rankings using raw model scores only (``predicted_score`` == ``raw_predicted_score``)."""
    _validate_mission_type(mission_type)
    return _rankings_payload(mission_type, None)


@app.post("/soldiers/rankings", response_model=RankingsResponse, tags=["Soldiers"])
def post_rankings(request: RankingsRequest):
    """Rankings with optional ``mission_context``; ``predicted_score`` may differ from ``raw_predicted_score``."""
    _validate_mission_type(request.mission_type)
    ctx = _mission_context_from_input(request.mission_context)
    return _rankings_payload(request.mission_type, ctx)


# ── All Soldier Profiles ──────────────────────────────────────
@app.get("/soldiers", response_model=SoldiersListResponse, tags=["Soldiers"])
def list_soldiers():
    profiles = _require_athena().get_all_profiles()
    return {"total": len(profiles), "soldiers": profiles}


# ── Single Soldier Profile ────────────────────────────────────
@app.get("/soldiers/{leader_identifier}", response_model=SoldierProfile, tags=["Soldiers"])
def get_soldier(leader_identifier: str):
    profile = _require_athena().get_soldier_profile(leader_identifier)
    if not profile:
        raise HTTPException(
            status_code=404,
            detail=f"Soldier '{leader_identifier}' not found."
        )
    return profile


_WEB_DIR = Path(__file__).resolve().parent / "web"
_UI_INDEX = _WEB_DIR / "index.html"


@app.get("/ui", include_in_schema=False)
def ui_strip_redirect():
    """Redirect /ui → /ui/ so browser relative URLs behave consistently."""
    return RedirectResponse(url="/ui/", status_code=307)


@app.get("/ui/", include_in_schema=False)
def team_selection_ui_page():
    """Serve the team-selection viewer (same origin as API; use http:// not https:// locally)."""
    if not _UI_INDEX.is_file():
        raise HTTPException(
            status_code=404,
            detail="Web UI missing: expected web/index.html next to main.py.",
        )
    return FileResponse(
        path=_UI_INDEX,
        media_type="text/html; charset=utf-8",
        headers={"Cache-Control": "no-cache"},
    )