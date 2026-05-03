"""
REST API for the Mountain Battalion dashboard — uses repo-root ``data_helpers`` (Postgres or CSV).

Run from repository root::

    pip install -r requirements.txt -r backend/requirements.txt
    uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
"""
from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env.local")

from backend.app.auth_database import bootstrap_primary_user_if_empty, init_auth_tables
from backend.app.routes_auth import router as auth_router
from backend.app.routes_dashboard import router as dashboard_router
from backend.app.routes_tel import router as tel_router
from backend.app.routes_athena import router as athena_router
from backend.app.routes_mission import router as mission_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_auth_tables()
    bootstrap_primary_user_if_empty()
    yield


app = FastAPI(title="MTN BN Leader Performance API", version="0.3.0", lifespan=lifespan)

_origins = __import__("os").getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _origins if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(dashboard_router)
app.include_router(tel_router)
app.include_router(athena_router)
app.include_router(mission_router)


@app.get("/health")
def health():
    return {"status": "ok"}
