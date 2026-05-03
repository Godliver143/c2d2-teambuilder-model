"""
Streamlit UI for the combat mission team builder (same ``Athena`` model as the FastAPI app).

Run locally from repo root::

    pip install -r requirements-streamlit.txt
    streamlit run streamlit_app.py

Deploy on Streamlit Community Cloud: main file ``streamlit_app.py``, requirements
``requirements-streamlit.txt``, Python 3.11.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent
APP_DIR = REPO_ROOT / "Combat Mission Model"

if not APP_DIR.is_dir():
    st.error(f"Expected app directory at `{APP_DIR}` — clone the full repository.")
    st.stop()

# Import model package from the spaced directory without changing its layout.
sys.path.insert(0, str(APP_DIR))
os.chdir(APP_DIR)

os.environ.setdefault("EVAL_PATH", str(APP_DIR / "evaluations_full.csv"))
os.environ.setdefault("DETAIL_PATH", str(APP_DIR / "detailed_full.csv"))
os.environ.setdefault("MODEL_PATH", str(APP_DIR / "athena.joblib"))

from model import AVAILABLE_MISSION_TYPES, Athena  # noqa: E402
from mission_context import (  # noqa: E402
    ALLOWED_DIFFICULTY,
    ALLOWED_ENEMY_FORCES,
    ALLOWED_TERRAIN,
    ALLOWED_WIND,
    MissionContext,
)


def _enum_select(label: str, allowed: frozenset[str], key: str) -> str | None:
    opts = [""] + sorted(allowed)
    i = st.selectbox(label, options=opts, format_func=lambda x: "— none —" if x == "" else x, key=key)
    return i or None


@st.cache_resource(show_spinner="Loading mission model…")
def load_athena() -> Athena:
    model_path = Path(os.environ["MODEL_PATH"])
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Missing trained bundle at {model_path}. Run `python train.py` in `Combat Mission Model/` "
            "and commit `athena.joblib`."
        )
    return Athena.load(str(model_path))


st.set_page_config(
    page_title="Team selection — Combat mission model",
    layout="wide",
)
st.title("Combat mission — team selection")
st.caption("Streamlit front-end for the same scoring / roster logic as the FastAPI service.")

try:
    athena = load_athena()
except Exception as e:
    st.exception(e)
    st.stop()

st.success(f"Model loaded — **{len(athena.profiles)}** leader profiles in memory.")

c1, c2, c3 = st.columns(3)
with c1:
    mission_type = st.selectbox("Mission type", options=AVAILABLE_MISSION_TYPES, index=0)
with c2:
    top_k = st.number_input("Candidate pool (top_k)", min_value=6, max_value=200, value=20, step=1)
with c3:
    num_team_options = st.number_input("Team options to show", min_value=1, max_value=10, value=2, step=1)

ctx: MissionContext | None = None
with st.expander("Mission context (optional — adjusts predicted scores)", expanded=False):
    sleep_hours = st.number_input("Sleep hours", min_value=0.0, max_value=24.0, value=7.0, step=0.5)
    use_temp = st.checkbox("Set temperature (°F)", value=False)
    temperature_f = (
        st.number_input("Temperature °F", value=60.0, step=1.0) if use_temp else None
    )
    wind = _enum_select("Wind", ALLOWED_WIND, "wind")
    terrain = _enum_select("Terrain", ALLOWED_TERRAIN, "terrain")
    difficulty = _enum_select("Difficulty", ALLOWED_DIFFICULTY, "difficulty")
    enemy_forces = _enum_select("Enemy forces", ALLOWED_ENEMY_FORCES, "enemy")
    notes_raw = st.text_input("Notes (echo only)", value="")

notes_trim = notes_raw.strip() or None
ctx = MissionContext(
    sleep_hours=float(sleep_hours),
    temperature_f=temperature_f,
    wind=wind,
    terrain=terrain,
    difficulty=difficulty,
    enemy_forces=enemy_forces,
    notes=notes_trim,
)

if st.button("Build team options", type="primary"):
    try:
        with st.spinner("Ranking candidates and composing rosters…"):
            result = athena.select_team(
                mission_type,
                top_k=int(top_k),
                context=ctx,
                num_team_options=int(num_team_options),
            )
    except ValueError as e:
        st.error(str(e))
        st.stop()

    st.subheader("Results")
    opts = result.get("team_options") or []
    for opt in opts:
        oid = opt.get("option_id", "?")
        score = opt.get("team_score")
        st.markdown(f"### Option {oid} — team score **{score}**")
        team = opt["team"]
        roles = {str(r["leader_identifier"]): r for r in (opt.get("breakdown") or {}).get("strength_roles", [])}
        rows = []
        for _, row in team.iterrows():
            lid = str(row["leader_identifier"])
            r = roles.get(lid, {})
            rows.append(
                {
                    "Name": row["leader_name"],
                    "ID": lid,
                    "Rank": row["leader_rank"],
                    "Unit": row["leader_unit"],
                    "Predicted": round(float(row["predicted_score"]), 4),
                    "Raw": round(float(row["raw_predicted_score"]), 4),
                    "Role": r.get("role_label") or "",
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        bd_raw = dict(opt.get("breakdown") or {})
        bd_raw.pop("strength_roles", None)
        with st.expander("Breakdown metrics"):
            st.json(bd_raw)

    if result.get("candidate_task_breakdowns"):
        with st.expander("Candidate pool — strengths / weaknesses summaries"):
            try:
                st.dataframe(
                    pd.json_normalize(result["candidate_task_breakdowns"]),
                    use_container_width=True,
                    hide_index=True,
                )
            except Exception:
                st.json(result["candidate_task_breakdowns"])

st.markdown("---")
st.caption(
    "API version of this tool lives under `Combat Mission Model/main.py` (FastAPI). "
    "This Streamlit app only imports `model` + `mission_context`."
)
