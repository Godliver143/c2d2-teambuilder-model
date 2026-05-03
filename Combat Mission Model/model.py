from __future__ import annotations

import json
import os
import re
import warnings
from datetime import datetime, timezone
from itertools import combinations, permutations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import cross_val_score

from mission_context import (
    MissionContext,
    combined_condition_modifier,
    context_public_dict,
)

_ROOT = Path(__file__).resolve().parent
MODELS_DIR = _ROOT / "models"
EVAL_PATH = os.getenv("EVAL_PATH", str(_ROOT / "evaluations_full.csv"))
DETAIL_PATH = os.getenv("DETAIL_PATH", str(_ROOT / "detailed_full.csv"))
MODEL_PATH = os.getenv("MODEL_PATH", str(_ROOT / "athena.joblib"))
MODEL_METADATA_PATH = os.getenv(
    "MODEL_METADATA_PATH", str(MODELS_DIR / "mission_model_metadata.json")
)

warnings.filterwarnings("ignore")

SKILL_COLS = ["planning","attention_to_detail","time_management","decisiveness","tactics"]
UMP_COLS   = [f"ump_{c}" for c in SKILL_COLS]

# Team of six: one slot per detailed-rubric ``task_group`` (mean ``rating_score`` in detailed CSV).
# Greedy assignment from the mission-ranked pool — each pick is best-in-slot for that task_group.
TEAM_TASK_GROUP_SLOTS: tuple[str, ...] = (
    "Planning",
    "Decisiveness",
    "Attention to Detail",
    "Tactics",
    "Time Management",
    "ST&EO",
)

MISSION_TYPE_MAP = {
    "LINEAR AMBUSH":"ambush","SQUAD AMBUSH":"ambush","AREA RECON":"recon",
    "MOVEMENT TO CONTACT":"movement_to_contact","SQUAD ATTACK":"attack",
    "REACT TO UAS":"react_uas","REACT TO IDF":"react_idf","BREAK CONTACT":"break_contact",
}
AVAILABLE_MISSION_TYPES = sorted(set(MISSION_TYPE_MAP.values()))


def extract_mission_type(event_name):
    if pd.isna(event_name): return "unknown"
    eu = str(event_name).upper()
    for kw, mt in MISSION_TYPE_MAP.items():
        if kw in eu: return mt
    return "unknown"


def load_data():
    evals   = pd.read_csv(EVAL_PATH)
    details = pd.read_csv(DETAIL_PATH)
    evals["mission_type"]   = evals["event_name"].apply(extract_mission_type)
    details["mission_type"] = details["event_name"].apply(extract_mission_type)
    evals["event_score"]    = evals[SKILL_COLS].mean(axis=1).round(3)
    evals["ump_score"]      = evals[UMP_COLS].mean(axis=1).round(3)
    return evals, details


def build_soldier_profiles(evals, details):
    profiles = []
    for lid, grp in evals.groupby("leader_identifier"):
        meta = grp.iloc[0]
        row  = {"leader_identifier":lid,"leader_name":meta["leader_name"],
                "leader_rank":meta["leader_rank"],"leader_unit":meta["leader_unit"]}
        for col in SKILL_COLS:
            row[f"avg_{col}"] = round(grp[col].mean(), 3)
            row[f"std_{col}"] = round(grp[col].std(ddof=0), 3)
        for col in UMP_COLS:
            row[f"avg_{col}"] = round(grp[col].mean(), 3)
        row["overall_avg"]        = round(grp["event_score"].mean(), 3)
        row["overall_ump_avg"]    = round(grp["ump_score"].mean(), 3)
        row["consistency"]        = round(grp["event_score"].std(ddof=0), 3)
        row["missions_completed"] = len(grp)
        for mt in AVAILABLE_MISSION_TYPES:
            s = grp[grp["mission_type"]==mt]["event_score"]
            row[f"score_{mt}"] = round(s.mean(),3) if len(s)>0 else row["overall_avg"]
        det = details[(details["leader_identifier"]==lid)&(details["task_group"]=="ST&EO")]
        row["avg_st_eo"] = round(det["rating_score"].mean(),3) if len(det)>0 else None
        profiles.append(row)
    df = pd.DataFrame(profiles)
    df["avg_st_eo"] = df["avg_st_eo"].fillna(df["avg_st_eo"].mean())
    return df


def build_training_data(evals, profiles):
    pf = [c for c in profiles.columns if c not in ["leader_identifier","leader_name","leader_rank","leader_unit"]]
    rows = []
    for _, ev in evals.iterrows():
        p = profiles[profiles["leader_identifier"]==ev["leader_identifier"]]
        if p.empty: continue
        row = {"mission_type":ev["mission_type"],"event_score":ev["event_score"]}
        for f in pf: row[f] = p.iloc[0][f]
        rows.append(row)
    return pd.DataFrame(rows)



def build_task_group_summaries(
    details: pd.DataFrame,
    profiles: pd.DataFrame | None = None,
) -> dict[str, dict]:
    """
    Per-leader mean ``rating_score`` by ``task_group`` from detailed rubric rows, plus
    relative strengths (top-2 task groups by mean) and weaknesses (bottom-2).
    """
    d = details.copy()
    d["rating_score"] = pd.to_numeric(d["rating_score"], errors="coerce")
    d = d.dropna(subset=["rating_score", "leader_identifier", "task_group"])
    name_lookup: dict[str, str] = {}
    if profiles is not None and len(profiles):
        name_lookup = profiles.set_index("leader_identifier")["leader_name"].to_dict()

    summaries: dict[str, dict] = {}
    for lid, grp in d.groupby("leader_identifier"):
        lid_s = str(lid)
        means = grp.groupby("task_group", dropna=True)["rating_score"].mean()
        by_tg: list[dict] = [
            {"task_group": str(tg), "mean_rating_score": round(float(m), 3)}
            for tg, m in means.items()
        ]
        by_tg.sort(key=lambda x: x["task_group"])

        ranked_hi = sorted(by_tg, key=lambda x: x["mean_rating_score"], reverse=True)
        ranked_lo = sorted(by_tg, key=lambda x: x["mean_rating_score"])
        if ranked_hi:
            strengths = [ranked_hi[0]["task_group"]]
            if len(ranked_hi) > 1:
                strengths.append(ranked_hi[1]["task_group"])
            weaknesses = [ranked_lo[0]["task_group"]]
            if len(ranked_lo) > 1:
                weaknesses.append(ranked_lo[1]["task_group"])
            weaknesses = [w for w in weaknesses if w not in strengths]
        else:
            strengths, weaknesses = [], []

        summaries[lid_s] = {
            "leader_identifier": lid_s,
            "leader_name": str(name_lookup.get(lid_s, grp.iloc[0].get("leader_name", ""))),
            "by_task_group": by_tg,
            "strengths_summary": strengths,
            "weaknesses_summary": weaknesses,
            "interpretation": (
                "Strengths: task groups with the highest mean rating_score (top 2). "
                "Weaknesses: lowest means (bottom 2), excluding groups already listed as strengths."
            ),
        }
    return summaries


def _task_group_means_for_leader(summaries: dict[str, dict], lid: str) -> dict[str, float]:
    s = summaries.get(str(lid), {})
    return {
        row["task_group"]: float(row["mean_rating_score"])
        for row in s.get("by_task_group", [])
    }


def _slot_score_task_group(
    means: dict[str, float],
    task_group: str,
) -> float:
    if task_group in means:
        return means[task_group]
    if means:
        return sum(means.values()) / len(means)
    return float("-inf")


def select_team_complementary_strengths(
    candidates: pd.DataFrame,
    summaries: dict[str, dict],
    slot_order: tuple[str, ...] | None = None,
) -> tuple[pd.DataFrame, list[dict]]:
    """
    Six distinct people from the pool: each slot is a ``task_group`` from the detailed
    rubric; pick the remaining candidate with the highest mean ``rating_score`` in that
    group (fallback: their overall mean across groups). Tie-break on ``predicted_score``.

    ``slot_order`` must be a permutation of ``TEAM_TASK_GROUP_SLOTS`` (order controls greedy
    picks; different orders yield different rosters).
    """
    if len(candidates) < 6:
        raise ValueError("Need at least 6 candidates for team selection.")
    order = slot_order if slot_order is not None else TEAM_TASK_GROUP_SLOTS
    if len(order) != len(TEAM_TASK_GROUP_SLOTS) or frozenset(order) != frozenset(
        TEAM_TASK_GROUP_SLOTS
    ):
        raise ValueError("slot_order must be a permutation of TEAM_TASK_GROUP_SLOTS.")
    remaining: list[int] = list(candidates.index)
    pick_order: list[int] = []
    assignments: list[dict] = []

    for tg in order:
        best_tuple = (float("-inf"), float("-inf"))
        best_idx = remaining[0]
        for idx in remaining:
            lid = str(candidates.loc[idx, "leader_identifier"])
            means = _task_group_means_for_leader(summaries, lid)
            sc = _slot_score_task_group(means, tg)
            pred = float(candidates.loc[idx, "predicted_score"])
            t = (sc, pred)
            if t > best_tuple:
                best_tuple = t
                best_idx = idx
        row = candidates.loc[best_idx]
        rk = re.sub(r"[^a-z0-9]+", "_", tg.lower()).strip("_")
        assignments.append(
            {
                "role_key": rk,
                "role_label": tg,
                "leader_identifier": str(row["leader_identifier"]),
            }
        )
        pick_order.append(best_idx)
        remaining.remove(best_idx)

    team = candidates.loc[pick_order].reset_index(drop=True)
    return team, assignments


def _collect_greedy_rosters_for_pool(
    candidates: pd.DataFrame,
    summaries: dict[str, dict],
    ind_scores: dict,
    evaluator: TeamEvaluator,
) -> dict[frozenset, dict]:
    """All distinct rosters reachable by greedy fills over permutations of task_group slots."""
    if len(candidates) < 6:
        return {}
    best_by_roster: dict[frozenset, dict] = {}
    for order in permutations(TEAM_TASK_GROUP_SLOTS):
        team, roles = select_team_complementary_strengths(
            candidates, summaries, slot_order=order
        )
        roster = frozenset(team["leader_identifier"].tolist())
        score = evaluator.score_team(team, ind_scores)
        br = evaluator.breakdown(team, ind_scores)
        br["strength_roles"] = roles
        rec = {
            "team": team,
            "strength_roles": roles,
            "team_score": score,
            "breakdown": br,
            "slot_order": list(order),
        }
        prev = best_by_roster.get(roster)
        if prev is None or score > prev["team_score"]:
            best_by_roster[roster] = rec
    return best_by_roster


def enumerate_distinct_team_options(
    candidates: pd.DataFrame,
    summaries: dict[str, dict],
    ind_scores: dict,
    evaluator: TeamEvaluator,
    n_options: int = 2,
) -> list[dict]:
    """
    Up to ``n_options`` **different** six-person rosters (distinct people sets), ranked by
    ``team_score``. Starts from all greedy outcomes over slot-order permutations; if fewer
    than ``n_options`` rosters exist, tries sub-pools that **exclude** one or two members
    from the current best team so the commander still gets multiple mixes when the raw
    greedy family is small.
    """
    if n_options < 1:
        raise ValueError("n_options must be at least 1.")
    merged = _collect_greedy_rosters_for_pool(
        candidates, summaries, ind_scores, evaluator
    )

    def _merge_extra(extra: dict[frozenset, dict]) -> None:
        for k, v in extra.items():
            prev = merged.get(k)
            if prev is None or v["team_score"] > prev["team_score"]:
                merged[k] = v

    if len(merged) < n_options and merged:
        top = max(merged.values(), key=lambda x: x["team_score"])
        for ban_lid in top["team"]["leader_identifier"].tolist():
            if len(merged) >= n_options:
                break
            sub = candidates[candidates["leader_identifier"] != ban_lid]
            if len(sub) < 6:
                continue
            _merge_extra(
                _collect_greedy_rosters_for_pool(sub, summaries, ind_scores, evaluator)
            )

    if len(merged) < n_options and merged:
        top = max(merged.values(), key=lambda x: x["team_score"])
        lids = top["team"]["leader_identifier"].tolist()
        for pair in combinations(lids, 2):
            if len(merged) >= n_options:
                break
            sub = candidates[~candidates["leader_identifier"].isin(pair)]
            if len(sub) < 6:
                continue
            _merge_extra(
                _collect_greedy_rosters_for_pool(sub, summaries, ind_scores, evaluator)
            )

    ranked = sorted(merged.values(), key=lambda x: x["team_score"], reverse=True)
    return [{**r, "option_id": i} for i, r in enumerate(ranked[:n_options], start=1)]


def candidate_task_breakdowns_for_pool(
    candidates: pd.DataFrame,
    summaries: dict[str, dict],
) -> list[dict]:
    """Per-candidate strengths/weaknesses (by task_group means) for the selection pool."""
    out: list[dict] = []
    for rank_i, (_, row) in enumerate(candidates.iterrows(), start=1):
        lid = str(row["leader_identifier"])
        base = summaries.get(lid)
        if base is None:
            entry = {
                "leader_identifier": lid,
                "leader_name": str(row["leader_name"]),
                "strengths_summary": [],
                "weaknesses_summary": [],
                "interpretation": "No detailed subtask rows for this leader in the detailed CSV.",
                "candidate_rank": rank_i,
                "predicted_score": round(float(row["predicted_score"]), 4),
            }
        else:
            entry = {
                "leader_identifier": base["leader_identifier"],
                "leader_name": base["leader_name"],
                "strengths_summary": base["strengths_summary"],
                "weaknesses_summary": base["weaknesses_summary"],
                "interpretation": base["interpretation"],
                "candidate_rank": rank_i,
                "predicted_score": round(float(row["predicted_score"]), 4),
            }
        out.append(entry)
    return out


class SoldierScoringModel:
    def __init__(self):
        self.model = GradientBoostingRegressor(
            n_estimators=300,learning_rate=0.04,max_depth=4,
            subsample=0.8,min_samples_leaf=2,random_state=42)
        self.le = LabelEncoder()
        self.scaler = StandardScaler()
        self.feature_cols = None

    def _prepare(self, df, fit):
        d = df[self.feature_cols].copy()
        if fit:
            d["mission_type"] = self.le.fit_transform(d["mission_type"].astype(str))
            return self.scaler.fit_transform(d.values.astype(float))
        else:
            d["mission_type"] = d["mission_type"].astype(str).map(
                lambda v: int(self.le.transform([v])[0]) if v in self.le.classes_
                else int(self.le.transform(["ambush"])[0]))
            return self.scaler.transform(d.values.astype(float))

    def train(self, train_df):
        self.feature_cols = [c for c in train_df.columns if c != "event_score"]
        X = self._prepare(train_df, fit=True)
        y = train_df["event_score"].values
        n = len(train_df)
        if n < 2:
            raise ValueError("Training data must contain at least 2 rows for cross-validation.")
        n_splits = min(5, max(2, n // 5), n)
        cv = cross_val_score(
            self.model, X, y, cv=n_splits, scoring="neg_mean_absolute_error"
        )
        self.model.fit(X, y)
        return {
            "cv_mae_mean": float(round(-cv.mean(), 4)),
            "cv_mae_std": float(round(cv.std(), 4)),
        }

    def score_soldier(self, soldier_row, mission_type):
        row = soldier_row.copy()
        row["mission_type"] = mission_type
        X = self._prepare(pd.DataFrame([row]), fit=False)
        return float(np.clip(self.model.predict(X)[0], 1, 5))

    def rank_soldiers(
        self,
        profiles,
        mission_type,
        context: MissionContext | None = None,
    ):
        pf = [
            c
            for c in profiles.columns
            if c not in ["leader_identifier", "leader_name", "leader_rank", "leader_unit"]
        ]
        ranked = profiles.copy()
        ranked["raw_predicted_score"] = ranked[pf].apply(
            lambda r: self.score_soldier(r, mission_type), axis=1
        ).round(4)
        if context is None:
            ranked["predicted_score"] = ranked["raw_predicted_score"]
        else:
            mod, _ = combined_condition_modifier(context)
            ranked["predicted_score"] = (
                ranked["raw_predicted_score"] * mod
            ).clip(1.0, 5.0).round(4)
        return ranked.sort_values("predicted_score", ascending=False).reset_index(drop=True)


class TeamEvaluator:
    def score_team(self, team, ind_scores):
        scores     = [ind_scores[str(lid)] for lid in team["leader_identifier"]]
        weak_pen   = max(0,(3.0 - np.min(scores))*0.5)
        unit_bonus = (team["leader_unit"].nunique()-1)*0.3
        synergy    = (team["avg_decisiveness"].mean()+team["avg_planning"].mean())/2
        syn_bonus  = (synergy-3.5)*0.4
        cons_bonus = max(0,(1.5-team["consistency"].mean())*0.3)
        return float(round(np.mean(scores)-weak_pen+unit_bonus+syn_bonus+cons_bonus, 4))

    def breakdown(self, team, ind_scores):
        scores = [float(ind_scores[str(lid)]) for lid in team["leader_identifier"]]
        return {
            "avg_predicted_score":  float(round(np.mean(scores), 3)),
            "min_predicted_score":  float(round(np.min(scores), 3)),
            "max_predicted_score":  float(round(np.max(scores), 3)),
            "units_represented":    sorted(str(u) for u in team["leader_unit"].unique().tolist()),
            "avg_planning":         float(round(team["avg_planning"].mean(), 3)),
            "avg_decisiveness":     float(round(team["avg_decisiveness"].mean(), 3)),
            "avg_tactics":          float(round(team["avg_tactics"].mean(), 3)),
            "avg_time_management":  float(round(team["avg_time_management"].mean(), 3)),
            "avg_attention_detail": float(round(team["avg_attention_to_detail"].mean(), 3)),
            "avg_consistency_std":  float(round(team["consistency"].mean(), 3)),
            "missions_completed":   int(team["missions_completed"].sum()),
        }


class Athena:
    """
    End-to-end pipeline: profiles from evaluations, GradientBoosting scorer, team search.

    The class name is historical (pickle compatibility). This object **is** the deployable
    mission-performance model artifact; a separate *Athena* product layer may consume its
    predictions later (e.g. commander coaching, gap analysis).
    """

    def __init__(self):
        self.scorer = SoldierScoringModel()
        self.evaluator = TeamEvaluator()
        self.profiles = None
        self.train_metrics = {}
        self.training_stats: dict | None = None
        self._task_group_summaries: dict[str, dict] | None = None

    def _ensure_task_group_summaries(self) -> dict[str, dict]:
        cached = getattr(self, "_task_group_summaries", None)
        if cached is None:
            _, details = load_data()
            self._task_group_summaries = build_task_group_summaries(
                details, self.profiles
            )
        return self._task_group_summaries

    def train(self):
        evals, details = load_data()
        self.profiles = build_soldier_profiles(evals, details)
        self._task_group_summaries = build_task_group_summaries(details, self.profiles)
        train_df = build_training_data(evals, self.profiles)
        self.train_metrics = self.scorer.train(train_df)
        self.training_stats = {
            "evaluation_rows": int(len(evals)),
            "training_rows": int(len(train_df)),
            "leader_profiles": int(len(self.profiles)),
            "target_mean": float(train_df["event_score"].mean()),
            "target_std": float(train_df["event_score"].std(ddof=0)),
        }
        return self.train_metrics

    def save(self, path=MODEL_PATH):
        joblib.dump(self, path)

    def build_metadata_payload(self) -> dict:
        """JSON-serialisable description of this trained instance for deployment / integrations."""
        if not self.train_metrics or self.scorer.feature_cols is None:
            raise ValueError("Train the model before exporting metadata.")
        stats = self.training_stats or {}
        return {
            "schema_version": 1,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "model": {
                "kind": "gradient_boosting_regressor",
                "target": "event_score",
                "score_clip_range": [1.0, 5.0],
                "feature_columns": list(self.scorer.feature_cols),
                "mission_type_classes": list(self.scorer.le.classes_),
            },
            "metrics": {
                "cv_mae_mean": self.train_metrics["cv_mae_mean"],
                "cv_mae_std": self.train_metrics["cv_mae_std"],
            },
            "data": {
                "evaluations_csv": EVAL_PATH,
                "detailed_csv": DETAIL_PATH,
                **stats,
            },
            "consumer_note": (
                "Downstream services (e.g. a future Athena coaching layer) should treat "
                "`raw_predicted_score` / rankings as model outputs; narrative guidance is out of scope here."
            ),
        }

    def save_metadata(self, path: str | Path | None = None) -> Path:
        """Write `build_metadata_payload()` next to the artifact for versioning and integration."""
        out = Path(path or MODEL_METADATA_PATH)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = self.build_metadata_payload()
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return out

    @staticmethod
    def load(path=MODEL_PATH):
        return joblib.load(path)

    def select_team(
        self,
        mission_type,
        top_k=20,
        context: MissionContext | None = None,
        num_team_options: int = 2,
    ):
        ranked = self.scorer.rank_soldiers(self.profiles, mission_type, context=context)
        ind_scores = {
            str(lid): float(score)
            for lid, score in zip(ranked["leader_identifier"], ranked["predicted_score"])
        }
        candidates = ranked.head(top_k)
        summaries = self._ensure_task_group_summaries()
        candidate_task_breakdowns = candidate_task_breakdowns_for_pool(
            candidates, summaries
        )
        team_opts = enumerate_distinct_team_options(
            candidates,
            summaries,
            ind_scores,
            self.evaluator,
            n_options=num_team_options,
        )
        primary = team_opts[0]
        out = {
            "mission_type": mission_type,
            "team_score": primary["team_score"],
            "team": primary["team"],
            "breakdown": primary["breakdown"],
            "candidate_task_breakdowns": candidate_task_breakdowns,
            "team_options": team_opts,
        }
        if context is not None:
            _, cond_mod_breakdown = combined_condition_modifier(context)
            out["condition_modifiers"] = cond_mod_breakdown
            out["mission_context"] = context_public_dict(context)
        return out

    def rank_soldiers(self, mission_type, context: MissionContext | None = None):
        return self.scorer.rank_soldiers(self.profiles, mission_type, context=context)

    def get_soldier_profile(self, leader_identifier):
        row = self.profiles[self.profiles["leader_identifier"]==leader_identifier]
        if row.empty: return None
        r = row.iloc[0]
        lid_val = r["leader_identifier"]
        return {
            "leader_identifier":       str(lid_val),
            "leader_name":             str(r["leader_name"]),
            "leader_rank":             str(r["leader_rank"]),
            "leader_unit":             str(r["leader_unit"]),
            "missions_completed":      int(r["missions_completed"]),
            "overall_avg":             float(r["overall_avg"]),
            "consistency":             float(r["consistency"]),
            "avg_planning":            float(r["avg_planning"]),
            "avg_attention_to_detail": float(r["avg_attention_to_detail"]),
            "avg_time_management":     float(r["avg_time_management"]),
            "avg_decisiveness":        float(r["avg_decisiveness"]),
            "avg_tactics":             float(r["avg_tactics"]),
            "avg_st_eo":               float(r["avg_st_eo"]) if pd.notna(r["avg_st_eo"]) else None,
            "per_mission_scores":      {mt: float(r[f"score_{mt}"]) for mt in AVAILABLE_MISSION_TYPES},
        }

    def get_all_profiles(self):
        return [self.get_soldier_profile(lid) for lid in self.profiles["leader_identifier"]]
