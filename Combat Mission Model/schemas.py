from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


WindLevel = Literal["calm", "moderate", "gusty", "high"]

TerrainType = Literal[
    "urban",
    "mountain",
    "jungle",
    "desert",
    "wooded",
    "forest",
    "open",
    "mixed",
]

DifficultyLevel = Literal[
    "low",
    "easy",
    "moderate",
    "medium",
    "high",
    "hard",
    "extreme",
]

EnemyPosture = Literal[
    "light",
    "few",
    "moderate",
    "medium",
    "heavy",
    "numerous",
    "overwhelming",
]


class MissionContextInput(BaseModel):
    """Operational context applied at inference (not learned from CSV history).

    ``wind``, ``terrain``, ``difficulty``, and ``enemy_forces`` are closed enums
    (see OpenAPI schema). Empty strings are treated as omitted.
    """

    sleep_hours: float = Field(default=7.0, ge=0, le=24)
    temperature_f: Optional[float] = Field(None, description="Ambient temperature (°F)")
    wind: Optional[WindLevel] = None
    terrain: Optional[TerrainType] = None
    difficulty: Optional[DifficultyLevel] = None
    enemy_forces: Optional[EnemyPosture] = None
    notes: Optional[str] = Field(None, description="Echoed in response only; does not change scores")

    @field_validator("wind", "terrain", "difficulty", "enemy_forces", mode="before")
    @classmethod
    def _normalize_enum_optional(cls, v: Any) -> Any:
        if v is None:
            return None
        if isinstance(v, str):
            s = v.strip()
            if s == "":
                return None
            return s.lower()
        return v


class TeamSelectRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "mission_type": "ambush",
                "top_k": 20,
                "num_team_options": 2,
                "mission_context": {
                    "sleep_hours": 4.5,
                    "temperature_f": 28,
                    "wind": "gusty",
                    "terrain": "wooded",
                    "difficulty": "high",
                    "enemy_forces": "heavy",
                },
            }
        }
    )

    mission_type: str
    top_k: int = Field(default=20, ge=6, le=200, description="Candidate pool size (must be ≥ team size 6).")
    num_team_options: int = Field(
        default=2,
        ge=1,
        le=25,
        description="How many alternative six-person rosters to return (distinct people sets; commander picks one). Default 2 fits small pools.",
    )
    mission_context: Optional[MissionContextInput] = None


class StrengthRoleAssignment(BaseModel):
    role_key: str
    role_label: str
    leader_identifier: str


class SoldierSummary(BaseModel):
    leader_identifier: str
    leader_name: str
    leader_rank: str
    leader_unit: str
    predicted_score: Optional[float] = None
    raw_predicted_score: Optional[float] = Field(
        None, description="GBM output before context multipliers (same as predicted if no context)"
    )
    strength_role_key: Optional[str] = Field(
        None, description="Complementary team role anchored on this soldier (distinct per slot)"
    )
    strength_role_label: Optional[str] = None


class CandidateTaskBreakdown(BaseModel):
    """Strengths and weaknesses by ``task_group`` (from detailed rubric means) for one pool member."""

    leader_identifier: str
    leader_name: str
    strengths_summary: List[str]
    weaknesses_summary: List[str]
    interpretation: str
    candidate_rank: int = Field(
        ..., ge=1, description="Position in mission-ranked pool (1 = highest predicted score)"
    )
    predicted_score: float


class TeamBreakdown(BaseModel):
    avg_predicted_score: float
    min_predicted_score: float
    max_predicted_score: float
    units_represented: List[str]
    avg_planning: float
    avg_decisiveness: float
    avg_tactics: float
    avg_time_management: float
    avg_attention_detail: float
    avg_consistency_std: float
    missions_completed: int
    strength_roles: Optional[List[StrengthRoleAssignment]] = Field(
        None,
        description="Ordered list: one strength anchor per team member (no duplicate people)",
    )


class TeamOptionResponse(BaseModel):
    """One of several distinct six-person rosters for the same mission (different mixes of people)."""

    option_id: int = Field(..., ge=1)
    team_score: float
    team: List[SoldierSummary]
    breakdown: TeamBreakdown
    slot_order: List[str] = Field(
        ...,
        min_length=6,
        max_length=6,
        description="Greedy fill order of task_group slots (a permutation of the six rubric groups)",
    )


class TeamSelectResponse(BaseModel):
    mission_type: str
    team_score: float
    team: List[SoldierSummary]
    breakdown: TeamBreakdown
    team_options: List[TeamOptionResponse] = Field(
        ...,
        description="Ranked alternatives: option 1 matches team / team_score / breakdown (best heuristic score among variants)",
    )
    num_team_options_requested: int = Field(
        ...,
        ge=1,
        description="Echo of the client's num_team_options request",
    )
    num_team_options_returned: int = Field(
        ...,
        ge=1,
        description="How many distinct rosters are in team_options (≤ requested if the pool yields fewer mixes)",
    )
    candidate_task_breakdowns: List[CandidateTaskBreakdown] = Field(
        ...,
        description="Per-candidate task_group / rating_score breakdown for the top_k pool (before team of six)",
    )
    mission_context: Optional[Dict[str, Any]] = None
    condition_modifiers: Optional[Dict[str, Any]] = Field(
        None,
        description="Per-factor multipliers and combined modifier applied to raw scores",
    )


class RankingsRequest(BaseModel):
    """Rank all soldiers for a mission with optional environmental context (same modifiers as ``/team/select``)."""

    mission_type: str
    mission_context: Optional[MissionContextInput] = None

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "mission_type": "ambush",
                "mission_context": {
                    "sleep_hours": 4.0,
                    "wind": "gusty",
                    "difficulty": "high",
                },
            }
        }
    )


class RankingsResponse(BaseModel):
    mission_type: str
    rankings: List[SoldierSummary]
    mission_context: Optional[Dict[str, Any]] = Field(
        None,
        description="Echo of request context when supplied (POST only, or null for GET)",
    )
    condition_modifiers: Optional[Dict[str, Any]] = Field(
        None,
        description="Per-factor multipliers when mission_context was applied",
    )


class SoldierProfile(BaseModel):
    leader_identifier: str
    leader_name: str
    leader_rank: str
    leader_unit: str
    missions_completed: int
    overall_avg: float
    consistency: float
    avg_planning: float
    avg_attention_to_detail: float
    avg_time_management: float
    avg_decisiveness: float
    avg_tactics: float
    avg_st_eo: Optional[float]
    per_mission_scores: Dict[str, float]


class SoldiersListResponse(BaseModel):
    total: int
    soldiers: List[SoldierProfile]


class HealthResponse(BaseModel):
    status: str
    model_trained: bool
    soldiers_loaded: int
    available_mission_types: List[str]
    train_metrics: Optional[Dict[str, float]] = None
    artifact_path: Optional[str] = Field(
        None, description="Resolved MODEL_PATH (joblib bundle) for ops / Athena wiring"
    )
    metadata_path: Optional[str] = Field(
        None, description="Resolved MODEL_METADATA_PATH (JSON sidecar from train.py)"
    )
    metadata_available: bool = Field(
        False, description="Whether the JSON metadata file exists on disk"
    )
