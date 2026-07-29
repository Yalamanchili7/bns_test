"""Pydantic v2 data models shared across the pv_agent pipeline.

These schemas are the contracts between the deterministic layer (cleaning,
physics, scoring) and the agent layer (judgment, explanation). Every model
here is a plain data container: no business logic lives in the schemas.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

# Three-state categoricals: the "Unknown" state is always explicit and is never
# collapsed into a value (see DECISIONS.md — missing/ambiguous data handling).
Tracking = Literal["Fixed", "Tracker", "Unknown"]
Ownership = Literal["Owner", "ThirdParty", "Unknown"]
Mounting = Literal["Roof", "Ground", "Unknown"]


class Recommendation(str, Enum):
    """Final triage band for a site, assigned from ``final_score``."""

    PRIORITIZE = "Prioritize"
    REVIEW = "Review"
    SKIP = "Skip"


class SiteRecord(BaseModel):
    """A single cleaned PV installation record.

    Produced by the cleaning layer from one raw CSV row. Sentinels (``-1`` /
    ``-9999``) are already nulled, the zip is zero-padded, and the three-state
    categoricals never collapse ``Unknown`` into a value. ``anomalies`` holds
    detected *facts* (not judgments) for the agent to weigh.
    """

    system_id: str
    state: str
    zip: str
    lat: float | None = None
    lon: float | None = None
    system_size_dc: float | None = None
    azimuth: float | None = None
    tilt: float | None = None
    module_quantity: float | None = None
    tracking: Tracking = "Unknown"
    third_party: Ownership = "Unknown"
    ground_mounted: Mounting = "Unknown"
    age_years: int | None = None
    anomalies: list[str] = Field(default_factory=list)


class SimulationResult(BaseModel):
    """Output of the pvlib physics + grid-search optimization for one site."""

    current_annual_poa: float
    optimal_annual_poa: float
    relative_energy_upside: float
    recommended_tilt: float
    recommended_azimuth: float


class JudgedScore(BaseModel):
    """An agent-produced score in [0, 1] with its grounding reason.

    Reused for both judged metrics (``data_confidence`` and ``actionability``).
    """

    score: float = Field(ge=0.0, le=1.0)
    reason: str


class Scorecard(BaseModel):
    """The explainable per-site result: physics + judgment + final ranking."""

    system_id: str
    energy_upside: float
    normalized_energy_upside: float
    data_confidence: JudgedScore
    actionability: JudgedScore
    final_score: float
    recommended_tilt: float
    recommended_azimuth: float
    recommendation: Recommendation
    flags: list[str] = Field(default_factory=list)


class ScoringConfig(BaseModel):
    """Tunable constants for normalization and recommendation banding."""

    upside_cap: float = 0.30
    prioritize_threshold: float = 0.5
    review_threshold: float = 0.2
