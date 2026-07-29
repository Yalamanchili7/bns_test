"""The single Pydantic-AI agent — the judgment + investigation layer.

This is the only LLM in the system. It investigates the dataset through the four
typed, row-bounded tools (the trust boundary) and returns a *judged* verdict for
each shortlisted candidate: ``data_confidence`` (can we trust this record?) and
``actionability`` (can we act on it?), each with a grounded one-line reason, plus
``flags`` for a human reviewer.

The agent does **no arithmetic**. It never computes ``final_score``, never
normalizes, never ranks — that is scoring.py's job. The split is deliberate: code
detects facts and does the math; the agent weighs the facts into judgment (see
DECISIONS.md).
"""

from __future__ import annotations

import os

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.agent import AgentRunResult

from pv_agent.schemas import JudgedScore, SimulationResult, SiteRecord
from pv_agent.tools import Dataset

# Cheap model by design (per the brief): the expensive judgment runs over a
# shortlist of ~15, never the full dataset. Pinned to a dated snapshot for
# reproducibility. Overridable via env for eval/testing.
DEFAULT_MODEL = "claude-haiku-4-5-20251001"


def _resolve_model() -> str:
    """Model string for pydantic-ai, defaulting to a cheap Haiku on Anthropic.

    Reads ``PV_AGENT_MODEL`` if set. A provider-less name is assumed to be an
    Anthropic model and gets the ``anthropic:`` prefix pydantic-ai expects.
    """
    name = os.environ.get("PV_AGENT_MODEL", DEFAULT_MODEL)
    return name if ":" in name else f"anthropic:{name}"


class CandidateJudgment(BaseModel):
    """The agent's judged verdict for one shortlisted candidate.

    Contains only judgment — no ``final_score``, no ranking. The two
    ``JudgedScore`` factors and the flags are the agent's contribution; scoring.py
    turns them into the final score and recommendation.
    """

    system_id: str
    data_confidence: JudgedScore = Field(
        description="Can we trust this record? 0-1 plus a one-line reason "
        "grounded in the record's anomalies/fields."
    )
    actionability: JudgedScore = Field(
        description="Can we act on this re-orientation? 0-1 plus a one-line "
        "reason (ownership, age, mounting)."
    )
    flags: list[str] = Field(
        default_factory=list,
        description="What a human should verify before acting.",
    )


_SYSTEM_PROMPT = """\
You are a solar-fleet analyst. Your job is to find the solar sites most worth
HUMAN REVIEW for panel re-orientation, and to judge how trustworthy and
actionable each candidate is. You do NOT compute scores or rankings — you only
investigate and judge.

Work in this order:
1. Call `get_dataset_summary` first to understand scale and data quality (counts,
   missingness, how many trackers / unknown-tracking / third-party / 0-0-default
   records there are). Return no rows.
2. Call `shortlist_candidates` to get the most-misaligned eligible sites. A cheap
   deterministic pre-filter has already run; you receive only the shortlist.
3. For EACH shortlisted candidate, call `simulate_site` to get its real energy
   upside and recommended orientation, and `get_site_details` to see its cleaned
   fields and detected anomalies (you may fetch details for several ids at once,
   up to 50). These anomalies are your evidence.
4. Judge EACH candidate and return a CandidateJudgment for it:
   - data_confidence (0-1 + one-line reason): can we trust this record?
   - actionability (0-1 + one-line reason): can we act on this re-orientation?
   - flags: what a human should verify.
   Every reason must be grounded in a specific detected fact or field.

Judgment guidance (weigh the detected anomalies — do not just copy them):
- orientation_default_0_0  -> VERY LOW data_confidence (~0.1-0.2). azimuth=0 and
  tilt=0 together is almost certainly a data-entry default, so the large raw
  upside is likely illusory. Flag "verify reported orientation".
- size_module_contradiction -> LOW data_confidence: system size and module count
  are physically inconsistent, so the record is unreliable.
- extreme_tilt -> LOW data_confidence: a near-vertical (>= 80 deg) fixed panel is
  almost certainly a data-entry error, and its very large raw upside is therefore
  illusory. Treat it like the 0/0-default. Flag "verify reported tilt".
- tracking_unknown -> mild data_confidence hit and flag "confirm fixed-tilt"
  (a tracker cannot be re-oriented, so we must confirm it is fixed).
- third_party_owned -> LOWER actionability: re-orientation needs owner approval.
- old_install -> LOWER actionability: an old system may be a repowering candidate
  rather than worth re-orienting; flag it.
- east_facing -> do NOT penalize. An east-facing array may be an intentional
  morning-load choice. Flag it as "confirm east-facing is intentional", but keep
  confidence/actionability as the rest of the evidence warrants.
- ground_mounted == "Ground" -> MORE actionable; "Roof" -> slightly less.
- A clean, internally-consistent record with known fields -> HIGH confidence.

Return the list of CandidateJudgment objects — one per shortlisted candidate.
Do NOT compute final scores, normalized upside, or any ranking.
"""


# The agent. deps is the Dataset instance the tools read through. defer_model_check
# keeps model/provider resolution (and the API-key requirement) at run time, so
# importing this module — and building tools/tests — needs no API key.
agent = Agent(
    _resolve_model(),
    deps_type=Dataset,
    output_type=list[CandidateJudgment],
    instructions=_SYSTEM_PROMPT,
    # temperature=0 for the most reproducible judgments the model allows. It does
    # not make an LLM bit-for-bit deterministic, but it removes sampling variance
    # so repeated runs land in the same bands (and usually the same ranking).
    model_settings={"temperature": 0.0},
    defer_model_check=True,
)


@agent.tool
def get_dataset_summary(ctx: RunContext[Dataset]) -> dict:
    """Aggregate counts/missingness for the whole dataset. Returns no raw rows."""
    return ctx.deps.get_dataset_summary()


@agent.tool
def shortlist_candidates(ctx: RunContext[Dataset], limit: int = 15) -> list[dict]:
    """Top-`limit` most-misaligned eligible sites (light dicts, bounded at 50)."""
    return ctx.deps.shortlist_candidates(limit)


@agent.tool
def get_site_details(
    ctx: RunContext[Dataset], system_ids: list[str]
) -> list[SiteRecord]:
    """Cleaned fields + detected anomalies for specific sites (up to 50)."""
    return ctx.deps.get_site_details(system_ids)


@agent.tool
def simulate_site(ctx: RunContext[Dataset], system_id: str) -> SimulationResult:
    """Physics + grid-search optimization for one site: POA, upside, orientation."""
    return ctx.deps.simulate_site_tool(system_id)


def run_agent(dataset: Dataset, shortlist_size: int = 15) -> AgentRunResult:
    """Run the agent and return the FULL result (output + message history).

    The full ``AgentRunResult`` is needed by the orchestrator to build the
    tool-call trace (``result.all_messages()``); ``.output`` alone hides which
    tools the agent called. Keeps the shortlist modest so simulation (one physics
    call per candidate) stays fast and cheap.
    """
    prompt = (
        f"Find and judge the top solar sites worth human review for panel "
        f"re-orientation. Use a shortlist of {shortlist_size} candidates. "
        f"Investigate with the tools, then return one CandidateJudgment per "
        f"shortlisted candidate."
    )
    return agent.run_sync(prompt, deps=dataset)


def judge_candidates(
    dataset: Dataset, shortlist_size: int = 15
) -> list[CandidateJudgment]:
    """Run the agent and return just its per-candidate judgments.

    Thin wrapper over :func:`run_agent` for callers that only need the output.
    The returned judgments feed scoring.py, which does the arithmetic and ranking
    the agent deliberately does not.
    """
    return run_agent(dataset, shortlist_size).output


# A second, separate agent whose only job is to write a short human-facing
# summary of the top sites (one call, plain-text output). It has no tools and no
# deps — it summarizes text it is handed, so it cannot touch the dataset or do
# any arithmetic. Kept distinct from the judgment agent for a clean separation.
summary_agent = Agent(
    _resolve_model(),
    output_type=str,
    instructions=(
        "You write a brief for a solar-fleet reviewer. Given the top few "
        "re-orientation candidates (each with its energy upside, data-confidence "
        "and actionability judgments, recommendation, and flags), write a short "
        "narrative — a few sentences — telling the reviewer which sites to look "
        "at first and what to watch for. Be concrete and reference the specific "
        "reasons/flags. Do not invent numbers; use only what you are given."
    ),
    model_settings={"temperature": 0.0},
    defer_model_check=True,
)


def summarize_top_sites(brief: str) -> str:
    """One LLM call: turn a formatted top-sites brief into reviewer guidance."""
    return summary_agent.run_sync(brief).output
