"""Pipeline orchestrator and entrypoint.

Ties the whole system together and produces the brief's required outputs:
top-5 scorecards, the "high raw upside pushed down" example, an agent-written
top-3 summary, and a compact tool-call trace.

Run it as a module::

    python -m pv_agent.run

or import :func:`run_pipeline` for a programmatic, testable result.

The division of labor is preserved end to end: the agent judges (agent.py),
deterministic code simulates and scores (physics.py / scoring.py), and this file
only orchestrates and presents. It computes no judgments and no physics itself.
"""

from __future__ import annotations

import os
from collections import Counter

from dotenv import load_dotenv

from pv_agent.agent import run_agent, summarize_top_sites
from pv_agent.schemas import Scorecard, ScoringConfig
from pv_agent.scoring import build_scorecard
from pv_agent.tools import Dataset

load_dotenv()

# Confidence at/below this marks a scorecard as a "suspicious" high-upside trap
# when hunting for the pushed-down example.
_LOW_CONFIDENCE = 0.3


def _tool_call_trace(result) -> dict[str, int]:
    """Count tool calls the agent made, keyed by tool name.

    Reads the run's message history: every ``ToolCallPart`` (``part_kind ==
    'tool-call'``) is one call the model issued through the trust boundary.
    """
    counts: Counter[str] = Counter()
    for message in result.all_messages():
        for part in getattr(message, "parts", []):
            if getattr(part, "part_kind", None) == "tool-call":
                counts[part.tool_name] += 1
    return dict(counts)


def _build_scorecards(judgments, dataset: Dataset, cfg: ScoringConfig) -> list[Scorecard]:
    """Simulate each judged candidate and assemble its full Scorecard.

    Re-simulates here (cheap: few candidates, and physics caches solar position
    per location) to get the real energy upside and recommended orientation, then
    hands the agent's judged scores + flags to build_scorecard for the arithmetic.
    """
    scorecards: list[Scorecard] = []
    for judgment in judgments:
        sim = dataset.simulate_site_tool(judgment.system_id)
        scorecards.append(
            build_scorecard(
                system_id=judgment.system_id,
                energy_upside=sim.relative_energy_upside,
                data_confidence=judgment.data_confidence,
                actionability=judgment.actionability,
                recommended_tilt=sim.recommended_tilt,
                recommended_azimuth=sim.recommended_azimuth,
                flags=judgment.flags,
                cfg=cfg,
            )
        )
    return scorecards


def _find_pushed_down(ranked: list[Scorecard]) -> dict | None:
    """Find the clearest "high raw upside, low final score" example.

    Among scorecards with low data-confidence (the suspicious-default traps),
    pick the one with the largest raw energy upside — the site whose big raw
    number was deliberately sunk by the trust factor. Returns its details and
    rank, or None if no such trap surfaced.
    """
    traps = [c for c in ranked if c.data_confidence.score <= _LOW_CONFIDENCE]
    if not traps:
        return None
    trap = max(traps, key=lambda c: c.energy_upside)
    return {
        "scorecard": trap,
        "rank": ranked.index(trap) + 1,
        "total": len(ranked),
    }


def _format_top_brief(top: list[Scorecard]) -> str:
    """Format the top scorecards into a compact brief for the summary agent."""
    lines = []
    for i, c in enumerate(top, 1):
        lines.append(
            f"{i}. {c.system_id}: raw_upside={c.energy_upside:.1%}, "
            f"final={c.final_score:.3f}, {c.recommendation.value}; "
            f"confidence={c.data_confidence.score:.2f} ({c.data_confidence.reason}); "
            f"actionability={c.actionability.score:.2f} ({c.actionability.reason}); "
            f"flags={c.flags}"
        )
    return "\n".join(lines)


def run_pipeline(
    dataset: Dataset,
    shortlist_size: int = 15,
    make_summary: bool = True,
) -> dict:
    """Run the full pipeline and return its results programmatically.

    Returns a dict with ``top5`` (list[Scorecard]), ``pushed_down_example``
    (dict | None), ``summary`` (str), ``trace`` (dict[str, int]), and
    ``all_scorecards`` (the full ranked list). ``make_summary=False`` skips the
    second LLM call (useful for tests / offline runs).
    """
    cfg = ScoringConfig()

    result = run_agent(dataset, shortlist_size)
    judgments = result.output
    trace = _tool_call_trace(result)

    scorecards = _build_scorecards(judgments, dataset, cfg)
    ranked = sorted(scorecards, key=lambda c: c.final_score, reverse=True)
    top5 = ranked[:5]

    pushed_down = _find_pushed_down(ranked)

    if make_summary and top5:
        summary = summarize_top_sites(_format_top_brief(top5[:3]))
    else:
        summary = _format_top_brief(top5[:3])

    return {
        "top5": top5,
        "pushed_down_example": pushed_down,
        "summary": summary,
        "trace": trace,
        "all_scorecards": ranked,
    }


# --------------------------------------------------------------------------- #
# Presentation
# --------------------------------------------------------------------------- #
def _print_scorecard(rank: int, c: Scorecard) -> None:
    print(f"  #{rank}  {c.system_id}   [{c.recommendation.value}]   final_score={c.final_score:.3f}")
    print(f"       raw energy upside : {c.energy_upside:.1%}   (normalized {c.normalized_energy_upside:.2f})")
    print(f"       data_confidence   : {c.data_confidence.score:.2f}  — {c.data_confidence.reason}")
    print(f"       actionability     : {c.actionability.score:.2f}  — {c.actionability.reason}")
    print(f"       recommend orient. : tilt {c.recommended_tilt:.0f}°, azimuth {c.recommended_azimuth:.0f}°")
    if c.flags:
        print(f"       flags             : {', '.join(c.flags)}")
    print()


def _print_report(results: dict) -> None:
    print("\n" + "=" * 72)
    print("TOP 5 SITES FOR RE-ORIENTATION REVIEW")
    print("=" * 72)
    for i, card in enumerate(results["top5"], 1):
        _print_scorecard(i, card)

    print("=" * 72)
    print("HIGH RAW UPSIDE, PUSHED DOWN")
    print("=" * 72)
    pd = results["pushed_down_example"]
    if pd is None:
        print("  (no low-confidence high-upside trap surfaced in this run)\n")
    else:
        c = pd["scorecard"]
        print(
            f"  High raw upside pushed down: {c.system_id} has {c.energy_upside:.1%} "
            f"upside but final score {c.final_score:.3f} due to low confidence "
            f"({c.data_confidence.score:.2f}) — ranked #{pd['rank']} of {pd['total']}."
        )
        print(f"  Why: {c.data_confidence.reason}\n")

    print("=" * 72)
    print("REVIEWER SUMMARY (agent-written)")
    print("=" * 72)
    print(results["summary"] + "\n")

    print("=" * 72)
    print("TOOL-CALL TRACE")
    print("=" * 72)
    trace = results["trace"]
    if trace:
        print("  " + ", ".join(f"{name}: {n}" for name, n in sorted(trace.items())))
    else:
        print("  (no tool calls recorded)")
    print()


def main() -> None:
    """Build the dataset, run the pipeline, and print all required outputs."""
    shortlist_size = int(os.environ.get("PV_AGENT_SHORTLIST", "15"))
    dataset = Dataset()
    results = run_pipeline(dataset, shortlist_size=shortlist_size)
    _print_report(results)


if __name__ == "__main__":
    main()
