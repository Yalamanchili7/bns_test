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


def _reconcile_judgments(judgments, expected_ids: set[str]) -> tuple[list, dict]:
    """Validate the agent's returned judgments against the requested shortlist.

    The shortlist is deterministic, so we know exactly which ids the agent was
    asked to judge. We do not blindly trust its output: a dropped, duplicated, or
    hallucinated id would otherwise propagate silently into the ranking.

    Returns ``(reconciled, report)`` where ``reconciled`` is the deduped judgments
    whose ids are in the shortlist (first occurrence wins), and ``report`` records
    what was ``dropped`` (requested but not judged), ``unexpected`` (judged but not
    requested — hallucinated / off-list), and ``duplicated``.
    """
    seen: set[str] = set()
    reconciled = []
    unexpected: list[str] = []
    duplicated: list[str] = []
    for j in judgments:
        if j.system_id in seen:
            duplicated.append(j.system_id)
            continue
        seen.add(j.system_id)
        if j.system_id in expected_ids:
            reconciled.append(j)
        else:
            unexpected.append(j.system_id)

    dropped = sorted(expected_ids - seen)
    report = {
        "requested": len(expected_ids),
        "judged": len(reconciled),
        "dropped": dropped,
        "unexpected": sorted(unexpected),
        "duplicated": sorted(set(duplicated)),
    }
    return reconciled, report


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


def _find_pushed_down(ranked: list[Scorecard], suspicious_ids: set[str]) -> dict | None:
    """Find the clearest "high raw upside, low final score" example.

    Required output #2 must not silently vanish, so we resolve it in two tiers:

    1. **Ideal** — among scorecards the agent judged low-confidence
       (``<= _LOW_CONFIDENCE``), take the largest raw upside: a big number the
       trust factor deliberately sank.
    2. **Fallback** — if the agent did not score any trap that low, fall back to
       the highest-upside *suspicious-default* (0/0) site it judged. Still an
       agent-judged card (we do not fabricate the demonstration), just without
       the confidence cliff that previously returned ``None``.

    Returns the chosen card, its rank, the total, and which tier fired — or
    ``None`` only if no suspicious-default candidate was judged at all.
    """
    low_conf = [c for c in ranked if c.data_confidence.score <= _LOW_CONFIDENCE]
    if low_conf:
        trap, method = max(low_conf, key=lambda c: c.energy_upside), "low_confidence"
    else:
        suspicious = [c for c in ranked if c.system_id in suspicious_ids]
        if not suspicious:
            return None
        trap = max(suspicious, key=lambda c: c.energy_upside)
        method = "suspicious_default_fallback"
    return {
        "scorecard": trap,
        "rank": ranked.index(trap) + 1,
        "total": len(ranked),
        "method": method,
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

    # The shortlist is deterministic, so compute the expected id set here to
    # reconcile the agent's output against, and to know which ids are the
    # suspicious 0/0-default traps (for the guaranteed pushed-down example).
    shortlist = dataset.shortlist_candidates(shortlist_size)
    expected_ids = {c["system_id"] for c in shortlist}
    suspicious_ids = {c["system_id"] for c in shortlist if c["is_suspicious_default"]}

    result = run_agent(dataset, shortlist_size)
    trace = _tool_call_trace(result)

    judgments, reconciliation = _reconcile_judgments(result.output, expected_ids)

    scorecards = _build_scorecards(judgments, dataset, cfg)
    ranked = sorted(scorecards, key=lambda c: c.final_score, reverse=True)
    top5 = ranked[:5]

    pushed_down = _find_pushed_down(ranked, suspicious_ids)

    if make_summary and top5:
        summary = summarize_top_sites(_format_top_brief(top5[:3]))
    else:
        summary = _format_top_brief(top5[:3])

    return {
        "top5": top5,
        "pushed_down_example": pushed_down,
        "summary": summary,
        "trace": trace,
        "reconciliation": reconciliation,
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
        print("  (no suspicious-default candidate was judged in this run)\n")
    else:
        c = pd["scorecard"]
        print(
            f"  High raw upside pushed down: {c.system_id} has {c.energy_upside:.1%} "
            f"upside but final score {c.final_score:.3f} due to low confidence "
            f"({c.data_confidence.score:.2f}) — ranked #{pd['rank']} of {pd['total']}."
        )
        print(f"  Why: {c.data_confidence.reason}")
        if pd["method"] == "suspicious_default_fallback":
            print(
                "  (fallback: no judged trap fell below the confidence threshold; "
                "showing the highest-upside 0/0-default the agent judged.)"
            )
        print()

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

    print("=" * 72)
    print("SHORTLIST RECONCILIATION")
    print("=" * 72)
    rec = results["reconciliation"]
    print(f"  requested {rec['requested']} shortlisted, judged {rec['judged']}.")
    for label in ("dropped", "unexpected", "duplicated"):
        if rec[label]:
            print(f"  {label}: {', '.join(rec[label])}")
    if not (rec["dropped"] or rec["unexpected"] or rec["duplicated"]):
        print("  clean: agent judged exactly the requested shortlist.")
    print()


def main() -> None:
    """Build the dataset, run the pipeline, and print all required outputs."""
    shortlist_size = int(os.environ.get("PV_AGENT_SHORTLIST", "15"))
    dataset = Dataset()
    results = run_pipeline(dataset, shortlist_size=shortlist_size)
    _print_report(results)


if __name__ == "__main__":
    main()
