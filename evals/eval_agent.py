"""Lightweight evaluation of the agent's non-deterministic judgments.

This is NOT a unit test with exact assertions. The agent is an LLM: its exact
scores vary run to run. This eval checks two cheaper, more robust things:

  Part 1 — Golden band checks. On a handful of hand-picked real records (one per
  known case type), do the judged data_confidence / actionability land in
  sensible BANDS, and do the required flags appear?

  Part 2 — Reason consistency (no labels). Across every judgment in the run, is
  each judged score accompanied by a non-empty reason, and does any low-confidence
  score (< 0.4) actually cite an anomaly term?

The bands are GENEROUS by design. The point is to catch regressions — a 0/0
trap scoring 0.9, a clean site scoring 0.1, an east-facing site penalized into
the ground, a missing "verify orientation" flag — NOT to assert exact values a
non-deterministic model can never reproduce.

Run it::

    python evals/eval_agent.py

Needs ANTHROPIC_API_KEY (loaded from .env).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from dotenv import load_dotenv

# Make the src/ package importable when run as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pv_agent.agent import CandidateJudgment, agent  # noqa: E402
from pv_agent.tools import Dataset  # noqa: E402

load_dotenv()

_GOLDEN_PATH = Path(__file__).resolve().parent / "golden.json"

# Terms a genuinely low-confidence reason should reference — the anomaly
# vocabulary the agent judges over.
_ANOMALY_TERMS = ("orientation", "default", "tracking", "contradiction", "size", "module")


def _load_golden() -> list[dict]:
    with open(_GOLDEN_PATH) as fh:
        return json.load(fh)


def _judge_golden_ids(dataset: Dataset, ids: list[str]) -> list[CandidateJudgment]:
    """Judge exactly the golden ids.

    Rather than hope the misalignment shortlist happens to contain low-misalignment
    cases (a clean or east-facing site never would), we ask the agent to judge the
    specific ids directly — it still investigates each via get_site_details and
    simulate_site, so the judgment path under test is the real one.
    """
    id_list = ", ".join(ids)
    prompt = (
        "Judge these specific solar sites for panel re-orientation, and ONLY "
        f"these: {id_list}. For each site, call get_site_details and "
        "simulate_site to gather evidence, then return exactly one "
        "CandidateJudgment per site (data_confidence, actionability, flags). "
        "Do not shortlist or add other sites."
    )
    return agent.run_sync(prompt, deps=dataset).output


def _flag_ok(flags: list[str], required) -> bool:
    """Case-insensitive flag check. `required` is a substring or any-of list."""
    haystack = " ".join(flags).lower()
    if isinstance(required, list):
        return any(sub.lower() in haystack for sub in required)
    return required.lower() in haystack


def _check_case(case: dict, judgment: CandidateJudgment) -> tuple[bool, list[str]]:
    """Return (passed, reasons_for_failure) for one golden case."""
    problems: list[str] = []

    conf = judgment.data_confidence.score
    if not (case["expected_confidence_min"] <= conf <= case["expected_confidence_max"]):
        problems.append(
            f"confidence {conf:.2f} outside "
            f"[{case['expected_confidence_min']}, {case['expected_confidence_max']}]"
        )

    act = judgment.actionability.score
    if not (case["expected_actionability_min"] <= act <= case["expected_actionability_max"]):
        problems.append(
            f"actionability {act:.2f} outside "
            f"[{case['expected_actionability_min']}, {case['expected_actionability_max']}]"
        )

    required = case.get("required_flag_substring")
    if required is not None and not _flag_ok(judgment.flags, required):
        problems.append(f"no flag matching {required!r} in {judgment.flags}")

    return (not problems), problems


def _reason_consistency(judgment: CandidateJudgment) -> tuple[bool, list[str]]:
    """Every judged score has a non-empty reason; low confidence cites an anomaly."""
    problems: list[str] = []
    if not judgment.data_confidence.reason.strip():
        problems.append("empty data_confidence reason")
    if not judgment.actionability.reason.strip():
        problems.append("empty actionability reason")
    if judgment.data_confidence.score < 0.4:
        reason = judgment.data_confidence.reason.lower()
        if not any(term in reason for term in _ANOMALY_TERMS):
            problems.append(
                f"low confidence {judgment.data_confidence.score:.2f} but reason "
                f"cites no anomaly term: {judgment.data_confidence.reason!r}"
            )
    return (not problems), problems


def main() -> int:
    golden = _load_golden()
    dataset = Dataset()
    ids = [c["system_id"] for c in golden]

    print(f"Judging {len(ids)} golden sites: {', '.join(ids)}\n")
    judgments = _judge_golden_ids(dataset, ids)
    by_id = {j.system_id: j for j in judgments}

    # ---- Part 1: golden band checks ----
    print("=" * 68)
    print("PART 1 — GOLDEN BAND CHECKS")
    print("=" * 68)
    passed = failed = skipped = 0
    for case in golden:
        sid, ctype = case["system_id"], case["case_type"]
        judgment = by_id.get(sid)
        if judgment is None:
            print(f"  SKIP  {sid:12} {ctype:28} (not returned by agent)")
            skipped += 1
            continue
        ok, problems = _check_case(case, judgment)
        if ok:
            print(
                f"  PASS  {sid:12} {ctype:28} "
                f"conf={judgment.data_confidence.score:.2f} act={judgment.actionability.score:.2f}"
            )
            passed += 1
        else:
            print(f"  FAIL  {sid:12} {ctype:28} " + "; ".join(problems))
            failed += 1
    print(f"\n  band checks: {passed} passed, {failed} failed, {skipped} skipped\n")

    # ---- Part 2: reason consistency (all judgments) ----
    print("=" * 68)
    print("PART 2 — REASON CONSISTENCY (all judgments)")
    print("=" * 68)
    consistent = 0
    for judgment in judgments:
        ok, problems = _reason_consistency(judgment)
        if ok:
            consistent += 1
        else:
            print(f"  INCONSISTENT  {judgment.system_id:12} " + "; ".join(problems))
    print(f"\n  reason consistency: {consistent}/{len(judgments)} judgments passed\n")

    # Exit non-zero if any band check failed (a real regression signal); skips and
    # reason-consistency are reported but do not fail the run on their own.
    print("=" * 68)
    print(f"SUMMARY: band {passed} pass / {failed} fail / {skipped} skip  |  "
          f"reasons {consistent}/{len(judgments)} consistent")
    print("=" * 68)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
