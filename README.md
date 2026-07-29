# PV Re-Orientation Prioritization Agent

Finds the **top 5 solar sites worth human review** for panel re-orientation from a
~10,000-record installation CSV. It combines a **deterministic physics engine**
(pvlib clear-sky POA + grid-search orientation optimization) that estimates how
much energy each site could gain, with a **single LLM agent** that judges what the
physics can't: *can we trust this record?* and *can we act on it?* The core
principle throughout: **code detects facts and does the math; the agent weighs the
fuzzy judgment.** A big theoretical upside on a record we can't trust ranks below a
solid, reliable one.

## Architecture

Three layers separated by one trust boundary:

```
Deterministic layer            →  Typed bounded tools  →  Single agent
(cleaning · physics · scoring)    (≤ 50 rows / call)      (investigate · judge · explain)
```

- **Deterministic** — CSV cleaning, pvlib physics, orientation optimization, and
  all scoring arithmetic. Fully testable, no LLM.
- **Trust boundary** — four typed tools; no tool returns more than 50 raw rows, so
  the full CSV never enters the model's context.
- **Agent** — investigates via the tools and returns a judged `data_confidence`
  and `actionability` (each 0–1 + a grounded reason) plus reviewer flags. It never
  computes the final score or ranking.

Full reasoning is in [DECISIONS.md](DECISIONS.md) and [PLAN.md](PLAN.md).

## Setup

Requires **Python 3.11+** (a dependency, pandas 3.x, drops 3.10).

```bash
pip install -e .            # or: uv pip install -e .
cp .env.example .env        # then add your ANTHROPIC_API_KEY
```

> **Note:** direct module/script runs use a `PYTHONPATH=src` prefix (an editable-
> install path quirk). The working commands below all include it.

## Running the pipeline

```bash
PYTHONPATH=src python -m pv_agent.run
```

Produces the **top-5 scorecards**, the **high-raw-upside-pushed-down** example, the
**agent-written top-3 summary**, and a compact **tool-call trace**. It makes ~2 LLM
calls (judgment + summary) and costs well under a dollar.

Optional environment variables:

| Variable | Default | Meaning |
|----------|---------|---------|
| `PV_AGENT_SHORTLIST` | `15` | How many candidates the agent is asked to shortlist |
| `PV_AGENT_MODEL` | `claude-haiku-4-5-20251001` | Model (Anthropic unless a `provider:` prefix is given) |

## Tests

```bash
pytest
```

Runs the **94 deterministic tests** (cleaning, physics, geo, tools, scoring).
**No API key needed** — the whole deterministic core is verified offline.

## Evaluation

```bash
PYTHONPATH=src python evals/eval_agent.py
```

The non-deterministic **agent eval**: golden band checks on one real record of each
known type (trap, clean, tracking-unknown, third-party, east-facing) plus a
reason-consistency pass. **Needs the API key.** Bands are **generous by design** —
the goal is catching regressions (a trap scoring 0.9, a clean site scoring 0.1),
not asserting exact values a non-deterministic model can't reproduce.

## Docker

Build once, then run the pipeline, tests, or eval in the container. The
`ANTHROPIC_API_KEY` is passed at runtime — never baked into the image.

```bash
docker build -t pv-agent .

docker run pv-agent pytest                                   # 94 tests, no key
docker run -e ANTHROPIC_API_KEY=sk-... pv-agent              # full pipeline (default)
docker run -e ANTHROPIC_API_KEY=sk-... pv-agent python evals/eval_agent.py
```

## Project layout

```
src/pv_agent/
  schemas.py    Pydantic models — the contracts between layers
  cleaning.py   Raw CSV rows → cleaned SiteRecords + detected anomalies (facts)
  geo.py        Offline zip → lat/lon geocoding (pgeocode), cached
  physics.py    pvlib clear-sky POA + grid-search orientation optimization
  scoring.py    Fixed-cap normalization, multiplicative final score, banding
  tools.py      Dataset + the four typed, ≤50-row tools (the trust boundary)
  agent.py      The single pydantic-ai agent: investigate + judge + summarize
  run.py        Pipeline orchestrator and entrypoint (python -m pv_agent.run)
tests/          94 deterministic unit tests (no API key)
evals/          Non-deterministic agent eval + golden.json (needs API key)
```

## Tools used

- **Claude Code** as the AI coding assistant used to build this project.
- **Anthropic Claude Haiku** (pinned `claude-haiku-4-5-20251001`) as the judgment
  agent, via **pydantic-ai**.
- **pvlib** for the solar physics; **pgeocode** for offline geocoding.

## Key design decisions

- **Deterministic / agent split** — anything with a correct answer (physics,
  arithmetic) is code; only the fuzzy, record-dependent judgment is the agent's.
- **≤50-row bounded tools** keep the full 10k-row CSV out of the model's context —
  there is no bulk accessor, and a shared guard enforces the limit.
- **Fixed-cap normalization + multiplicative score** —
  `final = normalized_upside × data_confidence × actionability`; any weak factor
  sinks the site, and the cap is stable/reproducible across runs.
- **Shortlist bucket-split** — 0/0-default records (no trustworthy orientation) are
  separated from genuine candidates, so a few traps are surfaced for the required
  "pushed-down" demonstration without flooding out real opportunities.

See [DECISIONS.md](DECISIONS.md) for the full rationale behind each.
