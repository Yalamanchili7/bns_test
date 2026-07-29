# PLAN — PV Re-Orientation Prioritization Agent

## Problem

Triage ~10,000 solar installation records down to the **5 sites most worth human
review** for panel re-orientation, and return an explainable scorecard for each.

The core insight from the brief: *worth review is not raw energy upside*. A large
theoretical gain on a record we cannot trust or act on must rank **below** a solid,
reliable one. So three things combine per site:

1. How large is the energy prize?  → deterministic physics
2. Can we trust this record?        → agent judgment
3. Can we act on it?                → agent judgment

`final_score = normalized_energy_upside × data_confidence × actionability`

The multiplication is deliberate: any factor near zero sinks the site.

## Data reality (from profiling the actual CSV)

The dataset is the LBNL "Tracking the Sun" sample. Profiling revealed several
things the brief does not state:

- **Missing-value sentinel is `-1`, not `-9999`** as the brief claims. Both are handled.
- **~15% of zip codes (1,461 of 10,000) have stripped leading zeros** (`5647`
  should be `05647`); they must be zero-padded before geocoding or they mislocate.
- **~2,960 of 10,000 records have azimuth + tilt + zip all present** (~2,760 after
  excluding trackers) — this is the simulatable candidate pool; the rest cannot be
  evaluated for re-orientation.
- **391 records have azimuth = 0 AND tilt = 0** — almost certainly a data-entry
  default, not a real north-flat array. These produce the largest *raw* upside and
  are the planted "high upside, low trust" trap. This is our required
  "high raw upside pushed down" example.
- Trackers (~200 in pool) cannot be re-oriented → excluded. ~850 have unknown
  tracking → simulated but flagged.
- Actionability signals (ownership, mounting) are largely unknown → judged under
  partial information, with flags rather than confident wrong answers.

## Architecture

Three layers separated by one trust boundary, with a cost-tiered funnel through it.

```
AGENT LAYER (LLM, Pydantic-AI): investigate · judge · explain
        ↕  TRUST BOUNDARY: 4 typed tools, ≤50 rows per call  ↕
DETERMINISTIC LAYER (code, testable): data · cleaning · physics · scoring
```

Two principles govern the whole design:
- **Code detects, the agent weighs.** Code reports facts (az=0, tracking unknown);
  the agent interprets them into a judged score with a reason.
- **Cost rises as volume falls.** Cheap filters over 10k → physics on ~15 →
  LLM judgment on the finalists. The expensive model never touches the full dataset.

## Pipeline (end to end)

```
1. get_dataset_summary()        → agent orients itself (counts, missingness); no rows
2. eligibility filter (all 10k) → ~2,760 with valid az+tilt+zip, non-tracker   [code]
3. shortlist_candidates(15)     → cheap misalignment heuristic, no physics      [code]
4. simulate_site(id) × ~15      → grid-search optimization → raw upside          [code]
5. get_site_details(ids)        → cleaned fields + detected anomalies (evidence)
6. agent judges each            → data_confidence, actionability, flags (+reasons)
7. scoring                      → normalize + final_score → rank → top 5         [code]
8. agent writes top-3 summary; code emits tool-call trace
```

## Deterministic layer — decisions

**Cleaning.** Both sentinels (`-1`, `-9999`) → null. Zip `str.split('.')[0].zfill(5)`.
Three-state categoricals for tracking / ownership / mounting (never collapse
"unknown" into a value). Emit `anomalies: list[str]` per record (facts, not judgments):
`orientation_default_0_0`, `tracking_unknown`, `third_party_owned`, `installed_<year>`,
`size_module_contradiction`, `east_facing`.

**Physics (per site, inside `simulate_site`).**
- Solar position from lat/lon/time via `pvlib.solarposition.get_solarposition`
  (computed, never read from data).
- Clear-sky (Ineichen) irradiance — brief-licensed proxy; no weather data needed.
- Transposition via `get_total_irradiance(model='haydavies')` — accounts for
  circumsolar diffuse, fully specified by clear-sky outputs, simpler than Perez.
- **Representative days**: one per month (12 days), hourly. Ranking is preserved
  under this sampling; absolute POA is not the deliverable.
- **Solar position computed once per site**, reused across all grid orientations.

**Orientation optimization (grid search).**
- Search tilt ∈ [0,60] step 5°, azimuth ∈ [90,270] step 10° → 247 evaluations.
- Bounds are physics-informed: N-hemisphere optimum is always in the southern half;
  optimal tilt tracks latitude and never exceeds ~60° in the US. Pruning the rest
  is free correctness.
- Grid search over `scipy.optimize`: the objective is smooth, unimodal, 2-D, and
  bounded — grid search is trivially correct, inspectable, and dependency-free.
  Pragmatic over clever.
- `upside = (optimal_POA − current_POA) / current_POA`.

**Pre-filter heuristic (cheap, over all eligible).**
`misalignment = |azimuth − 180|/180 + |tilt − latitude_est|/90`, a monotone proxy
for upside. Uses a state→latitude estimate so no geocoding is needed at this stage;
precise geocoding is deferred to the ~15 shortlisted sites. Shortlist = top 15
(3× the output size, so the heuristic's imperfection can't drop real winners).

**Scoring.**
- Normalize upside with a **fixed cap** (0.30 → 1.0), not min-max: stable across
  runs, reproducible, outlier-robust against the high-upside garbage records, and
  absolute/interpretable for a reviewer. Cap lives in `ScoringConfig`.
- `final_score = normalized_upside × data_confidence × actionability` (brief default).
- Bands: ≥0.5 Prioritize, 0.2–0.5 Review, <0.2 Skip (tunable constants).

## Agent layer — decisions

Single Pydantic-AI agent (brief: no multi-agent). Investigates via the four tools,
then for each simulated candidate returns a typed `Scorecard`:

- **data_confidence** (0–1 + reason): weighs the detected anomalies. 0/0-default →
  ~0.15; size/module contradiction → low; tracking unknown → mild down + flag;
  clean consistent record → high.
- **actionability** (0–1 + reason): third-party → down; old install → repowering
  flag; ground-mount → up, roof → mild down; **east-facing → flagged, not
  penalized** (may be an intentional morning-load choice).
- **flags**: the human reviewer's checklist ("verify reported orientation",
  "confirm fixed-tilt").
- Continuous scores with prompt anchors; eval checks they land in expected bands.

The agent never computes `final_score` — that arithmetic is code.

## Tools (the trust boundary)

| Tool | Returns | Bound |
|------|---------|-------|
| `get_dataset_summary()` | aggregate counts/missingness | no rows |
| `shortlist_candidates(limit=15)` | top-N by heuristic (2 buckets) | ≤50 |
| `get_site_details(ids)` | cleaned fields + anomalies | ≤50 |
| `simulate_site(id)` | POA / upside / recommended orientation | 1 site |

One shared `enforce_row_limit(rows, 50)` guard, unit-tested (raises at 51).

## Testing & evaluation

- **Deterministic tests**: physics sanity (south@lat ≈ optimal → ~0 upside;
  north-flat → large upside), normalization arithmetic, scoring arithmetic, and
  the ≤50-row tool bound.
- **Agent eval**: a ~6–8 record hand-labeled set (one of each trap: 0/0-default,
  tracker, third-party, old, east-facing, clean winner) asserting judged scores
  land in expected bands and flags contain expected markers; plus a
  reason-consistency check (every judged score carries a non-empty reason grounded
  in a detected fact).

## Deliverables (brief checklist)

Top-5 scorecards · the high-upside-pushed-down example called out explicitly ·
agent-written top-3 summary · compact tool-call trace · README · DECISIONS.md ·
deterministic tests + agent eval.

## Packaging & hygiene

Python 3.11+, `uv` (pyproject.toml + uv.lock, pinned deps), sane `src/` module
layout, one documented entrypoint for pipeline / tests / evals. Dockerfile if time
permits (API key via env var). Incremental commits throughout.

## Time budget (3–4 h)

scaffold + this plan (commit #1) → cleaning + tests → physics + tests → tools +
bound test + scoring → agent + judgment → eval → docs + Docker. Cut order if
overrunning: Docker → eval breadth → grid resolution. Never cut: the ≤50 bound,
the 0/0-default handling, DECISIONS.md, commit hygiene.
