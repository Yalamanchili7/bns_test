# DECISIONS

A one-page account of what is deterministic vs agent-driven and why, the tool
interface, how the full CSV stays out of the LLM context, how missing/ambiguous
data is handled, and the top next steps.

## Deterministic vs agent-driven — and why

The boundary follows one rule: **if it has a correct answer, it is code; if it is
fuzzy and record-dependent, it is the agent.** Code *detects facts*; the agent
*weighs them*.

| Deterministic (code) | Agent (LLM) |
|----------------------|-------------|
| CSV access, filtering, cleaning, geocoding | Investigation strategy (what to fetch, when to simulate) |
| Anomaly *detection* (az=0 and tilt=0, tracking unknown, contradictions) | *Interpreting* those anomalies into `data_confidence` |
| pvlib physics: solar position, clear-sky, transposition | `actionability` judgment (ownership, age, mounting, trackers) |
| Orientation optimization (grid search) | `flags` — what a human should verify |
| Normalization + `final_score` arithmetic | The human-facing top-3 summary |

Why the split lands where it does:
- The orientation optimum is a smooth 2-D maximization — arithmetic, so it is code.
  The brief is explicit: "keep it in plain code, not the LLM."
- Whether `az=0, tilt=0` means "real north-flat array" or "data-entry default" has
  no clean rule; it is judgment weighing co-occurrence and plausibility, so it is
  the agent. Code can *detect* the pattern but should not *decide* what it means.
- `final_score` is multiplication of three numbers — code. The agent supplies the
  judged factors; it never does the arithmetic or ranking.

## Tools the agent can call (the trust boundary)

All tools are Pydantic-typed and row-bounded. A single shared guard
(`enforce_row_limit`, `MAX_ROWS = 50`) enforces the limit and is unit-tested.

- `get_dataset_summary()` — aggregate counts and missingness only; returns **no raw
  records**. Lets the agent understand scale and data quality before fetching.
- `shortlist_candidates(limit, include_suspicious)` — runs the cheap deterministic
  misalignment heuristic over the eligible pool in code; returns only the top-N
  (bounded at 50). See the shortlist decision below.
- `get_site_details(ids)` — cleaned fields plus detected `anomalies` for specific
  sites (bounded at 50). The evidence feed for judgment.
- `simulate_site(id)` — the pvlib physics and grid-search optimization behind one
  call; returns POA, upside, and recommended orientation. Single-site by design so
  the agent simulates selectively, never in bulk.

## Keeping the full CSV out of context

The dataset lives only in an in-memory list of records behind the tools; the agent
never receives the CSV. Three mechanisms enforce this:
1. **No bulk tool.** There is no "return all rows" call. The only row-returning
   tools are `shortlist_candidates` and `get_site_details`, both capped at 50.
2. **A cheap pre-filter in code.** Filtering and the misalignment heuristic run
   deterministically over all ~10k; the agent sees only the shortlist. A naive
   fan-out of physics over 10k is structurally impossible via the tools.
3. **Cost-tiered funnel.** 10k cheap-filtered -> shortlist simulated -> finalists
   judged. Expensive work (physics, LLM) only ever touches small, pre-narrowed sets.
   The tool-call trace confirms this: one summary, one shortlist, one simulate per
   shortlisted site, one batched details call.

## The shortlist: why it is not pure misalignment ranking

Profiling found **391 records with azimuth = 0 and tilt = 0** — a data-entry
default, not a real north-flat array. These have the *maximum* apparent
misalignment, so a pure top-N-by-misalignment shortlist is flooded entirely by
them, and the final top-5 would be all traps — every one scored near zero and
marked Skip. That fails the brief's required output #1 ("top 5 sites *worth human
review*"): a queue of five "do not bother" records is not a useful review list.

**The principled fix:** a 0/0-default has *no trustworthy orientation*, so it
cannot legitimately be ranked by orientation-misalignment — you cannot measure how
misaligned a panel is when you do not believe its reported orientation. The
defaults therefore belong in a separate "suspicious" bucket, not the misalignment
queue. `shortlist_candidates` splits accordingly:
- **Plausible candidates** (no `orientation_default_0_0`), ranked by misalignment —
  the genuine re-orientation opportunities that populate the real top-5.
- **Suspicious defaults** — a small number (`include_suspicious`, default 3) are
  deliberately surfaced so the system still demonstrates a high-upside record being
  pushed down (required output #2), without flooding the queue.

This satisfies both required outputs: a useful top-5 of real opportunities, and an
explicit "high raw upside pushed down" example contrasted against it. It is also
supported by the brief's own wording — it asks the agent to simulate "plausible
candidates," and a data-entry default is not a plausible re-orientation candidate.

## The summary agent (interpreting "one agent")

The brief says "one agent plus deterministic tools" and "do not build multi-agent
orchestration." The single **judgment agent** does all investigation and judgment.
The top-3 reviewer summary is produced by a separate, **tool-less** summarizer that
has no data access and does no arithmetic — it only turns the already-computed
top-3 scorecards into human-facing prose. It is not a second *investigating* agent
coordinating the task, so it is consistent with the "no multi-agent orchestration"
intent. Flagged here explicitly rather than left to be discovered.

## Scoring

- **Normalization** uses a fixed 0.30 upside cap, not min-max: stable and
  reproducible across runs (a site's score does not depend on the batch) and robust
  to the high-upside garbage records. It intentionally saturates — beyond ~30%
  upside is uniformly "large," so what differentiates large-upside sites is whether
  we can trust and act on them (confidence x actionability). Min-max self-scales to
  the data but is pool-dependent and outlier-sensitive; noted as the alternative.
- `final_score = normalized_upside x data_confidence x actionability` (brief
  default). The multiplication is deliberate: any factor near zero sinks the site,
  so a large prize we cannot trust or act on still ranks low.
- Recommendation bands: >=0.5 Prioritize, 0.2-0.5 Review, <0.2 Skip (tunable).

## Physics assumptions

- **Clear-sky + representative days** (one day per month, hourly) are proxies
  adequate for *ranking* relative upside, not for quantifying real kWh. A constant
  timezone/sampling bias cancels in the optimal-vs-current ratio.
- **Solar position and clear-sky are cached per location** and reused across all
  247 grid orientations — the single biggest speedup in the physics layer.
- **Grid search** (tilt 0-60 step 5, azimuth 90-270 step 10) over scipy.optimize:
  the objective is smooth, unimodal, 2-D and bounded, so grid search is trivially
  correct and inspectable. Bounds are physics-informed (N-hemisphere optimum is
  always southern; optimal tilt tracks latitude, never exceeding ~60 in the US).

## Missing / ambiguous data

Profiling the file (not the brief) drove this. The sentinel is `-1`, not the
`-9999` the brief states; ~15% of zips (1,461 of 10,000) have stripped leading
zeros; only ~30% of records have the orientation (az+tilt+zip) needed to simulate.

- **Missing orientation** -> record is not a candidate (nothing to evaluate);
  excluded from the pool rather than guessed.
- **Unknown tracking/ownership/mounting** -> kept as an explicit `Unknown` state,
  never collapsed into a value. It lowers confidence/actionability and raises a flag
  rather than producing a confident wrong answer.
- **Stripped-leading-zero zips** (`5647` -> `05647`) are padded in cleaning before
  geocoding, or ~15% of records (heavily New England) would fail to geocode.
- **The 0/0-default pattern** -> detected in code, judged low-confidence by the
  agent, and flagged. It is the required "high raw upside pushed down" case.
- **East-facing arrays** -> flagged as possibly intentional (morning load), **not**
  penalized — an intentional choice is not a mistake.

## Known edge cases / limitations

- **Extreme tilt (e.g. 90 vertical panels)** currently pass as valid and can
  produce very large upside (a vertical panel is far from optimal). The agent trusts
  an internally-consistent record but correctly flags it for verification. A future
  `extreme_tilt` anomaly (tilt >= ~80) would lower confidence, since such tilts are
  often data or installation oddities on fixed systems. Noted, not yet built.
- **Re-simulation in `run.py`** — judged candidates are simulated again when
  building scorecards rather than threading the agent's tool results out. This is
  near-free because physics caches solar position per location; a production version
  would thread the results through to avoid the recompute entirely.
- **This dataset is rooftop-heavy** (Tracking the Sun); BrightNight's real fleet is
  utility-scale ground-mount, so actionability inputs (ownership, interconnection,
  land) would be weighted differently in production — the architecture is unchanged.

## Top three next steps

1. **Real production data** — replace clear-sky with actual weather/irradiance and,
   where available, metered generation, to move from a *prioritization* to a
   defensible *quantified* upside. Add shading/soiling checks and an `extreme_tilt`
   anomaly.
2. **Cost-adjusted actionability** — the true decision is net value: energy gain
   minus re-orientation cost and downtime. Add a cost model so the ranking reflects
   ROI, not just the gross prize.
3. **A growing, human-fed eval set** — promote the hand-labeled smoke test into a
   golden set that grows from reviewer corrections, closing a feedback loop that
   tightens the judged metrics over time.