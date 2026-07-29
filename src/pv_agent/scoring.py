"""Deterministic scoring arithmetic.

Pure functions — no agent, no LLM. The brief is explicit that the final-score
arithmetic and banding stay in code: the agent supplies the judged factors
(``data_confidence``, ``actionability``) and the flags; this module does the
multiplication and thresholding. That split is the code/agent boundary — the
agent never computes ``final_score`` or the recommendation.

    final_score = normalized_energy_upside × data_confidence × actionability
"""

from __future__ import annotations

from pv_agent.schemas import JudgedScore, Recommendation, Scorecard, ScoringConfig


def normalize_upside(raw_upside: float, cfg: ScoringConfig) -> float:
    """Normalize raw energy upside to [0, 1] with a fixed cap.

    Raw upside at or above ``cfg.upside_cap`` (0.30) maps to 1.0; negative upside
    clamps to 0.0. A fixed cap is chosen over min-max normalization deliberately
    (see DECISIONS.md): it is stable and reproducible across runs, robust to the
    high-upside garbage records (e.g. the 0/0-default trap), and absolute /
    interpretable for a reviewer rather than pool-dependent.
    """
    return min(max(raw_upside, 0.0) / cfg.upside_cap, 1.0)


def compute_final_score(
    normalized_upside: float, data_confidence: float, actionability: float
) -> float:
    """Combine the three factors into the final score by multiplication.

    ``normalized_upside × data_confidence × actionability``. The multiplication
    is deliberate: any factor near zero sinks the site. A large theoretical
    energy prize on a record we cannot trust or act on must rank below a solid,
    reliable one — a product enforces that, a sum would not.
    """
    return normalized_upside * data_confidence * actionability


def recommend(final_score: float, cfg: ScoringConfig) -> Recommendation:
    """Band a final score into a triage recommendation.

    ``>= prioritize_threshold`` (0.5) -> Prioritize; ``>= review_threshold``
    (0.2) -> Review; otherwise -> Skip.
    """
    if final_score >= cfg.prioritize_threshold:
        return Recommendation.PRIORITIZE
    if final_score >= cfg.review_threshold:
        return Recommendation.REVIEW
    return Recommendation.SKIP


def build_scorecard(
    system_id: str,
    energy_upside: float,
    data_confidence: JudgedScore,
    actionability: JudgedScore,
    recommended_tilt: float,
    recommended_azimuth: float,
    flags: list[str],
    cfg: ScoringConfig,
) -> Scorecard:
    """Assemble a full :class:`Scorecard` from physics + agent judgment.

    The agent supplies ``energy_upside`` (via simulation), the two
    ``JudgedScore`` objects, and ``flags``. This function does only the
    deterministic arithmetic — normalization, the final-score product, and the
    recommendation banding — and never second-guesses the judged factors.
    """
    normalized = normalize_upside(energy_upside, cfg)
    final_score = compute_final_score(
        normalized, data_confidence.score, actionability.score
    )
    return Scorecard(
        system_id=system_id,
        energy_upside=energy_upside,
        normalized_energy_upside=normalized,
        data_confidence=data_confidence,
        actionability=actionability,
        final_score=final_score,
        recommended_tilt=recommended_tilt,
        recommended_azimuth=recommended_azimuth,
        recommendation=recommend(final_score, cfg),
        flags=flags,
    )
