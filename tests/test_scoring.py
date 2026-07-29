"""Tests for the deterministic scoring arithmetic (brief-required).

These assert the exact arithmetic and banding the brief mandates live in code:
fixed-cap normalization, the multiplicative final score (where any weak factor
sinks the site), the recommendation thresholds, and the end-to-end scorecard —
including the ranking flip where an honest, modest site outranks a high-raw-
upside-but-untrustworthy one.
"""

from __future__ import annotations

import pytest

from pv_agent.schemas import JudgedScore, Recommendation, Scorecard, ScoringConfig
from pv_agent.scoring import (
    build_scorecard,
    compute_final_score,
    normalize_upside,
    recommend,
)

CFG = ScoringConfig()


# --------------------------------------------------------------------------- #
# normalize_upside — fixed-cap normalization
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "raw,expected",
    [
        (0.30, 1.0),   # exactly at the cap -> saturates to 1.0
        (0.15, 0.5),   # half the cap
        (0.45, 1.0),   # above the cap -> clamps, never exceeds 1.0
        (0.0, 0.0),    # no upside
        (-0.1, 0.0),   # negative -> clamped to 0.0
    ],
)
def test_normalize_upside(raw, expected):
    assert normalize_upside(raw, CFG) == pytest.approx(expected)


# --------------------------------------------------------------------------- #
# compute_final_score — multiplicative; any weak factor sinks it
# --------------------------------------------------------------------------- #
def test_final_score_all_ones():
    assert compute_final_score(1.0, 1.0, 1.0) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "norm,conf,act",
    [
        (0.0, 0.9, 0.9),  # no upside
        (0.9, 0.0, 0.9),  # no confidence
        (0.9, 0.9, 0.0),  # no actionability
    ],
)
def test_final_score_zero_factor_sinks_it(norm, conf, act):
    """Any single factor at zero drives the whole product to zero."""
    assert compute_final_score(norm, conf, act) == pytest.approx(0.0)


def test_final_score_known_case():
    assert compute_final_score(0.37, 0.85, 0.8) == pytest.approx(0.2516)


# --------------------------------------------------------------------------- #
# recommend — banding at the default thresholds
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "score,expected",
    [
        (0.6, Recommendation.PRIORITIZE),
        (0.5, Recommendation.PRIORITIZE),  # boundary, >=
        (0.3, Recommendation.REVIEW),
        (0.2, Recommendation.REVIEW),      # boundary, >=
        (0.1, Recommendation.SKIP),
    ],
)
def test_recommend_bands(score, expected):
    assert recommend(score, CFG) is expected


# --------------------------------------------------------------------------- #
# build_scorecard — the integration, incl. the ranking flip
# --------------------------------------------------------------------------- #
def _card(system_id, upside, conf, act):
    return build_scorecard(
        system_id=system_id,
        energy_upside=upside,
        data_confidence=JudgedScore(score=conf, reason="test"),
        actionability=JudgedScore(score=act, reason="test"),
        recommended_tilt=35.0,
        recommended_azimuth=180.0,
        flags=["verify orientation"],
        cfg=CFG,
    )


def test_trap_case_is_pushed_down():
    """High raw upside but low confidence -> normalized 1.0 yet Skip."""
    trap = _card("SITE_TRAP", upside=0.40, conf=0.15, act=0.5)
    assert trap.normalized_energy_upside == pytest.approx(1.0)
    assert trap.final_score == pytest.approx(0.075)
    assert trap.recommendation is Recommendation.SKIP


def test_honest_case_is_reviewable():
    """Modest upside but trustworthy and actionable -> Review."""
    honest = _card("SITE_HONEST", upside=0.11, conf=0.85, act=0.8)
    assert honest.final_score == pytest.approx(0.249, abs=1e-3)
    assert honest.recommendation is Recommendation.REVIEW


def test_honest_outranks_trap():
    """The key property: the honest site outranks the high-raw-upside trap."""
    trap = _card("SITE_TRAP", upside=0.40, conf=0.15, act=0.5)
    honest = _card("SITE_HONEST", upside=0.11, conf=0.85, act=0.8)
    assert honest.final_score > trap.final_score


def test_scorecard_is_valid_and_fully_populated():
    """build_scorecard returns a complete, valid Scorecard."""
    card = _card("SITE_HONEST", upside=0.11, conf=0.85, act=0.8)
    assert isinstance(card, Scorecard)
    assert card.system_id == "SITE_HONEST"
    assert card.energy_upside == pytest.approx(0.11)
    assert isinstance(card.data_confidence, JudgedScore)
    assert isinstance(card.actionability, JudgedScore)
    assert card.recommended_tilt == 35.0
    assert card.recommended_azimuth == 180.0
    assert card.flags == ["verify orientation"]
    assert isinstance(card.recommendation, Recommendation)
