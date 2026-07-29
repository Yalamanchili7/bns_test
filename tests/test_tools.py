"""Tests for the tool layer — the trust boundary.

The ≤50-row bound is explicitly required by the brief, so it is tested
thoroughly here: the shared guard, and every row-returning tool that relies on
it. The dataset is loaded once from the real CSV (fast enough) and shared across
tests via a module-scoped fixture.
"""

from __future__ import annotations

import pytest

from pv_agent import geo
from pv_agent.schemas import SimulationResult, SiteRecord
from pv_agent.tools import Dataset, MAX_ROWS, enforce_row_limit


@pytest.fixture(scope="module")
def dataset() -> Dataset:
    """The real dataset, cleaned once and reused across the module's tests."""
    return Dataset()


@pytest.fixture(scope="module")
def good_site_id(dataset: Dataset) -> str:
    """An eligible site whose zip actually geocodes (so it can be simulated)."""
    for record in dataset.eligible:
        if geo.geocode_zip(record.zip) is not None:
            return record.system_id
    pytest.skip("no geocodable eligible site in dataset")


def _missing_orientation_id(dataset: Dataset) -> str:
    """A system_id for a record with no azimuth/tilt (there are thousands)."""
    for record in dataset.records:
        if record.azimuth is None or record.tilt is None:
            return record.system_id
    pytest.skip("no record missing orientation")


# --------------------------------------------------------------------------- #
# The ≤50-row bound  (brief-required — thorough)
# --------------------------------------------------------------------------- #
def test_enforce_row_limit_passes_at_and_below_50():
    """Exactly 50 rows (and fewer) is allowed."""
    enforce_row_limit(list(range(50)))
    enforce_row_limit(list(range(10)))
    enforce_row_limit([])


def test_enforce_row_limit_raises_at_51():
    """51 rows exceeds the limit and raises."""
    with pytest.raises(ValueError):
        enforce_row_limit(list(range(51)))


def test_shortlist_51_raises_not_capped(dataset: Dataset):
    """The bound rejects an over-limit request; it does not silently cap."""
    with pytest.raises(ValueError):
        dataset.shortlist_candidates(51)


@pytest.mark.parametrize("limit", [1, 10, 30, 50])
def test_shortlist_within_bound_returns_normally(dataset: Dataset, limit: int):
    """A limit at or below the bound returns at most `limit` rows.

    Uses include_suspicious=0 to isolate the limit->count relationship: with a
    non-zero suspicious quota the result can hold up to `include_suspicious`
    extra rows when `limit` is smaller than that quota.
    """
    result = dataset.shortlist_candidates(limit, include_suspicious=0)
    assert len(result) <= limit


def test_get_site_details_over_50_ids_raises(dataset: Dataset):
    """More than 50 requested ids trips the input bound."""
    ids = [r.system_id for r in dataset.records[:51]]
    with pytest.raises(ValueError):
        dataset.get_site_details(ids)


def test_get_site_details_never_exceeds_50(dataset: Dataset):
    """A max-size valid request returns no more than 50 records."""
    ids = [r.system_id for r in dataset.records[:50]]
    result = dataset.get_site_details(ids)
    assert len(result) <= MAX_ROWS


# --------------------------------------------------------------------------- #
# Tool correctness
# --------------------------------------------------------------------------- #
def test_summary_is_aggregates_only(dataset: Dataset):
    """The summary carries counts/aggregates and NO raw SiteRecords."""
    summary = dataset.get_dataset_summary()
    for key in (
        "total_records",
        "eligible_records",
        "missing",
        "trackers",
        "tracking_unknown",
        "third_party_owned",
        "orientation_default_0_0",
        "azimuth_distribution",
    ):
        assert key in summary
    assert summary["total_records"] == len(dataset.records)
    # No raw record leaks through any value.
    assert not _contains_site_record(summary)


def _contains_site_record(obj) -> bool:
    """Recursively check a JSON-ish structure for any SiteRecord instance."""
    if isinstance(obj, SiteRecord):
        return True
    if isinstance(obj, dict):
        return any(_contains_site_record(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return any(_contains_site_record(v) for v in obj)
    return False


def test_shortlist_shape(dataset: Dataset):
    """Shortlist returns light dicts with the expected keys (incl. the bucket flag)."""
    result = dataset.shortlist_candidates(30)
    assert len(result) <= 30
    expected_keys = {
        "system_id", "state", "azimuth", "tilt", "misalignment_score",
        "anomalies", "is_suspicious_default",
    }
    for entry in result:
        assert set(entry.keys()) == expected_keys
        assert not isinstance(entry, SiteRecord)


def test_shortlist_plausible_bucket_sorted_descending(dataset: Dataset):
    """Within the plausible bucket, entries are sorted by misalignment descending."""
    result = dataset.shortlist_candidates(limit=15, include_suspicious=3)
    plausible_scores = [
        e["misalignment_score"] for e in result if not e["is_suspicious_default"]
    ]
    assert plausible_scores == sorted(plausible_scores, reverse=True)


def test_shortlist_bucket_split(dataset: Dataset):
    """limit=15, include_suspicious=3 -> 15 total with exactly 3 suspicious."""
    result = dataset.shortlist_candidates(limit=15, include_suspicious=3)
    assert len(result) == 15
    suspicious = [e for e in result if e["is_suspicious_default"]]
    assert len(suspicious) == 3


def test_shortlist_plausible_never_carry_default_anomaly(dataset: Dataset):
    """Plausible entries must not carry the orientation_default_0_0 anomaly."""
    result = dataset.shortlist_candidates(limit=15, include_suspicious=3)
    for entry in result:
        if not entry["is_suspicious_default"]:
            assert "orientation_default_0_0" not in entry["anomalies"]


def test_shortlist_suspicious_all_carry_default_anomaly(dataset: Dataset):
    """Suspicious entries must all carry the orientation_default_0_0 anomaly."""
    result = dataset.shortlist_candidates(limit=15, include_suspicious=3)
    for entry in result:
        if entry["is_suspicious_default"]:
            assert "orientation_default_0_0" in entry["anomalies"]


def test_shortlist_include_suspicious_zero(dataset: Dataset):
    """include_suspicious=0 yields no suspicious entries."""
    result = dataset.shortlist_candidates(limit=15, include_suspicious=0)
    assert all(not e["is_suspicious_default"] for e in result)


def test_shortlist_bound_trips_when_total_over_50(dataset: Dataset):
    """The combined result still trips the ≤50 bound when limit exceeds it."""
    with pytest.raises(ValueError):
        dataset.shortlist_candidates(limit=51)


def test_get_site_details_returns_matching_record(dataset: Dataset):
    """A known id returns its cleaned SiteRecord."""
    result = dataset.get_site_details(["SITE_00001"])
    assert len(result) == 1
    assert isinstance(result[0], SiteRecord)
    assert result[0].system_id == "SITE_00001"


def test_get_site_details_skips_unknown_ids(dataset: Dataset):
    """Unknown ids are skipped, not raised; a mixed list returns only the valid."""
    result = dataset.get_site_details(["SITE_00001", "NOPE_99999"])
    ids = [r.system_id for r in result]
    assert ids == ["SITE_00001"]


def test_simulate_good_site_returns_result(dataset: Dataset, good_site_id: str):
    """A geocodable eligible site simulates; optimal POA dominates current."""
    result = dataset.simulate_site_tool(good_site_id)
    assert isinstance(result, SimulationResult)
    assert result.optimal_annual_poa >= result.current_annual_poa


def test_simulate_unknown_id_raises_keyerror(dataset: Dataset):
    """An unknown system_id is a KeyError."""
    with pytest.raises(KeyError):
        dataset.simulate_site_tool("NOPE_99999")


def test_simulate_missing_orientation_raises_valueerror(dataset: Dataset):
    """A record with no azimuth/tilt cannot be simulated — ValueError."""
    site_id = _missing_orientation_id(dataset)
    with pytest.raises(ValueError):
        dataset.simulate_site_tool(site_id)


# --------------------------------------------------------------------------- #
# Eligibility
# --------------------------------------------------------------------------- #
def _record(**overrides) -> SiteRecord:
    base = dict(
        system_id="X", state="TX", zip="75150",
        azimuth=180.0, tilt=30.0, tracking="Fixed",
    )
    return SiteRecord(**{**base, **overrides})


def test_eligible_true_for_valid_fixed_site():
    assert Dataset._is_eligible(_record()) is True


def test_eligible_false_for_tracker():
    assert Dataset._is_eligible(_record(tracking="Tracker")) is False


@pytest.mark.parametrize(
    "overrides",
    [{"azimuth": None}, {"tilt": None}, {"zip": ""}, {"zip": "abcde"}],
)
def test_eligible_false_when_field_missing(overrides):
    assert Dataset._is_eligible(_record(**overrides)) is False
