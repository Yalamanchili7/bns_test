"""Tests for the geocoding layer.

These test *our* logic — the tuple/None contract, NaN handling, caching, and the
record-annotation rules — not pgeocode's coordinate accuracy. We never assert
exact lat/lon (that would be testing pgeocode's data), only that resolved
coordinates are two non-NaN floats within plausible continental-US bounds.
"""

from __future__ import annotations

import math

import pandas as pd

from pv_agent import geo
from pv_agent.schemas import SiteRecord

# Generous continental-US bounding box — a sanity envelope, not a precise check.
_US_LAT = (20.0, 50.0)
_US_LON = (-130.0, -65.0)


def test_valid_zip_returns_plausible_coords():
    """A real zip resolves to two non-NaN floats inside the US bounding box."""
    result = geo.geocode_zip("85016")  # Phoenix, AZ
    assert result is not None
    lat, lon = result
    assert isinstance(lat, float) and isinstance(lon, float)
    assert not math.isnan(lat) and not math.isnan(lon)
    assert _US_LAT[0] <= lat <= _US_LAT[1]
    assert _US_LON[0] <= lon <= _US_LON[1]


def test_unresolvable_zip_returns_none():
    """A well-formed but unknown zip resolves to None."""
    assert geo.geocode_zip("00000") is None


def test_malformed_zip_returns_none():
    """Empty and non-numeric zips are rejected before any lookup."""
    assert geo.geocode_zip("") is None
    assert geo.geocode_zip("abcde") is None


def test_nan_coords_treated_as_not_found(monkeypatch):
    """A row with NaN lat/lon maps to None (deterministic, no pgeocode data)."""
    geo._ZIP_CACHE.pop("99999", None)  # ensure the patched lookup is actually hit
    monkeypatch.setattr(
        geo._NOMINATIM,
        "query_postal_code",
        lambda _zip: pd.Series({"latitude": float("nan"), "longitude": float("nan")}),
    )
    assert geo.geocode_zip("99999") is None


def test_cache_populated_and_reused():
    """Repeated lookups populate the cache; a resolved zip caches its tuple."""
    geo._ZIP_CACHE.clear()
    first = geo.geocode_zip("85016")
    assert "85016" in geo._ZIP_CACHE
    second = geo.geocode_zip("85016")
    assert first == second == geo._ZIP_CACHE["85016"]


def test_failed_lookup_caches_none():
    """A failed lookup is cached as None so it is not re-queried."""
    geo._ZIP_CACHE.clear()
    assert geo.geocode_zip("00000") is None
    assert "00000" in geo._ZIP_CACHE
    assert geo._ZIP_CACHE["00000"] is None


def test_geocode_records_sets_coords_and_flags_failure():
    """Good record gets coords; bad-zip record stays None and gets geocode_failed."""
    good = SiteRecord(system_id="good", state="AZ", zip="85016")
    bad = SiteRecord(system_id="bad", state="XX", zip="00000")
    geo.geocode_records([good, bad])

    assert good.lat is not None and good.lon is not None
    assert bad.lat is None and bad.lon is None
    assert "geocode_failed" in bad.anomalies


def test_invalid_zip_record_not_double_flagged():
    """A record already carrying invalid_zip is not also given geocode_failed."""
    record = SiteRecord(system_id="malformed", state="XX", zip="", anomalies=["invalid_zip"])
    geo.geocode_records([record])

    assert record.lat is None and record.lon is None
    assert "geocode_failed" not in record.anomalies
    assert record.anomalies == ["invalid_zip"]
