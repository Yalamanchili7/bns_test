"""Tests for the deterministic cleaning + anomaly-detection layer.

Fixtures are small inline raw-row dicts (the shape a ``csv.DictReader`` yields),
never the real CSV, so tests are fast and deterministic. Age assertions are made
relative to ``REFERENCE_YEAR`` rather than a hardcoded number so they stay stable
over time.
"""

from __future__ import annotations

import pytest

from pv_agent.cleaning import REFERENCE_YEAR, clean_record

# A baseline well-formed raw row. Each test overrides only the field(s) under
# test so the fixture stays minimal and the intent stays obvious.
_BASELINE = {
    "system_ID": "SITE_TEST",
    "state": "TX",
    "zip_code": "75150",
    "system_size_DC": "5.0",
    "azimuth_1": "180.0",
    "tilt_1": "30.0",
    "module_quantity_1": "15.0",
    "tracking": "0.0",
    "installation_date": "2020-06-15",
    "third_party_owned": "0.0",
    "ground_mounted": "0.0",
}


def raw(**overrides) -> dict:
    """Build a raw row from the baseline with the given field overrides."""
    return {**_BASELINE, **overrides}


# --------------------------------------------------------------------------- #
# Sentinel handling
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("sentinel", [-1.0, -9999.0, "-1", "-9999", "-1.0", "-9999.0"])
def test_sentinel_becomes_none(sentinel):
    """All sentinel forms map numeric fields to None."""
    record = clean_record(
        raw(
            system_size_DC=sentinel,
            azimuth_1=sentinel,
            tilt_1=sentinel,
            module_quantity_1=sentinel,
        )
    )
    assert record.system_size_dc is None
    assert record.azimuth is None
    assert record.tilt is None
    assert record.module_quantity is None


def test_valid_numeric_passes_through():
    """A valid numeric value is parsed unchanged."""
    record = clean_record(raw(system_size_DC="12.5", azimuth_1="137.5"))
    assert record.system_size_dc == 12.5
    assert record.azimuth == 137.5


# --------------------------------------------------------------------------- #
# Zip normalization
# --------------------------------------------------------------------------- #
def test_zip_leading_zero_restored():
    """A stripped leading zero is restored to 5 digits."""
    assert clean_record(raw(zip_code="5647")).zip == "05647"


def test_zip_float_string_leading_zero_restored():
    """A float-string zip drops the decimal part and zero-pads."""
    assert clean_record(raw(zip_code="5647.0")).zip == "05647"


def test_zip_already_valid_unchanged():
    """An already-valid 5-digit zip passes through unchanged."""
    assert clean_record(raw(zip_code="75150")).zip == "75150"


@pytest.mark.parametrize("bad_zip", ["-1", ""])
def test_zip_missing_flagged_invalid(bad_zip):
    """A sentinel/empty zip becomes '' and is flagged invalid_zip."""
    record = clean_record(raw(zip_code=bad_zip))
    assert record.zip == ""
    assert "invalid_zip" in record.anomalies


# --------------------------------------------------------------------------- #
# Categorical mapping (three-state)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "value,expected", [("0.0", "Fixed"), ("1.0", "Tracker"), ("-1.0", "Unknown")]
)
def test_tracking_three_state(value, expected):
    assert clean_record(raw(tracking=value)).tracking == expected


@pytest.mark.parametrize(
    "value,expected", [("0.0", "Owner"), ("1.0", "ThirdParty"), ("-1.0", "Unknown")]
)
def test_third_party_three_state(value, expected):
    assert clean_record(raw(third_party_owned=value)).third_party == expected


@pytest.mark.parametrize(
    "value,expected", [("0.0", "Roof"), ("1.0", "Ground"), ("-1.0", "Unknown")]
)
def test_ground_mounted_three_state(value, expected):
    assert clean_record(raw(ground_mounted=value)).ground_mounted == expected


# --------------------------------------------------------------------------- #
# Age derivation
# --------------------------------------------------------------------------- #
def test_age_derived_from_reference_year():
    """age_years == REFERENCE_YEAR - install_year (asserted, not hardcoded)."""
    install_year = 2015
    record = clean_record(raw(installation_date=f"{install_year}-03-01"))
    assert record.age_years == REFERENCE_YEAR - install_year


# --------------------------------------------------------------------------- #
# Anomaly detection — one test each
# --------------------------------------------------------------------------- #
def test_orientation_default_detected():
    record = clean_record(raw(azimuth_1="0.0", tilt_1="0.0"))
    assert "orientation_default_0_0" in record.anomalies


def test_east_facing_detected():
    record = clean_record(raw(azimuth_1="90.0"))
    assert "east_facing" in record.anomalies


def test_well_oriented_has_no_orientation_anomalies():
    record = clean_record(raw(azimuth_1="180.0", tilt_1="30.0"))
    assert "orientation_default_0_0" not in record.anomalies
    assert "east_facing" not in record.anomalies


def test_tracking_unknown_detected():
    record = clean_record(raw(tracking="-1.0"))
    assert "tracking_unknown" in record.anomalies


def test_third_party_owned_detected():
    record = clean_record(raw(third_party_owned="1.0"))
    assert "third_party_owned" in record.anomalies


def test_size_module_contradiction_detected():
    """300 kW across 2 modules (~150 kW/module) is physically impossible."""
    record = clean_record(raw(system_size_DC="300.0", module_quantity_1="2.0"))
    assert "size_module_contradiction" in record.anomalies


def test_plausible_size_module_no_contradiction():
    """10 kW across 32 modules (~0.3 kW/module) is normal — no anomaly."""
    record = clean_record(raw(system_size_DC="10.0", module_quantity_1="32.0"))
    assert "size_module_contradiction" not in record.anomalies


def test_old_install_detected():
    """An install at least 20 years old is flagged old_install."""
    old_year = REFERENCE_YEAR - 20
    record = clean_record(raw(installation_date=f"{old_year}-01-01"))
    assert "old_install" in record.anomalies
