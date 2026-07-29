"""Deterministic cleaning + anomaly detection for raw PV records.

Turns raw CSV rows (``dict`` of strings) into :class:`~pv_agent.schemas.SiteRecord`
objects. Two responsibilities, both purely deterministic:

1. **Clean** — null out sentinels, restore stripped zip leading zeros, map the
   three-state categoricals, derive age.
2. **Detect** — populate ``anomalies`` with *facts* about the record.

The anomalies are deliberately **facts, not judgments**: this layer reports
"azimuth==0 and tilt==0" or "tracking unknown"; it never decides what that
*means* for trust or actionability. That weighing is the agent's job (see
DECISIONS.md — "Code detects, the agent weighs").

lat/lon are not present in the source CSV (Tracking the Sun), so they are left
as ``None`` here; geocoding is a later, separate step in the pipeline.
"""

from __future__ import annotations

from pv_agent.schemas import Mounting, Ownership, SiteRecord, Tracking

from datetime import datetime
REFERENCE_YEAR = datetime.now().year

# Values used as "missing" markers in the source. The data uses -1; the brief
# mentions -9999. Both are treated as missing.
_SENTINELS = {-1.0, -9999.0}

# Raw CSV column names (note the exact casing/suffixes in the source file).
_COL_SYSTEM_ID = "system_ID"
_COL_STATE = "state"
_COL_ZIP = "zip_code"
_COL_SIZE_DC = "system_size_DC"
_COL_AZIMUTH = "azimuth_1"
_COL_TILT = "tilt_1"
_COL_MODULE_QTY = "module_quantity_1"
_COL_TRACKING = "tracking"
_COL_INSTALL_DATE = "installation_date"
_COL_THIRD_PARTY = "third_party_owned"
_COL_GROUND_MOUNTED = "ground_mounted"


# --------------------------------------------------------------------------- #
# Small, individually testable parsing helpers
# --------------------------------------------------------------------------- #
def _is_sentinel(value: object) -> bool:
    """True if ``value`` is a missing-marker (-1 / -9999, int/float/str forms)."""
    if value is None:
        return False
    try:
        return float(value) in _SENTINELS
    except (TypeError, ValueError):
        return False


def to_float(value: object) -> float | None:
    """Parse ``value`` to float; return None if missing/sentinel/unparseable."""
    if value is None or _is_sentinel(value):
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def parse_zip(value: object) -> str:
    """Restore stripped leading zeros to a 5-digit zip.

    Returns ``""`` for missing/sentinel/empty input. Otherwise takes the integer
    part and left-pads to 5 digits (``"5647"`` -> ``"05647"``). A result that is
    not exactly 5 digits is returned as-is and detected as ``invalid_zip`` later.
    """
    if value is None or _is_sentinel(value):
        return ""
    text = str(value).strip()
    if text == "":
        return ""
    return text.split(".")[0].zfill(5)


def _is_valid_zip(zip_code: str) -> bool:
    """A normalized zip is valid iff it is exactly five digits."""
    return len(zip_code) == 5 and zip_code.isdigit()


def parse_year(value: object) -> int | None:
    """Extract the four-digit year from an installation date, else None."""
    if value is None or _is_sentinel(value):
        return None
    text = str(value).strip()
    if len(text) < 4 or not text[:4].isdigit():
        return None
    return int(text[:4])


def _age_years(value: object) -> int | None:
    """Derive age in years from an installation date, relative to REFERENCE_YEAR."""
    year = parse_year(value)
    if year is None:
        return None
    return REFERENCE_YEAR - year


def _map_category(value: object, zero: str, one: str, unknown: str = "Unknown") -> str:
    """Map a 0.0/1.0 flag to its two named states; sentinel/other -> unknown."""
    parsed = to_float(value)
    if parsed == 0.0:
        return zero
    if parsed == 1.0:
        return one
    return unknown


def _clean_range(value: float | None, low: float, high: float) -> float | None:
    """Keep ``value`` only if it falls in [low, high); else treat as missing."""
    if value is None:
        return None
    return value if low <= value < high else None


# --------------------------------------------------------------------------- #
# Anomaly detection — facts only, never judgments
# --------------------------------------------------------------------------- #
def detect_anomalies(record: SiteRecord) -> list[str]:
    """Return detected factual anomalies for a cleaned record.

    Every entry is an observable fact about the data (an orientation default, an
    unknown flag, a physical contradiction) — not an interpretation of whether
    the record is trustworthy or worth acting on.
    """
    anomalies: list[str] = []

    if record.azimuth == 0 and record.tilt == 0:
        anomalies.append("orientation_default_0_0")

    # East-facing: azimuth roughly due-east. Flagged (possibly intentional
    # morning-load), not penalized — the agent decides.
    if record.azimuth is not None and 70 <= record.azimuth <= 110:
        anomalies.append("east_facing")

    if record.tracking == "Unknown":
        anomalies.append("tracking_unknown")

    if record.third_party == "ThirdParty":
        anomalies.append("third_party_owned")

    if record.age_years is not None and record.age_years >= 20:
        anomalies.append("old_install")

    # Size/module contradiction: implied per-module capacity is physically
    # impossible. Real PV modules are well under ~1 kW each; an implied size of
    # >5 kW per module means the size and count cannot both be right.
    if (
        record.system_size_dc is not None
        and record.module_quantity is not None
        and record.module_quantity > 0
        and record.system_size_dc / record.module_quantity > 5.0
    ):
        anomalies.append("size_module_contradiction")

    if not _is_valid_zip(record.zip):
        anomalies.append("invalid_zip")

    return anomalies


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def clean_record(raw: dict) -> SiteRecord:
    """Clean one raw CSV row into a :class:`SiteRecord` with detected anomalies."""
    tracking: Tracking = _map_category(raw.get(_COL_TRACKING), "Fixed", "Tracker")  # type: ignore[assignment]
    third_party: Ownership = _map_category(  # type: ignore[assignment]
        raw.get(_COL_THIRD_PARTY), "Owner", "ThirdParty"
    )
    ground_mounted: Mounting = _map_category(  # type: ignore[assignment]
        raw.get(_COL_GROUND_MOUNTED), "Roof", "Ground"
    )

    record = SiteRecord(
        system_id=str(raw.get(_COL_SYSTEM_ID, "")).strip(),
        state=str(raw.get(_COL_STATE, "")).strip(),
        zip=parse_zip(raw.get(_COL_ZIP)),
        lat=None,  # not present in source CSV; geocoded later
        lon=None,
        system_size_dc=to_float(raw.get(_COL_SIZE_DC)),
        azimuth=_clean_range(to_float(raw.get(_COL_AZIMUTH)), 0.0, 360.0),
        tilt=_clean_range(to_float(raw.get(_COL_TILT)), 0.0, 90.0 + 1e-9),
        module_quantity=to_float(raw.get(_COL_MODULE_QTY)),
        tracking=tracking,
        third_party=third_party,
        ground_mounted=ground_mounted,
        age_years=_age_years(raw.get(_COL_INSTALL_DATE)),
    )
    record.anomalies = detect_anomalies(record)
    return record


def clean_dataset(rows: list[dict]) -> list[SiteRecord]:
    """Clean a list of raw CSV rows into cleaned :class:`SiteRecord` objects."""
    return [clean_record(row) for row in rows]
