"""Offline zip-code geocoding (zip -> lat/lon) via pgeocode.

This is the seam between the cleaned records (which carry a zip but no
coordinates) and the physics (which needs lat/lon). Geocoding is offline — no
network call — so it is reproducible and fast.

Per the cost-tiering design, geocoding is *deferred*: coordinates are only
needed for records that reach simulation, so in the pipeline this runs over the
small shortlisted set, not all 10k. The functions here are location-agnostic,
though — `geocode_records` will geocode whatever list it is handed — so the
caller decides the scope.
"""

from __future__ import annotations

import pandas as pd
import pgeocode

from pv_agent.schemas import SiteRecord

# Instantiating Nominatim loads a country data file, so do it exactly once at
# import and reuse it for every lookup.
_NOMINATIM = pgeocode.Nominatim("us")

# Per-zip result cache: many records share a zip, so cache the (lat, lon) (or
# None) outcome keyed by the 5-digit zip string.
_ZIP_CACHE: dict[str, tuple[float, float] | None] = {}


def geocode_zip(zip_code: str) -> tuple[float, float] | None:
    """Look up a 5-digit US zip, returning ``(lat, lon)`` or ``None``.

    Returns ``None`` when the zip is not five digits, is unknown to pgeocode, or
    resolves to NaN coordinates (pgeocode's "not found" signal). Results are
    cached per zip.
    """
    if zip_code is None or len(zip_code) != 5 or not zip_code.isdigit():
        return None

    if zip_code in _ZIP_CACHE:
        return _ZIP_CACHE[zip_code]

    row = _NOMINATIM.query_postal_code(zip_code)
    lat, lon = row["latitude"], row["longitude"]

    # pgeocode returns NaN lat/lon for unknown zips — treat that as not found.
    # pd.isna handles None, NaN, and non-float types without raising.
    result: tuple[float, float] | None
    if pd.isna(lat) or pd.isna(lon):
        result = None
    else:
        result = (float(lat), float(lon))

    _ZIP_CACHE[zip_code] = result
    return result


def geocode_records(records: list[SiteRecord]) -> list[SiteRecord]:
    """Populate ``lat``/``lon`` on each record from its zip, in place.

    Records that geocode successfully get their coordinates set. Records whose
    zip cannot be geocoded keep ``lat``/``lon`` as ``None`` (they are filtered
    out before simulation) and receive a ``geocode_failed`` anomaly so the
    failure is visible downstream. The same list is returned for convenience.
    """
    for record in records:
        coords = geocode_zip(record.zip)
        if coords is None:
            # A malformed zip is already explained by invalid_zip; reserve
            # geocode_failed for well-formed zips pgeocode couldn't resolve.
            if (
                "invalid_zip" not in record.anomalies
                and "geocode_failed" not in record.anomalies
            ):
                record.anomalies.append("geocode_failed")
            continue
        record.lat, record.lon = coords
    return records
