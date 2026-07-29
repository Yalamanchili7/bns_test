"""Deterministic solar physics and orientation optimization.

The physics heart of the pipeline. No agent, no scoring — just plane-of-array
(POA) irradiance from a clear-sky model and a grid search for the best
orientation. Everything here has a correct answer, so it is plain, inspectable
code.

Method (see PLAN.md / DECISIONS.md):
- **Representative sampling**: one representative day per month (12 days),
  hourly, in a fixed reference year. This is a proxy adequate for *ranking*
  relative upside, not for quantifying real kWh.
- **Solar position** is computed (never read from data).
- **Clear-sky** irradiance (pvlib's default Ineichen) is a weather-free proxy.
- **Transposition** uses the Hay-Davies model (accounts for circumsolar
  diffuse, fully specified by the clear-sky outputs).

Timezone is simplified: timestamps are naive (treated as UTC). This is
acceptable here because we compare a site against *itself* at a different
orientation and rank the relative upside — a constant tz offset shifts every
site's POA identically and cannot change the ranking. It would matter if we
were billing energy; we are not.
"""

from __future__ import annotations

import pandas as pd
import pvlib

from pv_agent.schemas import SimulationResult

# Fixed reference year so the representative timestamps are reproducible.
_REFERENCE_YEAR = 2021

# Orientation grid. Bounds are physics-informed, not arbitrary:
#   - Azimuth 90-270: in the northern hemisphere the optimum always lies in the
#     southern half; north-facing orientations are never optimal.
#   - Tilt 0-60: optimal tilt tracks latitude and never needs to exceed ~60° in
#     the continental US. Pruning the rest of the space is free correctness.
TILT_GRID = range(0, 61, 5)  # 0,5,...,60  -> 13 tilts
AZIMUTH_GRID = range(90, 271, 10)  # 90,100,...,270 -> 19 azimuths  (13*19 = 247)


def _build_representative_times() -> pd.DatetimeIndex:
    """One representative day (the 15th) per month, hourly, in the reference year.

    Naive timestamps (no tz) — see the module docstring for why that is fine for
    a relative-upside ranking.
    """
    parts = [
        pd.date_range(
            start=f"{_REFERENCE_YEAR}-{month:02d}-15", periods=24, freq="1h"
        )
        for month in range(1, 13)
    ]
    return parts[0].append(parts[1:])


# Computed once at import: the timestamp grid and extraterrestrial irradiance.
# Both depend only on time (not location or orientation), so they are constants.
_TIMES = _build_representative_times()
_DNI_EXTRA = pvlib.irradiance.get_extra_radiation(_TIMES)

# Per-location cache of (solar_position, clear_sky). Both depend only on
# (lat, lon) and time — never on orientation — so they are computed once per
# site and reused across all 247 grid orientations. Keyed by rounded coords so
# tiny float differences do not defeat the cache.
_LOCATION_CACHE: dict[tuple[float, float], tuple[pd.DataFrame, pd.DataFrame]] = {}


def _location_data(lat: float, lon: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return cached (solar_position, clear_sky) for a location, computing once."""
    key = (round(lat, 3), round(lon, 3))
    cached = _LOCATION_CACHE.get(key)
    if cached is None:
        solar_position = pvlib.solarposition.get_solarposition(_TIMES, lat, lon)
        clear_sky = pvlib.location.Location(lat, lon).get_clearsky(_TIMES)
        cached = (solar_position, clear_sky)
        _LOCATION_CACHE[key] = cached
    return cached


def annual_poa(lat: float, lon: float, tilt: float, azimuth: float) -> float:
    """Summed plane-of-array global irradiance over the representative timestamps.

    Uses cached solar position and clear-sky for the location (orientation
    independent) and transposes onto the given tilt/azimuth via Hay-Davies.
    The returned value is a relative quantity for ranking, not an annual kWh.
    """
    solar_position, clear_sky = _location_data(lat, lon)
    total = pvlib.irradiance.get_total_irradiance(
        surface_tilt=tilt,
        surface_azimuth=azimuth,
        solar_zenith=solar_position["apparent_zenith"],
        solar_azimuth=solar_position["azimuth"],
        dni=clear_sky["dni"],
        ghi=clear_sky["ghi"],
        dhi=clear_sky["dhi"],
        dni_extra=_DNI_EXTRA,
        model="haydavies",
    )
    return float(total["poa_global"].sum())


def optimize_orientation(lat: float, lon: float) -> tuple[float, float, float]:
    """Grid-search the best (tilt, azimuth) for a location.

    Returns ``(best_tilt, best_azimuth, best_poa)``. The objective is smooth,
    unimodal, 2-D and bounded, so an exhaustive grid over the physics-informed
    ranges (see TILT_GRID / AZIMUTH_GRID) is trivially correct and inspectable.
    """
    best_tilt = 0.0
    best_azimuth = 180.0
    best_poa = float("-inf")
    for tilt in TILT_GRID:
        for azimuth in AZIMUTH_GRID:
            poa = annual_poa(lat, lon, float(tilt), float(azimuth))
            if poa > best_poa:
                best_tilt, best_azimuth, best_poa = float(tilt), float(azimuth), poa
    return best_tilt, best_azimuth, best_poa


def simulate_site(
    lat: float, lon: float, current_tilt: float, current_azimuth: float
) -> SimulationResult:
    """Simulate a site: current POA, optimal POA, and relative upside.

    ``relative_energy_upside = (optimal - current) / current``, guarded against a
    zero current POA (a degenerate orientation with no sun on the plane) — in
    that case upside is reported as 0.0 rather than raising.
    """
    current_poa = annual_poa(lat, lon, current_tilt, current_azimuth)
    best_tilt, best_azimuth, best_poa = optimize_orientation(lat, lon)
    upside = (best_poa - current_poa) / current_poa if current_poa > 0 else 0.0
    return SimulationResult(
        current_annual_poa=current_poa,
        optimal_annual_poa=best_poa,
        relative_energy_upside=upside,
        recommended_tilt=best_tilt,
        recommended_azimuth=best_azimuth,
    )
