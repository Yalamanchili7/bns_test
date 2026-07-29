"""Deterministic-correctness tests for the physics layer.

Solar geometry has known ground truths, so these tests assert *physical
properties* (the optimizer rediscovers the textbook optimum, a well-oriented
panel has little to gain, the optimum dominates any input) rather than
hardcoded POA numbers. That is what makes them meaningful: they would catch a
grid search that returned constants or a transposition wired up backwards.

A mid-latitude northern-hemisphere site is used throughout:
    lat = 39.0, lon = -105.0  (Colorado Front Range)
"""

from __future__ import annotations

import pytest

from pv_agent import physics

_LAT = 39.0
_LON = -105.0

# Clear-sky at extreme orientations can emit RuntimeWarnings; they are not
# failures and would only clutter test output.
pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")


def test_optimizer_rediscovers_textbook_optimum():
    """N-hemisphere optimum is due south, tilt ~ latitude — physics, not constants."""
    best_tilt, best_azimuth, _ = physics.optimize_orientation(_LAT, _LON)
    assert best_azimuth == 180.0  # due south
    assert abs(best_tilt - _LAT) < 15.0  # optimal tilt tracks latitude


def test_well_oriented_panel_has_near_zero_upside():
    """A panel already near the optimum (south, tilt≈lat) has little to gain."""
    result = physics.simulate_site(_LAT, _LON, current_tilt=35.0, current_azimuth=180.0)
    assert result.relative_energy_upside < 0.05


def test_north_flat_panel_has_large_upside():
    """The 0/0-default trap: a north-flat array shows a large raw upside."""
    result = physics.simulate_site(_LAT, _LON, current_tilt=0.0, current_azimuth=0.0)
    assert result.relative_energy_upside > 0.10


@pytest.mark.parametrize(
    "tilt,azimuth",
    [(10.0, 120.0), (50.0, 220.0), (0.0, 0.0), (35.0, 180.0)],
)
def test_optimal_poa_never_below_current(tilt, azimuth):
    """The optimized orientation is by definition at least as good as any input."""
    result = physics.simulate_site(_LAT, _LON, current_tilt=tilt, current_azimuth=azimuth)
    assert result.optimal_annual_poa >= result.current_annual_poa


def test_divide_by_zero_guard(monkeypatch):
    """A degenerate orientation with ~zero current POA yields 0.0 upside, not a crash.

    Hard to trigger naturally (clear-sky always leaves some diffuse on the
    plane), so the guard is exercised directly by forcing POA to zero.
    """
    monkeypatch.setattr(physics, "annual_poa", lambda *args, **kwargs: 0.0)
    result = physics.simulate_site(_LAT, _LON, current_tilt=90.0, current_azimuth=0.0)
    assert result.relative_energy_upside == 0.0


def test_location_cache_reused_across_calls():
    """Solar position + clear-sky are computed once per location, then reused.

    Two simulate_site calls at the same coordinates must leave exactly one cache
    entry — proof the 247 grid orientations do not each recompute solar geometry.
    """
    physics._LOCATION_CACHE.clear()
    physics.simulate_site(_LAT, _LON, current_tilt=20.0, current_azimuth=160.0)
    physics.simulate_site(_LAT, _LON, current_tilt=40.0, current_azimuth=200.0)
    assert len(physics._LOCATION_CACHE) == 1
