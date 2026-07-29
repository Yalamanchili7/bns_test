"""The typed, bounded tool interface — the trust boundary.

The agent reaches the data **only** through the four tools on :class:`Dataset`,
and no row-returning tool ever hands back more than :data:`MAX_ROWS` records.
That single constraint is what keeps the full 10k-row CSV out of the LLM
context: there is no bulk accessor, the cheap pre-filter runs in code, and the
expensive per-site physics is single-site by design.

Layer responsibilities (see PLAN.md / DECISIONS.md):
- Cleaning, eligibility, the misalignment heuristic, geocoding, and physics are
  all deterministic and live below this boundary.
- The tools only *expose* those results, typed and row-bounded. They contain no
  judgment — that is the agent's job.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

from pv_agent import geo, physics
from pv_agent.cleaning import clean_dataset
from pv_agent.schemas import SimulationResult, SiteRecord

# tools.py is at src/pv_agent/tools.py, so the repo root is two parents up.
_FILE_RELATIVE_CSV = Path(__file__).resolve().parents[2] / "data" / "pv_sites_sample.csv"
# Relative to the current working directory (e.g. the container's WORKDIR).
_CWD_CSV = Path("data") / "pv_sites_sample.csv"


def _resolve_default_csv() -> Path:
    """Locate the dataset CSV, trying (in order) env var, repo path, then CWD.

    - ``PV_AGENT_CSV`` if set (explicit override; used by the Docker image).
    - The path relative to this file (works for editable installs / repo runs).
    - ``./data/pv_sites_sample.csv`` relative to the working directory (works for
      a non-editable install where the package lives in site-packages).

    Raises a clear :class:`FileNotFoundError` listing every path tried if none
    resolve, so a misconfigured run fails obviously rather than cryptically.
    """
    tried: list[Path] = []

    env_path = os.environ.get("PV_AGENT_CSV")
    if env_path:
        candidate = Path(env_path)
        if candidate.exists():
            return candidate
        tried.append(candidate)

    for candidate in (_FILE_RELATIVE_CSV, _CWD_CSV):
        if candidate.exists():
            return candidate
        tried.append(candidate)

    raise FileNotFoundError(
        "Could not locate the dataset CSV. Set PV_AGENT_CSV or place it at one "
        "of these paths:\n  " + "\n  ".join(str(p) for p in tried)
    )

# The row bound that defines the trust boundary. Enforced by every tool that
# returns records.
MAX_ROWS = 50

# Approximate state-centroid latitudes for the cheap pre-filter only. This is a
# coarse stand-in so the misalignment heuristic needs no geocoding over 10k
# rows; precise lat/lon is geocoded later, per site, at simulation time. Default
# ~39 (roughly the middle of the continental US) for anything unlisted.
_STATE_LATITUDE: dict[str, float] = {
    "CA": 37.0, "IL": 40.0, "VT": 44.0, "MA": 42.3, "NH": 43.7, "CO": 39.0,
    "NM": 34.5, "MN": 46.3, "OH": 40.4, "DC": 38.9, "CT": 41.6, "TX": 31.5,
    "PA": 40.9, "OR": 44.0, "WI": 44.5, "AZ": 34.2, "ME": 45.4, "FL": 28.6,
    "MD": 39.0, "NY": 42.9, "DE": 39.0, "WA": 47.4, "RI": 41.7, "NJ": 40.2,
    "UT": 39.3, "VA": 37.5, "AR": 34.8,
}
_DEFAULT_LATITUDE = 39.0


def enforce_row_limit(rows: list, limit: int = MAX_ROWS) -> None:
    """Raise if ``rows`` exceeds ``limit`` — the shared trust-boundary guard."""
    if len(rows) > limit:
        raise ValueError(
            f"Tool returned {len(rows)} rows, exceeds limit of {limit}"
        )


def _state_latitude(state: str) -> float:
    """Coarse centroid latitude for a state (pre-filter heuristic only)."""
    return _STATE_LATITUDE.get(state.upper(), _DEFAULT_LATITUDE)


def _misalignment(record: SiteRecord) -> float:
    """Cheap monotone proxy for re-orientation upside (no physics, no geocoding).

    ``|azimuth - 180|/180 + |tilt - latitude_est|/90`` — larger means further
    from the south-at-latitude ideal. Used only to shortlist; the real upside
    comes from the physics simulation on the narrowed set.
    """
    lat_est = _state_latitude(record.state)
    return abs(record.azimuth - 180.0) / 180.0 + abs(record.tilt - lat_est) / 90.0


class Dataset:
    """Loads and cleans the CSV once, then serves the agent through four tools.

    The cleaned records and the precomputed eligible pool are held in memory;
    the agent never receives the underlying dataframe or CSV.
    """

    def __init__(self, csv_path: str | Path | None = None) -> None:
        if csv_path is None:
            csv_path = _resolve_default_csv()
        with open(csv_path, newline="") as fh:
            raw_rows = list(csv.DictReader(fh))
        self.records: list[SiteRecord] = clean_dataset(raw_rows)
        self._by_id: dict[str, SiteRecord] = {r.system_id: r for r in self.records}

        # Eligible pool: simulatable, re-orientable sites — valid azimuth AND
        # tilt AND zip, and not a tracker (trackers cannot be re-oriented).
        self.eligible: list[SiteRecord] = [
            r for r in self.records if self._is_eligible(r)
        ]

    @staticmethod
    def _is_eligible(record: SiteRecord) -> bool:
        """A record is eligible iff it can be simulated and re-oriented."""
        return (
            record.azimuth is not None
            and record.tilt is not None
            and len(record.zip) == 5
            and record.zip.isdigit()
            and record.tracking != "Tracker"
        )

    # ----------------------------------------------------------------- tools #
    def get_dataset_summary(self) -> dict:
        """Aggregate data-shape stats only — **no raw records**.

        Lets the agent understand scale and data quality (counts, missingness,
        trap prevalence, azimuth distribution) before it fetches anything.
        """
        total = len(self.records)
        missing_azimuth = sum(1 for r in self.records if r.azimuth is None)
        missing_tilt = sum(1 for r in self.records if r.tilt is None)
        missing_zip = sum(
            1 for r in self.records if not (len(r.zip) == 5 and r.zip.isdigit())
        )

        # Compass buckets for present azimuths (N includes the 315-360/0-45 wrap).
        azimuth_buckets = {"N": 0, "E": 0, "S": 0, "W": 0}
        for r in self.records:
            if r.azimuth is None:
                continue
            az = r.azimuth % 360.0
            if az < 45 or az >= 315:
                azimuth_buckets["N"] += 1
            elif az < 135:
                azimuth_buckets["E"] += 1
            elif az < 225:
                azimuth_buckets["S"] += 1
            else:
                azimuth_buckets["W"] += 1

        return {
            "total_records": total,
            "eligible_records": len(self.eligible),
            "missing": {
                "azimuth": missing_azimuth,
                "tilt": missing_tilt,
                "zip": missing_zip,
            },
            "trackers": sum(1 for r in self.records if r.tracking == "Tracker"),
            "tracking_unknown": sum(
                1 for r in self.records if r.tracking == "Unknown"
            ),
            "third_party_owned": sum(
                1 for r in self.records if r.third_party == "ThirdParty"
            ),
            "orientation_default_0_0": sum(
                1 for r in self.records if "orientation_default_0_0" in r.anomalies
            ),
            "azimuth_distribution": azimuth_buckets,
        }

    def shortlist_candidates(
        self, limit: int = 30, include_suspicious: int = 3
    ) -> list[dict]:
        """Top eligible sites by misalignment, split into two buckets.

        A 0/0-default record (``orientation_default_0_0``) has azimuth=0 AND
        tilt=0, which is almost certainly a data-entry default, not a real
        north-flat array. Such a record has *no trustworthy orientation*, so it
        cannot be meaningfully ranked by orientation-misalignment — by raw
        misalignment it dominates the list (391 of them do), flooding out the
        genuine re-orientation opportunities. So we separate them:

        - **Plausible candidates**: eligible records WITHOUT the 0/0-default
          anomaly, ranked by misalignment descending — the real opportunities.
          We take the top ``limit - include_suspicious``.
        - **Suspicious defaults**: eligible records WITH the 0/0-default anomaly.
          We include up to ``include_suspicious`` of them so the system still
          surfaces the required "high raw upside pushed down" trap example
          without letting it swamp the shortlist.

        Each returned dict carries ``is_suspicious_default`` so downstream (and
        the agent) knows which bucket a candidate came from. The combined result
        is bounded by ``MAX_ROWS``.
        """
        plausible = sorted(
            (r for r in self.eligible if "orientation_default_0_0" not in r.anomalies),
            key=_misalignment,
            reverse=True,
        )
        suspicious = sorted(
            (r for r in self.eligible if "orientation_default_0_0" in r.anomalies),
            key=_misalignment,
            reverse=True,
        )

        n_plausible = max(limit - include_suspicious, 0)
        selected = [(r, False) for r in plausible[:n_plausible]]
        selected += [(r, True) for r in suspicious[:include_suspicious]]

        enforce_row_limit(selected)
        return [
            {
                "system_id": r.system_id,
                "state": r.state,
                "azimuth": r.azimuth,
                "tilt": r.tilt,
                "misalignment_score": _misalignment(r),
                "anomalies": list(r.anomalies),
                "is_suspicious_default": is_suspicious,
            }
            for r, is_suspicious in selected
        ]

    def get_site_details(self, system_ids: list[str]) -> list[SiteRecord]:
        """Full cleaned records + anomalies for specific sites (≤50).

        The evidence feed for agent judgment. Unknown ids are skipped rather
        than raising, so a partially-valid request still returns what it can.
        """
        enforce_row_limit(system_ids)
        details = [
            self._by_id[sid] for sid in system_ids if sid in self._by_id
        ]
        enforce_row_limit(details)
        return details

    def simulate_site_tool(self, system_id: str) -> SimulationResult:
        """Geocode one site, then run the physics simulation. Single-site by design.

        Raises ``KeyError`` for an unknown id, and ``ValueError`` if the site
        lacks orientation or cannot be geocoded — a clear failure the agent can
        act on, rather than a silently wrong result.
        """
        record = self._by_id.get(system_id)
        if record is None:
            raise KeyError(f"Unknown system_id: {system_id!r}")
        if record.azimuth is None or record.tilt is None:
            raise ValueError(
                f"Site {system_id!r} has no azimuth/tilt to simulate"
            )

        coords = geo.geocode_zip(record.zip)
        if coords is None:
            raise ValueError(
                f"Site {system_id!r} zip {record.zip!r} could not be geocoded"
            )

        lat, lon = coords
        return physics.simulate_site(lat, lon, record.tilt, record.azimuth)
