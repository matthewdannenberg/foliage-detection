"""
download_npn.py — Download USA-NPN fall foliage observations.

Queries the USA-NPN Status and Intensity API for observations of the two
relevant fall phenophases across all training/val/test years. By default
queries Vermont only; pass --states to expand to the full Northeast or
any other set of states.

    Phenophase 498 — "Colored leaves"
        intensity_value strings map to foliage stages:
            "Less than 5%"  → early
            "5-24%"         → early
            "25-49%"        → peak
            "50-74%"        → peak
            "75-94%"        → peak
            "95% or more"   → peak
        phenophase_status = 0 (No)  → no_transition

    Phenophase 499 — "Falling leaves"
        intensity "25-49%" or higher → late
        (lower intensities excluded — some leaf drop occurs even at peak)

Output is the RAW observation CSV saved to:
    data/raw/observer_reports/npn_{region}.csv

where {region} is derived from the states queried (e.g. "vermont" or
"northeast"). This file preserves the original stage as a string name
and includes source metadata in the notes field. Run
process_observations.py afterward to produce standardised output.

No API key is required. A request_source (your name) is required by the
NPN on an honor system basis — set it via --request-source or the
NPN_REQUEST_SOURCE environment variable.

Usage:
    python scripts/download_npn.py --request-source "Your Name"
    python scripts/download_npn.py --request-source "Your Name" --states ME NH VT MA NY
    python scripts/download_npn.py --request-source "Your Name" --years 2018 2019
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd
import requests

from config import OBSERVER_RAW, TRAIN_YEARS, VAL_YEARS, TEST_YEARS

# ---------------------------------------------------------------------------
# NPN API constants
# ---------------------------------------------------------------------------

NPN_BASE_URL = "https://services.usanpn.org/npn_portal/observations/getObservations.json"

# Phenophase IDs relevant to fall foliage transition
PHENOPHASE_COLORED_LEAVES = 498
PHENOPHASE_FALLING_LEAVES = 499

# Months to query — Aug through Nov covers the full transition window
FALL_MONTHS = {8, 9, 10, 11}

# Seconds to wait between year/state requests — be polite to the API
REQUEST_DELAY = 1.0

# Northeast states available for querying
NORTHEAST_STATES = ["ME", "NH", "VT", "MA", "RI", "CT", "NY", "NJ", "PA"]

# Bounding boxes per state — used as a geographic backstop after API filtering.
# Bounds are conservative (slightly padded) to avoid clipping border sites.
STATE_BOUNDS: dict[str, tuple[float, float, float, float]] = {
    # (lat_min, lat_max, lon_min, lon_max)
    "ME": (43.05, 47.47, -71.08, -66.95),
    "NH": (42.69, 45.31, -72.56, -70.61),
    "VT": (42.72, 45.02, -73.44, -71.46),
    "MA": (41.23, 42.89, -73.51, -69.93),
    "RI": (41.15, 42.02, -71.91, -71.12),
    "CT": (40.95, 42.05, -73.73, -71.79),
    "NY": (40.50, 45.02, -79.76, -71.86),
    "NJ": (38.93, 41.36, -75.56, -73.89),
    "PA": (39.72, 42.27, -80.52, -74.69),
}

def _region_bounds(states: list[str]) -> tuple[float, float, float, float]:
    """Return the bounding box enclosing all requested states."""
    lat_mins, lat_maxs, lon_mins, lon_maxs = zip(
        *[STATE_BOUNDS[s] for s in states if s in STATE_BOUNDS]
    )
    return min(lat_mins), max(lat_maxs), min(lon_mins), max(lon_maxs)

# ---------------------------------------------------------------------------
# Intensity string → stage mapping
# ---------------------------------------------------------------------------
# The NPN API returns intensity_value as a human-readable string, e.g.
# "50-74%" or "Less than 5%". The value -9999 means not recorded.
#
# Colored leaves intensity → foliage stage:
COLORED_INTENSITY_TO_STAGE: dict[str, str] = {
    "less than 5%":  "early",
    "5-24%":         "early",
    "25-49%":        "peak",
    "50-74%":        "peak",
    "75-94%":        "peak",
    "95% or more":   "peak",
}

# Falling leaves intensity → stage.
# Only intensities of 25%+ are classified as "late" — lower values occur
# even during peak foliage and are not reliable indicators of post-peak.
FALLING_INTENSITY_TO_STAGE: dict[str, str] = {
    "less than 5%":  None,    # too ambiguous — skip
    "5-24%":         None,    # still within normal peak-period leaf drop
    "25-49%":        "late",
    "50-74%":        "late",
    "75-94%":        "late",
    "95% or more":   "late",
}


def _parse_intensity(raw_value) -> str | None:
    """Normalise an NPN intensity_value to a lowercase string key.

    Returns None if the value is missing or the -9999 sentinel.
    """
    if raw_value is None:
        return None
    s = str(raw_value).strip().lower()
    if s in ("-9999", ""):
        return None
    return s


# ---------------------------------------------------------------------------
# API querying
# ---------------------------------------------------------------------------

def _query_year(
    year: int,
    request_source: str,
    phenophase_ids: list[int],
    state: str = "VT",
) -> list[dict]:
    """Query NPN observations for one state and year.

    Returns a list of raw observation dicts, or [] on failure.
    """
    params = [
        ("request_src", request_source),
        ("start_date",  f"{year}-01-01"),
        ("end_date",    f"{year}-12-31"),
        ("state",       state),
        ("num_days_quality_filter", 30),
    ]
    for pid in phenophase_ids:
        params.append(("phenophase_id[]", pid))

    try:
        response = requests.get(NPN_BASE_URL, params=params, timeout=120)
        response.raise_for_status()
        data = response.json()
        return data if isinstance(data, list) else []
    except requests.exceptions.RequestException as e:
        print(f"  [ERROR] {state} {year}: {e}")
        return []


# ---------------------------------------------------------------------------
# Record parsing and stage assignment
# ---------------------------------------------------------------------------

def _parse_records(
    raw_records: list[dict],
    verbose: bool = True,
) -> list[dict]:
    """Parse raw NPN dicts into standardised observer records.

    Each raw record is one observation of one phenophase on one individual
    plant on one date. Returns a list of dicts with keys:
        date, latitude, longitude, stage, source, notes

    Args:
        raw_records: Raw dicts from _query_year.
        verbose:     If True, print a drop-reason summary after parsing.
    """
    parsed = []
    drops: Counter = Counter()

    for rec in raw_records:
        try:
            lat      = float(rec.get("latitude",  0) or 0)
            lon      = float(rec.get("longitude", 0) or 0)
            date_str = rec.get("observation_date", "")

            if not lat or not lon or not date_str:
                drops["missing_location_or_date"] += 1
                continue

            phenophase_status = int(rec.get("phenophase_status", -1))
            if phenophase_status == -1:
                drops["uncertain_observation"] += 1
                continue

            # Parse date and filter to fall months
            try:
                obs_date = pd.to_datetime(date_str).date()
            except Exception:
                drops["bad_date"] += 1
                continue

            if obs_date.month not in FALL_MONTHS:
                drops["wrong_month"] += 1
                continue

            phenophase_id = int(rec.get("phenophase_id", 0))
            intensity_key = _parse_intensity(rec.get("intensity_value"))

            # -----------------------------------------------------------
            # Stage assignment
            # -----------------------------------------------------------
            stage = None
            stage_source = "intensity"

            if phenophase_id == PHENOPHASE_COLORED_LEAVES:
                if phenophase_status == 0:
                    # Explicit "No colored leaves" — reliable no_transition
                    stage = "no_transition"
                    stage_source = "phenophase_status"
                elif phenophase_status == 1:
                    if intensity_key is not None:
                        stage = COLORED_INTENSITY_TO_STAGE.get(intensity_key)
                        if stage is None:
                            drops["unrecognised_colored_intensity"] += 1
                    else:
                        drops["colored_leaves_no_intensity"] += 1

            elif phenophase_id == PHENOPHASE_FALLING_LEAVES:
                if phenophase_status == 1:
                    if intensity_key is not None:
                        stage = FALLING_INTENSITY_TO_STAGE.get(intensity_key)
                        if stage is None:
                            # Intensity too low to confidently call "late"
                            drops["falling_leaves_intensity_too_low"] += 1
                    else:
                        drops["falling_leaves_no_intensity"] += 1
                # phenophase_status == 0 for falling leaves is not useful
                # as a positive label — skip silently

            if stage is None:
                drops["no_stage_assigned"] += 1
                continue

            parsed.append({
                "date":      obs_date.isoformat(),
                "latitude":  lat,
                "longitude": lon,
                "stage":     stage,
                "source":    "USA-NPN",
                "notes": (
                    f"species={rec.get('common_name', '')} "
                    f"site_id={rec.get('site_id', '')} "
                    f"phenophase={phenophase_id} "
                    f"intensity={rec.get('intensity_value', '')} "
                    f"stage_source={stage_source}"
                ),
            })

        except (KeyError, ValueError, TypeError):
            drops["exception"] += 1
            continue

    if verbose and drops:
        print(f"  Drop reasons: {dict(drops)}")

    return parsed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    all_years = sorted(set(TRAIN_YEARS + VAL_YEARS + TEST_YEARS))
    p = argparse.ArgumentParser(
        description="Download USA-NPN fall foliage observations."
    )
    p.add_argument(
        "--request-source",
        default=os.environ.get("NPN_REQUEST_SOURCE", ""),
        help=(
            "Your name or organisation (required by NPN). "
            "Can also be set via NPN_REQUEST_SOURCE environment variable."
        ),
    )
    p.add_argument(
        "--states", nargs="+", default=["VT"],
        choices=NORTHEAST_STATES,
        help=(
            "States to query (default: VT). "
            f"Available: {' '.join(NORTHEAST_STATES)}"
        ),
    )
    p.add_argument(
        "--years", nargs="+", type=int, default=all_years,
        help="Years to download (default: all split years).",
    )
    p.add_argument(
        "--out", type=Path, default=None,
        help=(
            "Output CSV path. Defaults to "
            "data/raw/observer_reports/npn_{region}.csv "
            "where region is derived from --states."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.request_source:
        print(
            "[error] --request-source is required by the NPN API.\n"
            "  Set it with: --request-source \"Your Name\"\n"
            "  Or:          set NPN_REQUEST_SOURCE=Your Name"
        )
        sys.exit(1)

    # Derive output path from states if not explicitly provided
    if args.out is None:
        if args.states == ["VT"]:
            region_tag = "vermont"
        elif set(args.states) == set(NORTHEAST_STATES):
            region_tag = "northeast"
        else:
            region_tag = "_".join(sorted(s.lower() for s in args.states))
        args.out = OBSERVER_RAW / f"npn_{region_tag}.csv"

    args.out.parent.mkdir(parents=True, exist_ok=True)

    region_bounds  = _region_bounds(args.states)
    phenophase_ids = [PHENOPHASE_COLORED_LEAVES, PHENOPHASE_FALLING_LEAVES]

    print(f"[npn] Querying {len(args.states)} state(s): {' '.join(sorted(args.states))}")
    print(f"[npn] Years: {sorted(args.years)}")
    print(f"[npn] Region bounds (for reference): lat=[{region_bounds[0]}, {region_bounds[1]}] "
          f"lon=[{region_bounds[2]}, {region_bounds[3]}]")

    all_records: list[dict] = []

    for state in sorted(args.states):
        print(f"\n[npn] State: {state}")
        for year in sorted(args.years):
            print(f"  Querying {year} ...")
            raw    = _query_year(year, args.request_source, phenophase_ids, state)
            parsed = _parse_records(raw, verbose=True)
            all_records.extend(parsed)
            print(f"  → {len(raw)} raw records, {len(parsed)} usable observations")
            time.sleep(REQUEST_DELAY)

    if not all_records:
        print("[npn] No usable records after parsing. Check parameters.")
        sys.exit(1)

    df = pd.DataFrame(all_records)

    # Deduplicate — same observation may appear if a site is near a state border
    before = len(df)
    df = df.drop_duplicates(
        subset=["date", "latitude", "longitude", "stage"]
    ).reset_index(drop=True)
    if len(df) < before:
        print(f"[npn] Removed {before - len(df)} duplicate records (border sites)")

    print(f"\n[npn] Total: {len(df)} observations across "
          f"{df['date'].nunique()} dates and "
          f"{df.groupby(['latitude','longitude']).ngroups} unique sites")
    print("  Stage distribution:")
    for stage, count in df["stage"].value_counts().items():
        print(f"    {stage:20s}: {count:6d}")

    df.to_csv(args.out, index=False)
    print(f"\n[npn] Saved → {args.out}")


if __name__ == "__main__":
    main()