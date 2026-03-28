"""
download_npn.py — Download USA-NPN fall foliage observations for Vermont.

Queries the USA-NPN Status and Intensity API for Vermont observations of
the two relevant fall phenophases across all training/val/test years:

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
    data/raw/observer_reports/npn_vermont.csv

This file preserves the original stage as a string name (e.g. "early")
and includes source metadata in the notes field. It is not intended for
direct use by the model — run process_observations.py afterward to
produce the standardised, consolidated, confidence-scored output.

No API key is required. A request_source (your name) is required by the
NPN on an honor system basis — set it via --request-source or the
NPN_REQUEST_SOURCE environment variable.

Usage:
    python scripts/download_npn.py --request-source "Your Name"
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

# Seconds to wait between year requests — be polite to the API
REQUEST_DELAY = 1.0

# Vermont bounding box — hard backstop in case state filter is unreliable
VT_LAT_MIN, VT_LAT_MAX = 42.72, 45.02
VT_LON_MIN, VT_LON_MAX = -73.44, -71.46

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
) -> list[dict]:
    """Query NPN observations for Vermont for one year.

    Filters by state='VT' and the given phenophase IDs. The API can only
    filter by full calendar years, so fall-month filtering happens in
    _parse_records.

    Returns a list of raw observation dicts, or [] on failure.
    """
    # Use a list of tuples so the same key can appear multiple times
    params = [
        ("request_src", request_source),
        ("start_date",  f"{year}-01-01"),
        ("end_date",    f"{year}-12-31"),
        ("state",       "VT"),
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
        print(f"  [ERROR] Year {year}: {e}")
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

            # Vermont bounding box backstop
            if not (VT_LAT_MIN <= lat <= VT_LAT_MAX and
                    VT_LON_MIN <= lon <= VT_LON_MAX):
                drops["outside_vermont"] += 1
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
        description="Download USA-NPN fall foliage observations for Vermont."
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
        "--years", nargs="+", type=int, default=all_years,
        help="Years to download (default: all split years).",
    )
    p.add_argument(
        "--out", type=Path,
        default=OBSERVER_RAW / "npn_vermont.csv",
        help="Output CSV path.",
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

    args.out.parent.mkdir(parents=True, exist_ok=True)
    phenophase_ids = [PHENOPHASE_COLORED_LEAVES, PHENOPHASE_FALLING_LEAVES]

    all_records: list[dict] = []

    for year in sorted(args.years):
        print(f"[npn] Querying year {year} ...")
        raw    = _query_year(year, args.request_source, phenophase_ids)
        parsed = _parse_records(raw, verbose=True)
        all_records.extend(parsed)
        print(f"  → {len(raw)} raw records, {len(parsed)} usable observations")
        time.sleep(REQUEST_DELAY)

    if not all_records:
        print("[npn] No usable records after parsing. Check parameters.")
        sys.exit(1)

    df = pd.DataFrame(all_records)

    print(f"\n[npn] Total: {len(df)} observations across {df['date'].nunique()} dates")
    print("  Stage distribution:")
    for stage, count in df["stage"].value_counts().items():
        print(f"    {stage:20s}: {count:6d}")

    df.to_csv(args.out, index=False)
    print(f"\n[npn] Saved → {args.out}")


if __name__ == "__main__":
    main()