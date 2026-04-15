"""
download_phenocam.py — Download PhenoCam GCC/RCC time series and derive
foliage stage labels for Northeast US deciduous/mixed forest sites.

Stage derivation uses PELT changepoint detection on the 2D (GCC, RCC)
signal, calibrated against 1,700 human-labeled images across 120 site-years.
Best parameters found: penalty=0.7, min_size=14, rcc_weight=0.1,
assignment_rule=max_rcc_anchor (macro-F1=0.748).

    max_rcc_anchor rule:
        PELT finds changepoints freely. The segment with the highest mean
        RCC is assigned 'peak'. All segments after it are 'late'. The
        segment immediately before peak is 'early'. All before that are
        'no_transition'. Temporal order is enforced via a forward pass.

Confidence is based on distance from segment boundaries — days near the
centre of a segment are more reliable than days near a changepoint.

Only sites with roi_type DB (deciduous broadleaf) or MX (mixed) are
included. Sites must have at least MIN_SITE_YEARS of data.

Geographic bounds default to the full Northeast (Maine → Pennsylvania/NJ)
to match the NPN observer coverage.

Output:
    data/raw/observer_reports/phenocam_northeast.csv

Run process_observations.py afterward to consolidate with NPN data.

Usage:
    python scripts/download_phenocam.py --request-source "Your Name"
    python scripts/download_phenocam.py --no-mx   (DB sites only)
    python scripts/download_phenocam.py --years 2019 2020 2021
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from datetime import date
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.signal import savgol_filter

from config import (
    OBSERVER_RAW,
    STAGES,
    STAGE_NAMES,
    TRAIN_YEARS,
    VAL_YEARS,
    TEST_YEARS,
)

try:
    import ruptures as rpt
    _RUPTURES_AVAILABLE = True
except ImportError:
    _RUPTURES_AVAILABLE = False
    print("[phenocam] WARNING: ruptures not installed. "
          "Install with: pip install ruptures")

# ---------------------------------------------------------------------------
# Search region
# ---------------------------------------------------------------------------
DEFAULT_LAT_MIN =  38.8
DEFAULT_LAT_MAX =  47.6
DEFAULT_LON_MIN = -80.6
DEFAULT_LON_MAX = -66.8

# ---------------------------------------------------------------------------
# Site filtering
# ---------------------------------------------------------------------------
VALID_ROI_TYPES = {"DB", "MX"}
MIN_SITE_YEARS  = 1.0

# ---------------------------------------------------------------------------
# PELT parameters — calibrated against 1,700 human labels, 120 site-years
# macro-F1 = 0.748 on held-out labels
# ---------------------------------------------------------------------------
PELT_PENALTY    = 0.7
PELT_MIN_SIZE   = 14
RCC_WEIGHT      = 0.1
SG_WINDOW       = 15    # Savitzky-Golay smoothing window (must be odd)
SG_POLYORDER    = 3

# ---------------------------------------------------------------------------
# Confidence
# ---------------------------------------------------------------------------
MAX_CONFIDENCE      = 0.85   # ceiling — labels are derived, not directly observed
BOUNDARY_DECAY_DAYS = 7      # days over which confidence ramps up from boundary

# ---------------------------------------------------------------------------
# Fall analysis window
# ---------------------------------------------------------------------------
FALL_MONTH_START = 8
FALL_MONTH_END   = 11

# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
API_ROILISTS  = "https://phenocam.nau.edu/api/roilists/"
REQUEST_DELAY = 0.5


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def _get_all_pages(url: str) -> list[dict]:
    """Fetch all pages of a paginated PhenoCam API endpoint."""
    results  = []
    next_url: str | None = url
    while next_url:
        resp = requests.get(next_url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        results.extend(data.get("results", []))
        next_url = data.get("next")
        if next_url:
            time.sleep(REQUEST_DELAY)
    return results


def _fetch_csv(url: str) -> pd.DataFrame | None:
    """Download a PhenoCam 1-day summary CSV, skipping comment lines."""
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"    [WARN] Failed to fetch {url}: {e}")
        return None

    lines      = [l for l in resp.text.splitlines() if not l.startswith("#")]
    if not lines:
        return None
    try:
        return pd.read_csv(StringIO("\n".join(lines)))
    except Exception as e:
        print(f"    [WARN] Failed to parse CSV: {e}")
        return None


# ---------------------------------------------------------------------------
# Signal preprocessing
# ---------------------------------------------------------------------------

def _smooth_series(values: np.ndarray) -> np.ndarray:
    """Interpolate gaps then apply Savitzky-Golay smoothing."""
    s = pd.Series(values.astype(float))
    s = s.interpolate(method="linear", limit=7).ffill().bfill()
    if s.isna().all():
        return values
    return savgol_filter(s.values, window_length=SG_WINDOW,
                         polyorder=SG_POLYORDER)


def _preprocess_fall(df: pd.DataFrame, year: int) -> pd.DataFrame | None:
    """Extract and quality-filter fall season data for one year.

    Returns a DataFrame indexed 0..N-1 with columns:
        date, gcc_90, rcc_90  (plus original flag columns)
    or None if fewer than 30 valid days remain.
    """
    required = {"date", "gcc_90", "rcc_90", "outlierflag_gcc_90", "snow_flag"}
    if not required.issubset(df.columns):
        return None

    fall = df[
        (df["date"].apply(lambda d: d.year) == year) &
        (df["date"].apply(lambda d: FALL_MONTH_START <= d.month <= FALL_MONTH_END))
    ].copy()

    if len(fall) < 10:
        return None

    valid = (
        (fall["outlierflag_gcc_90"].fillna(0) == 0) &
        (fall["snow_flag"].fillna(0) == 0) &
        fall["gcc_90"].notna() &
        fall["rcc_90"].notna()
    )
    fall = fall[valid].copy()

    if len(fall) < 30:
        return None

    # Reindex to complete daily sequence so gaps are explicit
    all_dates = pd.date_range(
        date(year, FALL_MONTH_START, 1),
        date(year, FALL_MONTH_END, 30)
    ).date
    fall = (fall.set_index("date")
                .reindex(all_dates)
                .reset_index()
                .rename(columns={"index": "date"}))
    return fall


# ---------------------------------------------------------------------------
# PELT changepoint detection
# ---------------------------------------------------------------------------

def _build_signal(gcc_s: np.ndarray, rcc_s: np.ndarray) -> np.ndarray:
    """Stack GCC and weighted RCC into a 2D signal for PELT."""
    return np.column_stack([gcc_s, rcc_s * RCC_WEIGHT])


def _detect_changepoints(signal: np.ndarray) -> list[int]:
    """Run PELT and return changepoint indices (last element = len(signal))."""
    if signal.ndim == 1:
        signal = signal.reshape(-1, 1)
    algo = rpt.Pelt(model="rbf", min_size=PELT_MIN_SIZE, jump=1)
    algo.fit(signal)
    return algo.predict(pen=PELT_PENALTY)


# ---------------------------------------------------------------------------
# Stage assignment — max_rcc_anchor rule
# ---------------------------------------------------------------------------

def _assign_max_rcc_anchor(segments: list[tuple[int, int]],
                            rcc_s: np.ndarray) -> list[str]:
    """Assign stages anchored on the segment with highest mean RCC.

    Segments after the max-RCC segment → late
    Max-RCC segment                    → peak
    Segment immediately before peak    → early
    All earlier segments               → no_transition

    Temporal order is guaranteed by construction.
    """
    n_segs = len(segments)

    if n_segs == 1:
        return ["no_transition"]
    if n_segs == 2:
        return ["no_transition", "late"]
    if n_segs == 3:
        return ["no_transition", "peak", "late"]

    rcc_means = [float(np.nanmean(rcc_s[s:e])) for s, e in segments]
    peak_idx  = max(1, min(int(np.argmax(rcc_means)), n_segs - 2))

    stages = []
    for i in range(n_segs):
        if   i < peak_idx - 1: stages.append("no_transition")
        elif i == peak_idx - 1: stages.append("early")
        elif i == peak_idx:     stages.append("peak")
        else:                   stages.append("late")
    return stages


# ---------------------------------------------------------------------------
# Confidence scoring
# ---------------------------------------------------------------------------

def _boundary_confidence(segments: list[tuple[int, int]],
                          n_days: int) -> np.ndarray:
    """Per-day confidence based on distance from nearest segment boundary.

    Days near the centre of a segment are reliable; days near a changepoint
    boundary are uncertain. Confidence ramps linearly from 0.5 at the
    boundary to 1.0 at BOUNDARY_DECAY_DAYS or more from any boundary.
    """
    # Collect all interior boundary positions
    boundaries = set()
    for i in range(1, len(segments)):
        boundaries.add(segments[i][0])

    conf = np.ones(n_days, dtype=float)
    for b in boundaries:
        for offset in range(BOUNDARY_DECAY_DAYS):
            ramp = 0.5 + 0.5 * (offset / BOUNDARY_DECAY_DAYS)
            for pos in (b - offset - 1, b + offset):
                if 0 <= pos < n_days:
                    conf[pos] = min(conf[pos], ramp)
    return conf


# ---------------------------------------------------------------------------
# Per-site-year processing
# ---------------------------------------------------------------------------

def _process_site_year(
    site: str,
    lat: float,
    lon: float,
    year: int,
    df_full: pd.DataFrame,
    verbose: bool = False,
) -> list[dict]:
    """Derive PELT-based stage labels for one site-year.

    Returns a list of observation records, one per valid fall day.
    """
    tag = f"  [{site} {year}]"

    fall = _preprocess_fall(df_full, year)
    if fall is None:
        if verbose:
            print(f"{tag} SKIP: insufficient valid fall data")
        return []

    gcc_raw = fall["gcc_90"].values
    rcc_raw = fall["rcc_90"].values

    gcc_s = _smooth_series(gcc_raw)
    rcc_s = _smooth_series(rcc_raw)
    sig   = _build_signal(gcc_s, rcc_s)

    try:
        cps = _detect_changepoints(sig)
    except Exception as e:
        if verbose:
            print(f"{tag} SKIP: PELT failed — {e}")
        return []

    boundaries = [0] + cps
    segments   = [(boundaries[i], boundaries[i + 1])
                  for i in range(len(boundaries) - 1)]

    seg_stages = _assign_max_rcc_anchor(segments, rcc_s)
    boundary_conf = _boundary_confidence(segments, len(fall))

    # Expand segment labels to per-day records
    day_stage: list[str | None] = [None] * len(fall)
    for (start, end), stage in zip(segments, seg_stages):
        for j in range(start, end):
            day_stage[j] = stage

    records = []
    dates   = fall["date"].tolist()

    for i, (d, stage) in enumerate(zip(dates, day_stage)):
        if stage is None or not isinstance(d, date):
            continue

        stage_int = STAGES.get(stage)
        if stage_int is None:
            continue

        confidence = round(float(boundary_conf[i]) * MAX_CONFIDENCE, 4)

        records.append({
            "date":      d.isoformat(),
            "latitude":  lat,
            "longitude": lon,
            "stage":     stage_int,
            "confidence": confidence,
            "source":    "PhenoCam",
            "notes": (
                f"site={site} year={year} "
                f"n_segments={len(segments)} "
                f"stage_source=pelt_max_rcc_anchor "
                f"penalty={PELT_PENALTY} min_size={PELT_MIN_SIZE} "
                f"rcc_weight={RCC_WEIGHT}"
            ),
        })

    if verbose and records:
        counts = Counter(r["stage"] for r in records)
        print(f"{tag} {len(records)} days — "
              + " ".join(f"{STAGE_NAMES[k]}:{v}"
                         for k, v in sorted(counts.items())))

    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    all_years = sorted(set(TRAIN_YEARS + VAL_YEARS + TEST_YEARS))
    p = argparse.ArgumentParser(
        description="Download PhenoCam GCC/RCC and derive foliage stage labels."
    )
    p.add_argument("--lat-min",  type=float, default=DEFAULT_LAT_MIN)
    p.add_argument("--lat-max",  type=float, default=DEFAULT_LAT_MAX)
    p.add_argument("--lon-min",  type=float, default=DEFAULT_LON_MIN)
    p.add_argument("--lon-max",  type=float, default=DEFAULT_LON_MAX)
    p.add_argument(
        "--years", nargs="+", type=int, default=all_years,
        help="Years to include (default: all split years).",
    )
    p.add_argument(
        "--no-mx", action="store_true",
        help="Exclude mixed (MX) ROIs — use deciduous broadleaf (DB) only.",
    )
    p.add_argument(
        "--out", type=Path,
        default=OBSERVER_RAW / "phenocam_northeast.csv",
        help="Output CSV path.",
    )
    p.add_argument(
        "--verbose", action="store_true",
        help="Print per-site-year stage counts.",
    )
    return p.parse_args()


def main() -> None:
    if not _RUPTURES_AVAILABLE:
        print("[phenocam] ERROR: ruptures is required. "
              "Install with: pip install ruptures scipy")
        sys.exit(1)

    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    valid_types = {"DB"} if args.no_mx else VALID_ROI_TYPES

    # 1. Fetch ROI list and filter to region + veg type
    print("[phenocam] Fetching ROI list ...")
    all_rois = _get_all_pages(API_ROILISTS)
    print(f"  Total ROIs in network: {len(all_rois)}")

    region_rois = [
        r for r in all_rois
        if r["roitype"] in valid_types
        and args.lat_min <= r["lat"] <= args.lat_max
        and args.lon_min <= r["lon"] <= args.lon_max
        and float(r["site_years"]) >= MIN_SITE_YEARS
    ]
    print(f"  ROIs matching region + veg type {valid_types}: {len(region_rois)}")

    if not region_rois:
        print("[phenocam] No ROIs found. Check bounds.")
        sys.exit(1)

    print(f"\n  {'ROI':45s} {'lat':>7} {'lon':>8} {'years':>6}")
    print(f"  {'-'*45} {'-'*7} {'-'*8} {'-'*6}")
    for roi in region_rois:
        print(f"  {roi['roi_name']:45s} "
              f"{roi['lat']:7.3f} {roi['lon']:8.3f} "
              f"{roi['site_years']:>6}")

    # 2. Download time series and derive labels
    all_records: list[dict] = []
    target_years = set(args.years)
    n_skip = 0

    for roi in region_rois:
        site    = roi["site"]
        lat     = roi["lat"]
        lon     = roi["lon"]
        csv_url = roi["one_day_summary"]

        print(f"\n[phenocam] {roi['roi_name']} ...")

        df = _fetch_csv(csv_url)
        if df is None or df.empty:
            print("  Skipped — no data")
            n_skip += 1
            continue

        try:
            df["date"] = pd.to_datetime(df["date"]).dt.date
        except Exception:
            print("  Skipped — date parse failed")
            n_skip += 1
            continue

        required = {"date", "gcc_90", "rcc_90", "outlierflag_gcc_90", "snow_flag"}
        if not required.issubset(df.columns):
            print(f"  Skipped — missing columns: {required - set(df.columns)}")
            n_skip += 1
            continue

        site_total = 0
        for year in sorted(target_years):
            df_year = df[df["date"].apply(lambda d: d.year) == year]
            if len(df_year) < 20:
                continue

            records = _process_site_year(
                site, lat, lon, year, df, verbose=args.verbose
            )
            all_records.extend(records)
            site_total += len(records)

        if site_total:
            print(f"  {site_total} records across {len(target_years)} years")

        time.sleep(REQUEST_DELAY)

    # 3. Write output
    if not all_records:
        print("\n[phenocam] No records derived.")
        sys.exit(1)

    df_out = pd.DataFrame(all_records)
    print(f"\n[phenocam] Total records : {len(df_out)}")
    print(f"           Sites skipped  : {n_skip}")
    print("           Stage distribution:")
    stage_lookup = {v: k for k, v in STAGES.items()}
    for stage_int, count in sorted(Counter(df_out["stage"].tolist()).items()):
        print(f"             {stage_lookup.get(stage_int, '?'):20s}: {count:6d}")

    df_out.to_csv(args.out, index=False)
    print(f"\n[phenocam] Saved → {args.out}")
    print(
        "\nNext step: python scripts/process_observations.py "
        "--sources npn_northeast.csv phenocam_northeast.csv"
    )


if __name__ == "__main__":
    main()