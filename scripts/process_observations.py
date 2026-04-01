"""
process_observations.py — Consolidate raw observer CSVs into a standardised
processed observation file.

Reads one or more raw observer CSVs from data/raw/observer_reports/ and
produces a single standardised CSV at data/processed/observations/observations.csv.

Processing steps applied to each source:
  1. Parse and validate fields (date, lat/lon, stage string).
  2. Map stage strings to integer indices (0–3).
  3. Assign a per-record confidence score based on source and intensity metadata.
  4. Snap coordinates to CONSOLIDATION_COORD_DECIMALS decimal places.
  5. Consolidate records sharing the same (date, lat, lon) into one record
     via plurality vote, with confidence = agreement fraction.

The output schema (OBSERVATION_COLUMNS) is:
    date        : YYYY-MM-DD
    latitude    : float
    longitude   : float
    stage       : int (0=no_transition, 1=early, 2=peak, 3=late)
    confidence  : float [0, 1]
    source      : str
    notes       : str

Multiple raw source files can be combined in one run — they are pooled
before consolidation so that, for example, an NPN record and an iNaturalist
record at the same location on the same day are consolidated together.

Usage:
    python scripts/process_observations.py
    python scripts/process_observations.py --sources npn_vermont.csv
    python scripts/process_observations.py --sources npn_vermont.csv inaturalist.csv
    python scripts/process_observations.py --no-consolidate
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import numpy as np
from pyproj import Transformer

from config import (
    CONSOLIDATION_COORD_DECIMALS,
    OBSERVATION_COLUMNS,
    OBSERVATIONS_DIR,
    OBSERVER_RAW,
    STAGES,
    STAGE_NAMES,
    NUM_CLASSES,
    TARGET_CRS,
)

# ---------------------------------------------------------------------------
# ARD tile grid constants (duplicated from diagnose_ard_tiles.py to keep
# process_observations.py self-contained)
# ---------------------------------------------------------------------------
ARD_TILE_SIZE_M  = 150_000
ARD_GRID_ORIGIN_X = -2_565_000
ARD_GRID_ORIGIN_Y =  3_314_805

# ---------------------------------------------------------------------------
# Confidence heuristics
# ---------------------------------------------------------------------------
# These values encode how much we trust a stage label given the information
# available. They are intentionally conservative — better to understate
# confidence than to train on mislabeled data with high confidence.
#
# Confidence is assigned per-record before consolidation. After consolidation
# within a (date, location) group, confidence is replaced by the agreement
# fraction if multiple records were consolidated, or kept as-is if only one
# record was present.

# NPN colored leaves: confidence varies by intensity string ambiguity.
# Boundary intensities (e.g. "25-49%" sits right at early/peak boundary)
# get lower confidence than unambiguous intensities.
NPN_COLORED_CONFIDENCE: dict[str, float] = {
    "less than 5%":  0.85,   # clearly early
    "5-24%":         0.85,   # clearly early
    "25-49%":        0.70,   # boundary — could be read as early or peak
    "50-74%":        0.90,   # clearly peak
    "75-94%":        0.95,   # clearly peak
    "95% or more":   0.95,   # clearly peak
}

# NPN colored leaves status=0 (explicit No): high confidence for no_transition
NPN_NO_TRANSITION_CONFIDENCE = 0.90

# NPN falling leaves → late
NPN_FALLING_CONFIDENCE: dict[str, float] = {
    "25-49%":        0.80,
    "50-74%":        0.90,
    "75-94%":        0.95,
    "95% or more":   0.95,
}

# Default confidence when source-specific rules don't apply
DEFAULT_CONFIDENCE = 0.70

# Date-heuristic labels (added by generate_synthetic.py later)
DATE_HEURISTIC_CONFIDENCE = 0.50


# ---------------------------------------------------------------------------
# Stage string normalisation
# ---------------------------------------------------------------------------

# Accepts stage as either a string name or an integer string
def _parse_stage(raw: str) -> int | None:
    """Convert a raw stage value to an integer index.

    Accepts:
        - String names: "no_transition", "early", "peak", "late"
        - Integer strings: "0", "1", "2", "3"

    Returns None if unrecognised.
    """
    s = str(raw).strip().lower()
    if s in STAGES:
        return STAGES[s]
    try:
        idx = int(s)
        if 0 <= idx < NUM_CLASSES:
            return idx
    except ValueError:
        pass
    return None


# ---------------------------------------------------------------------------
# Confidence extraction
# ---------------------------------------------------------------------------

def _extract_confidence(row: pd.Series) -> float:
    """Derive a confidence score from a raw observation row.

    Reads the 'notes' field to recover intensity and phenophase metadata
    written by download_npn.py. Falls back to DEFAULT_CONFIDENCE if the
    notes don't contain recognisable metadata.
    """
    notes = str(row.get("notes", "")).lower()
    source = str(row.get("source", "")).upper()

    # Date-heuristic synthetic records
    if "stage_source=date_heuristic" in notes:
        return DATE_HEURISTIC_CONFIDENCE

    if source == "USA-NPN":
        # Extract intensity from notes: "intensity=50-74%"
        intensity = None
        for part in notes.split():
            if part.startswith("intensity="):
                intensity = part.replace("intensity=", "").strip()
                break

        # Extract phenophase
        phenophase = None
        for part in notes.split():
            if part.startswith("phenophase="):
                try:
                    phenophase = int(part.replace("phenophase=", ""))
                except ValueError:
                    pass
                break

        if phenophase == 498:  # colored leaves
            stage_str = str(row.get("stage", "")).lower()
            if stage_str == "no_transition":
                return NPN_NO_TRANSITION_CONFIDENCE
            if intensity and intensity in NPN_COLORED_CONFIDENCE:
                return NPN_COLORED_CONFIDENCE[intensity]

        elif phenophase == 499:  # falling leaves
            if intensity and intensity in NPN_FALLING_CONFIDENCE:
                return NPN_FALLING_CONFIDENCE[intensity]

    return DEFAULT_CONFIDENCE


# ---------------------------------------------------------------------------
# Loading and validation
# ---------------------------------------------------------------------------

def load_raw_csv(path: Path) -> pd.DataFrame:
    """Load a raw observer CSV and return a validated DataFrame.

    Drops rows with missing or unparseable date, coordinates, or stage.
    Adds a 'stage_int' column with the integer stage index.
    Adds a 'confidence' column derived from source metadata.
    """
    df = pd.read_csv(path)

    # Normalise column names
    df.columns = [c.strip().lower() for c in df.columns]

    required = ["date", "latitude", "longitude", "stage"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV {path.name} missing columns: {missing}")

    # Add optional columns if absent
    for col in ["source", "notes"]:
        if col not in df.columns:
            df[col] = ""

    n_raw = len(df)
    drops = Counter()

    # Parse dates
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    bad_dates = df["date"].isna()
    drops["bad_date"] += bad_dates.sum()
    df = df[~bad_dates].copy()

    # Parse coordinates
    df["latitude"]  = pd.to_numeric(df["latitude"],  errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    bad_coords = df["latitude"].isna() | df["longitude"].isna()
    drops["bad_coords"] += bad_coords.sum()
    df = df[~bad_coords].copy()

    # Parse stage
    df["stage_int"] = df["stage"].apply(_parse_stage)
    bad_stage = df["stage_int"].isna()
    drops["bad_stage"] += bad_stage.sum()
    df = df[~bad_stage].copy()
    df["stage_int"] = df["stage_int"].astype(int)

    # Assign confidence
    df["confidence"] = df.apply(_extract_confidence, axis=1)

    n_kept = len(df)
    if drops:
        print(f"  [{path.name}] {n_kept}/{n_raw} rows kept. Drops: {dict(drops)}")

    return df


# ---------------------------------------------------------------------------
# Consolidation
# ---------------------------------------------------------------------------

def consolidate(df: pd.DataFrame) -> pd.DataFrame:
    """Consolidate multiple records at the same (date, location) into one.

    Coordinates are snapped to CONSOLIDATION_COORD_DECIMALS decimal places
    before grouping, so that observations at the same monitoring site are
    treated as identical regardless of minor GPS jitter.

    Within each group:
      - Stage: plurality vote (most common stage_int value).
      - Confidence: if group size == 1, keep original confidence.
                    if group size > 1, confidence = agreement fraction
                    (fraction of records that match the plurality stage).
      - Source: joined unique sources e.g. "USA-NPN+iNaturalist".
      - Notes: joined notes from all records in the group.

    Returns a DataFrame with OBSERVATION_COLUMNS schema.
    """
    if df.empty:
        return pd.DataFrame(columns=OBSERVATION_COLUMNS)
    
    # Snap coordinates
    df = df.copy()
    d = CONSOLIDATION_COORD_DECIMALS
    df["lat_snap"] = df["latitude"].round(d)
    df["lon_snap"] = df["longitude"].round(d)

    records = []
    group_keys = ["date", "lat_snap", "lon_snap"]

    for (date, lat, lon), group in df.groupby(group_keys):
        n = len(group)

        # Plurality stage
        stage_counts = Counter(group["stage_int"].tolist())
        plurality_stage, plurality_count = stage_counts.most_common(1)[0]

        # Confidence
        if n == 1:
            confidence = float(group["confidence"].iloc[0])
        else:
            agreement_fraction = plurality_count / n
            # Scale: if all agree, use the mean individual confidence;
            # if partial agreement, weight by agreement fraction
            mean_conf = float(group["confidence"].mean())
            confidence = round(agreement_fraction * mean_conf, 4)

        # Source and notes
        sources = "+".join(sorted(group["source"].dropna().unique()))
        notes_parts = group["notes"].dropna().tolist()
        notes = " | ".join(str(n) for n in notes_parts if str(n).strip())

        records.append({
            "date":       date.isoformat(),
            "latitude":   lat,
            "longitude":  lon,
            "stage":      plurality_stage,
            "confidence": confidence,
            "source":     sources,
            "notes":      notes,
        })

    result = pd.DataFrame(records, columns=OBSERVATION_COLUMNS)
    return result.sort_values(["date", "latitude", "longitude"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _observations_to_tile_list(df: pd.DataFrame) -> list[str]:
    """Derive the set of ARD tile IDs needed to cover all observation sites.

    Returns a sorted list of tile ID strings e.g. ['028004', '029005', ...].
    """
    transformer = Transformer.from_crs("EPSG:4326", TARGET_CRS, always_xy=True)
    tile_ids: set[str] = set()

    for _, row in df.drop_duplicates(subset=["latitude", "longitude"]).iterrows():
        x, y = transformer.transform(float(row["longitude"]), float(row["latitude"]))
        h = int((x - ARD_GRID_ORIGIN_X) / ARD_TILE_SIZE_M) + 1
        v = int((ARD_GRID_ORIGIN_Y - y) / ARD_TILE_SIZE_M) + 1
        tile_ids.add(f"{h:03d}{v:03d}")

    return sorted(tile_ids)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Consolidate raw observer CSVs into standardised observations."
    )
    p.add_argument(
        "--sources", nargs="+", type=str, default=None,
        help=(
            "Raw CSV filenames to process (looked up in OBSERVER_RAW). "
            "Defaults to all *.csv files in OBSERVER_RAW."
        ),
    )
    p.add_argument(
        "--out", type=Path,
        default=OBSERVATIONS_DIR / "observations.csv",
        help="Output path for the standardised CSV.",
    )
    p.add_argument(
        "--tile-list", type=Path,
        default=None,
        help=(
            "If provided, write a text file of ARD tile IDs needed to cover "
            "all observation sites. Defaults to "
            "data/processed/observations/ard_tile_list.txt."
        ),
    )
    p.add_argument(
        "--no-tile-list", action="store_true",
        help="Skip tile list generation.",
    )
    p.add_argument(
        "--no-consolidate", action="store_true",
        help="Skip consolidation — write one row per raw observation.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Discover source files
    if args.sources:
        source_paths = [OBSERVER_RAW / s for s in args.sources]
    else:
        source_paths = sorted(OBSERVER_RAW.glob("*.csv"))

    if not source_paths:
        print(f"[process] No CSV files found in {OBSERVER_RAW}")
        sys.exit(1)

    print(f"[process] Loading {len(source_paths)} source file(s):")

    dfs = []
    for path in source_paths:
        if not path.exists():
            print(f"  [skip] Not found: {path}")
            continue
        print(f"  {path.name}")
        dfs.append(load_raw_csv(path))

    if not dfs:
        print("[process] No data loaded.")
        sys.exit(1)

    combined = pd.concat(dfs, ignore_index=True)
    print(f"\n[process] Combined: {len(combined)} raw observations")

    if args.no_consolidate:
        # Write without consolidation — use integer stages directly
        out_df = combined[["date", "latitude", "longitude",
                            "stage_int", "confidence", "source", "notes"]].copy()
        out_df = out_df.rename(columns={"stage_int": "stage"})
        out_df["date"] = out_df["date"].apply(
            lambda d: d.isoformat() if hasattr(d, "isoformat") else str(d)
        )
    else:
        print("[process] Consolidating by (date, location) ...")
        out_df = consolidate(combined)
        n_before = len(combined)
        n_after  = len(out_df)
        print(f"  {n_before} → {n_after} records "
              f"({n_before - n_after} merged by consolidation)")

    # Summary
    print("\n[process] Stage distribution:")
    stage_counts = Counter(out_df["stage"].tolist())
    for idx in range(NUM_CLASSES):
        print(f"  {STAGE_NAMES[idx]:20s}: {stage_counts.get(idx, 0):6d}")

    print(f"\n[process] Confidence distribution:")
    conf = pd.to_numeric(out_df["confidence"], errors="coerce")
    print(f"  mean={conf.mean():.3f}  "
          f"min={conf.min():.3f}  "
          f"max={conf.max():.3f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"\n[process] Saved → {args.out}  ({len(out_df)} observations)")

    # --- Tile list ---
    if not args.no_tile_list:
        tile_list_path = args.tile_list or (args.out.parent / "ard_tile_list.txt")
        tile_ids = _observations_to_tile_list(out_df)
        tile_list_path.write_text("\n".join(tile_ids) + "\n")
        print(f"[process] ARD tile list ({len(tile_ids)} tiles) → {tile_list_path}")


if __name__ == "__main__":
    main()