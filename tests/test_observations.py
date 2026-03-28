"""
test_observations.py — Unit tests for process_observations.py pure functions.

Tests cover:
  - _parse_stage: string and integer stage normalisation
  - _extract_confidence: confidence scoring from NPN metadata notes
  - consolidate: plurality voting, coordinate snapping, single/multi-record groups
"""

import sys
from datetime import date
from pathlib import Path

import pandas as pd
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from process_observations import (
    _parse_stage,
    _extract_confidence,
    consolidate,
    DEFAULT_CONFIDENCE,
    DATE_HEURISTIC_CONFIDENCE,
    NPN_NO_TRANSITION_CONFIDENCE,
    NPN_COLORED_CONFIDENCE,
    NPN_FALLING_CONFIDENCE,
)


# ---------------------------------------------------------------------------
# _parse_stage
# ---------------------------------------------------------------------------

class TestParseStage:

    def test_string_no_transition(self):
        assert _parse_stage("no_transition") == 0

    def test_string_early(self):
        assert _parse_stage("early") == 1

    def test_string_peak(self):
        assert _parse_stage("peak") == 2

    def test_string_late(self):
        assert _parse_stage("late") == 3

    def test_integer_string_zero(self):
        assert _parse_stage("0") == 0

    def test_integer_string_three(self):
        assert _parse_stage("3") == 3

    def test_strips_whitespace(self):
        assert _parse_stage("  peak  ") == 2

    def test_case_insensitive(self):
        assert _parse_stage("PEAK")  == 2
        assert _parse_stage("Early") == 1

    def test_invalid_string_returns_none(self):
        assert _parse_stage("full color") is None
        assert _parse_stage("") is None
        assert _parse_stage("unknown") is None

    def test_out_of_range_integer_returns_none(self):
        assert _parse_stage("4")  is None
        assert _parse_stage("-1") is None

    def test_non_numeric_garbage_returns_none(self):
        assert _parse_stage("abc123") is None


# ---------------------------------------------------------------------------
# _extract_confidence
# ---------------------------------------------------------------------------

def _npn_row(stage: str, phenophase: int, intensity: str, source: str = "USA-NPN") -> pd.Series:
    """Helper to build a row as process_observations expects it."""
    notes = (
        f"species=sugar maple site_id=5668 "
        f"phenophase={phenophase} intensity={intensity} "
        f"stage_source=intensity"
    )
    return pd.Series({
        "stage":  stage,
        "source": source,
        "notes":  notes,
    })


class TestExtractConfidence:

    def test_date_heuristic_records(self):
        row = pd.Series({
            "stage":  "no_transition",
            "source": "iNaturalist",
            "notes":  "taxon=Acer saccharum stage_source=date_heuristic",
        })
        assert _extract_confidence(row) == DATE_HEURISTIC_CONFIDENCE

    def test_npn_no_transition_high_confidence(self):
        row = _npn_row("no_transition", phenophase=498, intensity="-9999")
        assert _extract_confidence(row) == NPN_NO_TRANSITION_CONFIDENCE

    def test_npn_colored_leaves_unambiguous(self):
        row = _npn_row("peak", phenophase=498, intensity="75-94%")
        assert _extract_confidence(row) == NPN_COLORED_CONFIDENCE["75-94%"]

    def test_npn_colored_leaves_boundary(self):
        # 25-49% is at the early/peak boundary — should have lower confidence
        row = _npn_row("peak", phenophase=498, intensity="25-49%")
        conf = _extract_confidence(row)
        assert conf == NPN_COLORED_CONFIDENCE["25-49%"]
        # Boundary confidence should be lower than unambiguous
        assert conf < NPN_COLORED_CONFIDENCE["75-94%"]

    def test_npn_falling_leaves(self):
        row = _npn_row("late", phenophase=499, intensity="50-74%")
        assert _extract_confidence(row) == NPN_FALLING_CONFIDENCE["50-74%"]

    def test_unknown_source_returns_default(self):
        row = pd.Series({
            "stage":  "peak",
            "source": "SomeOtherSource",
            "notes":  "no useful metadata here",
        })
        assert _extract_confidence(row) == DEFAULT_CONFIDENCE

    def test_npn_missing_intensity_returns_default(self):
        row = pd.Series({
            "stage":  "peak",
            "source": "USA-NPN",
            "notes":  "phenophase=498 intensity=-9999",
        })
        # -9999 intensity for a "Yes" colored leaves → falls back to default
        assert _extract_confidence(row) == DEFAULT_CONFIDENCE

    def test_confidence_values_in_valid_range(self):
        # All confidence values in the module should be in [0, 1]
        all_values = (
            [DATE_HEURISTIC_CONFIDENCE, NPN_NO_TRANSITION_CONFIDENCE, DEFAULT_CONFIDENCE]
            + list(NPN_COLORED_CONFIDENCE.values())
            + list(NPN_FALLING_CONFIDENCE.values())
        )
        for v in all_values:
            assert 0.0 <= v <= 1.0, f"Confidence {v} out of range"


# ---------------------------------------------------------------------------
# consolidate
# ---------------------------------------------------------------------------

def _make_df(records: list[dict]) -> pd.DataFrame:
    """Build a DataFrame in the format expected by consolidate()."""
    if not records:
        return pd.DataFrame({
            "date":       pd.Series([], dtype="object"),
            "latitude":   pd.Series([], dtype="float64"),
            "longitude":  pd.Series([], dtype="float64"),
            "stage_int":  pd.Series([], dtype="int64"),
            "confidence": pd.Series([], dtype="float64"),
            "source":     pd.Series([], dtype="object"),
            "notes":      pd.Series([], dtype="object"),
        })

    rows = []
    for r in records:
        rows.append({
            "date":       r.get("date", date(2019, 10, 1)),
            "latitude":   r.get("latitude",  44.0),
            "longitude":  r.get("longitude", -72.5),
            "stage_int":  r.get("stage_int", 2),
            "confidence": r.get("confidence", 0.90),
            "source":     r.get("source",  "USA-NPN"),
            "notes":      r.get("notes",   ""),
        })
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


class TestConsolidate:

    def test_single_record_unchanged(self):
        df = _make_df([{}])
        result = consolidate(df)
        assert len(result) == 1
        assert result.iloc[0]["stage"] == 2
        assert abs(result.iloc[0]["confidence"] - 0.90) < 1e-4

    def test_two_identical_records_merged(self):
        df = _make_df([{}, {}])
        result = consolidate(df)
        assert len(result) == 1

    def test_plurality_stage_wins(self):
        # 2 peak, 1 early — peak should win
        df = _make_df([
            {"stage_int": 2},   # peak
            {"stage_int": 2},   # peak
            {"stage_int": 1},   # early
        ])
        result = consolidate(df)
        assert len(result) == 1
        assert result.iloc[0]["stage"] == 2  # peak

    def test_full_agreement_confidence(self):
        # All agree → agreement_fraction = 1.0, confidence = mean_conf
        df = _make_df([
            {"stage_int": 2, "confidence": 0.90},
            {"stage_int": 2, "confidence": 0.80},
        ])
        result = consolidate(df)
        # agreement_fraction = 1.0, mean_conf = 0.85
        expected = round(1.0 * 0.85, 4)
        assert abs(result.iloc[0]["confidence"] - expected) < 1e-3

    def test_partial_agreement_reduces_confidence(self):
        # 2 of 3 agree → agreement_fraction = 2/3
        df = _make_df([
            {"stage_int": 2, "confidence": 0.90},
            {"stage_int": 2, "confidence": 0.90},
            {"stage_int": 1, "confidence": 0.90},
        ])
        result = consolidate(df)
        assert result.iloc[0]["confidence"] < 0.90

    def test_different_locations_not_merged(self):
        df = _make_df([
            {"latitude": 44.0, "longitude": -72.5},
            {"latitude": 44.5, "longitude": -72.5},
        ])
        result = consolidate(df)
        assert len(result) == 2

    def test_different_dates_not_merged(self):
        df = _make_df([
            {"date": date(2019, 10, 1)},
            {"date": date(2019, 10, 2)},
        ])
        result = consolidate(df)
        assert len(result) == 2

    def test_coordinate_snapping_merges_nearby(self):
        # Two observations 0.00001° apart — should snap together
        df = _make_df([
            {"latitude": 44.00000, "longitude": -72.50000},
            {"latitude": 44.00001, "longitude": -72.50001},
        ])
        result = consolidate(df)
        assert len(result) == 1

    def test_sources_joined(self):
        df = _make_df([
            {"source": "USA-NPN"},
            {"source": "iNaturalist"},
        ])
        result = consolidate(df)
        src = result.iloc[0]["source"]
        assert "USA-NPN" in src
        assert "iNaturalist" in src

    def test_output_columns_present(self):
        from config import OBSERVATION_COLUMNS
        df = _make_df([{}])
        result = consolidate(df)
        for col in OBSERVATION_COLUMNS:
            assert col in result.columns, f"Missing column: {col}"

    def test_sorted_by_date(self):
        df = _make_df([
            {"date": date(2019, 10, 15), "latitude": 44.0},
            {"date": date(2019, 9,  1),  "latitude": 44.1},
            {"date": date(2019, 11, 1),  "latitude": 44.2},
        ])
        result = consolidate(df)
        dates = result["date"].tolist()
        assert dates == sorted(dates)

    def test_empty_dataframe(self):
        df = _make_df([])
        result = consolidate(df)
        assert len(result) == 0

    def test_tie_broken_deterministically(self):
        # Exact tie between two stages — result should be consistent across runs
        df = _make_df([
            {"stage_int": 1},   # early
            {"stage_int": 2},   # peak
        ])
        result1 = consolidate(df)
        result2 = consolidate(df)
        assert result1.iloc[0]["stage"] == result2.iloc[0]["stage"]
