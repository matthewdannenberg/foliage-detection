"""
test_npn.py — Unit tests for download_npn.py pure functions.

Tests cover:
  - _parse_intensity: normalisation of raw intensity strings
  - COLORED_INTENSITY_TO_STAGE: stage mapping for colored leaves phenophase
  - FALLING_INTENSITY_TO_STAGE: stage mapping for falling leaves phenophase
  - _parse_records: full record parsing pipeline including all filter paths
"""

import sys
from pathlib import Path
from datetime import date

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from download_npn import (
    _parse_intensity,
    _parse_records,
    COLORED_INTENSITY_TO_STAGE,
    FALLING_INTENSITY_TO_STAGE,
    PHENOPHASE_COLORED_LEAVES,
    PHENOPHASE_FALLING_LEAVES,
)


# ---------------------------------------------------------------------------
# Helpers — minimal valid NPN record
# ---------------------------------------------------------------------------

def _make_record(**overrides) -> dict:
    """Return a minimal valid NPN observation record for Vermont."""
    base = {
        "observation_id":    123456,
        "latitude":          44.0,
        "longitude":         -72.5,
        "observation_date":  "2019-10-01",
        "phenophase_id":     PHENOPHASE_COLORED_LEAVES,
        "phenophase_status": 1,
        "intensity_value":   "50-74%",
        "state":             "VT",
        "common_name":       "sugar maple",
        "site_id":           5668,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# _parse_intensity
# ---------------------------------------------------------------------------

class TestParseIntensity:

    def test_standard_string(self):
        assert _parse_intensity("50-74%") == "50-74%"

    def test_strips_and_lowercases(self):
        assert _parse_intensity("  50-74%  ") == "50-74%"
        assert _parse_intensity("Less than 5%") == "less than 5%"

    def test_sentinel_minus_9999_returns_none(self):
        assert _parse_intensity(-9999)   is None
        assert _parse_intensity("-9999") is None

    def test_none_returns_none(self):
        assert _parse_intensity(None) is None

    def test_empty_string_returns_none(self):
        assert _parse_intensity("") is None

    def test_numeric_string_returned_normalised(self):
        # Some API responses may return numeric-looking strings
        result = _parse_intensity("95% or more")
        assert result == "95% or more"

    def test_whitespace_only_returns_none(self):
        assert _parse_intensity("   ") is None


# ---------------------------------------------------------------------------
# COLORED_INTENSITY_TO_STAGE mapping
# ---------------------------------------------------------------------------

class TestColoredIntensityMapping:

    def test_less_than_5_is_early(self):
        assert COLORED_INTENSITY_TO_STAGE["less than 5%"] == "early"

    def test_5_24_is_early(self):
        assert COLORED_INTENSITY_TO_STAGE["5-24%"] == "early"

    def test_25_49_is_peak(self):
        assert COLORED_INTENSITY_TO_STAGE["25-49%"] == "peak"

    def test_50_74_is_peak(self):
        assert COLORED_INTENSITY_TO_STAGE["50-74%"] == "peak"

    def test_75_94_is_peak(self):
        assert COLORED_INTENSITY_TO_STAGE["75-94%"] == "peak"

    def test_95_plus_is_peak(self):
        assert COLORED_INTENSITY_TO_STAGE["95% or more"] == "peak"

    def test_all_keys_lowercase(self):
        for key in COLORED_INTENSITY_TO_STAGE:
            assert key == key.lower(), f"Key '{key}' is not lowercase"

    def test_no_missing_stages(self):
        stages = set(COLORED_INTENSITY_TO_STAGE.values())
        assert stages <= {"early", "peak", "no_transition"}


# ---------------------------------------------------------------------------
# FALLING_INTENSITY_TO_STAGE mapping
# ---------------------------------------------------------------------------

class TestFallingIntensityMapping:

    def test_less_than_5_excluded(self):
        assert FALLING_INTENSITY_TO_STAGE["less than 5%"] is None

    def test_5_24_excluded(self):
        assert FALLING_INTENSITY_TO_STAGE["5-24%"] is None

    def test_25_49_is_late(self):
        assert FALLING_INTENSITY_TO_STAGE["25-49%"] == "late"

    def test_50_74_is_late(self):
        assert FALLING_INTENSITY_TO_STAGE["50-74%"] == "late"

    def test_75_94_is_late(self):
        assert FALLING_INTENSITY_TO_STAGE["75-94%"] == "late"

    def test_95_plus_is_late(self):
        assert FALLING_INTENSITY_TO_STAGE["95% or more"] == "late"


# ---------------------------------------------------------------------------
# _parse_records
# ---------------------------------------------------------------------------

class TestParseRecords:

    def test_valid_colored_leaves_peak(self):
        records = _parse_records([_make_record()], verbose=False)
        assert len(records) == 1
        assert records[0]["stage"] == "peak"

    def test_valid_colored_leaves_early(self):
        rec = _make_record(intensity_value="5-24%")
        records = _parse_records([rec], verbose=False)
        assert len(records) == 1
        assert records[0]["stage"] == "early"

    def test_status_zero_is_no_transition(self):
        rec = _make_record(phenophase_status=0, intensity_value=-9999)
        records = _parse_records([rec], verbose=False)
        assert len(records) == 1
        assert records[0]["stage"] == "no_transition"

    def test_falling_leaves_high_intensity_is_late(self):
        rec = _make_record(
            phenophase_id=PHENOPHASE_FALLING_LEAVES,
            intensity_value="50-74%",
        )
        records = _parse_records([rec], verbose=False)
        assert len(records) == 1
        assert records[0]["stage"] == "late"

    def test_falling_leaves_low_intensity_dropped(self):
        rec = _make_record(
            phenophase_id=PHENOPHASE_FALLING_LEAVES,
            intensity_value="less than 5%",
        )
        records = _parse_records([rec], verbose=False)
        assert len(records) == 0

    def test_falling_leaves_no_intensity_dropped(self):
        rec = _make_record(
            phenophase_id=PHENOPHASE_FALLING_LEAVES,
            intensity_value=-9999,
        )
        records = _parse_records([rec], verbose=False)
        assert len(records) == 0

    def test_wrong_month_dropped(self):
        rec = _make_record(observation_date="2019-07-15")  # July
        records = _parse_records([rec], verbose=False)
        assert len(records) == 0

    def test_all_fall_months_accepted(self):
        for month in [8, 9, 10, 11]:
            rec = _make_record(observation_date=f"2019-{month:02d}-15")
            records = _parse_records([rec], verbose=False)
            assert len(records) == 1, f"Month {month} should be accepted"

    def test_uncertain_observation_dropped(self):
        rec = _make_record(phenophase_status=-1)
        records = _parse_records([rec], verbose=False)
        assert len(records) == 0

    def test_missing_location_dropped(self):
        rec = _make_record(latitude=None, longitude=None)
        records = _parse_records([rec], verbose=False)
        assert len(records) == 0

    def test_output_fields_present(self):
        records = _parse_records([_make_record()], verbose=False)
        assert len(records) == 1
        r = records[0]
        for field in ["date", "latitude", "longitude", "stage", "source", "notes"]:
            assert field in r, f"Missing field: {field}"

    def test_source_is_usa_npn(self):
        records = _parse_records([_make_record()], verbose=False)
        assert records[0]["source"] == "USA-NPN"

    def test_colored_no_intensity_dropped(self):
        rec = _make_record(phenophase_status=1, intensity_value=-9999)
        records = _parse_records([rec], verbose=False)
        assert len(records) == 0

    def test_multiple_records_mixed(self):
        recs = [
            _make_record(),                                       # valid peak
            _make_record(phenophase_status=-1),                   # uncertain — dropped
            _make_record(observation_date="2019-07-01"),          # wrong month — dropped
            _make_record(intensity_value="5-24%"),                # valid early
        ]
        records = _parse_records(recs, verbose=False)
        assert len(records) == 2
        stages = {r["stage"] for r in records}
        assert stages == {"peak", "early"}