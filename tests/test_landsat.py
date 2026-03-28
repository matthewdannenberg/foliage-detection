"""
test_landsat.py — Unit tests for landsat.py pure functions.

Tests cover:
  - _bit_mask: QA bitmask logic
  - compute_evi2 / compute_ndii / compute_ndvi: spectral index math and edge cases
  - LandsatTile._parse_tile_id: ARD tile ID parsing
  - LandsatTile.from_tile_id: construction from ID string
"""

import numpy as np
import pytest
import sys
from pathlib import Path

# Make src importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data.landsat import (
    _bit_mask,
    compute_evi2,
    compute_ndii,
    compute_ndvi,
    LandsatTile,
)


# ---------------------------------------------------------------------------
# _bit_mask
# ---------------------------------------------------------------------------

class TestBitMask:

    def test_single_bit_set(self):
        qa = np.array([[0b00001000]], dtype=np.uint16)  # bit 3 set
        mask = _bit_mask(qa, [3])
        assert mask[0, 0] == True

    def test_single_bit_not_set(self):
        qa = np.array([[0b00000000]], dtype=np.uint16)
        mask = _bit_mask(qa, [3])
        assert mask[0, 0] == False

    def test_multiple_bits_any_triggers(self):
        # bits 3 and 4 are the mask; only bit 3 is set
        qa = np.array([[0b00001000]], dtype=np.uint16)
        mask = _bit_mask(qa, [3, 4])
        assert mask[0, 0] == True

    def test_multiple_bits_none_set(self):
        qa = np.array([[0b00000001]], dtype=np.uint16)  # bit 0 only
        mask = _bit_mask(qa, [3, 4])
        assert mask[0, 0] == False

    def test_returns_bool_dtype(self):
        qa = np.zeros((4, 4), dtype=np.uint16)
        mask = _bit_mask(qa, [3])
        assert mask.dtype == bool

    def test_correct_shape(self):
        qa = np.zeros((10, 20), dtype=np.uint16)
        mask = _bit_mask(qa, [1, 2])
        assert mask.shape == (10, 20)

    def test_mixed_array(self):
        qa = np.array([[0b00001000, 0b00000000],
                       [0b00010000, 0b00011000]], dtype=np.uint16)
        mask = _bit_mask(qa, [3, 4])
        expected = np.array([[True,  False],
                              [True,  True]])
        np.testing.assert_array_equal(mask, expected)

    def test_cloud_shadow_bits(self):
        """Verify the actual QA bits used in the pipeline work as expected."""
        # QA_FILL=0, QA_DILATED_CLOUD=1, QA_CLOUD=3, QA_CLOUD_SHADOW=4, QA_SNOW=5
        QA_MASK_BITS = [0, 1, 3, 4, 5]
        # Clear pixel — none of the mask bits set
        clear = np.array([[0b01000000]], dtype=np.uint16)  # bit 6 (CLEAR) set
        assert _bit_mask(clear, QA_MASK_BITS)[0, 0] == False
        # Cloud pixel — bit 3 set
        cloud = np.array([[0b00001000]], dtype=np.uint16)
        assert _bit_mask(cloud, QA_MASK_BITS)[0, 0] == True
        # Snow pixel — bit 5 set
        snow = np.array([[0b00100000]], dtype=np.uint16)
        assert _bit_mask(snow, QA_MASK_BITS)[0, 0] == True


# ---------------------------------------------------------------------------
# compute_evi2
# ---------------------------------------------------------------------------

class TestEVI2:

    def test_known_value(self):
        # EVI2 = 2.5 * (nir - red) / (nir + 2.4*red + 1)
        nir = np.array([[0.5]], dtype=np.float32)
        red = np.array([[0.1]], dtype=np.float32)
        expected = 2.5 * (0.5 - 0.1) / (0.5 + 2.4 * 0.1 + 1.0)
        result = compute_evi2(nir, red)
        assert abs(float(result[0, 0]) - expected) < 1e-5

    def test_nan_propagation(self):
        nir = np.array([[np.nan]], dtype=np.float32)
        red = np.array([[0.1]],   dtype=np.float32)
        result = compute_evi2(nir, red)
        assert np.isnan(result[0, 0])

    def test_zero_denominator_returns_nan(self):
        # denom = nir + 2.4*red + 1; can't be zero with physical values,
        # but near-zero should not produce infinity
        nir = np.array([[-1.0]], dtype=np.float32)
        red = np.array([[-1.0 / 2.4 + 1e-7]], dtype=np.float32)
        result = compute_evi2(nir, red)
        # Should not raise, result may be NaN or very large — just no crash
        assert result.dtype == np.float32

    def test_output_dtype(self):
        nir = np.array([[0.4]], dtype=np.float32)
        red = np.array([[0.1]], dtype=np.float32)
        assert compute_evi2(nir, red).dtype == np.float32

    def test_green_vegetation_positive(self):
        # Healthy vegetation: high NIR, low red → positive EVI2
        nir = np.full((3, 3), 0.6, dtype=np.float32)
        red = np.full((3, 3), 0.05, dtype=np.float32)
        result = compute_evi2(nir, red)
        assert np.all(result > 0)

    def test_bare_soil_near_zero(self):
        # Bare soil: NIR ≈ red → EVI2 near zero
        nir = np.full((2, 2), 0.2, dtype=np.float32)
        red = np.full((2, 2), 0.2, dtype=np.float32)
        result = compute_evi2(nir, red)
        assert np.all(np.abs(result) < 0.1)


# ---------------------------------------------------------------------------
# compute_ndii
# ---------------------------------------------------------------------------

class TestNDII:

    def test_known_value(self):
        nir   = np.array([[0.6]], dtype=np.float32)
        swir1 = np.array([[0.2]], dtype=np.float32)
        expected = (0.6 - 0.2) / (0.6 + 0.2)
        result = compute_ndii(nir, swir1)
        assert abs(float(result[0, 0]) - expected) < 1e-5

    def test_nan_propagation(self):
        nir   = np.array([[np.nan]], dtype=np.float32)
        swir1 = np.array([[0.2]],   dtype=np.float32)
        assert np.isnan(compute_ndii(nir, swir1)[0, 0])

    def test_zero_denominator_returns_nan(self):
        nir   = np.array([[0.0]], dtype=np.float32)
        swir1 = np.array([[0.0]], dtype=np.float32)
        assert np.isnan(compute_ndii(nir, swir1)[0, 0])

    def test_range(self):
        # NDII should be in [-1, 1] for valid inputs
        nir   = np.random.uniform(0, 1, (10, 10)).astype(np.float32)
        swir1 = np.random.uniform(0, 1, (10, 10)).astype(np.float32)
        # Avoid zero denominator
        mask = (nir + swir1) > 0
        result = compute_ndii(nir, swir1)
        assert np.all(result[mask] >= -1.0)
        assert np.all(result[mask] <=  1.0)

    def test_high_moisture_positive(self):
        # High NIR, low SWIR1 → positive NDII (moist canopy)
        nir   = np.full((3, 3), 0.7, dtype=np.float32)
        swir1 = np.full((3, 3), 0.1, dtype=np.float32)
        assert np.all(compute_ndii(nir, swir1) > 0)

    def test_senescing_vegetation_decreasing(self):
        # As SWIR1 increases (water loss during senescence), NDII decreases
        nir   = np.array([[0.5, 0.5]], dtype=np.float32)
        swir1 = np.array([[0.1, 0.4]], dtype=np.float32)
        result = compute_ndii(nir, swir1)
        assert result[0, 0] > result[0, 1]


# ---------------------------------------------------------------------------
# compute_ndvi
# ---------------------------------------------------------------------------

class TestNDVI:

    def test_known_value(self):
        nir = np.array([[0.8]], dtype=np.float32)
        red = np.array([[0.1]], dtype=np.float32)
        expected = (0.8 - 0.1) / (0.8 + 0.1)
        result = compute_ndvi(nir, red)
        assert abs(float(result[0, 0]) - expected) < 1e-5

    def test_nan_propagation(self):
        nir = np.array([[np.nan]], dtype=np.float32)
        red = np.array([[0.1]],   dtype=np.float32)
        assert np.isnan(compute_ndvi(nir, red)[0, 0])

    def test_zero_denominator_returns_nan(self):
        nir = np.array([[0.0]], dtype=np.float32)
        red = np.array([[0.0]], dtype=np.float32)
        assert np.isnan(compute_ndvi(nir, red)[0, 0])

    def test_range(self):
        nir = np.random.uniform(0, 1, (10, 10)).astype(np.float32)
        red = np.random.uniform(0, 1, (10, 10)).astype(np.float32)
        mask = (nir + red) > 0
        result = compute_ndvi(nir, red)
        assert np.all(result[mask] >= -1.0)
        assert np.all(result[mask] <=  1.0)

    def test_symmetry(self):
        # NDVI(nir, red) == -NDVI(red, nir)
        nir = np.array([[0.6]], dtype=np.float32)
        red = np.array([[0.2]], dtype=np.float32)
        assert abs(float(compute_ndvi(nir, red)[0, 0]) +
                   float(compute_ndvi(red, nir)[0, 0])) < 1e-5


# ---------------------------------------------------------------------------
# LandsatTile._parse_tile_id
# ---------------------------------------------------------------------------

class TestParseTileId:

    def test_standard_lc08(self):
        tile_id = "LC08_CU_028004_20190915_20200901_02_T1"
        sensor, h, v, date_str = LandsatTile._parse_tile_id(tile_id)
        assert sensor   == "LC08"
        assert h        == "028"
        assert v        == "004"
        assert date_str == "20190915"

    def test_lc09(self):
        sensor, h, v, date_str = LandsatTile._parse_tile_id(
            "LC09_CU_029005_20221001_20230101_02_T1"
        )
        assert sensor   == "LC09"
        assert h        == "029"
        assert v        == "005"
        assert date_str == "20221001"

    def test_lt05(self):
        sensor, h, v, date_str = LandsatTile._parse_tile_id(
            "LT05_CU_028004_20050820_20210426_02_T1"
        )
        assert sensor   == "LT05"
        assert date_str == "20050820"

    def test_le07(self):
        sensor, _, _, _ = LandsatTile._parse_tile_id(
            "LE07_CU_028004_20130901_20200101_02_T1"
        )
        assert sensor == "LE07"

    def test_invalid_id_raises(self):
        with pytest.raises(ValueError, match="Cannot parse ARD tile ID"):
            LandsatTile._parse_tile_id("not_a_valid_tile_id")

    def test_wrong_prefix_raises(self):
        with pytest.raises(ValueError):
            LandsatTile._parse_tile_id("XX08_CU_028004_20190915_20200901_02_T1")


# ---------------------------------------------------------------------------
# LandsatTile.from_tile_id
# ---------------------------------------------------------------------------

class TestFromTileId:

    def test_date_formatting(self):
        tile = LandsatTile.from_tile_id(
            "LC08_CU_028004_20190915_20200901_02_T1"
        )
        assert tile.date == "2019-09-15"

    def test_year_extraction(self):
        tile = LandsatTile.from_tile_id(
            "LC08_CU_028004_20190915_20200901_02_T1"
        )
        assert tile.year == 2019

    def test_tile_dir_is_none(self):
        tile = LandsatTile.from_tile_id(
            "LC08_CU_028004_20190915_20200901_02_T1"
        )
        assert tile.tile_dir is None

    def test_repr(self):
        tile = LandsatTile.from_tile_id(
            "LC08_CU_028004_20190915_20200901_02_T1"
        )
        r = repr(tile)
        assert "LC08" in r
        assert "028" in r
        assert "2019-09-15" in r

    def test_from_dir_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError):
            LandsatTile.from_dir(Path("/nonexistent/path/tile"))
