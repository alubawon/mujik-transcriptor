"""Tests for quantize/groove.py."""
from __future__ import annotations

import pytest

from mujik.quantize.groove import (
    DEFAULT_SWING_RATIO,
    groove_offset,
    groove_offset_seconds,
    is_16th_off_offbeat,
    is_offbeat_position,
)


class TestIsOffbeat:
    def test_8th_offbeat_in_8th_grid(self):
        # 8th grid (grid_resolution=8): offbeat at idx 4
        assert is_offbeat_position(4, grid_resolution=8) is True
        assert is_offbeat_position(0, grid_resolution=8) is False
        assert is_offbeat_position(2, grid_resolution=8) is False

    def test_8th_offbeat_in_16th_grid(self):
        # 16th grid: offbeat at idx 8
        assert is_offbeat_position(8, grid_resolution=16) is True
        assert is_offbeat_position(4, grid_resolution=16) is False


class TestIs16thOffOffbeat:
    def test_at_16th_offsets(self):
        # 16th grid: 8th offbeat=8, ±1 = 7 and 9
        assert is_16th_off_offbeat(7, grid_resolution=16) is True
        assert is_16th_off_offbeat(9, grid_resolution=16) is True
        assert is_16th_off_offbeat(8, grid_resolution=16) is False  # 8th offbeat itself
        assert is_16th_off_offbeat(0, grid_resolution=16) is False

    def test_grid_res_too_low(self):
        # 8th grid: 概念不适用
        assert is_16th_off_offbeat(3, grid_resolution=8) is False


class TestGrooveOffsetStraight:
    def test_straight_is_zero(self):
        # straight 模板任何位置都返回 0
        assert groove_offset(0.0, 0, 16, "straight") == 0.0
        assert groove_offset(0.5, 8, 16, "straight") == 0.0
        assert groove_offset(0.25, 4, 16, "straight") == 0.0


class TestGrooveOffsetSwing16:
    def test_at_downbeat_zero(self):
        # downbeat 不偏移
        assert groove_offset(0.0, 0, 16, "swing16", ratio=0.6) == 0.0

    def test_at_8th_offbeat_positive(self):
        # 8th offbeat 在 ratio=0.6 → 偏移 0.1 拍
        result = groove_offset(0.5, 8, 16, "swing16", ratio=0.6)
        assert result == pytest.approx(0.1)

    def test_at_8th_offbeat_default_ratio(self):
        # 默认 ratio=0.6 → 0.1 拍
        result = groove_offset(0.5, 8, 16, "swing16")
        assert result == pytest.approx(DEFAULT_SWING_RATIO - 0.5)

    def test_at_16th_off_offbeat_half_magnitude(self):
        # 16分 偏移位置 (idx 7, 9) → 0.05 拍（half of 0.1）
        result = groove_offset(0.4375, 7, 16, "swing16", ratio=0.6)
        assert result == pytest.approx(0.05)

    def test_straight_ratio_zero_offset(self):
        # ratio=0.5 即 straight → 0 偏移
        assert groove_offset(0.5, 8, 16, "swing16", ratio=0.5) == 0.0


class TestGrooveOffsetSeconds:
    def test_120bpm_8th_offbeat_swing(self):
        # 120bpm: 1 beat = 0.5s；0.1 beat = 0.05s
        result = groove_offset_seconds(0.5, 8, 16, "swing16", bpm=120.0, ratio=0.6)
        assert result == pytest.approx(0.05)

    def test_60bpm_8th_offbeat_swing(self):
        # 60bpm: 1 beat = 1.0s；0.1 beat = 0.1s
        result = groove_offset_seconds(0.5, 8, 16, "swing16", bpm=60.0, ratio=0.6)
        assert result == pytest.approx(0.1)

    def test_straight_zero_seconds(self):
        result = groove_offset_seconds(0.5, 8, 16, "straight", bpm=120.0)
        assert result == 0.0


class TestGrooveErrors:
    def test_unknown_template(self):
        with pytest.raises(ValueError, match="unknown groove template"):
            groove_offset(0.0, 0, 16, "shuffle")

    def test_invalid_ratio(self):
        with pytest.raises(ValueError, match="ratio"):
            groove_offset(0.0, 0, 16, "swing16", ratio=1.5)
        with pytest.raises(ValueError, match="ratio"):
            groove_offset(0.0, 0, 16, "swing16", ratio=-0.1)

    def test_invalid_grid_resolution(self):
        with pytest.raises(ValueError, match="grid_resolution"):
            groove_offset(0.0, 0, 0, "swing16")

    def test_zero_bpm_raises(self):
        with pytest.raises(ValueError, match="bpm"):
            groove_offset_seconds(0.0, 0, 16, "swing16", bpm=0)
