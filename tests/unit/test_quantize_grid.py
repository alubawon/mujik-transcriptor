"""Tests for quantize/grid.py."""
from __future__ import annotations

import pytest

from mujik.quantize.grid import (
    beat_duration_sec,
    beat_index_at_time,
    is_8th_offbeat_position,
    snap_to_grid,
    time_at_beat_index,
)
from mujik.time_signature.model import TimeSignatureSegment


def _seg(sig=(4, 4), start=0.0, end=10.0) -> TimeSignatureSegment:
    return TimeSignatureSegment(
        start_time=start,
        end_time=end,
        time_signature=sig,
        confidence=1.0,
        source="manual",
    )


class TestBeatDuration:
    def test_120bpm(self):
        assert beat_duration_sec(120.0) == pytest.approx(0.5)

    def test_60bpm(self):
        assert beat_duration_sec(60.0) == pytest.approx(1.0)

    def test_240bpm(self):
        assert beat_duration_sec(240.0) == pytest.approx(0.25)

    def test_invalid_bpm_raises(self):
        with pytest.raises(ValueError):
            beat_duration_sec(0)
        with pytest.raises(ValueError):
            beat_duration_sec(-1)


class TestBeatIndexAtTime:
    def test_at_segment_start(self):
        seg = _seg(start=0.0)
        assert beat_index_at_time(0.0, seg, 120.0) == 0.0

    def test_one_beat_in(self):
        seg = _seg(start=0.0)
        # 120 bpm = 0.5s per beat
        assert beat_index_at_time(0.5, seg, 120.0) == pytest.approx(1.0)

    def test_subdivision(self):
        seg = _seg(start=0.0)
        # 0.25s at 120bpm = half beat
        assert beat_index_at_time(0.25, seg, 120.0) == pytest.approx(0.5)

    def test_before_segment_clamps_to_zero(self):
        seg = _seg(start=2.0)
        assert beat_index_at_time(1.0, seg, 120.0) == 0.0

    def test_past_segment_clamps(self):
        seg = _seg(start=0.0, end=5.0)  # 10 beats at 120bpm
        result = beat_index_at_time(100.0, seg, 120.0)
        # 5.0s at 120bpm = 10.0 beats
        assert result == pytest.approx(10.0)

    def test_offset_segment(self):
        seg = _seg(start=10.0, end=20.0)
        # 0.5s after segment start = 1 beat
        assert beat_index_at_time(10.5, seg, 120.0) == pytest.approx(1.0)


class TestTimeAtBeatIndex:
    def test_zero(self):
        seg = _seg(start=0.0)
        assert time_at_beat_index(0.0, seg, 120.0) == 0.0

    def test_one_beat(self):
        seg = _seg(start=0.0)
        assert time_at_beat_index(1.0, seg, 120.0) == pytest.approx(0.5)

    def test_offset_segment(self):
        seg = _seg(start=10.0, end=20.0)
        assert time_at_beat_index(2.0, seg, 120.0) == pytest.approx(11.0)

    def test_inverse_of_beat_index(self):
        seg = _seg(start=5.0, end=15.0)
        t = 7.25
        bi = beat_index_at_time(t, seg, 120.0)
        recovered = time_at_beat_index(bi, seg, 120.0)
        assert recovered == pytest.approx(7.25)


class TestSnapToGrid:
    def test_already_on_grid(self):
        """已在 grid 点上：不动。"""
        seg = _seg(sig=(4, 4), start=0.0, end=10.0)
        # 0.25s = 0.5 beat = 8th note in 4/4 at 120bpm
        assert snap_to_grid(0.25, seg, 120.0, grid_resolution=8) == pytest.approx(0.25)

    def test_snap_to_nearest_16th(self):
        seg = _seg(sig=(4, 4), start=0.0, end=10.0)
        # 0.123s at 120bpm = 0.246 beat → 16th 索引 3.94 → round 4 → 0.25 beat → 0.125s
        snapped = snap_to_grid(0.123, seg, 120.0, grid_resolution=16)
        assert snapped == pytest.approx(0.125, abs=1e-6)

    def test_snap_to_8th(self):
        seg = _seg(sig=(4, 4), start=0.0, end=10.0)
        # 0.20s at 120bpm = 0.4 beat → 8th 索引 3.2 → round 3 → 0.375 beat → 0.1875s
        snapped = snap_to_grid(0.20, seg, 120.0, grid_resolution=8)
        assert snapped == pytest.approx(0.1875, abs=1e-6)

    def test_snap_to_32nd(self):
        seg = _seg(sig=(4, 4), start=0.0, end=10.0)
        # 0.123s at 120bpm: 16th = 0.125s, 32nd = 0.0625 or 0.1875
        # 0.123/0.5 * 32 = 7.872 → round 8 → 0.125s
        snapped = snap_to_grid(0.123, seg, 120.0, grid_resolution=32)
        assert snapped == pytest.approx(0.125, abs=1e-6)

    def test_3_4_grid_uses_quarter_beat(self):
        """3/4 grid 仍按 quarter note 划分（grid_resolution 是每拍）。"""
        seg = _seg(sig=(3, 4), start=0.0, end=10.0)
        # 8th note in 3/4 = 0.25s at 120bpm
        snapped = snap_to_grid(0.24, seg, 120.0, grid_resolution=8)
        assert snapped == pytest.approx(0.25, abs=1e-6)

    def test_6_8_compound(self):
        """6/8 在 120bpm 下，beat 仍按 quarter note 计算（0.5s）。"""
        seg = _seg(sig=(6, 8), start=0.0, end=10.0)
        # 0.20s at 120bpm = 0.4 beat → 8th 索引 3.2 → round 3 → 0.375 beat → 0.1875s
        snapped = snap_to_grid(0.20, seg, 120.0, grid_resolution=8)
        assert snapped == pytest.approx(0.1875, abs=1e-6)

    def test_before_segment_snaps_to_start(self):
        seg = _seg(start=5.0, end=15.0)
        snapped = snap_to_grid(0.0, seg, 120.0, grid_resolution=8)
        assert snapped == 5.0

    def test_after_segment_snaps_inside(self):
        seg = _seg(start=0.0, end=5.0)
        # t = 100s, 端 5s = 10 beats at 120bpm → snap 到 5.0
        snapped = snap_to_grid(100.0, seg, 120.0, grid_resolution=16)
        # 端值 5.0 = beat 10.0 → 16th idx 160
        assert snapped == pytest.approx(5.0, abs=1e-6)

    def test_zero_grid_resolution_raises(self):
        with pytest.raises(ValueError):
            snap_to_grid(1.0, _seg(), 120.0, grid_resolution=0)


class TestIs8thOffbeat:
    def test_at_offbeat(self):
        # 0.5 beat = 8th offbeat
        assert is_8th_offbeat_position(0.5, grid_resolution=8) is True

    def test_at_downbeat(self):
        assert is_8th_offbeat_position(0.0, grid_resolution=8) is False

    def test_at_16th_off_offbeat(self):
        # 0.75 beat = 16th position 12 of 16 → not at 8th offbeat (idx 4)
        assert is_8th_offbeat_position(0.75, grid_resolution=16) is False

    def test_grid_res_too_low(self):
        # grid_resolution < 8: 8 分概念不适用
        assert is_8th_offbeat_position(0.5, grid_resolution=4) is False
