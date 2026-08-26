"""Tests for time_signature model and operations."""
from __future__ import annotations

import pytest

from mujik.time_signature.model import (
    TimeSignatureSegment,
    build_default_segments,
    find_segment_for_time,
)
from mujik.time_signature.operations import (
    change_time_signature_at_boundary,
    redraw_bars_under_new_time_signature,
)


class TestTimeSignatureSegment:
    def test_basic(self):
        s = TimeSignatureSegment(
            start_time=0.0, end_time=10.0,
            time_signature=(4, 4), confidence=0.9, source="manual",
        )
        assert s.duration() == 10.0

    def test_invalid_order(self):
        with pytest.raises(ValueError):
            TimeSignatureSegment(
                start_time=10.0, end_time=5.0,
                time_signature=(4, 4), confidence=0.9, source="manual",
            )

    def test_invalid_confidence(self):
        with pytest.raises(ValueError):
            TimeSignatureSegment(
                start_time=0.0, end_time=10.0,
                time_signature=(4, 4), confidence=1.5, source="manual",
            )

    def test_invalid_denominator(self):
        with pytest.raises(ValueError):
            TimeSignatureSegment(
                start_time=0.0, end_time=10.0,
                time_signature=(3, 5), confidence=0.9, source="manual",
            )

    def test_bar_duration_4_4_at_120bpm(self):
        s = TimeSignatureSegment(
            start_time=0.0, end_time=10.0,
            time_signature=(4, 4), confidence=1.0, source="manual",
        )
        # 4/4 @ 120bpm = 2 秒/小节
        assert abs(s.bar_duration_sec(120.0) - 2.0) < 1e-9

    def test_bar_duration_3_4_at_120bpm(self):
        s = TimeSignatureSegment(
            start_time=0.0, end_time=10.0,
            time_signature=(3, 4), confidence=1.0, source="manual",
        )
        # 3/4 @ 120bpm = 1.5 秒/小节
        assert abs(s.bar_duration_sec(120.0) - 1.5) < 1e-9

    def test_bar_duration_7_8_at_120bpm(self):
        s = TimeSignatureSegment(
            start_time=0.0, end_time=10.0,
            time_signature=(7, 8), confidence=1.0, source="manual",
        )
        # 7/8 @ 120bpm = 7 * 0.5 / 2 = 1.75 秒/小节
        assert abs(s.bar_duration_sec(120.0) - 1.75) < 1e-9


class TestBuildDefaultSegments:
    def test_4_4_default(self):
        segs = build_default_segments(100.0)
        assert len(segs) == 1
        assert segs[0].time_signature == (4, 4)
        assert segs[0].source == "default_4_4"
        assert segs[0].end_time == 100.0

    def test_custom_fallback(self):
        segs = build_default_segments(100.0, fallback=(3, 4))
        assert segs[0].time_signature == (3, 4)
        assert segs[0].source == "manual"

    def test_zero_duration_raises(self):
        with pytest.raises(ValueError):
            build_default_segments(0.0)


class TestFindSegmentForTime:
    def test_within_segment(self):
        s = TimeSignatureSegment(
            start_time=0.0, end_time=10.0,
            time_signature=(4, 4), confidence=1.0, source="manual",
        )
        assert find_segment_for_time([s], 5.0) is s

    def test_outside(self):
        s = TimeSignatureSegment(
            start_time=0.0, end_time=10.0,
            time_signature=(4, 4), confidence=1.0, source="manual",
        )
        assert find_segment_for_time([s], 15.0) is None


class TestRedrawUnderNewTimeSignature:
    def test_replaces_full_range(self):
        segs = [TimeSignatureSegment(
            start_time=0.0, end_time=20.0,
            time_signature=(4, 4), confidence=1.0, source="manual",
        )]
        out = redraw_bars_under_new_time_signature(segs, (5.0, 15.0), (7, 8))
        # 应有 3 段：0-5, 5-15 (7/8), 15-20
        assert len(out) == 3
        assert out[0].time_signature == (4, 4)
        assert out[1].time_signature == (7, 8)
        assert out[2].time_signature == (4, 4)
        assert out[1].source == "manual"

    def test_invalid_range(self):
        segs = [TimeSignatureSegment(
            start_time=0.0, end_time=20.0,
            time_signature=(4, 4), confidence=1.0, source="manual",
        )]
        with pytest.raises(ValueError):
            redraw_bars_under_new_time_signature(segs, (10.0, 5.0), (3, 4))


class TestChangeAtBoundary:
    def test_split_at_change_time(self):
        segs = [TimeSignatureSegment(
            start_time=0.0, end_time=20.0,
            time_signature=(4, 4), confidence=1.0, source="manual",
        )]
        out = change_time_signature_at_boundary(segs, 10.0, (7, 8))
        # 0-10 4/4, 10-20 7/8
        assert len(out) == 2
        assert out[0].time_signature == (4, 4)
        assert out[0].end_time == 10.0
        assert out[1].time_signature == (7, 8)
        assert out[1].start_time == 10.0

    def test_change_time_at_zero(self):
        segs = [TimeSignatureSegment(
            start_time=0.0, end_time=20.0,
            time_signature=(4, 4), confidence=1.0, source="manual",
        )]
        out = change_time_signature_at_boundary(segs, 0.0, (3, 4))
        assert out[0].time_signature == (3, 4)

    def test_change_time_after_all(self):
        segs = [TimeSignatureSegment(
            start_time=0.0, end_time=10.0,
            time_signature=(4, 4), confidence=1.0, source="manual",
        )]
        out = change_time_signature_at_boundary(segs, 20.0, (3, 4))
        # change_time 在所有段之后，原 0-10 4/4 段保留，不构造零长度段
        assert len(out) == 1
        assert out[0].time_signature == (4, 4)
        assert out[0].end_time == 10.0
