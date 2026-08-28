"""Tests for score/time_helpers.py."""
from __future__ import annotations

import pytest

from mujik.midi.model import TempoSegment
from mujik.score.time_helpers import (
    bpm_at_time,
    measure_index_at_time,
    seconds_to_ticks,
    time_signature_at_time,
)
from mujik.time_signature.model import TimeSignatureSegment


def _ts(start: float, end: float, sig=(4, 4)) -> TimeSignatureSegment:
    return TimeSignatureSegment(
        start_time=start, end_time=end,
        time_signature=sig, confidence=1.0, source="manual",
    )


class TestSecondsToTicks:
    def test_zero(self):
        seg = _ts(0.0, 10.0)
        assert seconds_to_ticks(0.0, seg, 120.0) == 0

    def test_one_beat_at_120bpm(self):
        # 0.5s = 1 beat = 480 ticks (PPQ=480)
        seg = _ts(0.0, 10.0)
        assert seconds_to_ticks(0.5, seg, 120.0, ppq=480) == 480

    def test_two_beats_at_120bpm(self):
        # 1.0s = 2 beats = 960 ticks
        seg = _ts(0.0, 10.0)
        assert seconds_to_ticks(1.0, seg, 120.0, ppq=480) == 960

    def test_different_ppq(self):
        # 0.5s = 1 beat at 120bpm
        # PPQ=960 → 960 ticks
        seg = _ts(0.0, 10.0)
        assert seconds_to_ticks(0.5, seg, 120.0, ppq=960) == 960

    def test_different_bpm(self):
        # 0.5s at 60bpm = 0.5 beat = 240 ticks
        seg = _ts(0.0, 10.0)
        assert seconds_to_ticks(0.5, seg, 60.0, ppq=480) == 240

    def test_invalid_bpm_raises(self):
        seg = _ts(0.0, 10.0)
        with pytest.raises(ValueError):
            seconds_to_ticks(0.0, seg, 0)
        with pytest.raises(ValueError):
            seconds_to_ticks(0.0, seg, -1)

    def test_invalid_ppq_raises(self):
        seg = _ts(0.0, 10.0)
        with pytest.raises(ValueError):
            seconds_to_ticks(0.0, seg, 120.0, ppq=0)


class TestBpmAtTime:
    def test_single_segment(self):
        tempo = [TempoSegment(0.0, 10.0, 120.0)]
        assert bpm_at_time(0.0, tempo) == 120.0
        assert bpm_at_time(5.0, tempo) == 120.0
        assert bpm_at_time(9.99, tempo) == 120.0

    def test_multiple_segments(self):
        tempo = [
            TempoSegment(0.0, 5.0, 120.0),
            TempoSegment(5.0, 10.0, 140.0),
        ]
        assert bpm_at_time(2.0, tempo) == 120.0
        assert bpm_at_time(7.0, tempo) == 140.0

    def test_past_last_segment_uses_last(self):
        tempo = [TempoSegment(0.0, 5.0, 120.0)]
        assert bpm_at_time(100.0, tempo) == 120.0

    def test_empty_tempo_uses_default(self):
        assert bpm_at_time(0.0, []) == 120.0
        assert bpm_at_time(0.0, [], default=80.0) == 80.0


class TestTimeSignatureAtTime:
    def test_in_segment(self):
        segs = [_ts(0.0, 5.0), _ts(5.0, 10.0, sig=(3, 4))]
        s1 = time_signature_at_time(2.0, segs)
        assert s1 is not None
        assert s1.time_signature == (4, 4)
        s2 = time_signature_at_time(7.0, segs)
        assert s2 is not None
        assert s2.time_signature == (3, 4)

    def test_outside_returns_none(self):
        segs = [_ts(0.0, 5.0)]
        assert time_signature_at_time(10.0, segs) is None


class TestMeasureIndexAtTime:
    def test_first_measure(self):
        seg = _ts(0.0, 10.0, sig=(4, 4))
        # 0.5s at 120bpm = bar 0 (4/4 = 2s per bar)
        assert measure_index_at_time(0.5, seg, 120.0) == 0

    def test_second_measure(self):
        seg = _ts(0.0, 10.0, sig=(4, 4))
        # 2.5s → bar 1
        assert measure_index_at_time(2.5, seg, 120.0) == 1

    def test_third_measure(self):
        seg = _ts(0.0, 10.0, sig=(4, 4))
        # 4.5s → bar 2
        assert measure_index_at_time(4.5, seg, 120.0) == 2

    def test_offset_segment(self):
        seg = _ts(10.0, 20.0, sig=(4, 4))
        # 10.5s relative to start = 0.5s → bar 0
        assert measure_index_at_time(10.5, seg, 120.0) == 0
        # 12.5s relative = 2.5s → bar 1
        assert measure_index_at_time(12.5, seg, 120.0) == 1

    def test_3_4_bar(self):
        seg = _ts(0.0, 10.0, sig=(3, 4))
        # 3/4 at 120bpm = 1.5s per bar
        assert measure_index_at_time(0.5, seg, 120.0) == 0
        assert measure_index_at_time(1.5, seg, 120.0) == 1
        assert measure_index_at_time(3.5, seg, 120.0) == 2

    def test_clamp_before_segment(self):
        seg = _ts(5.0, 15.0)
        assert measure_index_at_time(0.0, seg, 120.0) == 0
