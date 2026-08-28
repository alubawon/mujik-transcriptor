"""Tests for postprocess/pitch_bend.py."""
from __future__ import annotations

import numpy as np
import pytest

from mujik.midi.model import Note
from mujik.postprocess.pitch_bend import (
    DEFAULT_FRAME_RATE_HZ,
    PITCH_BEND_CENTER,
    PITCH_BEND_MAX,
    bend_to_pretty_pitch,
    extract_pitch_bends_from_pretty_midi,
    inject_pitch_bends_to_pretty_midi,
    pretty_pitch_to_bend,
)


def _mock_instrument(name: str = "test"):
    """构造一个 minimal pretty_midi.Instrument mock。"""
    import pretty_midi as pm

    inst = pm.Instrument(program=0, name=name, is_drum=False)
    return inst


class TestBendConversion:
    def test_zero_bend_center(self):
        assert bend_to_pretty_pitch(0.0) == 8192  # PITCH_BEND_CENTER

    def test_positive_full_bend(self):
        assert bend_to_pretty_pitch(1.0) == 16383

    def test_negative_full_bend(self):
        assert bend_to_pretty_pitch(-1.0) == 0

    def test_clamp_high(self):
        assert bend_to_pretty_pitch(1.5) == 16383

    def test_clamp_low(self):
        assert bend_to_pretty_pitch(-1.5) == 0

    def test_round_trip(self):
        for v in (-1.0, -0.5, 0.0, 0.5, 1.0):
            pretty = bend_to_pretty_pitch(v)
            recovered = pretty_pitch_to_bend(pretty)
            assert recovered == pytest.approx(v, abs=0.001)

    def test_pretty_pitch_to_bend(self):
        assert pretty_pitch_to_bend(0) == -1.0
        assert pretty_pitch_to_bend(16383) == 1.0
        # 8192 接近中心，但 int 舍入后非完全 0；用 approx
        assert pretty_pitch_to_bend(8192) == pytest.approx(0.0, abs=1e-3)


class TestInjectPitchBends:
    def test_no_bend_no_events(self):
        inst = _mock_instrument()
        notes = [Note(0.0, 1.0, 60, 100)]
        n = inject_pitch_bends_to_pretty_midi(inst, notes)
        assert n == 0
        assert len(inst.pitch_bends) == 0

    def test_single_note_with_bend(self):
        inst = _mock_instrument()
        # 100 frames, 1 second, 100 fps
        bend_seq = tuple(0.0 for _ in range(100))  # 无实际弯音
        notes = [Note(0.0, 1.0, 60, 100, pitch_bend=bend_seq)]
        n = inject_pitch_bends_to_pretty_midi(inst, notes)
        assert n == 100
        assert len(inst.pitch_bends) == 100

    def test_events_sorted_by_time(self):
        inst = _mock_instrument()
        bend_seq = (0.0,) * 50 + (0.5,) * 50
        notes = [Note(0.0, 1.0, 60, 100, pitch_bend=bend_seq)]
        n = inject_pitch_bends_to_pretty_midi(inst, notes)
        assert n == 100
        times = [pb.time for pb in inst.pitch_bends]
        assert times == sorted(times)

    def test_bend_values_in_pretty_range(self):
        inst = _mock_instrument()
        bend_seq = (-1.0, -0.5, 0.0, 0.5, 1.0)
        notes = [Note(0.0, 0.05, 60, 100, pitch_bend=bend_seq)]
        inject_pitch_bends_to_pretty_midi(inst, notes)
        for pb in inst.pitch_bends:
            assert 0 <= pb.pitch <= 16383

    def test_multiple_notes(self):
        inst = _mock_instrument()
        notes = [
            Note(0.0, 0.5, 60, 100, pitch_bend=(0.0,) * 50),
            Note(0.5, 1.0, 62, 90, pitch_bend=(0.5,) * 50),
        ]
        n = inject_pitch_bends_to_pretty_midi(inst, notes)
        assert n == 100

    def test_zero_length_note_skipped(self):
        inst = _mock_instrument()
        notes = [Note(0.0, 0.0, 60, 100, pitch_bend=(0.0,) * 10)]
        n = inject_pitch_bends_to_pretty_midi(inst, notes)
        assert n == 0

    def test_invalid_frame_rate(self):
        inst = _mock_instrument()
        with pytest.raises(ValueError):
            inject_pitch_bends_to_pretty_midi(
                _mock_instrument(), [], frame_rate_hz=0
            )


class TestExtractPitchBends:
    def test_empty_instrument(self):
        notes = [Note(0.0, 1.0, 60, 100)]
        result = extract_pitch_bends_from_pretty_midi(_mock_instrument(), notes)
        assert result == notes
        assert result[0].pitch_bend == ()

    def test_round_trip(self):
        inst = _mock_instrument()
        original = [
            Note(0.0, 1.0, 60, 100, pitch_bend=(0.0, 0.5, 1.0, 0.5, 0.0)),
            Note(1.0, 2.0, 62, 90, pitch_bend=(0.0, -0.5, -1.0, -0.5, 0.0)),
        ]
        inject_pitch_bends_to_pretty_midi(inst, original)
        extracted = extract_pitch_bends_from_pretty_midi(inst, original)

        assert len(extracted) == 2
        # 提取的 bend 数量应等于帧率 × duration
        for orig, ext in zip(original, extracted):
            assert len(ext.pitch_bend) == DEFAULT_FRAME_RATE_HZ  # 100 帧/秒

    def test_preserves_other_fields(self):
        inst = _mock_instrument()
        original = [Note(0.0, 1.0, 60, 100, articulation="slide", pitch_bend=(0.0,) * 5)]
        inject_pitch_bends_to_pretty_midi(inst, original)
        extracted = extract_pitch_bends_from_pretty_midi(inst, original)
        assert extracted[0].articulation == "slide"
        assert extracted[0].velocity == 100
        assert extracted[0].pitch == 60
