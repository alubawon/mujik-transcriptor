"""Tests for benchmarks/metrics.py (v0.5.0)."""
from __future__ import annotations

import pytest

from mujik.benchmarks.metrics import (
    BeatTrackingMetrics,
    ChordRecognitionMetrics,
    NoteTranscriptionMetrics,
)


class TestNoteTranscription:
    def test_perfect_match(self):
        notes = [(60, 0.0, 0.5), (62, 0.5, 1.0)]
        calc = NoteTranscriptionMetrics()
        result = calc.compute(
            {"notes": notes},
            {"notes": notes},
        )
        assert result["f1"] == 1.0
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0

    def test_no_pred(self):
        calc = NoteTranscriptionMetrics()
        result = calc.compute(
            {"notes": []},
            {"notes": [(60, 0.0, 0.5)]},
        )
        assert result["f1"] == 0.0
        assert result["precision"] == 0.0
        assert result["recall"] == 0.0

    def test_no_gt(self):
        calc = NoteTranscriptionMetrics()
        result = calc.compute(
            {"notes": [(60, 0.0, 0.5)]},
            {"notes": []},
        )
        assert result["f1"] == 0.0

    def test_both_empty(self):
        calc = NoteTranscriptionMetrics()
        result = calc.compute({"notes": []}, {"notes": []})
        assert result["f1"] == 1.0  # 视为完美

    def test_onset_tolerance_50ms(self):
        """v0.5.0: onset 偏差 50ms 内算 TP。"""
        calc = NoteTranscriptionMetrics()
        result = calc.compute(
            {"notes": [(60, 0.03, 0.5)]},  # onset 偏 30ms
            {"notes": [(60, 0.0, 0.5)]},
        )
        assert result["f1"] == 1.0  # 30ms 在 ±50ms 内

    def test_onset_outside_tolerance(self):
        calc = NoteTranscriptionMetrics()
        result = calc.compute(
            {"notes": [(60, 0.1, 0.5)]},  # onset 偏 100ms
            {"notes": [(60, 0.0, 0.5)]},
        )
        assert result["f1"] == 0.0

    def test_pitch_mismatch(self):
        calc = NoteTranscriptionMetrics()
        result = calc.compute(
            {"notes": [(62, 0.0, 0.5)]},  # pitch 不同
            {"notes": [(60, 0.0, 0.5)]},
        )
        assert result["f1"] == 0.0


class TestBeatTracking:
    def test_perfect_match(self):
        beats = [0.0, 0.5, 1.0, 1.5]
        calc = BeatTrackingMetrics()
        result = calc.compute(
            {"beats": beats},
            {"beats": beats},
        )
        assert "cmlt" in result
        assert "amlt" in result
        assert "n_pred" in result
        assert "n_gt" in result

    def test_empty_gt(self):
        calc = BeatTrackingMetrics()
        result = calc.compute(
            {"beats": [0.0, 0.5]},
            {"beats": []},
        )
        # 空 ground truth → metrics 0
        assert result["cmlt"] == 0.0
        assert result["amlt"] == 0.0


class TestChordRecognition:
    def test_perfect_match(self):
        chords = [(0.0, 2.0, "C", ""), (2.0, 4.0, "F", "")]
        calc = ChordRecognitionMetrics()
        result = calc.compute(
            {"chords": chords},
            {"chords": chords},
        )
        assert "majmin" in result
        assert "root" in result
        assert "sevenths" in result

    def test_empty_gt(self):
        calc = ChordRecognitionMetrics()
        result = calc.compute(
            {"chords": [(0.0, 2.0, "C", "")]},
            {"chords": []},
        )
        assert result["majmin"] == 0.0
        assert result["root"] == 0.0
