"""Tests for rhythm.model.BeatTrack."""
from __future__ import annotations

import pytest

from mujik.rhythm.model import BeatTrack


class TestBeatTrack:
    def test_default(self):
        bt = BeatTrack()
        assert bt.beats == []
        assert bt.downbeats == []
        assert bt.bpm == 120.0
        assert bt.tempo_confidence == 0.0
        assert bt.beat_count == 0

    def test_basic(self):
        bt = BeatTrack(
            beats=[0.0, 0.5, 1.0, 1.5, 2.0],
            downbeats=[0.0, 2.0],
            bpm=120.0,
            tempo_confidence=0.95,
        )
        assert bt.beat_count == 5
        assert bt.duration() == 2.0

    def test_invalid_bpm(self):
        with pytest.raises(ValueError):
            BeatTrack(bpm=0.0)
        with pytest.raises(ValueError):
            BeatTrack(bpm=-10.0)

    def test_invalid_confidence(self):
        with pytest.raises(ValueError):
            BeatTrack(tempo_confidence=1.5)
        with pytest.raises(ValueError):
            BeatTrack(tempo_confidence=-0.1)

    def test_to_dict(self):
        bt = BeatTrack(beats=[0.0, 0.5], bpm=120.0, tempo_confidence=0.8)
        d = bt.to_dict()
        assert d["beats"] == [0.0, 0.5]
        assert d["bpm"] == 120.0
        assert d["tempo_confidence"] == 0.8

    def test_empty_duration(self):
        assert BeatTrack().duration() == 0.0
