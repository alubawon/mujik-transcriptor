"""Tests for midi.model.Note / Track / Project."""
from __future__ import annotations

import pytest

from mujik.midi.model import Note, Track, TempoSegment, ChordEvent, Project
from mujik.time_signature.model import build_default_segments


class TestNote:
    def test_basic(self):
        n = Note(start=0.0, end=1.0, pitch=60, velocity=100)
        assert n.duration() == 1.0
        assert n.pitch == 60
        assert n.channel == 0

    def test_invalid_pitch(self):
        with pytest.raises(ValueError):
            Note(start=0.0, end=1.0, pitch=128, velocity=100)

    def test_invalid_velocity(self):
        with pytest.raises(ValueError):
            Note(start=0.0, end=1.0, pitch=60, velocity=128)

    def test_end_before_start(self):
        with pytest.raises(ValueError):
            Note(start=1.0, end=0.5, pitch=60, velocity=100)

    def test_pitch_bend_range(self):
        n = Note(start=0.0, end=1.0, pitch=60, velocity=100, pitch_bend=(0.0, 0.5, -0.5))
        assert len(n.pitch_bend) == 3

    def test_pitch_bend_out_of_range(self):
        with pytest.raises(ValueError):
            Note(start=0.0, end=1.0, pitch=60, velocity=100, pitch_bend=(2.0,))

    def test_articulation(self):
        n = Note(start=0.0, end=1.0, pitch=60, velocity=100, articulation="slide")
        assert n.articulation == "slide"


class TestTrack:
    def test_add_and_sort(self):
        t = Track(stem_name="vocals")
        t.add(Note(start=1.0, end=2.0, pitch=60, velocity=100))
        t.add(Note(start=0.0, end=1.0, pitch=62, velocity=100))
        t.add(Note(start=0.5, end=1.5, pitch=64, velocity=80))
        t.sort_by_start()
        assert [n.pitch for n in t.notes] == [62, 64, 60]

    def test_duration(self):
        t = Track(stem_name="vocals")
        t.add(Note(start=0.0, end=1.0, pitch=60, velocity=100))
        t.add(Note(start=2.0, end=5.0, pitch=62, velocity=100))
        assert t.duration() == 5.0

    def test_empty(self):
        t = Track(stem_name="drums")
        assert t.duration() == 0.0


class TestTempoSegment:
    def test_basic(self):
        ts = TempoSegment(start_time=0.0, end_time=10.0, bpm=120.0)
        assert ts.bpm == 120.0

    def test_invalid_bpm(self):
        with pytest.raises(ValueError):
            TempoSegment(start_time=0.0, end_time=10.0, bpm=0.0)
        with pytest.raises(ValueError):
            TempoSegment(start_time=0.0, end_time=10.0, bpm=1000.0)


class TestProject:
    def test_get_track_creates(self):
        p = Project(
            audio_path="song.wav",
            duration=100.0,
            sample_rate=44100,
            time_signatures=build_default_segments(100.0),
            tempo_map=[TempoSegment(0.0, 100.0, 120.0)],
        )
        t = p.get_track("vocals")
        assert t.stem_name == "vocals"
        assert "vocals" in p.tracks

    def test_total_notes(self):
        p = Project(
            audio_path="song.wav",
            duration=100.0,
            sample_rate=44100,
            time_signatures=build_default_segments(100.0),
            tempo_map=[TempoSegment(0.0, 100.0, 120.0)],
        )
        p.get_track("vocals").add(Note(start=0.0, end=1.0, pitch=60, velocity=100))
        p.get_track("bass").add(Note(start=0.0, end=1.0, pitch=40, velocity=100))
        p.get_track("bass").add(Note(start=1.0, end=2.0, pitch=42, velocity=100))
        assert p.total_notes() == 3
