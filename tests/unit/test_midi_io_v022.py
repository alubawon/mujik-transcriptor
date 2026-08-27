"""Tests for midi/io v0.2.2 multi-segment tempo."""
from __future__ import annotations

from pathlib import Path

import pytest

from mujik.midi.io import read_midi_to_project, write_project_to_midi
from mujik.midi.model import Note, Project, TempoSegment
from mujik.time_signature.model import build_default_segments


def _make_project_with_tempos(tempos: list[TempoSegment], duration: float = 10.0) -> Project:
    return Project(
        audio_path="song.wav",
        duration=duration,
        sample_rate=44100,
        time_signatures=build_default_segments(duration),
        tempo_map=tempos,
    )


class TestMultiSegmentTempo:
    def test_single_tempo(self, tmp_path: Path):
        """单段 tempo 仍可正确写读。"""
        p = _make_project_with_tempos([TempoSegment(0.0, 10.0, 120.0)])
        p.get_track("vocals").add(Note(0.0, 1.0, 60, 100))

        midi_path = tmp_path / "out.mid"
        write_project_to_midi(p, midi_path)
        loaded = read_midi_to_project(midi_path)

        # 第一段 BPM 保留
        assert loaded.tempo_map[0].bpm == pytest.approx(120.0, abs=0.1)

    def test_two_tempos_preserved(self, tmp_path: Path):
        """两段 tempo 写入后，pretty-midi 至少识别第一段。"""
        p = _make_project_with_tempos(
            [TempoSegment(0.0, 5.0, 120.0), TempoSegment(5.0, 10.0, 140.0)],
            duration=10.0,
        )
        p.get_track("vocals").add(Note(0.0, 1.0, 60, 100))

        midi_path = tmp_path / "out.mid"
        write_project_to_midi(p, midi_path)
        loaded = read_midi_to_project(midi_path)

        # 至少第一段 BPM 在
        assert len(loaded.tempo_map) >= 1
        assert loaded.tempo_map[0].bpm == pytest.approx(120.0, abs=0.1)

    def test_three_tempos_does_not_crash(self, tmp_path: Path):
        p = _make_project_with_tempos([
            TempoSegment(0.0, 3.0, 100.0),
            TempoSegment(3.0, 6.0, 120.0),
            TempoSegment(6.0, 10.0, 90.0),
        ])
        p.get_track("vocals").add(Note(0.0, 1.0, 60, 100))

        midi_path = tmp_path / "out.mid"
        write_project_to_midi(p, midi_path)
        assert midi_path.exists()


class TestEdgeCases:
    def test_empty_tempo_map(self, tmp_path: Path):
        p = _make_project_with_tempos([], duration=5.0)
        p.get_track("vocals").add(Note(0.0, 1.0, 60, 100))
        midi_path = tmp_path / "out.mid"
        write_project_to_midi(p, midi_path)
        loaded = read_midi_to_project(midi_path)
        # 默认 120
        assert loaded.tempo_map[0].bpm == pytest.approx(120.0, abs=0.1)

    def test_high_bpm(self, tmp_path: Path):
        """极高 BPM 也应正确处理。"""
        p = _make_project_with_tempos([TempoSegment(0.0, 5.0, 240.0)])
        p.get_track("vocals").add(Note(0.0, 0.1, 60, 100))
        midi_path = tmp_path / "out.mid"
        write_project_to_midi(p, midi_path)
        loaded = read_midi_to_project(midi_path)
        # 240 在合理范围
        assert loaded.tempo_map[0].bpm == pytest.approx(240.0, abs=1.0)
