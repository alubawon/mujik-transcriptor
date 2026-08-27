"""Tests for quantize/core.py."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from mujik.config.schema import QuantizeConfig
from mujik.midi.model import Note, Project, TempoSegment, Track
from mujik.quantize.core import (
    QuantizeReport,
    TrackQuantizeStats,
    load_beat_track_from_json,
    quantize_project,
    quantize_track,
    write_quantize_report,
)
from mujik.rhythm.model import BeatTrack
from mujik.time_signature.model import TimeSignatureSegment


def _seg(sig=(4, 4), start=0.0, end=20.0) -> TimeSignatureSegment:
    return TimeSignatureSegment(
        start_time=start,
        end_time=end,
        time_signature=sig,
        confidence=1.0,
        source="manual",
    )


def _track(notes: list[Note], stem="vocals") -> Track:
    t = Track(stem_name=stem)  # type: ignore[arg-type]
    for n in notes:
        t.add(n)
    return t


def _beat(bpm=120.0) -> BeatTrack:
    return BeatTrack(
        beats=[0.0, 0.5, 1.0, 1.5, 2.0],
        downbeats=[0.0, 2.0],
        bpm=bpm,
    )


class TestQuantizeTrack:
    def test_strength_zero_no_change(self):
        """strength=0 → note 时间完全不变。"""
        notes = [Note(0.123, 0.5, 60, 100), Note(0.987, 1.5, 62, 90)]
        track = _track(notes)
        cfg = QuantizeConfig(enabled=True, strength=0.0, grid_resolution=16)

        new_track, stats = quantize_track(track, _beat(120.0), [_seg()], cfg, duration=10.0)

        assert len(new_track.notes) == 2
        # 量化器仍 snap，但不应用
        for orig, new in zip(notes, new_track.notes):
            assert new.start == pytest.approx(orig.start, abs=1e-9)
            assert new.end == pytest.approx(orig.end, abs=1e-9)
        assert stats.mean_shift_ms == pytest.approx(0.0, abs=1e-6)

    def test_strength_one_full_snap(self):
        """strength=1 → 完全 snap 到最近 16 分。"""
        notes = [Note(0.123, 0.5, 60, 100)]
        track = _track(notes)
        cfg = QuantizeConfig(enabled=True, strength=1.0, grid_resolution=16)

        new_track, _ = quantize_track(track, _beat(120.0), [_seg()], cfg, duration=10.0)

        # 0.123s at 120bpm, 16th grid: snap 到 0.125
        assert new_track.notes[0].start == pytest.approx(0.125, abs=1e-6)
        # end 0.5s 已在 grid 上不动
        assert new_track.notes[0].end == pytest.approx(0.5, abs=1e-6)

    def test_strength_partial(self):
        """strength=0.5 → 半数移动。"""
        notes = [Note(0.123, 0.5, 60, 100)]
        track = _track(notes)
        cfg = QuantizeConfig(enabled=True, strength=0.5, grid_resolution=16)

        new_track, _ = quantize_track(track, _beat(120.0), [_seg()], cfg, duration=10.0)

        # snapped=0.125, orig=0.123, strength=0.5 → 0.123 + 0.5*(0.125-0.123) = 0.124
        assert new_track.notes[0].start == pytest.approx(0.124, abs=1e-6)

    def test_velocity_unchanged(self):
        notes = [Note(0.123, 0.5, 60, 100)]
        track = _track(notes)
        cfg = QuantizeConfig(enabled=True, strength=1.0, grid_resolution=16)
        new_track, _ = quantize_track(track, _beat(120.0), [_seg()], cfg, duration=10.0)
        assert new_track.notes[0].velocity == 100

    def test_pitch_unchanged(self):
        notes = [Note(0.123, 0.5, 60, 100)]
        track = _track(notes)
        cfg = QuantizeConfig(enabled=True, strength=1.0, grid_resolution=16)
        new_track, _ = quantize_track(track, _beat(120.0), [_seg()], cfg, duration=10.0)
        assert new_track.notes[0].pitch == 60

    def test_pitch_bend_unchanged(self):
        notes = [Note(0.123, 0.5, 60, 100, pitch_bend=(0.1, 0.2, 0.3))]
        track = _track(notes)
        cfg = QuantizeConfig(enabled=True, strength=1.0, grid_resolution=16)
        new_track, _ = quantize_track(track, _beat(120.0), [_seg()], cfg, duration=10.0)
        assert new_track.notes[0].pitch_bend == (0.1, 0.2, 0.3)

    def test_articulation_preserved(self):
        notes = [Note(0.123, 0.5, 60, 100, articulation="slide")]
        track = _track(notes)
        cfg = QuantizeConfig(enabled=True, strength=1.0, grid_resolution=16)
        new_track, _ = quantize_track(track, _beat(120.0), [_seg()], cfg, duration=10.0)
        assert new_track.notes[0].articulation == "slide"

    def test_swing16_shifts_8th_offbeat(self):
        notes = [Note(0.25, 0.7, 60, 100)]  # 8th offbeat at 120bpm
        track = _track(notes)
        cfg = QuantizeConfig(
            enabled=True, strength=1.0, grid_resolution=16,
            groove_template="swing16",
        )
        new_track, _ = quantize_track(track, _beat(120.0), [_seg()], cfg, duration=10.0)
        # snapped 0.25 + swing offset 0.1 beat * 0.5s = 0.05s → 0.30s
        assert new_track.notes[0].start == pytest.approx(0.30, abs=1e-6)

    def test_straight_groove_no_shift(self):
        notes = [Note(0.5, 0.7, 60, 100)]
        track = _track(notes)
        cfg = QuantizeConfig(
            enabled=True, strength=1.0, grid_resolution=16,
            groove_template="straight",
        )
        new_track, _ = quantize_track(track, _beat(120.0), [_seg()], cfg, duration=10.0)
        # straight：offbeat snap 后不动
        assert new_track.notes[0].start == pytest.approx(0.5, abs=1e-6)

    def test_empty_track(self):
        track = _track([], stem="vocals")
        cfg = QuantizeConfig(enabled=True, strength=1.0, grid_resolution=16)
        new_track, stats = quantize_track(track, _beat(120.0), [_seg()], cfg, duration=10.0)
        assert new_track.notes == []
        assert stats.notes_before == 0
        assert stats.notes_after == 0
        assert stats.mean_shift_ms == 0.0

    def test_stats_recorded(self):
        notes = [Note(0.123, 0.5, 60, 100), Note(0.487, 0.9, 62, 90)]
        track = _track(notes)
        cfg = QuantizeConfig(enabled=True, strength=1.0, grid_resolution=16)
        _, stats = quantize_track(track, _beat(120.0), [_seg()], cfg, duration=10.0)
        assert stats.notes_before == 2
        assert stats.notes_after == 2
        assert stats.grid_resolution == 16
        assert stats.mean_shift_ms > 0
        assert stats.max_shift_ms >= stats.mean_shift_ms

    def test_time_signature_change_at_midpoint(self):
        """4/4 段 0-5s，3/4 段 5-20s；note 跨段时按各自段量化。"""
        segs = [_seg(sig=(4, 4), start=0.0, end=5.0), _seg(sig=(3, 4), start=5.0, end=20.0)]
        notes = [Note(0.123, 0.5, 60, 100), Note(5.123, 5.5, 62, 90)]  # 各段一个
        track = _track(notes)
        cfg = QuantizeConfig(enabled=True, strength=1.0, grid_resolution=16)
        new_track, _ = quantize_track(track, _beat(120.0), segs, cfg, duration=20.0)
        # 0.123s snap 到 0.125s (4/4 段)
        assert new_track.notes[0].start == pytest.approx(0.125, abs=1e-6)
        # 5.123s snap 到 5.125s (3/4 段)
        assert new_track.notes[1].start == pytest.approx(5.125, abs=1e-6)

    def test_drum_channel_preserved(self):
        """鼓轨（channel=9）量化时 channel 不变。"""
        notes = [Note(0.0, 0.1, 36, 100, channel=9)]
        track = _track(notes, stem="drums")
        cfg = QuantizeConfig(enabled=True, strength=1.0, grid_resolution=16)
        new_track, _ = quantize_track(track, _beat(120.0), [_seg()], cfg, duration=10.0)
        assert new_track.notes[0].channel == 9


class TestQuantizeReport:
    def test_to_dict(self):
        r = QuantizeReport(
            total_notes_before=10,
            total_notes_after=10,
            duration_sec=5.0,
            strength=0.8,
            grid_resolution=16,
            groove_template="swing16",
        )
        d = r.to_dict()
        assert d["total_notes_before"] == 10
        assert d["strength"] == 0.8
        assert d["groove_template"] == "swing16"
        assert d["per_track"] == {}

    def test_per_track_to_dict(self):
        r = QuantizeReport(
            per_track={
                "vocals": TrackQuantizeStats(  # type: ignore[arg-type]
                    stem_name="vocals",
                    notes_before=5,
                    notes_after=5,
                    mean_shift_ms=2.5,
                    max_shift_ms=10.0,
                    grid_resolution=16,
                    groove_template="straight",
                ),
            },
        )
        d = r.to_dict()
        assert "vocals" in d["per_track"]
        assert d["per_track"]["vocals"]["mean_shift_ms"] == 2.5


class TestWriteQuantizeReport:
    def test_writes_valid_json(self, tmp_path: Path):
        r = QuantizeReport(
            total_notes_before=5,
            total_notes_after=5,
            duration_sec=3.0,
        )
        p = tmp_path / "deep" / "report.json"
        write_quantize_report(r, p)
        assert p.exists()
        data = json.loads(p.read_text())
        assert data["total_notes_before"] == 5
        assert data["total_notes_after"] == 5

    def test_creates_parent_dir(self, tmp_path: Path):
        r = QuantizeReport()
        p = tmp_path / "a" / "b" / "report.json"
        write_quantize_report(r, p)
        assert p.exists()


class TestLoadBeatTrackFromJson:
    def test_loads_valid(self, tmp_path: Path):
        p = tmp_path / "beats.json"
        p.write_text(json.dumps({
            "beats": [0.0, 0.5, 1.0],
            "downbeats": [0.0, 1.0],
            "bpm": 120.0,
            "tempo_confidence": 0.9,
        }))
        bt = load_beat_track_from_json(p)
        assert bt.bpm == 120.0
        assert len(bt.beats) == 3
        assert bt.tempo_confidence == 0.9

    def test_missing_keys_uses_defaults(self, tmp_path: Path):
        p = tmp_path / "beats.json"
        p.write_text(json.dumps({"bpm": 100.0}))
        bt = load_beat_track_from_json(p)
        assert bt.bpm == 100.0
        assert bt.beats == []


class TestQuantizeProject:
    def _make_project_midi(self, tmp_path: Path) -> Path:
        """构造一个最小 project.mid 用于 quantize_project 测试。"""
        proj = Project(
            audio_path="song.wav",
            duration=5.0,
            sample_rate=44100,
            time_signatures=[_seg(end=5.0)],
            tempo_map=[TempoSegment(0.0, 5.0, 120.0)],
        )
        proj.get_track("vocals").add(Note(0.123, 0.5, 60, 100))
        proj.get_track("vocals").add(Note(0.987, 1.5, 62, 90))
        proj.get_track("drums").add(Note(1.0, 1.05, 36, 100, channel=9))
        midi_path = tmp_path / "project.mid"
        from mujik.midi.io import write_project_to_midi
        write_project_to_midi(proj, midi_path)
        return midi_path

    def test_quantize_enabled(self, tmp_path: Path):
        midi_in = self._make_project_midi(tmp_path)
        bt = _beat(120.0)
        cfg = QuantizeConfig(enabled=True, strength=1.0, grid_resolution=16)

        midi_out = tmp_path / "out.mid"
        new_proj, report = quantize_project(
            midi_in, bt, [_seg()], cfg, output_midi_path=midi_out,
        )
        assert midi_out.exists()
        # 2 vocals + 1 drum = 3 notes
        assert report.total_notes_before == 3
        assert report.total_notes_after == 3
        assert "vocals" in report.per_track
        assert "drums" in report.per_track

    def test_quantize_disabled_no_op(self, tmp_path: Path):
        midi_in = self._make_project_midi(tmp_path)
        bt = _beat(120.0)
        cfg = QuantizeConfig(enabled=False)

        midi_out = tmp_path / "out.mid"
        new_proj, report = quantize_project(
            midi_in, bt, [_seg()], cfg, output_midi_path=midi_out,
        )
        assert midi_out.exists()
        # 量化禁用，stats 全 0 shift
        for stats in report.per_track.values():
            assert stats.mean_shift_ms == 0.0

    def test_overwrites_input_when_no_output(self, tmp_path: Path):
        midi_in = self._make_project_midi(tmp_path)
        bt = _beat(120.0)
        cfg = QuantizeConfig(enabled=True, strength=1.0, grid_resolution=16)

        quantize_project(midi_in, bt, [_seg()], cfg)
        # 应已覆盖
        assert midi_in.exists()
        from mujik.midi.io import read_midi_to_project
        loaded = read_midi_to_project(midi_in)
        # 0.123 → 0.125 (snap 到 16分)
        vocals = loaded.tracks.get("vocals")
        assert vocals is not None
        assert vocals.notes[0].start == pytest.approx(0.125, abs=1e-6)
