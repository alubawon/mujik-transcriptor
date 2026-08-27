"""Tests for merge/core.py."""
from __future__ import annotations

import pytest

from mujik.config.schema import MergeConfig
from mujik.merge.core import MergeReport, merge_tracks
from mujik.midi.model import Note, Track


def _n(start, end, pitch, vel, **kw) -> Note:
    return Note(start=start, end=end, pitch=pitch, velocity=vel, **kw)


def _track(notes: list[Note], stem: str, channel: int = 0) -> Track:
    t = Track(stem_name=stem, channel=channel)  # type: ignore[arg-type]
    for n in notes:
        t.add(n)
    return t


class TestMergeAllMode:
    def test_collapses_non_drum_vocal(self):
        """all 模式：non-drum + non-vocal 合并到 combined 轨。"""
        tracks = {
            "vocals": _track([_n(0.0, 0.5, 60, 100)], "vocals"),
            "drums": _track([_n(0.0, 0.1, 36, 100, channel=9)], "drums", channel=9),
            "bass": _track([_n(0.0, 0.5, 40, 90)], "bass"),
            "other": _track([_n(0.0, 0.5, 64, 80)], "other"),
        }
        cfg = MergeConfig(mode="all", preserve_drums=True, preserve_voice_separate=True)
        out, report = merge_tracks(tracks, cfg)

        assert "vocals" in out
        assert "drums" in out
        assert "combined" in out
        assert "bass" not in out
        assert "other" not in out
        # combined 包含 bass + other
        assert len(out["combined"].notes) == 2

    def test_drums_preserved_in_all_mode(self):
        drum_notes = [_n(0.0, 0.1, 36, 100, channel=9)]
        tracks = {"drums": _track(drum_notes, "drums", channel=9)}
        cfg = MergeConfig(mode="all")
        out, _ = merge_tracks(tracks, cfg)
        assert "drums" in out
        assert out["drums"].notes == drum_notes

    def test_density_filter_applied(self):
        # 13 个非 drum note 同时发声，max=12 → 丢 1
        notes = [_n(0.0, 0.5, 60 + i, 100) for i in range(13)]
        tracks = {"bass": _track(notes, "bass")}
        cfg = MergeConfig(mode="all", density_filter=True, max_simultaneous_notes=12)
        out, report = merge_tracks(tracks, cfg)
        assert report.dropped_by_density == 1
        assert len(out["combined"].notes) == 12

    def test_density_filter_disabled(self):
        notes = [_n(0.0, 0.5, 60 + i, 100) for i in range(15)]
        tracks = {"bass": _track(notes, "bass")}
        cfg = MergeConfig(mode="all", density_filter=False, max_simultaneous_notes=12)
        out, report = merge_tracks(tracks, cfg)
        assert report.dropped_by_density == 0
        assert len(out["combined"].notes) == 15


class TestMergePianoReductionMode:
    def test_piano_track_created(self):
        tracks = {
            "bass": _track([_n(0.0, 0.5, 40, 90), _n(0.0, 0.5, 47, 80)], "bass"),
            "other": _track([_n(0.0, 0.5, 64, 100)], "other"),
        }
        cfg = MergeConfig(mode="piano_reduction", max_simultaneous_notes=4)
        out, report = merge_tracks(tracks, cfg)
        assert "piano_reduction" in out
        # K = max(1, 4//2) = 2
        assert len(out["piano_reduction"].notes) == 2
        # bass + other 都没保留为独立轨
        assert "bass" not in out
        assert "other" not in out

    def test_keeps_drums_and_vocals(self):
        tracks = {
            "vocals": _track([_n(0.0, 0.5, 60, 100)], "vocals"),
            "drums": _track([_n(0.0, 0.1, 36, 100, channel=9)], "drums", channel=9),
            "other": _track([_n(0.0, 0.5, 64, 100)], "other"),
        }
        cfg = MergeConfig(mode="piano_reduction", preserve_voice_separate=True)
        out, _ = merge_tracks(tracks, cfg)
        assert "vocals" in out
        assert "drums" in out
        assert "piano_reduction" in out


class TestMergeScoreMode:
    def test_no_op_passes_through(self):
        """score 模式：所有 stem 保留为独立轨。"""
        tracks = {
            "vocals": _track([_n(0.0, 0.5, 60, 100)], "vocals"),
            "bass": _track([_n(0.0, 0.5, 40, 90)], "bass"),
            "drums": _track([_n(0.0, 0.1, 36, 100, channel=9)], "drums", channel=9),
        }
        cfg = MergeConfig(mode="score")
        out, _ = merge_tracks(tracks, cfg)
        assert set(out.keys()) == {"vocals", "bass", "drums"}
        assert len(out["vocals"].notes) == 1
        assert len(out["bass"].notes) == 1

    def test_density_applied_per_track(self):
        # 15 notes 同 onset → density filter 应用
        notes = [_n(0.0, 0.5, 60 + i, 100) for i in range(15)]
        tracks = {"bass": _track(notes, "bass")}
        cfg = MergeConfig(mode="score", density_filter=True, max_simultaneous_notes=12)
        out, report = merge_tracks(tracks, cfg)
        assert report.dropped_by_density == 3
        assert len(out["bass"].notes) == 12


class TestPreserveFlags:
    def test_preserve_voice_separate_false_collapses_vocals(self):
        tracks = {
            "vocals": _track([_n(0.0, 0.5, 60, 100)], "vocals"),
            "bass": _track([_n(0.0, 0.5, 40, 90)], "bass"),
        }
        cfg = MergeConfig(mode="all", preserve_voice_separate=False)
        out, _ = merge_tracks(tracks, cfg)
        # vocals 不应保留为独立轨
        assert "vocals" not in out
        # vocals note 进入 combined
        assert any(n.pitch == 60 for n in out["combined"].notes)

    def test_preserve_drums_false_still_uses_drum_detection(self):
        """即便 preserve_drums=False，drum 轨也不合并到 combined。"""
        # 实际上：preserve_drums=False 时 _is_drum_track 仍返回 True
        # 设计选择：drum 是 GM 惯例，无法合并到 pitched 轨
        tracks = {
            "drums": _track([_n(0.0, 0.1, 36, 100, channel=9)], "drums", channel=9),
        }
        cfg = MergeConfig(mode="all", preserve_drums=False)
        out, _ = merge_tracks(tracks, cfg)
        # drums channel=9 → 仍是 drum 轨
        assert "drums" in out
        assert "combined" not in out


class TestEdgeCases:
    def test_empty_tracks(self):
        cfg = MergeConfig(mode="all")
        out, report = merge_tracks({}, cfg)
        # all 模式：空 input → 空 output
        assert out == {}
        assert report.notes_in == 0
        assert report.notes_out == 0

    def test_unknown_mode_raises(self):
        cfg = MergeConfig(mode="all")
        # bypass pydantic literal check
        object.__setattr__(cfg, "mode", "bogus")
        with pytest.raises(ValueError, match="unknown merge mode"):
            merge_tracks({}, cfg)

    def test_report_to_dict(self):
        report = MergeReport(
            mode="all",
            output_tracks=["vocals", "combined"],
            notes_in=10,
            notes_out=8,
            dropped_by_density=2,
        )
        d = report.to_dict()
        assert d["mode"] == "all"
        assert d["notes_in"] == 10
        assert d["dropped_by_density"] == 2
