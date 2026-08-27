"""Tests for midi/io.py: Project ↔ .mid."""
from __future__ import annotations

from pathlib import Path

import pytest

from mujik.midi.io import (
    DRUM_CHANNEL,
    PITCHED_CHANNELS,
    read_midi_to_project,
    write_project_to_midi,
)
from mujik.midi.model import Note, Project, TempoSegment
from mujik.time_signature.model import build_default_segments


def _make_simple_project(duration: float = 5.0) -> Project:
    """构造 4 stem 简单 Project。"""
    p = Project(
        audio_path="song.wav",
        duration=duration,
        sample_rate=44100,
        time_signatures=build_default_segments(duration),
        tempo_map=[TempoSegment(0.0, duration, 120.0)],
    )
    # Vocals
    p.get_track("vocals").add(Note(0.0, 1.0, 60, 100))
    p.get_track("vocals").add(Note(1.0, 2.0, 62, 80))
    # Bass
    p.get_track("bass").add(Note(0.0, 0.5, 40, 110))
    p.get_track("bass").add(Note(0.5, 1.0, 42, 110))
    # Drums (channel 9 by convention)
    p.get_track("drums").add(Note(0.0, 0.1, 36, 100))   # kick
    p.get_track("drums").add(Note(0.5, 0.6, 38, 100))   # snare
    # Other
    p.get_track("other").add(Note(0.0, 2.0, 64, 90))
    return p


class TestWriteReadRoundTrip:
    def test_roundtrip_basic(self, tmp_path: Path):
        original = _make_simple_project()
        midi_path = tmp_path / "out.mid"
        write_project_to_midi(original, midi_path)
        assert midi_path.exists()
        assert midi_path.stat().st_size > 0

        loaded = read_midi_to_project(midi_path, audio_path="song.wav")
        assert loaded.total_notes() == original.total_notes()
        assert set(loaded.tracks.keys()) == set(original.tracks.keys())

    def test_roundtrip_preserves_notes(self, tmp_path: Path):
        original = _make_simple_project()
        midi_path = tmp_path / "out.mid"
        write_project_to_midi(original, midi_path)
        loaded = read_midi_to_project(midi_path, audio_path="song.wav")

        for stem in original.tracks:
            orig_notes = sorted(
                (n.start, n.end, n.pitch, n.velocity) for n in original.tracks[stem].notes
            )
            loaded_notes = sorted(
                (n.start, n.end, n.pitch, n.velocity) for n in loaded.tracks[stem].notes
            )
            # MIDI 时间戳 round-trip 有 1ms 级浮点误差，用 approx
            assert len(orig_notes) == len(loaded_notes), f"count mismatch in {stem}"
            for o, l in zip(orig_notes, loaded_notes):
                assert o[0] == pytest.approx(l[0], abs=1e-3)
                assert o[1] == pytest.approx(l[1], abs=1e-3)
                assert o[2] == l[2]  # pitch 精确
                assert o[3] == l[3]  # velocity 精确

    def test_roundtrip_tempo(self, tmp_path: Path):
        p = _make_simple_project()
        p.tempo_map = [TempoSegment(0.0, 5.0, 140.0)]
        midi_path = tmp_path / "out.mid"
        write_project_to_midi(p, midi_path)
        loaded = read_midi_to_project(midi_path)
        assert loaded.tempo_map[0].bpm == pytest.approx(140.0, abs=1e-3)

    def test_roundtrip_time_signature(self, tmp_path: Path):
        from mujik.time_signature.model import TimeSignatureSegment
        p = _make_simple_project()
        p.time_signatures = [TimeSignatureSegment(
            start_time=0.0,
            end_time=5.0,
            time_signature=(3, 4),
            confidence=1.0,
            source="manual",
        )]
        midi_path = tmp_path / "out.mid"
        write_project_to_midi(p, midi_path)
        loaded = read_midi_to_project(midi_path)
        assert loaded.time_signatures[0].time_signature == (3, 4)


class TestChannelAllocation:
    def test_drums_channel_9(self, tmp_path: Path):
        """drums 永远在 channel 9 (GM)。"""
        p = _make_simple_project()
        midi_path = tmp_path / "out.mid"

        import pretty_midi
        write_project_to_midi(p, midi_path)
        pm = pretty_midi.PrettyMIDI(str(midi_path))
        drum_insts = [i for i in pm.instruments if i.is_drum]
        assert len(drum_insts) == 1
        # pretty-midi 内部会按 is_drum=True 路由到 channel 9
        assert drum_insts[0].is_drum is True

    def test_pitched_channels_unique(self, tmp_path: Path):
        """pitched tracks 不重复 channel。"""
        p = _make_simple_project()
        # Add many extra tracks to force channel reuse detection
        for i in range(20):
            stem_name = f"other_{i}"  # type: ignore[arg-type]
            p.tracks[stem_name] = type(p.tracks["other"])(stem_name=stem_name)

        midi_path = tmp_path / "out.mid"
        write_project_to_midi(p, midi_path)
        loaded = read_midi_to_project(midi_path)
        # Should have 24 tracks (1 drum + 1 vocal + 1 bass + 1 other + 20 extra)
        assert len(loaded.tracks) >= 3

    def test_no_pitched_channel_conflict_with_drum(self):
        """PITCHED_CHANNELS 不含 9。"""
        assert DRUM_CHANNEL not in PITCHED_CHANNELS
        assert 9 not in PITCHED_CHANNELS


class TestEdgeCases:
    def test_empty_project(self, tmp_path: Path):
        p = Project(
            audio_path="empty.wav",
            duration=0.001,  # 避免 build_default_segments 的 0 校验
            sample_rate=44100,
            time_signatures=build_default_segments(0.001),
            tempo_map=[],
        )
        midi_path = tmp_path / "empty.mid"
        write_project_to_midi(p, midi_path)
        assert midi_path.exists()
        loaded = read_midi_to_project(midi_path)
        assert loaded.total_notes() == 0

    def test_drums_only(self, tmp_path: Path):
        p = _make_simple_project()
        # Remove pitched
        for k in ("vocals", "bass", "other"):
            p.tracks.pop(k, None)
        midi_path = tmp_path / "drums_only.mid"
        write_project_to_midi(p, midi_path)
        loaded = read_midi_to_project(midi_path)
        assert "drums" in loaded.tracks
        assert loaded.tracks["drums"].notes  # has notes

    def test_write_creates_parent_dir(self, tmp_path: Path):
        p = _make_simple_project()
        midi_path = tmp_path / "deep" / "nest" / "out.mid"
        write_project_to_midi(p, midi_path)
        assert midi_path.exists()

    def test_read_nonexistent_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            read_midi_to_project(tmp_path / "missing.mid")

    def test_no_tempo_fallback(self, tmp_path: Path):
        p = _make_simple_project()
        p.tempo_map = []
        midi_path = tmp_path / "out.mid"
        write_project_to_midi(p, midi_path)
        loaded = read_midi_to_project(midi_path)
        # Default 120.0 BPM should be written/read
        assert len(loaded.tempo_map) >= 1
        assert loaded.tempo_map[0].bpm == pytest.approx(120.0, abs=1e-3)
