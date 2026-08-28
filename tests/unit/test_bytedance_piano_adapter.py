"""Tests for transcribe/bytedance_piano_adapter.py (mocked subprocess + pretty_midi)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from mujik.midi.io import write_project_to_midi
from mujik.midi.model import Note, Project, TempoSegment
from mujik.time_signature.model import build_default_segments
from mujik.transcribe.bytedance_piano_adapter import (
    PIANO_TRANSCRIPTION_MODULE,
    ByteDancePianoAdapterError,
    check_bytedance_piano_available,
    transcribe_piano_bytedance,
    _write_wrapper,
)


def _write_piano_midi(path: Path) -> None:
    """写一个简单的 piano MIDI 测试文件（含 note track + pedal track）。"""
    proj = Project(
        audio_path="song.wav",
        duration=2.0,
        sample_rate=44100,
        time_signatures=build_default_segments(2.0),
        tempo_map=[TempoSegment(0.0, 2.0, 120.0)],
    )
    proj.get_track("vocals").add(Note(0.0, 0.5, 60, 100))
    proj.get_track("vocals").add(Note(0.5, 1.0, 64, 90))
    proj.get_track("vocals").add(Note(1.0, 1.5, 67, 100))
    write_project_to_midi(proj, path)


class TestCheckAvailable:
    def test_available(self):
        with patch.dict(sys.modules, {PIANO_TRANSCRIPTION_MODULE: MagicMock()}):
            assert check_bytedance_piano_available() is True

    def test_not_available(self):
        with patch.dict(sys.modules, {PIANO_TRANSCRIPTION_MODULE: None}):
            assert check_bytedance_piano_available() is False


class TestWriteWrapper:
    def test_creates_temp_file(self, tmp_path: Path):
        wrapper = _write_wrapper(tmp_path / "in.wav", tmp_path / "out.mid")
        assert wrapper.exists()
        content = wrapper.read_text()
        assert "piano_transcription_inference" in content
        assert "load_audio" in content
        assert "transcribe" in content
        # 清理
        wrapper.unlink()


class TestTranscribe:
    def test_basic(self, tmp_path: Path):
        in_wav = tmp_path / "song.wav"
        in_wav.write_bytes(b"RIFF" * 100)
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        output_midi = out_dir / "song_bytedance.mid"
        _write_piano_midi(output_midi)

        def fake_run(cmd, **kwargs):
            # wrapper script path is cmd[1]
            wrapper_path = Path(cmd[1])
            # Extract args from cmd
            assert len(cmd) >= 4
            assert cmd[2] == str(in_wav)
            assert cmd[3] == str(output_midi)
            assert cmd[4] in ("cuda", "cpu")
            return MagicMock(returncode=0, stderr="")

        with patch.dict(sys.modules, {PIANO_TRANSCRIPTION_MODULE: MagicMock()}), \
             patch("mujik.transcribe.bytedance_piano_adapter.subprocess.run",
                   side_effect=fake_run):
            notes = transcribe_piano_bytedance(in_wav, out_dir=out_dir)

        assert len(notes) == 3
        assert all(n.pitch in (60, 64, 67) for n in notes)
        # 排序后按 start
        assert notes[0].start == 0.0

    def test_input_not_found(self, tmp_path: Path):
        with patch.dict(sys.modules, {PIANO_TRANSCRIPTION_MODULE: MagicMock()}):
            with pytest.raises(FileNotFoundError):
                transcribe_piano_bytedance(tmp_path / "missing.wav")

    def test_module_not_installed(self, tmp_path: Path):
        in_wav = tmp_path / "song.wav"
        in_wav.write_bytes(b"RIFF" * 100)
        with patch.dict(sys.modules, {PIANO_TRANSCRIPTION_MODULE: None}):
            with pytest.raises(ByteDancePianoAdapterError, match="not installed"):
                transcribe_piano_bytedance(in_wav)

    def test_subprocess_failure(self, tmp_path: Path):
        in_wav = tmp_path / "song.wav"
        in_wav.write_bytes(b"RIFF" * 100)

        with patch.dict(sys.modules, {PIANO_TRANSCRIPTION_MODULE: MagicMock()}), \
             patch("mujik.transcribe.bytedance_piano_adapter.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stderr="error msg")
            with pytest.raises(ByteDancePianoAdapterError, match="exit=1"):
                transcribe_piano_bytedance(in_wav)

    def test_subprocess_timeout(self, tmp_path: Path):
        in_wav = tmp_path / "song.wav"
        in_wav.write_bytes(b"RIFF" * 100)

        with patch.dict(sys.modules, {PIANO_TRANSCRIPTION_MODULE: MagicMock()}), \
             patch("mujik.transcribe.bytedance_piano_adapter.subprocess.run",
                   side_effect=subprocess.TimeoutExpired(cmd="bytedance", timeout=1800)):
            with pytest.raises(ByteDancePianoAdapterError, match="timeout"):
                transcribe_piano_bytedance(in_wav)

    def test_no_output_midi_raises(self, tmp_path: Path):
        in_wav = tmp_path / "song.wav"
        in_wav.write_bytes(b"RIFF" * 100)

        with patch.dict(sys.modules, {PIANO_TRANSCRIPTION_MODULE: MagicMock()}), \
             patch("mujik.transcribe.bytedance_piano_adapter.subprocess.run",
                   return_value=MagicMock(returncode=0)):
            with pytest.raises(ByteDancePianoAdapterError, match="output midi not found"):
                transcribe_piano_bytedance(in_wav)
