"""Tests for transcribe/muscriptor_adapter.py (mocked subprocess + MIDI parse)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from mujik.midi.io import write_project_to_midi
from mujik.midi.model import Note, Project, TempoSegment, Track
from mujik.time_signature.model import build_default_segments
from mujik.transcribe.muscriptor_adapter import (
    MUSCRIPTOR_TIMEOUT_DEFAULT,
    VALID_MUSCRIPTOR_MODELS,
    MuscriptorAdapterError,
    check_muscriptor_available,
    transcribe_multitrack,
)


def _write_multitrack_midi(
    path: Path,
    stems: list[tuple[str, list[Note]]] | None = None,
) -> None:
    """写一个 muscriptor 风格的多轨 MIDI（instrument name 用标准名）。"""
    if stems is None:
        stems = [
            ("Vocals", [Note(0.0, 0.5, 60, 100)]),
            ("Drum Kit", [Note(1.0, 1.05, 36, 100)]),
            ("Electric Bass", [Note(0.5, 1.5, 40, 100)]),
            ("Acoustic Grand Piano", [Note(0.0, 1.0, 60, 100)]),
            ("Electric Guitar", [Note(0.5, 1.0, 67, 100)]),
        ]
    proj = Project(
        audio_path="song.wav",
        duration=2.0,
        sample_rate=44100,
        time_signatures=build_default_segments(2.0),
        tempo_map=[TempoSegment(0.0, 2.0, 120.0)],
    )
    for name, notes in stems:
        t = Track(stem_name="other")  # 临时 stem
        for n in notes:
            t.add(n)
        t.instrument = name  # 让 write_project_to_midi 用这个 name
        proj.tracks[name] = t
    write_project_to_midi(proj, path)


class TestCheckAvailable:
    def test_uvx_available(self):
        with patch("shutil.which", return_value="/usr/bin/uvx"):
            assert check_muscriptor_available() is True

    def test_uvx_not_available(self):
        with patch("shutil.which", return_value=None):
            assert check_muscriptor_available() is False


class TestParseError:
    """v0.4.2: muscriptor stderr → 友好错误信息。"""

    def test_hf_401_error(self):
        from mujik.transcribe.muscriptor_adapter import _parse_error
        hint = _parse_error("HTTPError 401: Unauthorized - HF_TOKEN required")
        assert "HuggingFace" in hint
        assert "HF_TOKEN" in hint or "huggingface" in hint.lower()

    def test_gated_repo_error(self):
        from mujik.transcribe.muscriptor_adapter import _parse_error
        hint = _parse_error("Repository is gated, please accept the license")
        assert "HuggingFace" in hint or "license" in hint.lower()

    def test_oom_error(self):
        from mujik.transcribe.muscriptor_adapter import _parse_error
        hint = _parse_error("CUDA out of memory")
        assert "model" in hint.lower() or "GPU" in hint or "memory" in hint.lower()

    def test_unknown_error_returns_truncated_stderr(self):
        from mujik.transcribe.muscriptor_adapter import _parse_error
        stderr = "Some weird error " * 100
        hint = _parse_error(stderr)
        assert len(hint) <= 500
        assert "weird error" in hint


class TestTranscribeMultitrack:
    def test_invalid_model_raises(self, tmp_path: Path):
        audio = tmp_path / "song.wav"
        audio.write_bytes(b"RIFF" * 100)
        with pytest.raises(MuscriptorAdapterError, match="invalid muscriptor model"):
            transcribe_multitrack(audio, model="huge")  # type: ignore[arg-type]

    def test_audio_not_found(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="audio not found"):
            transcribe_multitrack(tmp_path / "missing.wav")

    def test_uvx_not_found(self, tmp_path: Path):
        audio = tmp_path / "song.wav"
        audio.write_bytes(b"RIFF" * 100)
        with patch("shutil.which", return_value=None):
            with pytest.raises(FileNotFoundError, match="uvx"):
                transcribe_multitrack(audio)

    def test_successful_transcription(self, tmp_path: Path):
        """v0.4.2: 成功路径，mock subprocess + read_midi_to_project。"""
        audio = tmp_path / "song.wav"
        audio.write_bytes(b"RIFF" * 100)
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        # Pre-create the expected output MIDI（muscriptor 默认写 <stem>.mid）
        output_midi = out_dir / "song.mid"
        _write_multitrack_midi(output_midi)

        # Mock read_midi_to_project 解析多轨（muscriptor 标准 instrument name → stem）
        def fake_read(midi_path, audio_path="", sample_rate=44100):
            proj = Project(
                audio_path=audio_path or str(audio),
                duration=2.0,
                sample_rate=sample_rate,
                time_signatures=build_default_segments(2.0),
                tempo_map=[TempoSegment(0.0, 2.0, 120.0)],
            )
            proj.tracks["vocals"] = Track(stem_name="vocals")
            proj.tracks["vocals"].add(Note(0.0, 0.5, 60, 100))
            proj.tracks["drums"] = Track(stem_name="drums")
            proj.tracks["drums"].add(Note(1.0, 1.05, 36, 100))
            proj.tracks["bass"] = Track(stem_name="bass")
            proj.tracks["bass"].add(Note(0.5, 1.5, 40, 100))
            proj.tracks["piano"] = Track(stem_name="piano")
            proj.tracks["piano"].add(Note(0.0, 1.0, 60, 100))
            proj.tracks["guitar"] = Track(stem_name="guitar")
            proj.tracks["guitar"].add(Note(0.5, 1.0, 67, 100))
            return proj

        def fake_run(cmd, **kwargs):
            return MagicMock(returncode=0, stderr="")

        with patch("shutil.which", return_value="/usr/bin/uvx"), \
             patch("mujik.transcribe.muscriptor_adapter.subprocess.run",
                   side_effect=fake_run), \
             patch("mujik.midi.io.read_midi_to_project",
                   side_effect=fake_read):
            project = transcribe_multitrack(
                audio, out_dir=out_dir, model="small",
            )

        # 验证 Project 含 5 stem
        assert "vocals" in project.tracks
        assert "drums" in project.tracks
        assert "bass" in project.tracks
        assert "piano" in project.tracks
        assert "guitar" in project.tracks
        assert project.total_notes() == 5

    def test_subprocess_failure_raises(self, tmp_path: Path):
        audio = tmp_path / "song.wav"
        audio.write_bytes(b"RIFF" * 100)

        with patch("shutil.which", return_value="/usr/bin/uvx"), \
             patch("mujik.transcribe.muscriptor_adapter.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=1, stderr="HTTPError 401: HF_TOKEN required",
            )
            with pytest.raises(MuscriptorAdapterError, match="muscriptor failed"):
                transcribe_multitrack(audio)

    def test_subprocess_timeout_raises(self, tmp_path: Path):
        audio = tmp_path / "song.wav"
        audio.write_bytes(b"RIFF" * 100)

        with patch("shutil.which", return_value="/usr/bin/uvx"), \
             patch("mujik.transcribe.muscriptor_adapter.subprocess.run",
                   side_effect=subprocess.TimeoutExpired(cmd="muscriptor", timeout=1800)):
            with pytest.raises(MuscriptorAdapterError, match="timeout"):
                transcribe_multitrack(audio, timeout_sec=1800)

    def test_no_output_midi_raises(self, tmp_path: Path):
        audio = tmp_path / "song.wav"
        audio.write_bytes(b"RIFF" * 100)
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        # out_dir 中不放任何 .mid

        with patch("shutil.which", return_value="/usr/bin/uvx"), \
             patch("mujik.transcribe.muscriptor_adapter.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="")
            with pytest.raises(MuscriptorAdapterError, match="no MIDI file"):
                transcribe_multitrack(audio, out_dir=out_dir)

    def test_default_timeout_constant(self):
        """v0.4.2: 默认 30 分钟超时。"""
        assert MUSCRIPTOR_TIMEOUT_DEFAULT == 1800

    def test_valid_models(self):
        """v0.4.2: 仅 small/medium/large 三个有效尺寸。"""
        assert set(VALID_MUSCRIPTOR_MODELS) == {"small", "medium", "large"}


class TestHFTokenWarning:
    def test_hf_token_missing_logs_warning(self, tmp_path: Path):
        """v0.4.2: HF_TOKEN 缺失时记 warning（loguru 通过 stderr 输出）。"""
        import io
        import sys
        from loguru import logger

        audio = tmp_path / "song.wav"
        audio.write_bytes(b"RIFF" * 100)
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        output_midi = out_dir / "song.mid"
        _write_multitrack_midi(output_midi)

        captured = io.StringIO()
        handler_id = logger.add(captured, level="WARNING")

        try:
            with patch("shutil.which", return_value="/usr/bin/uvx"), \
                 patch("mujik.transcribe.muscriptor_adapter.subprocess.run",
                       return_value=MagicMock(returncode=0, stderr="")), \
                 patch.dict("os.environ", {}, clear=True):
                transcribe_multitrack(audio, out_dir=out_dir)
        finally:
            logger.remove(handler_id)

        assert "HF_TOKEN" in captured.getvalue()
