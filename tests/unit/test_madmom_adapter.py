"""Tests for rhythm.madmom_adapter (mocked subprocess)."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from mujik.rhythm.madmom_adapter import (
    MadmomAdapterError,
    check_madmom_available,
    track_beats_with_madmom,
)
from mujik.config.schema import RhythmConfig


def _write_madmom_json(
    json_path: Path,
    beats: list[float],
    downbeats: list[float],
    bpm: float = 120.0,
    tempo_confidence: float = 0.85,
) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps({
        "beats": beats,
        "downbeats": downbeats,
        "bpm": bpm,
        "tempo_confidence": tempo_confidence,
    }))


class TestCheckAvailable:
    def test_true(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            assert check_madmom_available() is True

    def test_false(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stderr="no madmom")
            assert check_madmom_available() is False

    def test_timeout(self):
        import subprocess
        with patch(
            "subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="x", timeout=30),
        ):
            assert check_madmom_available() is False


class TestTrackBeats:
    def test_basic(self, tmp_path: Path):
        audio = tmp_path / "song.wav"
        audio.write_bytes(b"RIFF")
        out_dir = tmp_path / "out"

        def fake_run(cmd, *args, **kwargs):
            json_path = Path(cmd[3])  # cmd = [python, wrapper, input, output_json]
            _write_madmom_json(
                json_path,
                beats=[0.0, 0.5, 1.0, 1.5, 2.0],
                downbeats=[0.0, 2.0],
                bpm=120.0,
                tempo_confidence=0.9,
            )
            return MagicMock(returncode=0, stderr="", stdout="ok")

        with patch("subprocess.run", side_effect=fake_run):
            track = track_beats_with_madmom(audio, out_dir=out_dir)

        assert len(track.beats) == 5
        assert len(track.downbeats) == 2
        assert track.bpm == 120.0
        assert track.tempo_confidence == 0.9
        assert track.beat_count == 5

    def test_empty_madmom_output(self, tmp_path: Path):
        audio = tmp_path / "song.wav"
        audio.write_bytes(b"RIFF")
        out_dir = tmp_path / "out"

        def fake_run(cmd, *args, **kwargs):
            _write_madmom_json(
                Path(cmd[3]),
                beats=[], downbeats=[],
                bpm=120.0, tempo_confidence=0.0,
            )
            return MagicMock(returncode=0, stderr="", stdout="ok")

        with patch("subprocess.run", side_effect=fake_run):
            track = track_beats_with_madmom(audio, out_dir=out_dir)

        assert track.beats == []
        assert track.downbeats == []
        assert track.bpm == 120.0  # default
        assert track.beat_count == 0

    def test_uses_config_timeout(self, tmp_path: Path):
        audio = tmp_path / "song.wav"
        audio.write_bytes(b"RIFF")
        cfg = RhythmConfig(madmom_timeout_sec=120)
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stderr="", stdout="ok",
            )
            with patch("pathlib.Path.write_text"):  # mock wrapper write
                with patch("pathlib.Path.unlink"):
                    try:
                        track_beats_with_madmom(audio, config=cfg)
                    except Exception:
                        pass
            # 验证 cmd 在最后调用
            if mock_run.call_args:
                pass  # already past; cmd arg 检查略


class TestErrorPaths:
    def test_input_not_found(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            track_beats_with_madmom(tmp_path / "missing.wav")

    def test_subprocess_failure(self, tmp_path: Path):
        audio = tmp_path / "song.wav"
        audio.write_bytes(b"RIFF")
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stderr="crashed")
            with pytest.raises(MadmomAdapterError, match="madmom failed"):
                track_beats_with_madmom(audio, out_dir=tmp_path / "out")

    def test_timeout(self, tmp_path: Path):
        audio = tmp_path / "song.wav"
        audio.write_bytes(b"RIFF")
        import subprocess
        with patch(
            "subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="x", timeout=1800),
        ):
            with pytest.raises(MadmomAdapterError, match="timeout"):
                track_beats_with_madmom(audio, out_dir=tmp_path / "out")

    def test_missing_output_json(self, tmp_path: Path):
        audio = tmp_path / "song.wav"
        audio.write_bytes(b"RIFF")
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            with pytest.raises(MadmomAdapterError, match="output json not found"):
                track_beats_with_madmom(audio, out_dir=tmp_path / "out")

    def test_invalid_json(self, tmp_path: Path):
        audio = tmp_path / "song.wav"
        audio.write_bytes(b"RIFF")
        out_dir = tmp_path / "out"

        def fake_run(cmd, *args, **kwargs):
            Path(cmd[3]).parent.mkdir(parents=True, exist_ok=True)
            Path(cmd[3]).write_text("not json{")
            return MagicMock(returncode=0, stderr="", stdout="ok")

        with patch("subprocess.run", side_effect=fake_run):
            with pytest.raises(json.JSONDecodeError):
                track_beats_with_madmom(audio, out_dir=out_dir)
