"""Tests for Demucs adapter (mocked subprocess)."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from mujik.separate.demucs_adapter import (
    separate_with_demucs,
    check_demucs_available,
    DemucsAdapterError,
)
from mujik.config.schema import SourceSeparationConfig


def _make_fake_stems(model_dir: Path, input_stem: str, out_format: str) -> None:
    """Generate fake demucs output."""
    track_dir = model_dir / input_stem
    track_dir.mkdir(parents=True, exist_ok=True)
    for name in ("vocals", "drums", "bass", "other"):
        (track_dir / f"{name}.{out_format}").write_bytes(b"RIFF" * 100)


def test_check_demucs_available_true():
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        assert check_demucs_available() is True


def test_check_demucs_available_false():
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=1)
        assert check_demucs_available() is False


def test_check_demucs_available_timeout():
    import subprocess
    with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="x", timeout=30)):
        assert check_demucs_available() is False


def test_separate_with_demucs_success(tmp_path: Path):
    # 准备假输入
    input_path = tmp_path / "song.wav"
    input_path.write_bytes(b"RIFF" * 100)
    out_dir = tmp_path / "stems"

    # 准备假 demucs 输出
    fake_stems_dir = out_dir / "htdemucs_ft" / "song"
    _make_fake_stems(out_dir / "htdemucs_ft", "song", "wav")

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stderr="", stdout="ok")
        stems = separate_with_demucs(input_path, out_dir)

    assert stems.stem_count == 4
    assert set(stems.names) == {"vocals", "drums", "bass", "other"}
    assert stems.separation_model == "demucs/htdemucs_ft"
    assert stems.get("vocals") is not None


def test_separate_input_not_found(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        separate_with_demucs(tmp_path / "missing.wav", tmp_path / "out")


def test_separate_demucs_failure(tmp_path: Path):
    input_path = tmp_path / "song.wav"
    input_path.write_bytes(b"x")

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=1, stderr="crashed")
        with pytest.raises(DemucsAdapterError, match="Demucs failed"):
            separate_with_demucs(input_path, tmp_path / "out")


def test_separate_timeout(tmp_path: Path):
    input_path = tmp_path / "song.wav"
    input_path.write_bytes(b"x")

    import subprocess
    with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="x", timeout=3600)):
        with pytest.raises(DemucsAdapterError, match="timeout"):
            separate_with_demucs(input_path, tmp_path / "out")


def test_separate_missing_output(tmp_path: Path):
    input_path = tmp_path / "song.wav"
    input_path.write_bytes(b"x")
    out_dir = tmp_path / "stems"

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        with pytest.raises(DemucsAdapterError, match="Demucs output not found"):
            separate_with_demucs(input_path, out_dir)


def test_separate_missing_one_stem(tmp_path: Path):
    input_path = tmp_path / "song.wav"
    input_path.write_bytes(b"x")
    out_dir = tmp_path / "stems"

    track_dir = out_dir / "htdemucs_ft" / "song"
    track_dir.mkdir(parents=True)
    for name in ("vocals", "drums", "bass"):  # 缺 other
        (track_dir / f"{name}.wav").write_bytes(b"x")

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        with pytest.raises(DemucsAdapterError, match="missing stem"):
            separate_with_demucs(input_path, out_dir)


def test_separate_uses_config(tmp_path: Path):
    input_path = tmp_path / "song.wav"
    input_path.write_bytes(b"x")
    out_dir = tmp_path / "stems"
    _make_fake_stems(out_dir / "mdx_q", "song", "mp3")

    cfg = SourceSeparationConfig(
        model="demucs", variant="mdx_q", device="cpu",
        out_format="mp3", segment_length=10.0, overlap=0.5, jobs=2,
    )

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        separate_with_demucs(input_path, out_dir, config=cfg)

    cmd = mock_run.call_args[0][0]
    # -n mdx_q
    assert "mdx_q" in cmd
    # --device cpu
    assert "cpu" in cmd
    # --out-format mp3
    assert "mp3" in cmd
    # --segment 10.0
    assert "10.0" in cmd
