"""Tests for separate/htdemucs_6s_adapter.py (mocked subprocess)."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from mujik.separate.htdemucs_6s_adapter import (
    HTDEMUCS_6S_STEMS,
    Htdemucs6sAdapterError,
    check_htdemucs_6s_available,
    separate_with_htdemucs_6s,
)


def _setup_demucs_output(track_dir: Path, stems: list[str] | None = None) -> None:
    """模拟 demucs 写出 6 个 stem 文件。"""
    stems = stems or list(HTDEMUCS_6S_STEMS)
    track_dir.mkdir(parents=True, exist_ok=True)
    for stem in stems:
        wav = track_dir / f"{stem}.wav"
        wav.write_bytes(b"RIFF" * 100)


class TestCheckAvailable:
    def test_via_python_module(self):
        # mock shutil.which: demucs CLI 不可用, python 可用
        def fake_which(name):
            if name == "python":
                return "/usr/bin/python"
            return None

        with patch("shutil.which", side_effect=fake_which):
            with patch.dict("sys.modules", {"demucs": MagicMock()}):
                assert check_htdemucs_6s_available() is True

    def test_no_demucs(self):
        # mock shutil.which: 两个都不可用
        with patch("shutil.which", return_value=None):
            import sys
            saved = sys.modules.pop("demucs", None)
            try:
                assert check_htdemucs_6s_available() is False
            finally:
                if saved is not None:
                    sys.modules["demucs"] = saved


class TestSeparate:
    def test_basic(self, tmp_path: Path):
        in_wav = tmp_path / "song.wav"
        in_wav.write_bytes(b"RIFF" * 100)
        out_dir = tmp_path / "stems"

        def fake_run(cmd, **kwargs):
            # cmd: ['python', '-m', 'demucs', '-n', 'htdemucs_6s', ..., 'song.wav']
            tmp_out = Path(cmd[cmd.index("--out") + 1])
            track_dir = tmp_out / "htdemucs_6s" / in_wav.stem
            _setup_demucs_output(track_dir)
            mock_proc = MagicMock()
            mock_proc.returncode = 0
            mock_proc.stderr = ""
            return mock_proc

        with patch("mujik.separate.htdemucs_6s_adapter.subprocess.run",
                   side_effect=fake_run):
            stems = separate_with_htdemucs_6s(in_wav, out_dir)

        assert len(stems.stems) == 6
        assert "vocals" in stems.stems
        assert "piano" in stems.stems
        assert "guitar" in stems.stems
        # 输出文件应被复制到 out_dir
        assert (out_dir / "song_piano.wav").exists()
        assert (out_dir / "song_guitar.wav").exists()
        assert stems.separation_model == "demucs/htdemucs_6s"

    def test_partial_stems(self, tmp_path: Path):
        in_wav = tmp_path / "song.wav"
        in_wav.write_bytes(b"RIFF" * 100)
        out_dir = tmp_path / "stems"

        def fake_run(cmd, **kwargs):
            tmp_out = Path(cmd[cmd.index("--out") + 1])
            track_dir = tmp_out / "htdemucs_6s" / in_wav.stem
            # 只写 4 个 stem（demucs 偶尔会缺失）
            _setup_demucs_output(track_dir, ["vocals", "drums", "bass", "other"])
            mock_proc = MagicMock()
            mock_proc.returncode = 0
            return mock_proc

        with patch("mujik.separate.htdemucs_6s_adapter.subprocess.run",
                   side_effect=fake_run):
            stems = separate_with_htdemucs_6s(in_wav, out_dir)

        # 应有 4 个 stem（partial）
        assert len(stems.stems) == 4
        assert "piano" not in stems.stems

    def test_no_stems_raises(self, tmp_path: Path):
        in_wav = tmp_path / "song.wav"
        in_wav.write_bytes(b"RIFF" * 100)
        out_dir = tmp_path / "stems"

        def fake_run(cmd, **kwargs):
            tmp_out = Path(cmd[cmd.index("--out") + 1])
            track_dir = tmp_out / "htdemucs_6s" / in_wav.stem
            track_dir.mkdir(parents=True, exist_ok=True)
            # 不创建任何 wav
            return MagicMock(returncode=0, stderr="")

        with patch("mujik.separate.htdemucs_6s_adapter.subprocess.run",
                   side_effect=fake_run):
            with pytest.raises(Htdemucs6sAdapterError, match="no stem files"):
                separate_with_htdemucs_6s(in_wav, out_dir)

    def test_input_not_found(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            separate_with_htdemucs_6s(tmp_path / "missing.wav", tmp_path / "out")

    def test_subprocess_failure(self, tmp_path: Path):
        in_wav = tmp_path / "song.wav"
        in_wav.write_bytes(b"RIFF" * 100)

        with patch("mujik.separate.htdemucs_6s_adapter.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stderr="segfault")
            with pytest.raises(Htdemucs6sAdapterError, match="exit=1"):
                separate_with_htdemucs_6s(in_wav, tmp_path / "out")

    def test_subprocess_timeout(self, tmp_path: Path):
        in_wav = tmp_path / "song.wav"
        in_wav.write_bytes(b"RIFF" * 100)

        with patch("mujik.separate.htdemucs_6s_adapter.subprocess.run",
                   side_effect=subprocess.TimeoutExpired(cmd="demucs", timeout=3600)):
            with pytest.raises(Htdemucs6sAdapterError, match="timeout"):
                separate_with_htdemucs_6s(in_wav, tmp_path / "out")
