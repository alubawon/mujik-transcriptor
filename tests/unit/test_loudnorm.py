"""Tests for preprocess.loudnorm (mocked soundfile + pyloudnorm).

真实音频 E2E 在 Step 11 验证；这里只测控制流 + 边界。
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# 注入假模块：让 patch("pyloudnorm.*") 不依赖真实安装
# 真实依赖在 Step 9 重建镜像后由 integration test 验证
if "pyloudnorm" not in sys.modules:
    _fake_pyln = MagicMock()
    _fake_pyln.Meter.return_value.integrated_loudness.return_value = -20.0
    _fake_pyln.normalize.loudness = lambda audio, src_lufs, tgt_lufs: audio
    sys.modules["pyloudnorm"] = _fake_pyln

from mujik.preprocess.loudnorm import (
    LoudnormError,
    normalize_loudness,
)
from mujik.config.schema import LoudnormConfig


def _mock_sf_read(audio: np.ndarray, sr: int = 44100):
    return audio, sr


class TestLoudnormControlFlow:
    def test_input_not_found(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            normalize_loudness(tmp_path / "missing.wav")

    def test_creates_out_path_parent(self, tmp_path: Path):
        audio = np.ones(44100, dtype=np.float32) * 0.1
        in_path = tmp_path / "in.wav"
        in_path.write_bytes(b"RIFF")
        out_path = tmp_path / "deep" / "nest" / "out.wav"

        with patch("soundfile.read", return_value=_mock_sf_read(audio, 44100)), \
             patch("soundfile.write") as mock_write, \
             patch("pyloudnorm.Meter") as MockMeter, \
             patch("pyloudnorm.normalize.loudness", return_value=audio * 5.0):
            MockMeter.return_value.integrated_loudness.return_value = -20.0
            normalize_loudness(in_path, out_path=out_path)

        # sf.write 被调且写到 deep/nest/ 路径
        assert mock_write.called
        args, _ = mock_write.call_args
        assert str(args[0]) == str(out_path)


class TestLoudnormMocked:
    def test_silent_falls_back_to_peak(self, tmp_path: Path):
        audio = np.zeros(44100, dtype=np.float32)
        in_path = tmp_path / "silent.wav"
        in_path.write_bytes(b"RIFF")
        out_path = tmp_path / "out.wav"

        with patch("soundfile.read", return_value=_mock_sf_read(audio, 44100)), \
             patch("soundfile.write") as mock_write, \
             patch("pyloudnorm.Meter") as MockMeter:
            MockMeter.return_value.integrated_loudness.side_effect = ValueError("silent")
            normalize_loudness(in_path, out_path=out_path)

        assert mock_write.called
        args, _ = mock_write.call_args
        assert np.all(args[1] == 0.0)

    def test_lufs_mode_called_with_target(self, tmp_path: Path):
        audio = np.ones(44100, dtype=np.float32) * 0.1
        in_path = tmp_path / "in.wav"
        in_path.write_bytes(b"RIFF")
        out_path = tmp_path / "out.wav"
        cfg = LoudnormConfig(target_lufs=-20.0)

        with patch("soundfile.read", return_value=_mock_sf_read(audio, 44100)), \
             patch("soundfile.write"), \
             patch("pyloudnorm.Meter") as MockMeter, \
             patch("pyloudnorm.normalize.loudness", return_value=audio) as mock_norm:
            MockMeter.return_value.integrated_loudness.return_value = -30.0
            normalize_loudness(in_path, config=cfg, out_path=out_path)

        args = mock_norm.call_args[0]
        assert args[2] == -20.0

    def test_clipping_protection(self, tmp_path: Path):
        audio = np.ones(44100, dtype=np.float32) * 0.1
        in_path = tmp_path / "in.wav"
        in_path.write_bytes(b"RIFF")
        out_path = tmp_path / "out.wav"
        clipped = np.ones(44100, dtype=np.float64) * 1.5

        with patch("soundfile.read", return_value=_mock_sf_read(audio, 44100)), \
             patch("soundfile.write") as mock_write, \
             patch("pyloudnorm.Meter") as MockMeter, \
             patch("pyloudnorm.normalize.loudness", return_value=clipped):
            MockMeter.return_value.integrated_loudness.return_value = -20.0
            normalize_loudness(in_path, out_path=out_path)

        args, _ = mock_write.call_args
        written = args[1]
        assert float(np.max(np.abs(written))) <= 0.99 + 1e-6

    def test_default_config_uses_minus_14(self, tmp_path: Path):
        audio = np.ones(44100, dtype=np.float32) * 0.1
        in_path = tmp_path / "in.wav"
        in_path.write_bytes(b"RIFF")
        out_path = tmp_path / "out.wav"

        with patch("soundfile.read", return_value=_mock_sf_read(audio, 44100)), \
             patch("soundfile.write"), \
             patch("pyloudnorm.Meter") as MockMeter, \
             patch("pyloudnorm.normalize.loudness", return_value=audio) as mock_norm:
            MockMeter.return_value.integrated_loudness.return_value = -30.0
            normalize_loudness(in_path, out_path=out_path)

        args = mock_norm.call_args[0]
        assert args[2] == -14.0


class TestConfigValidation:
    def test_default_values(self):
        cfg = LoudnormConfig()
        assert cfg.target_lufs == -14.0
        assert cfg.peak_dbfs == -1.0
        assert cfg.enabled is True

    def test_target_range(self):
        with pytest.raises(Exception):
            LoudnormConfig(target_lufs=0.0)
        with pytest.raises(Exception):
            LoudnormConfig(target_lufs=-50.0)
