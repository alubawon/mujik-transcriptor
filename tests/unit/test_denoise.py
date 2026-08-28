"""Tests for preprocess/denoise.py (mocked nnnoiseless + demucs)."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from mujik.config.schema import PreprocessConfig
from mujik.preprocess.denoise import (
    DenoiseError,
    denoise,
    denoise_with_demucs_mode,
    denoise_with_nnnoiseless,
)


def _write_wav(path: Path, sr: int = 48000, dur: float = 1.0) -> None:
    """写测试 wav 文件。"""
    import soundfile as sf
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    audio = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sr)


def _mock_nnnoiseless_module():
    """构造一个 mock nnnoiseless 模块。"""
    mock_mod = MagicMock()

    class _FakeRNNoise:
        def __init__(self, sample_rate=48000):
            self.sr = sample_rate

        def process_frames(self, frames):
            # 简单：原样返回（相当于"无效"去噪）
            return frames

    mock_mod.RNNoise = _FakeRNNoise
    return mock_mod


class TestDenoiseWithNnnoiseless:
    def test_basic(self, tmp_path: Path):
        in_wav = tmp_path / "in.wav"
        _write_wav(in_wav, sr=48000, dur=1.0)
        out_wav = tmp_path / "out.wav"

        mock_mod = _mock_nnnoiseless_module()
        with patch.dict(sys.modules, {"nnnoiseless": mock_mod}):
            result = denoise_with_nnnoiseless(in_wav, out_path=out_wav)

        assert result == out_wav
        assert out_wav.exists()
        # 验证 wav 可读
        import soundfile as sf
        data, sr = sf.read(str(out_wav))
        assert sr > 0
        assert len(data) > 0

    def test_resample_when_not_48k(self, tmp_path: Path):
        in_wav = tmp_path / "in_44100.wav"
        _write_wav(in_wav, sr=44100, dur=1.0)
        out_wav = tmp_path / "out.wav"

        mock_mod = _mock_nnnoiseless_module()
        with patch.dict(sys.modules, {"nnnoiseless": mock_mod}):
            result = denoise_with_nnnoiseless(in_wav, out_path=out_wav)

        assert result == out_wav
        # 应被重采样到 48000 后再回 44100
        import soundfile as sf
        _, sr = sf.read(str(out_wav))
        assert sr == 44100

    def test_input_not_found(self, tmp_path: Path):
        with patch.dict(sys.modules, {"nnnoiseless": _mock_nnnoiseless_module()}):
            with pytest.raises(DenoiseError, match="not found"):
                denoise_with_nnnoiseless(tmp_path / "missing.wav")

    def test_nnnoiseless_not_installed(self, tmp_path: Path):
        in_wav = tmp_path / "in.wav"
        _write_wav(in_wav)
        # 移除 nnnoiseless
        saved = sys.modules.pop("nnnoiseless", None)
        try:
            with patch.dict(sys.modules, {"nnnoiseless": None}):
                with pytest.raises(DenoiseError, match="nnnoiseless is not installed"):
                    denoise_with_nnnoiseless(in_wav)
        finally:
            if saved is not None:
                sys.modules["nnnoiseless"] = saved

    def test_short_audio_raises(self, tmp_path: Path):
        in_wav = tmp_path / "short.wav"
        # 写 100 样本（远小于 480 帧）
        import soundfile as sf
        audio = np.zeros(100, dtype=np.float32)
        in_wav.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(in_wav), audio, 48000)

        with patch.dict(sys.modules, {"nnnoiseless": _mock_nnnoiseless_module()}):
            with pytest.raises(DenoiseError, match="too short"):
                denoise_with_nnnoiseless(in_wav)


class TestDenoiseWithDemucsMode:
    def test_basic(self, tmp_path: Path):
        in_wav = tmp_path / "in.wav"
        _write_wav(in_wav)
        out_wav = tmp_path / "out.wav"

        # mock demucs.api.Separator
        mock_separator_class = MagicMock()
        mock_separator = MagicMock()
        mock_separator.samplerate = 44100
        # 模拟分离：4 个 stem (vocals, drums, bass, other)
        mock_audio_vocals = np.zeros((1, 44100), dtype=np.float32)
        mock_audio_other = np.ones((1, 44100), dtype=np.float32) * 0.5
        mock_separator.separate_audio_file.return_value = (
            mock_audio_vocals,
            {"vocals": mock_audio_vocals, "drums": mock_audio_other,
             "bass": mock_audio_other, "other": mock_audio_other},
        )
        mock_separator_class.return_value = mock_separator

        # mock torch 模块（v0.4.0 测试环境无 torch）
        mock_torch = MagicMock()
        with patch.dict(sys.modules, {
            "demucs": MagicMock(),
            "demucs.api": MagicMock(Separator=mock_separator_class),
            "torch": mock_torch,
        }):
            result = denoise_with_demucs_mode(in_wav, out_path=out_wav)

        assert result == out_wav
        assert out_wav.exists()


class TestDenoiseUnifiedEntry:
    def test_disabled_returns_input(self, tmp_path: Path):
        in_wav = tmp_path / "in.wav"
        _write_wav(in_wav)
        cfg = PreprocessConfig(denoise_enabled=False)
        result = denoise(in_wav, config=cfg)
        assert result == in_wav

    def test_default_backend_nnnoiseless(self, tmp_path: Path):
        in_wav = tmp_path / "in.wav"
        _write_wav(in_wav)
        out_wav = tmp_path / "out.wav"
        cfg = PreprocessConfig(denoise_enabled=True, denoise_backend="nnnoiseless")

        with patch.dict(sys.modules, {"nnnoiseless": _mock_nnnoiseless_module()}):
            result = denoise(in_wav, config=cfg, out_path=out_wav)
        assert result == out_wav

    def test_unknown_backend_caught_by_pydantic(self):
        # pydantic Literal 在配置层就拒绝非法值
        with pytest.raises(Exception):  # pydantic ValidationError
            PreprocessConfig(denoise_enabled=True, denoise_backend="bogus")
