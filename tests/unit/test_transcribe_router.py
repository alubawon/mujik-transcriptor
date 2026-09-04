"""Tests for transcribe.router."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from mujik.transcribe.router import RouterError, transcribe_stem
from mujik.config.schema import TranscribeConfig
from mujik.separate.model import Stem


def _make_stem(name: str = "vocals", audio: str = "/tmp/x.wav") -> Stem:
    return Stem(
        name=name,  # type: ignore[arg-type]
        audio_path=Path(audio),
        sample_rate=44100,
        duration=5.0,
        source_model="demucs/htdemucs_ft",
    )


class TestRouting:
    def test_vocals_to_basic_pitch(self):
        stem = _make_stem("vocals")
        with patch(
            "mujik.transcribe.basic_pitch_adapter.transcribe_with_basic_pitch"
        ) as mock_bp:
            mock_bp.return_value = []
            transcribe_stem(stem)
            assert mock_bp.called

    def test_bass_to_basic_pitch(self):
        stem = _make_stem("bass")
        with patch(
            "mujik.transcribe.basic_pitch_adapter.transcribe_with_basic_pitch"
        ) as mock_bp:
            mock_bp.return_value = []
            transcribe_stem(stem)
            assert mock_bp.called

    def test_other_to_basic_pitch(self):
        stem = _make_stem("other")
        with patch(
            "mujik.transcribe.basic_pitch_adapter.transcribe_with_basic_pitch"
        ) as mock_bp:
            mock_bp.return_value = []
            transcribe_stem(stem)
            assert mock_bp.called

    def test_drums_to_drumscript(self):
        """v0.5.2: drums 默认路由到 drumscript（替代 adtof）。"""
        stem = _make_stem("drums")
        with patch(
            "mujik.transcribe.drumscript_adapter.transcribe_drums_with_drumscript"
        ) as mock_ds:
            mock_ds.return_value = []
            transcribe_stem(stem)
            assert mock_ds.called

    def test_piano_dispatches_to_bytedance(self):
        """v0.4.0: piano 路由到 bytedance adapter（没装模块时抛 ByteDancePianoAdapterError）。"""
        from mujik.transcribe.bytedance_piano_adapter import ByteDancePianoAdapterError
        stem = _make_stem("piano")
        with patch("mujik.transcribe.bytedance_piano_adapter.check_bytedance_piano_available",
                   return_value=False):
            with pytest.raises(ByteDancePianoAdapterError):
                transcribe_stem(stem)

    def test_guitar_not_implemented(self):
        """v0.4.1: guitar 仍未实现（Apollo 仓库未公开），留 v0.5+。"""
        stem = _make_stem("guitar")
        with pytest.raises(RouterError, match="guitar"):
            transcribe_stem(stem)


class TestConfigOverride:
    def test_custom_drums_adapter(self):
        """config.drums = 'foo' 时调 foo（不存在则抛 RouterError）。"""
        stem = _make_stem("drums")
        cfg = TranscribeConfig(drums="unknown-adapter")  # type: ignore[arg-type]
        with pytest.raises(RouterError):
            transcribe_stem(stem, config=cfg)


class TestUnknownStem:
    def test_unknown_stem(self):
        # 强制构造一个不在 VALID_STEM_NAMES 里的 stem
        # Stem dataclass 会校验；用 mutation bypass
        stem = _make_stem("vocals")
        object.__setattr__(stem, "name", "unknown")  # type: ignore[arg-type]
        with pytest.raises(RouterError, match="unknown stem"):
            transcribe_stem(stem)


class TestOutDirPropagates:
    def test_out_dir_passed_through(self):
        stem = _make_stem("vocals")
        with patch(
            "mujik.transcribe.basic_pitch_adapter.transcribe_with_basic_pitch"
        ) as mock_bp:
            mock_bp.return_value = []
            transcribe_stem(stem, out_dir="/tmp/my_out")
            kwargs = mock_bp.call_args.kwargs
            assert kwargs.get("out_dir") == "/tmp/my_out"


class TestPerStemBasicPitch:
    """v0.5.3: stem_basic_pitch 覆盖 + onset_interval_min_ms /100 类型说谎修复。"""

    def test_bass_uses_low_frequency_band(self):
        stem = _make_stem("bass")
        with patch(
            "mujik.transcribe.basic_pitch_adapter.transcribe_with_basic_pitch"
        ) as mock_bp:
            mock_bp.return_value = []
            transcribe_stem(stem)
            cfg = mock_bp.call_args.kwargs["config"]
        assert cfg.min_frequency == 27.0
        assert cfg.max_frequency == 440.0
        assert cfg.onset_threshold == 0.6

    def test_vocals_uses_vocal_band(self):
        stem = _make_stem("vocals")
        with patch(
            "mujik.transcribe.basic_pitch_adapter.transcribe_with_basic_pitch"
        ) as mock_bp:
            mock_bp.return_value = []
            transcribe_stem(stem)
            cfg = mock_bp.call_args.kwargs["config"]
        assert cfg.min_frequency == 130.0
        assert cfg.max_frequency == 1050.0

    def test_drums_not_routed_through_basic_pitch(self):
        stem = _make_stem("drums")
        with patch(
            "mujik.transcribe.basic_pitch_adapter.transcribe_with_basic_pitch"
        ) as mock_bp, patch(
            "mujik.transcribe.drumscript_adapter.transcribe_drums_with_drumscript"
        ) as mock_ds:
            mock_ds.return_value = []
            transcribe_stem(stem)
            assert not mock_bp.called
            assert mock_ds.called

    def test_unlisted_stem_falls_back_to_global(self):
        # piano → bytedance；6-stem 场景里未列出的 stem 用全局 basic_pitch
        from mujik.config.schema import TranscribeConfig

        cfg = TranscribeConfig(stem_basic_pitch={})
        stem = _make_stem("vocals")
        with patch(
            "mujik.transcribe.basic_pitch_adapter.transcribe_with_basic_pitch"
        ) as mock_bp:
            mock_bp.return_value = []
            transcribe_stem(stem, config=cfg)
            sent = mock_bp.call_args.kwargs["config"]
        assert sent.onset_threshold == cfg.basic_pitch.onset_threshold
        assert sent.min_frequency is None

    def test_onset_threshold_no_longer_derived_from_ms(self):
        # 旧实现 onset_threshold = onset_interval_min_ms / 100（类型说谎）；
        # 现在阈值只来自 BasicPitchConfig，与毫秒参数无关
        from mujik.config.schema import TranscribeConfig

        cfg = TranscribeConfig()
        assert not hasattr(cfg, "onset_interval_min_ms")
        assert 0.0 <= cfg.basic_pitch.onset_threshold <= 1.0
