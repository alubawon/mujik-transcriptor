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

    def test_drums_to_adtof(self):
        stem = _make_stem("drums")
        with patch(
            "mujik.transcribe.adtof_adapter.transcribe_drums_with_adtof"
        ) as mock_adtof:
            mock_adtof.return_value = []
            transcribe_stem(stem)
            assert mock_adtof.called

    def test_piano_dispatches_to_bytedance(self):
        """v0.4.0: piano 路由到 bytedance adapter（没装模块时抛 ByteDancePianoAdapterError）。"""
        from mujik.transcribe.bytedance_piano_adapter import ByteDancePianoAdapterError
        stem = _make_stem("piano")
        with patch("mujik.transcribe.bytedance_piano_adapter.check_bytedance_piano_available",
                   return_value=False):
            with pytest.raises(ByteDancePianoAdapterError):
                transcribe_stem(stem)

    def test_guitar_not_implemented(self):
        """v0.4.0: guitar 仍未实现，留 v0.4.1。"""
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
