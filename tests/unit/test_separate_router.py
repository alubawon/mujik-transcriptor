"""Tests for separate/router.py（v0.5.2 分离后端路由）。"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mujik.config.schema import SourceSeparationConfig
from mujik.separate.bsroformer_adapter import BsRoformerAdapterError
from mujik.separate.router import (
    ROFORMER_MODELS,
    SeparationBackendError,
    separate_audio,
)


def _stems_mock():
    m = MagicMock()
    m.separation_model = "demucs/test"
    return m


class TestRouteDemucs:
    def test_default_routes_to_demucs_adapter(self, tmp_path: Path):
        cfg = SourceSeparationConfig()  # demucs / htdemucs_ft / 4
        with patch(
            "mujik.separate.demucs_adapter.separate_with_demucs",
            return_value=_stems_mock(),
        ) as mock_sep:
            separate_audio(tmp_path / "a.wav", tmp_path / "out", cfg)
        mock_sep.assert_called_once()

    def test_htdemucs_6s_routes_to_6s_adapter(self, tmp_path: Path):
        cfg = SourceSeparationConfig(variant="htdemucs_6s", stem_count=6)
        with patch(
            "mujik.separate.htdemucs_6s_adapter.separate_with_htdemucs_6s",
            return_value=_stems_mock(),
        ) as mock_sep:
            separate_audio(tmp_path / "a.wav", tmp_path / "out", cfg)
        mock_sep.assert_called_once()


class TestRouteBsRoformer:
    """v0.5.3: bsroformer 已实现 → 必须真路由到 adapter，不再 fail-loud。"""

    def test_bsroformer_routes_to_adapter(self, tmp_path: Path):
        cfg = SourceSeparationConfig(model="bsroformer", stem_count=6)
        with patch(
            "mujik.separate.bsroformer_adapter.separate_with_bsroformer",
            return_value=_stems_mock(),
        ) as mock_sep:
            separate_audio(tmp_path / "a.wav", tmp_path / "out", cfg)
        mock_sep.assert_called_once()

    def test_bsroformer_not_in_fail_loud_list(self):
        # 回归防线：bsroformer 若被误加回 ROFORMER_MODELS，
        # TestRouteRoformerFailLoud 的 parametrize 会静默扩容而本测试会红
        assert "bsroformer" not in ROFORMER_MODELS

    def test_bsroformer_does_not_fall_back_to_demucs(self, tmp_path: Path):
        """CLI 缺失时必须报错，不许静默降级到 demucs（config 不说谎）。"""
        cfg = SourceSeparationConfig(model="bsroformer", stem_count=6)
        src = tmp_path / "a.wav"
        src.write_bytes(b"RIFF")  # 存在即可；_resolve_bin 在读音频之前就炸
        with (
            patch(
                "mujik.separate.bsroformer_adapter._resolve_bin",
                side_effect=BsRoformerAdapterError("no cli"),
            ),
            patch("mujik.separate.demucs_adapter.separate_with_demucs") as mock_demucs,
            pytest.raises(BsRoformerAdapterError),
        ):
            separate_audio(src, tmp_path / "out", cfg)
        mock_demucs.assert_not_called()

    def test_stem_count_mismatch_warns_but_proceeds(self, tmp_path: Path):
        """stem_count=4 与 6-stem 输出不符 → warning，以实际产出为准。"""
        cfg = SourceSeparationConfig(model="bsroformer", stem_count=4)
        with (
            patch(
                "mujik.separate.bsroformer_adapter.separate_with_bsroformer",
                return_value=_stems_mock(),
            ) as mock_sep,
            patch("mujik.separate.router.logger") as mock_logger,
        ):
            separate_audio(tmp_path / "a.wav", tmp_path / "out", cfg)
        mock_sep.assert_called_once()
        mock_logger.warning.assert_called_once()


class TestRouteRoformerFailLoud:
    """未实现的后端 → fail-loud，绝不静默降级（v0.5.2 前的 bug）。"""

    @pytest.mark.parametrize("model", ROFORMER_MODELS)
    def test_roformer_models_raise(self, tmp_path: Path, model: str):
        cfg = SourceSeparationConfig(model=model)
        with (
            patch("mujik.separate.demucs_adapter.separate_with_demucs") as mock_demucs,
            patch("mujik.separate.htdemucs_6s_adapter.separate_with_htdemucs_6s") as mock_6s,
            pytest.raises(SeparationBackendError, match=model),
        ):
            separate_audio(tmp_path / "a.wav", tmp_path / "out", cfg)
        # 不许有任何静默兜底
        mock_demucs.assert_not_called()
        mock_6s.assert_not_called()

    def test_unimplemented_list_is_not_empty(self):
        # parametrize 空列表会让整个 fail-loud 测试类静默变成 0 个用例
        assert len(ROFORMER_MODELS) > 0
