"""Tests for separate/router.py（v0.5.2 分离后端路由）。"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mujik.config.schema import SourceSeparationConfig
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


class TestRouteRoformerFailLoud:
    """Roformer 家族未实现 → fail-loud，绝不静默降级（v0.5.2 前的 bug）。"""

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
