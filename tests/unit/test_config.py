"""Tests for config schema."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from mujik.config.schema import (
    PipelineConfig,
    SourceSeparationConfig,
    QuantizeConfig,
)


class TestSourceSeparationConfig:
    def test_defaults(self):
        c = SourceSeparationConfig()
        assert c.stem_count == 4
        assert c.model == "demucs"
        assert c.variant == "htdemucs_ft"

    def test_invalid_stem_count(self):
        with pytest.raises(ValidationError):
            SourceSeparationConfig(stem_count=7)  # type: ignore[arg-type]

    def test_invalid_model(self):
        with pytest.raises(ValidationError):
            SourceSeparationConfig(model="not-a-model")  # type: ignore[arg-type]

    def test_segment_length_bounds(self):
        with pytest.raises(ValidationError):
            SourceSeparationConfig(segment_length=0.5)
        with pytest.raises(ValidationError):
            SourceSeparationConfig(segment_length=120.0)


class TestQuantizeConfig:
    def test_default(self):
        c = QuantizeConfig()
        assert c.groove_template == "straight"
        assert c.grid_resolution == 16

    def test_strength_bounds(self):
        with pytest.raises(ValidationError):
            QuantizeConfig(strength=1.5)
        with pytest.raises(ValidationError):
            QuantizeConfig(strength=-0.1)


class TestPipelineConfig:
    def test_from_yaml(self, tmp_path):
        yaml_content = """
input_path: "song.wav"
output_dir: "./out"
preset: "pop"
source_separation:
  stem_count: 4
  model: "demucs"
"""
        p = tmp_path / "cfg.yaml"
        p.write_text(yaml_content)
        cfg = PipelineConfig.from_yaml(p)
        assert cfg.preset == "pop"
        assert cfg.input_path == "song.wav"

    def test_to_yaml_roundtrip(self, tmp_path):
        cfg = PipelineConfig(input_path="x.wav", output_dir="./o")
        p = tmp_path / "out.yaml"
        cfg.to_yaml(p)
        cfg2 = PipelineConfig.from_yaml(p)
        assert cfg2.input_path == cfg.input_path
        assert cfg2.preset == cfg.preset

    def test_empty_input_raises(self):
        with pytest.raises(ValidationError):
            PipelineConfig(input_path="", output_dir="./o")

    def test_apply_preset_jazz(self):
        cfg = PipelineConfig(input_path="x.wav", output_dir="./o")
        cfg2 = cfg.apply_preset("jazz")
        assert cfg2.preset == "jazz"
        # v0.5.2 修：原 mdx23c/stem_count=5 从未生效（Roformer 未实现，
        # 被 demucs 路由静默忽略）——preset 必须说真话
        assert cfg2.source_separation.stem_count == 4
        assert cfg2.source_separation.model == "demucs"
        assert cfg2.quantize.groove_template == "swing16"
        assert cfg2.chord.enabled is True

    def test_apply_preset_metal(self):
        cfg = PipelineConfig(input_path="x.wav", output_dir="./o")
        cfg2 = cfg.apply_preset("metal")
        assert cfg2.quantize.grid_resolution == 32

    def test_apply_preset_pop(self):
        cfg = PipelineConfig(input_path="x.wav", output_dir="./o")
        cfg2 = cfg.apply_preset("pop")
        assert cfg2.quantize.groove_template == "straight"

    def test_unknown_preset_keeps_state(self):
        cfg = PipelineConfig(input_path="x.wav", output_dir="./o")
        cfg2 = cfg.apply_preset("custom")
        # no-op except deepcopy
        assert cfg2.preset == "custom"
        assert cfg2.source_separation.stem_count == cfg.source_separation.stem_count
