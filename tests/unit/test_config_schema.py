"""Tests for mujik.config.schema (v0.5.3 per-stem basic-pitch 配置)。"""
from __future__ import annotations

import pytest

from mujik.config.schema import BasicPitchConfig, TranscribeConfig


class TestPerStemBasicPitchSchema:
    """v0.5.3: TranscribeConfig basic_pitch / stem_basic_pitch。"""

    def test_default_overrides_present(self):
        cfg = TranscribeConfig()
        assert set(cfg.stem_basic_pitch) == {"vocals", "bass", "other"}
        assert cfg.stem_basic_pitch["bass"].max_frequency == 440.0

    def test_freq_band_validator(self):
        with pytest.raises(ValueError, match="must be <"):
            BasicPitchConfig(min_frequency=500.0, max_frequency=440.0)

    def test_max_frequency_can_be_low(self):
        # v0.5.3 前 max_frequency ge=2000，低频带写不进去
        cfg = BasicPitchConfig(min_frequency=27.0, max_frequency=440.0)
        assert cfg.max_frequency == 440.0

    def test_yaml_roundtrip(self, tmp_path):
        import yaml

        from mujik.config.schema import PipelineConfig

        p = tmp_path / "cfg.yaml"
        p.write_text(yaml.safe_dump({
            "input_path": "x.wav",
            "output_dir": "out",
            "transcribe": {
                "vocals": "basic-pitch",
                "min_note_length_ms": 50.0,
                "basic_pitch": {"onset_threshold": 0.5},
                "stem_basic_pitch": {
                    "bass": {"min_frequency": 27.0, "max_frequency": 440.0},
                },
            },
        }), encoding="utf-8")
        cfg = PipelineConfig.from_yaml(str(p))
        assert cfg.transcribe.stem_basic_pitch["bass"].min_frequency == 27.0
