"""Pipeline configuration schemas (pydantic v2).

See docs/design.md §5 for full specification.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class SourceSeparationConfig(BaseModel):
    """源分离配置。"""

    stem_count: Literal[4, 5, 6] = 4
    model: Literal[
        "demucs", "mdx23c", "bsroformer", "melbandroformer"
    ] = "demucs"
    variant: str = "htdemucs_ft"
    device: Literal["cuda", "cpu", "mps"] = "cuda"
    precision: Literal["fp32", "fp16", "bf16"] = "fp16"
    segment_length: float = Field(default=7.5, ge=1.0, le=60.0)
    overlap: float = Field(default=0.25, ge=0.0, le=0.9)
    jobs: int = Field(default=1, ge=1, le=16)
    out_bitrate: int = 256
    out_format: Literal["wav", "flac", "mp3"] = "wav"


class LoudnormConfig(BaseModel):
    """响度归一配置。"""

    target_lufs: float = Field(default=-14.0, ge=-36.0, le=-6.0)
    peak_dbfs: float = Field(default=-1.0, ge=-12.0, le=0.0)
    enabled: bool = True


class TranscribeConfig(BaseModel):
    """转录配置：按 stem 路由到不同转录器。"""

    vocals: str = "basic-pitch"
    bass: str = "basic-pitch"
    drums: str = "adtof"
    piano: str = "bytedance-piano"
    guitar: str = "apollo"
    other: str = "basic-pitch"

    polyphonic_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    onset_interval_min_ms: float = Field(default=50.0, ge=10.0, le=500.0)
    velocity_threshold: int = Field(default=30, ge=1, le=127)
    min_note_length_ms: float = Field(default=50.0, ge=10.0, le=1000.0)
    max_polyphony: int = Field(default=6, ge=1, le=32)


class BasicPitchConfig(BaseModel):
    """Spotify basic-pitch 配置（subprocess 调用）。"""

    onset_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    frame_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    min_note_length_ms: float = Field(default=50.0, ge=0.0, le=1000.0)
    min_frequency: float | None = Field(default=None, ge=20.0, le=2000.0)
    max_frequency: float | None = Field(default=None, ge=2000.0, le=8000.0)
    timeout_sec: int = Field(default=1800, ge=60, le=7200)


class AdtofConfig(BaseModel):
    """adtof 配置（subprocess 调用）。"""

    model: Literal["adtof-5class", "adtof-9class"] = "adtof-5class"
    onset_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    min_note_length_ms: float = Field(default=50.0, ge=10.0, le=1000.0)
    device: Literal["cpu", "cuda"] = "cpu"
    timeout_sec: int = Field(default=1800, ge=60, le=7200)


class RhythmConfig(BaseModel):
    """节拍/下拍/时间签名配置。"""

    beat_tracker: Literal["beat-transformer", "beatnet"] = "beat-transformer"
    time_signature_model: str = "resnet18-meter2800"
    time_signature_fallback: tuple[int, int] = (4, 4)
    allow_user_override: bool = True
    confidence_threshold: float = Field(default=0.6, ge=0.0, le=1.0)


class ChordConfig(BaseModel):
    """和弦识别配置。"""

    enabled: bool = False
    models: list[str] = Field(default_factory=lambda: ["btc-hcqt", "chord-cnn-lstm"])
    vocab: Literal["root", "root-quality", "extended"] = "extended"


class QuantizeConfig(BaseModel):
    """节拍量化配置。"""

    enabled: bool = True
    grid_resolution: int = Field(default=16)
    strength: float = Field(default=0.8, ge=0.0, le=1.0)
    groove_template: str = "straight"
    custom_groove_path: str | None = None


class MergeConfig(BaseModel):
    """多轨合并配置。"""

    mode: Literal["all", "piano_reduction", "score"] = "piano_reduction"
    density_filter: bool = True
    max_simultaneous_notes: int = Field(default=12, ge=1, le=64)
    preserve_drums: bool = True
    preserve_voice_separate: bool = True


class RenderConfig(BaseModel):
    """渲染配置：决定 PDF 后端。"""

    pdf_backend: Literal["verovio", "lilypond", "musescore"] = "verovio"
    lilypond_url: str = "http://localhost:5001"
    musescore_url: str = "http://localhost:5002"
    include_chord_symbols: bool = True
    include_lyrics: bool = False
    page_size: Literal["A4", "Letter"] = "A4"
    staff_count: int = Field(default=2, ge=1, le=20)
    timeout_sec: int = Field(default=60, ge=1, le=600)


class PipelineConfig(BaseModel):
    """管线总配置。"""

    input_path: str
    output_dir: str
    preset: Literal["pop", "jazz", "metal", "custom"] = "custom"

    source_separation: SourceSeparationConfig = Field(default_factory=SourceSeparationConfig)
    transcribe: TranscribeConfig = Field(default_factory=TranscribeConfig)
    rhythm: RhythmConfig = Field(default_factory=RhythmConfig)
    chord: ChordConfig = Field(default_factory=ChordConfig)
    quantize: QuantizeConfig = Field(default_factory=QuantizeConfig)
    merge: MergeConfig = Field(default_factory=MergeConfig)
    render: RenderConfig = Field(default_factory=RenderConfig)

    @field_validator("input_path", "output_dir")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("path must be non-empty")
        return v

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineConfig":
        """从 YAML 加载配置。"""
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)

    def to_yaml(self, path: str | Path) -> None:
        """导出 YAML。"""
        import yaml
        with open(path, "w") as f:
            yaml.safe_dump(
                self.model_dump(mode="json"),
                f,
                sort_keys=False,
                allow_unicode=True,
            )

    def apply_preset(self, preset: str) -> "PipelineConfig":
        """应用预设（覆盖部分字段）。"""
        import copy
        cfg = copy.deepcopy(self)
        cfg.preset = preset  # type: ignore[assignment]
        if preset == "pop":
            cfg.source_separation.stem_count = 4
            cfg.source_separation.model = "demucs"
            cfg.quantize.groove_template = "straight"
        elif preset == "jazz":
            cfg.source_separation.stem_count = 5
            cfg.source_separation.model = "mdx23c"
            cfg.quantize.groove_template = "swing16"
            cfg.chord.enabled = True
        elif preset == "metal":
            cfg.source_separation.stem_count = 4
            cfg.source_separation.model = "demucs"
            cfg.quantize.groove_template = "straight"
            cfg.quantize.grid_resolution = 32
        return cfg


__all__ = [
    "SourceSeparationConfig",
    "LoudnormConfig",
    "TranscribeConfig",
    "BasicPitchConfig",
    "AdtofConfig",
    "RhythmConfig",
    "ChordConfig",
    "QuantizeConfig",
    "MergeConfig",
    "RenderConfig",
    "PipelineConfig",
]
