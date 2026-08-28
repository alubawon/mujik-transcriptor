"""Stems 容器：源分离产出的多轨音频。

支持 4/5/6+ stem 可插拔（v0.1 默认 4-stem）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from mujik.midi.model import StemName, VALID_STEM_NAMES

SeparationModel = Literal[
    "demucs", "mdx23c", "bsroformer", "melbandroformer"
]


@dataclass
class Stem:
    """单个 stem 的音频 + 元数据。"""

    name: StemName
    audio_path: Path
    sample_rate: int
    duration: float
    source_model: str
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.name not in VALID_STEM_NAMES:
            raise ValueError(f"invalid stem name: {self.name}")
        if self.sample_rate <= 0:
            raise ValueError(f"sample_rate must be > 0")
        if self.duration < 0:
            raise ValueError(f"duration must be >= 0")


@dataclass
class Stems:
    """一次源分离产出的所有 stem。"""

    stems: dict[StemName, Stem] = field(default_factory=dict)
    separation_model: str = "demucs"
    separation_time: float = 0.0
    sample_rate: int = 44100
    total_duration: float = 0.0

    @property
    def stem_count(self) -> int:
        return len(self.stems)

    @property
    def names(self) -> list[StemName]:
        return list(self.stems.keys())

    def add(self, stem: Stem) -> None:
        self.stems[stem.name] = stem

    def get(self, name: StemName) -> Stem | None:
        return self.stems.get(name)

    def require(self, name: StemName) -> Stem:
        """必须存在的 stem；不存在抛错。"""
        if name not in self.stems:
            raise KeyError(f"stem '{name}' not present, have: {self.names}")
        return self.stems[name]

    def primary_stems(self) -> list[Stem]:
        """返回所有 stem（v0.4.0 改为全集，5/6-stem 自动覆盖）。"""
        return list(self.stems.values())


__all__ = [
    "Stem",
    "Stems",
    "SeparationModel",
]
