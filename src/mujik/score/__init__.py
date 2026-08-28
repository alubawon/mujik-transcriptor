"""MusicXML builder（v0.2.4）：Project → MusicXML。

公开 API：
    build_musicxml(project, config, layout) -> str
    MusicXMLBuilderError
    seconds_to_ticks
    bpm_at_time
    time_signature_at_time
    measure_index_at_time

依赖：music21（已在 `midi` extra 中）
"""
from __future__ import annotations

from mujik.score.builder import (
    LayoutMode,
    MusicXMLBuilderError,
    build_musicxml,
)
from mujik.score.time_helpers import (
    bpm_at_time,
    measure_index_at_time,
    seconds_to_ticks,
    time_signature_at_time,
)

__all__ = [
    "LayoutMode",
    "MusicXMLBuilderError",
    "build_musicxml",
    "seconds_to_ticks",
    "bpm_at_time",
    "time_signature_at_time",
    "measure_index_at_time",
]
