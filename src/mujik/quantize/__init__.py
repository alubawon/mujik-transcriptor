"""Quantize 模块：grid 量化 + groove 模板 + 报告。

公开 API（v0.2.3）：
    quantize_track
    quantize_project
    QuantizeReport / TrackQuantizeStats
    load_beat_track_from_json
    write_quantize_report

注：`QuantizeConfig` 仍从 `mujik.config.schema` 导入（保持 schema 集中）。
"""
from __future__ import annotations

from mujik.quantize.core import (
    QuantizeReport,
    TrackQuantizeStats,
    load_beat_track_from_json,
    quantize_project,
    quantize_track,
    write_quantize_report,
)

__all__ = [
    "QuantizeReport",
    "TrackQuantizeStats",
    "quantize_project",
    "quantize_track",
    "load_beat_track_from_json",
    "write_quantize_report",
]
