"""Merge 模块：多 stem 合并为单轨 / 总谱。

公开 API（v0.2.3）：
    merge_tracks
    MergeReport
    apply_density_filter
    piano_reduce

注：`MergeConfig` 仍从 `mujik.config.schema` 导入（保持 schema 集中）。
"""
from __future__ import annotations

from mujik.merge.core import MergeReport, merge_tracks
from mujik.merge.density import apply_density_filter
from mujik.merge.reduce import piano_reduce

__all__ = [
    "MergeReport",
    "merge_tracks",
    "apply_density_filter",
    "piano_reduce",
]
