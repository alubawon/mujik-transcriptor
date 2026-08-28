"""Time helpers for MusicXML builder（v0.2.4）。

职责：
- seconds → ticks（按指定 PPQ）
- 任意时间查 tempo / time signature

约定：
- 假设整轨 bpm 单一（rubato 留 v0.4+ 完整多 tempo）
- PPQ 默认 480（music21 默认）
- 所有函数对 boundary 行为一致：start_time ≤ t < end_time
"""
from __future__ import annotations

from mujik.midi.model import TempoSegment
from mujik.time_signature.model import (
    TimeSignatureSegment,
    find_segment_for_time,
)


def seconds_to_ticks(
    t: float,
    segment: TimeSignatureSegment,
    bpm: float,
    ppq: int = 480,
) -> int:
    """把绝对时间 t 转 ticks。

    算法：1 拍 = 60/bpm 秒；ticks = t / (60/bpm) * ppq = t * bpm * ppq / 60。
    segment 参数保留以备 v0.4+ 处理复合拍号（v0.2.4 不直接用）。
    """
    if bpm <= 0:
        raise ValueError(f"bpm must be > 0, got {bpm}")
    if ppq <= 0:
        raise ValueError(f"ppq must be > 0, got {ppq}")
    return int(round(t * bpm * ppq / 60.0))


def bpm_at_time(
    t: float,
    tempo_map: list[TempoSegment],
    default: float = 120.0,
) -> float:
    """返回 t 时刻的 bpm。

    若 tempo_map 为空或 t 落在所有段外，回退 default。
    """
    if not tempo_map:
        return default
    for seg in tempo_map:
        if seg.start_time <= t < seg.end_time:
            return float(seg.bpm)
    # t > 所有段尾 → 用最后一段的 bpm
    if t >= tempo_map[-1].end_time:
        return float(tempo_map[-1].bpm)
    # t < 第一段起点（理论不应发生）→ 用第一段
    return float(tempo_map[0].bpm)


def time_signature_at_time(
    t: float,
    segments: list[TimeSignatureSegment],
) -> TimeSignatureSegment | None:
    """返回 t 所属的 TimeSignatureSegment。无匹配返回 None。"""
    return find_segment_for_time(segments, t)


def measure_index_at_time(
    t: float,
    segment: TimeSignatureSegment,
    bpm: float,
) -> int:
    """计算 t 在 segment 内的 measure index（从 0 开始）。"""
    rel = max(0.0, t - segment.start_time)
    bar_dur = segment.bar_duration_sec(bpm)
    if bar_dur <= 0:
        return 0
    return int(rel / bar_dur)


__all__ = [
    "seconds_to_ticks",
    "bpm_at_time",
    "time_signature_at_time",
    "measure_index_at_time",
]
