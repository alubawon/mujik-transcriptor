"""Quantize grid 工具（pure functions, no I/O）。

职责：把时间戳（秒）在给定的拍号段 + bpm + grid_resolution 下做 grid snap。

约定（v0.2.3）：
- "beat" = 一个 quarter note（无论拍号分母）
- grid_resolution = 每拍划分的子分数（16 → 16 分音符；8 → 8 分；32 → 32 分）
- BPM 视为整轨常量（rubato/变速留 v0.2.4）
- 一个拍号段内的 grid 是均匀的；不同段（拍号变化时）独立处理
- 当 t 落在某段边界外时，用段起点对齐（clamp）
"""
from __future__ import annotations

from mujik.time_signature.model import TimeSignatureSegment


def beat_duration_sec(bpm: float) -> float:
    """一个 quarter note 的秒数（不依赖拍号）。"""
    if bpm <= 0:
        raise ValueError(f"bpm must be > 0, got {bpm}")
    return 60.0 / bpm


def beat_index_at_time(
    t: float,
    segment: TimeSignatureSegment,
    bpm: float,
) -> float:
    """计算 t 距离 segment.start 的 beat 数（连续值，可小数）。

    若 t < segment.start_time，返回 0（clamp 到段起点）。
    若 t >= segment.end_time，返回 floor((end-start) / beat_dur)（clamp 到段尾）。
    """
    bd = beat_duration_sec(bpm)
    rel = t - segment.start_time
    if rel <= 0:
        return 0.0
    if rel >= segment.duration():
        # 已过本段，回到本段最后一拍
        last_beat_idx = segment.duration() / bd
        # 防止极小浮点误差
        return max(0.0, last_beat_idx)
    return rel / bd


def time_at_beat_index(
    beat_idx: float,
    segment: TimeSignatureSegment,
    bpm: float,
) -> float:
    """beat_idx 距离 segment.start_time 的绝对时间（秒）。

    beat_idx 可负（返回早于段起点）或越界（返回段尾）。
    """
    return segment.start_time + beat_idx * beat_duration_sec(bpm)


def snap_to_grid(
    t: float,
    segment: TimeSignatureSegment,
    bpm: float,
    grid_resolution: int,
) -> float:
    """把时间 t snap 到最近的 grid 细分点。

    算法：
      1. beat_idx = (t - seg.start) / beat_dur
      2. g = round(beat_idx * grid_resolution)
      3. snapped_beat = g / grid_resolution
      4. return seg.start + snapped_beat * beat_dur

    若 t < seg.start_time，snap 到 seg.start_time。
    若 t > seg.end_time，snap 到 seg.end_time 上一个 grid 点。
    """
    if grid_resolution <= 0:
        raise ValueError(f"grid_resolution must be > 0, got {grid_resolution}")

    # clamp 到段内
    clamped = max(segment.start_time, min(t, segment.end_time))
    beat_idx = beat_index_at_time(clamped, segment, bpm)
    g = round(beat_idx * grid_resolution)
    snapped_beat = g / grid_resolution
    return time_at_beat_index(snapped_beat, segment, bpm)


def is_8th_offbeat_position(
    beat_idx: float,
    grid_resolution: int,
) -> bool:
    """判断 beat_idx 是否落在 8 分 offbeat（即 0.5 拍附近 ± tolerance）。

    仅在 grid_resolution >= 8 时有意义。
    """
    if grid_resolution < 8:
        return False
    # 0.5 拍附近的 grid 索引 = 0.5 * grid_resolution
    offbeat_grid = 0.5 * grid_resolution
    nearest = round(beat_idx * grid_resolution)
    return abs(nearest - offbeat_grid) < 1e-6


__all__ = [
    "beat_duration_sec",
    "beat_index_at_time",
    "time_at_beat_index",
    "snap_to_grid",
    "is_8th_offbeat_position",
]
