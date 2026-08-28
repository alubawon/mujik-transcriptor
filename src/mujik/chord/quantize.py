"""Chord track quantize（v0.4.5）：把 madmom 100ms 帧粒度的 chord_track
对齐到 bar/beat 网格，合并相邻同 chord，过滤短片段。

设计动机：
- madmom CRNN 输出 10 fps（100ms 帧）粒度，对 sheet music 来说太碎
- 直接喂给 score/builder.py 会出现同一 chord 在相邻 measure 重复显示
- 把 chord snap 到最近的 bar/beat 边界后，<harmony> 在乐谱上更可读

约定（v0.4.5）：
- grid_per_bar: 每小节内的 grid 点数
  - 1 = 整 bar（最粗粒度）
  - 2 = half-bar（半小节）
  - 4 = beat（每拍 1 个，最常见）
  - 8 = 8th（8 分音符粒度）
- 跨 TimeSignatureSegment 段独立 snap（变拍子下，每段 grid 不同）
- 合并：相邻 root+quality 相同的 chord 合并为单一事件
- 过滤：丢弃 < min_duration_sec 的短片段（madmom 误识别通常很短）
- 纯函数：no I/O，不修改入参
"""
from __future__ import annotations

import logging

from mujik.midi.model import ChordEvent
from mujik.time_signature.model import (
    TimeSignatureSegment,
    find_segment_for_time,
)

logger = logging.getLogger(__name__)


def _bar_duration_sec(segment: TimeSignatureSegment, bpm: float) -> float:
    """单个小节的秒数。"""
    return segment.bar_duration_sec(bpm)


def _beat_duration_sec(bpm: float) -> float:
    """一个 quarter note 的秒数。"""
    if bpm <= 0:
        raise ValueError(f"bpm must be > 0, got {bpm}")
    return 60.0 / bpm


def _grid_step_sec(segment: TimeSignatureSegment, bpm: float, grid_per_bar: int) -> float:
    """一个 grid 点的秒数。"""
    if grid_per_bar <= 0:
        raise ValueError(f"grid_per_bar must be > 0, got {grid_per_bar}")
    return _bar_duration_sec(segment, bpm) / grid_per_bar


def _snap_t_to_grid(
    t: float,
    segment: TimeSignatureSegment,
    bpm: float,
    grid_per_bar: int,
) -> float:
    """把 t snap 到最近 grid 点。

    若 t < segment.start_time，clamp 到 segment.start_time。
    若 t >= segment.end_time，clamp 到段尾最后一个 grid 点。

    段尾需要钳制，避免量化后 chord 跨段（v0.4.5 简化：每段独立 snap，
    跨段时不强约束，下游 find_chord_at_time 仍按 start/end 命中）。
    """
    step = _grid_step_sec(segment, bpm, grid_per_bar)
    if step <= 0:
        raise ValueError(f"grid step must be > 0, got {step}")
    if t < segment.start_time:
        return segment.start_time
    if t >= segment.end_time:
        # 钳制到段内最后一个 grid 点
        n = int((segment.end_time - segment.start_time) / step)
        return segment.start_time + n * step
    rel = t - segment.start_time
    g = round(rel / step)
    return segment.start_time + g * step


def _resolve_segment(
    t: float,
    time_signatures: list[TimeSignatureSegment],
    duration: float,
) -> TimeSignatureSegment:
    """找 t 所属段；找不到时兜底为单段 4/4 覆盖 [0, duration]。"""
    seg = find_segment_for_time(time_signatures, t)
    if seg is not None:
        return seg
    # 兜底：默认 4/4（与 quantize/core.py:_resolve_segment_or_default 一致）
    return TimeSignatureSegment(
        start_time=0.0,
        end_time=max(duration, 1.0),
        time_signature=(4, 4),
        confidence=0.0,
        source="default_4_4",
    )


def snap_chord_to_grid(
    chord: ChordEvent,
    time_signatures: list[TimeSignatureSegment],
    bpm: float,
    grid_per_bar: int = 4,
    duration: float = 0.0,
) -> ChordEvent:
    """把单个 chord 的 start/end snap 到最近 grid 点。

    Args:
        chord: 输入 ChordEvent
        time_signatures: 拍号段列表（v0.2.2 节奏步产出）
        bpm: 全局 BPM（v0.2.3 单值常量；rubato 留 v0.5+）
        grid_per_bar: 每小节 grid 点数（1/2/4/8）
        duration: 工程时长（用于兜底段构造）

    Returns:
        新 ChordEvent 实例（frozen）
    """
    if bpm <= 0:
        raise ValueError(f"bpm must be > 0, got {bpm}")
    if grid_per_bar not in (1, 2, 4, 8):
        raise ValueError(f"grid_per_bar must be in (1,2,4,8), got {grid_per_bar}")

    if not time_signatures:
        # 兜底段
        time_signatures = [
            TimeSignatureSegment(
                start_time=0.0,
                end_time=max(duration, chord.end + 1e-3),
                time_signature=(4, 4),
                confidence=0.0,
                source="default_4_4",
            )
        ]

    seg = _resolve_segment(chord.start, time_signatures, duration)
    new_start = _snap_t_to_grid(chord.start, seg, bpm, grid_per_bar)
    new_end = _snap_t_to_grid(chord.end, seg, bpm, grid_per_bar)

    # 防御：end 必须 > start（grid 退化时 start==end）
    if new_end <= new_start:
        step = _grid_step_sec(seg, bpm, grid_per_bar)
        new_end = new_start + step

    return ChordEvent(
        start=new_start,
        end=new_end,
        root=chord.root,
        quality=chord.quality,
        bass=chord.bass,
    )


def merge_consecutive_chords(
    chord_track: list[ChordEvent],
) -> list[ChordEvent]:
    """合并相邻 root+quality 相同的 chord。

    示例：
        [C 0-2, C 2-4, F 4-6] → [C 0-4, F 4-6]
        [C 0-2, C 2-4 bass="E" 4-6] → 不合并（bass 不同）
    """
    if not chord_track:
        return []
    merged: list[ChordEvent] = [chord_track[0]]
    for cur in chord_track[1:]:
        prev = merged[-1]
        if (
            cur.start <= prev.end + 1e-6
            and cur.root == prev.root
            and cur.quality == prev.quality
            and cur.bass == prev.bass
        ):
            # 合并：扩展 end
            merged[-1] = ChordEvent(
                start=prev.start,
                end=max(prev.end, cur.end),
                root=prev.root,
                quality=prev.quality,
                bass=prev.bass,
            )
        else:
            merged.append(cur)
    return merged


def filter_short_chords(
    chord_track: list[ChordEvent],
    min_duration_sec: float = 0.5,
) -> list[ChordEvent]:
    """丢弃持续时间 < min_duration_sec 的 chord。

    短片段通常是 madmom 误识别（< 0.5s 的 chord 在流行/爵士里罕见）。
    设为 0 时不过滤。
    """
    if min_duration_sec <= 0:
        return list(chord_track)
    return [c for c in chord_track if (c.end - c.start) >= min_duration_sec]


def quantize_chord_track(
    chord_track: list[ChordEvent],
    time_signatures: list[TimeSignatureSegment],
    bpm: float,
    grid_per_bar: int = 4,
    merge_consecutive: bool = True,
    min_duration_sec: float = 0.5,
    duration: float = 0.0,
) -> list[ChordEvent]:
    """端到端 chord quantize。

    步骤：
    1. snap start/end 到 grid（按所在拍号段）
    2. 合并相邻 root+quality 相同的 chord（可选）
    3. 过滤短片段（可选）

    Args:
        chord_track: 输入 chord 列表（madmom 输出 100ms 粒度）
        time_signatures: 拍号段列表
        bpm: 全局 BPM
        grid_per_bar: 每小节 grid 点数（1/2/4/8）
        merge_consecutive: 是否合并相邻同 chord
        min_duration_sec: 丢弃短于此秒数的 chord（0 = 不过滤）
        duration: 工程时长（兜底用）

    Returns:
        量化后的 ChordEvent 列表（新实例，不修改入参）
    """
    if not chord_track:
        return []

    # 1. snap
    snapped: list[ChordEvent] = [
        snap_chord_to_grid(c, time_signatures, bpm, grid_per_bar, duration)
        for c in chord_track
    ]

    # 2. merge
    if merge_consecutive:
        snapped = merge_consecutive_chords(snapped)

    # 3. filter
    if min_duration_sec > 0:
        snapped = filter_short_chords(snapped, min_duration_sec)

    logger.debug(
        "chord quantize: {n_in} → {n_out} (grid={g}, merge={m}, min_dur={d}s)",
        n_in=len(chord_track),
        n_out=len(snapped),
        g=grid_per_bar,
        m=merge_consecutive,
        d=min_duration_sec,
    )
    return snapped


__all__ = [
    "snap_chord_to_grid",
    "merge_consecutive_chords",
    "filter_short_chords",
    "quantize_chord_track",
]
