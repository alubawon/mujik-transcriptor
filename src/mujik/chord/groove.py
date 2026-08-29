"""Chord track groove 应用（v0.4.9）。

把 v0.4.5 quantize 后的 chord_track 按 groove 模板（swing16）
二次偏移每个 chord 边界（start + end），让 <harmony> 在乐谱上
体现 swing 感。

设计动机（v0.4.9）：
- v0.4.5 quantize 已 snap 到 bar/beat，但 jazz/swing 曲的 offbeat
  chord 仍贴在网格上，与 swing 听感错位
- 对每个 chord 边界重新应用 groove_offset_seconds，复用
  mujik.quantize.groove 已有的 swing16 模板
- **默认关闭**（apply_groove=False）：保护 audio 准确性；opt-in 启用

约定（v0.4.9）：
- 与 quantize.groove 共享模板（"straight" / "swing16"）
- chord 可独立于 note 选择模板
- strength: 0=noop, 1=full offset, 0.5=half
- start 和 end 独立计算 offset
- 跨 TimeSignatureSegment 段独立处理
"""
from __future__ import annotations

import logging

from mujik.midi.model import ChordEvent
from mujik.quantize.groove import DEFAULT_SWING_RATIO, groove_offset_seconds
from mujik.time_signature.model import (
    TimeSignatureSegment,
    find_segment_for_time,
)

logger = logging.getLogger(__name__)


# 内部 grid_resolution 用于 groove offbeat 检测（4 = 16th per beat）
_CHORD_GROOVE_GRID_RESOLUTION = 4


def _resolve_segment(
    t: float,
    time_signatures: list[TimeSignatureSegment],
    duration: float,
) -> TimeSignatureSegment | None:
    """找 t 所属段；找不到返回 None（caller fallback）。"""
    seg = find_segment_for_time(time_signatures, t)
    if seg is not None:
        return seg
    if duration <= 0:
        return None
    return TimeSignatureSegment(
        start_time=0.0,
        end_time=duration,
        time_signature=(4, 4),
        confidence=0.0,
        source="default_4_4",
    )


def _beat_index_at_time(t: float, segment: TimeSignatureSegment, bpm: float) -> float:
    """t 距离 segment.start 的 beat 数（连续值）。"""
    if bpm <= 0:
        raise ValueError(f"bpm must be > 0, got {bpm}")
    beat_dur = 60.0 / bpm
    if t < segment.start_time:
        return 0.0
    if t >= segment.end_time:
        return max(0.0, (segment.end_time - segment.start_time) / beat_dur)
    return (t - segment.start_time) / beat_dur


def _groove_offset_at(
    t: float,
    segment: TimeSignatureSegment,
    bpm: float,
    template: str,
    ratio: float,
    strength: float,
) -> float:
    """t 时刻的 groove 偏移（秒）。"""
    if template == "straight" or strength == 0.0:
        return 0.0
    beat_idx = _beat_index_at_time(t, segment, bpm)
    # grid_pos 用 mod 拿到本 beat 内的 local position
    # （与 quantize.groove 的 is_offbeat_position 检查兼容）
    grid_pos_global = round(beat_idx * _CHORD_GROOVE_GRID_RESOLUTION)
    grid_pos_local = grid_pos_global % _CHORD_GROOVE_GRID_RESOLUTION
    offset = groove_offset_seconds(
        beat_position=beat_idx - int(beat_idx),
        grid_position=grid_pos_local,
        grid_resolution=_CHORD_GROOVE_GRID_RESOLUTION,
        template=template,
        bpm=bpm,
        ratio=ratio,
    )
    return offset * strength


def apply_groove_to_chord_track(
    chord_track: list[ChordEvent],
    time_signatures: list[TimeSignatureSegment],
    bpm: float,
    template: str = "straight",
    strength: float = 1.0,
    ratio: float = DEFAULT_SWING_RATIO,
    duration: float = 0.0,
) -> list[ChordEvent]:
    """按 groove 模板对每个 chord 边界做偏移。

    Args:
        chord_track: v0.4.5 quantize 输出（已 snap 到 grid）
        time_signatures: 拍号段列表
        bpm: 全局 BPM
        template: "straight" | "swing16"
        strength: 0=noop, 1=full offset
        ratio: swing 比例（默认 0.6）
        duration: 工程时长（兜底用）

    Returns:
        新 ChordEvent 列表（不修改入参）
    """
    if not chord_track or template == "straight" or strength == 0.0:
        return list(chord_track)

    new_track: list[ChordEvent] = []
    for chord in chord_track:
        seg = _resolve_segment(chord.start, time_signatures, duration)
        if seg is None:
            new_track.append(chord)
            continue
        offset_start = _groove_offset_at(
            chord.start, seg, bpm, template, ratio, strength,
        )
        offset_end = _groove_offset_at(
            chord.end, seg, bpm, template, ratio, strength,
        )
        new_start = chord.start + offset_start
        new_end = chord.end + offset_end
        if new_end <= new_start:
            beat_dur = 60.0 / bpm
            new_end = new_start + beat_dur * 0.5
        new_track.append(ChordEvent(
            start=new_start,
            end=new_end,
            root=chord.root,
            quality=chord.quality,
            bass=chord.bass,
        ))

    logger.debug(
        "chord groove: %d chords shifted (template=%s, strength=%s)",
        len(chord_track), template, strength,
    )
    return new_track


__all__ = [
    "apply_groove_to_chord_track",
]
