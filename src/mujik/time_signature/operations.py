"""Time signature operations: change_time_signature().

Implements two accumulation modes (设计文档 §4):
- 模式 A: 按现有时间轴在新拍号下堆积 (redraw_bars_under_new_time_signature)
- 模式 B: 按当前小节改拍号后填充/阶段 (change_time_signature_at_boundary)

Notes retain absolute time stamps; only the bar grid and segment boundaries change.
"""
from __future__ import annotations

from typing import Literal

from mujik.time_signature.model import (
    TimeSignatureSegment,
    build_default_segments,
    find_segment_for_time,
)

ChangeMode = Literal["preserve_time", "regrid"]


def _split_segment_at(
    seg: TimeSignatureSegment,
    t: float,
) -> tuple[TimeSignatureSegment, TimeSignatureSegment]:
    """把一段在时间 t 切分为两段。"""
    if t <= seg.start_time or t >= seg.end_time:
        return seg, seg
    left = TimeSignatureSegment(
        start_time=seg.start_time,
        end_time=t,
        time_signature=seg.time_signature,
        confidence=seg.confidence,
        source=seg.source,
    )
    right = TimeSignatureSegment(
        start_time=t,
        end_time=seg.end_time,
        time_signature=seg.time_signature,
        confidence=seg.confidence,
        source=seg.source,
    )
    return left, right


def redraw_bars_under_new_time_signature(
    segments: list[TimeSignatureSegment],
    apply_range: tuple[float, float],
    new_signature: tuple[int, int],
) -> list[TimeSignatureSegment]:
    """模式 A：在 apply_range 内按新拍号重画小节线。

    Note 绝对时间戳不变；只有 TimeSignatureSegment 在该范围内被替换。
    """
    start, end = apply_range
    if start >= end:
        raise ValueError(f"apply_range start must < end, got {apply_range}")
    if not (0.0 <= start < end):
        raise ValueError(f"apply_range out of bounds, got {apply_range}")

    out: list[TimeSignatureSegment] = []
    replaced = False
    for seg in segments:
        # 段完全在范围内 → 替换
        if seg.start_time >= start and seg.end_time <= end:
            if not replaced:
                out.append(
                    TimeSignatureSegment(
                        start_time=start,
                        end_time=end,
                        time_signature=new_signature,
                        confidence=1.0,
                        source="manual",
                    )
                )
                replaced = True
            continue
        # 段完全在范围外 → 保持
        if seg.end_time <= start or seg.start_time >= end:
            out.append(seg)
            continue
        # 段与范围有交集 → 切分并替换中间
        left, right = _split_segment_at(seg, start)
        _, right2 = _split_segment_at(right, end)
        if left.start_time < left.end_time:
            out.append(left)
        if not replaced:
            out.append(
                TimeSignatureSegment(
                    start_time=start,
                    end_time=end,
                    time_signature=new_signature,
                    confidence=1.0,
                    source="manual",
                )
            )
            replaced = True
        if right2.start_time < right2.end_time:
            out.append(right2)

    if not replaced:
        # 范围在所有段之外（理论上应已被 catch）
        out.append(
            TimeSignatureSegment(
                start_time=start,
                end_time=end,
                time_signature=new_signature,
                confidence=1.0,
                source="manual",
            )
        )

    out.sort(key=lambda s: s.start_time)
    return _merge_adjacent_same_signature(out)


def change_time_signature_at_boundary(
    segments: list[TimeSignatureSegment],
    change_time: float,
    new_signature: tuple[int, int],
    mode: ChangeMode = "preserve_time",
) -> list[TimeSignatureSegment]:
    """模式 B：在 change_time 处分段，前段保持原拍号，后段换新拍号。

    mode:
      - "preserve_time": 后段 note 绝对时间戳保留（仅重画小节线）
      - "regrid":  后段按新拍号 grid 重排 note（调用方需在 note 层另行处理）
    """
    if change_time < 0:
        raise ValueError(f"change_time must be >= 0, got {change_time}")

    out: list[TimeSignatureSegment] = []
    inserted_marker = False
    for seg in segments:
        # 段完全在 change_time 之前 → 原样保留
        if seg.end_time <= change_time:
            out.append(seg)
            continue
        # 段完全在 change_time 之后 → 换新拍号
        if seg.start_time >= change_time:
            out.append(
                TimeSignatureSegment(
                    start_time=seg.start_time,
                    end_time=seg.end_time,
                    time_signature=new_signature,
                    confidence=1.0,
                    source="manual",
                )
            )
            inserted_marker = True
            continue
        # 段跨越 change_time → 切分：左半保留原拍号，右半换新拍号
        left, right = _split_segment_at(seg, change_time)
        if left.start_time < left.end_time:
            out.append(left)
        if right.start_time < right.end_time:
            out.append(
                TimeSignatureSegment(
                    start_time=right.start_time,
                    end_time=right.end_time,
                    time_signature=new_signature,
                    confidence=1.0,
                    source="manual",
                )
            )
        inserted_marker = True

    # change_time 超出所有段 → 原样返回，不构造零长度段
    return _merge_adjacent_same_signature(out)


def _merge_adjacent_same_signature(
    segments: list[TimeSignatureSegment],
) -> list[TimeSignatureSegment]:
    """合并相邻同拍号的段。"""
    if not segments:
        return []
    merged: list[TimeSignatureSegment] = [segments[0]]
    for seg in segments[1:]:
        prev = merged[-1]
        if (
            abs(prev.end_time - seg.start_time) < 1e-9
            and prev.time_signature == seg.time_signature
            and prev.source == seg.source
        ):
            merged[-1] = TimeSignatureSegment(
                start_time=prev.start_time,
                end_time=seg.end_time,
                time_signature=prev.time_signature,
                confidence=min(prev.confidence, seg.confidence),
                source=prev.source,
            )
        else:
            merged.append(seg)
    return merged


__all__ = [
    "redraw_bars_under_new_time_signature",
    "change_time_signature_at_boundary",
    "ChangeMode",
]
