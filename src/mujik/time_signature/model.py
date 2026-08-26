"""Time signature data model.

Supports variable time signatures (变拍子) via a segment list.
Each segment covers a time range and declares the time signature in effect.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

TimeSigSource = Literal[
    "auto_resnet18",   # ResNet18 / METER2800
    "auto_beatnet",    # BeatNet
    "manual",          # 用户手动
    "default_4_4",     # 默认兜底
]


@dataclass(frozen=True)
class TimeSignatureSegment:
    """一段拍号区间。

    所有时间戳用秒。
    start_time 是区间起点，end_time 是区间终点（开区间）。
    """

    start_time: float
    end_time: float
    time_signature: tuple[int, int]
    confidence: float
    source: TimeSigSource

    def __post_init__(self) -> None:
        if self.start_time >= self.end_time:
            raise ValueError(
                f"start_time ({self.start_time}) must precede end_time ({self.end_time})"
            )
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0,1], got {self.confidence}")
        num, den = self.time_signature
        if num < 1 or num > 32:
            raise ValueError(f"numerator out of range: {num}")
        if den not in (1, 2, 4, 8, 16, 32):
            raise ValueError(f"denominator must be a power of 2 up to 32, got {den}")

    def duration(self) -> float:
        return self.end_time - self.start_time

    def beats_per_second(self, bpm: float) -> float:
        """在该段以 bpm 计算每秒拍数。"""
        return bpm / 60.0

    def bar_duration_sec(self, bpm: float) -> float:
        """在该段以 bpm 计算单个小节的时长（秒）。"""
        num, den = self.time_signature
        # 一拍 = 60/bpm 秒；num/den 个四分音符的拍 = num * (4/den) 个 quarter
        return (60.0 / bpm) * num * (4.0 / den)


def find_segment_for_time(
    segments: list[TimeSignatureSegment],
    t: float,
) -> TimeSignatureSegment | None:
    """在分段列表中查询时间 t 所属的段。"""
    for seg in segments:
        if seg.start_time <= t < seg.end_time:
            return seg
    return None


def build_default_segments(
    duration: float,
    fallback: tuple[int, int] = (4, 4),
) -> list[TimeSignatureSegment]:
    """生成一段覆盖整个音频的默认拍号（4/4 或用户指定）。"""
    if duration <= 0:
        raise ValueError(f"duration must be > 0, got {duration}")
    return [
        TimeSignatureSegment(
            start_time=0.0,
            end_time=duration,
            time_signature=fallback,
            confidence=0.0,
            source="default_4_4" if fallback == (4, 4) else "manual",
        )
    ]


__all__ = [
    "TimeSignatureSegment",
    "TimeSigSource",
    "find_segment_for_time",
    "build_default_segments",
]
