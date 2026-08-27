"""Rhythm 数据模型：BeatTrack。

承载 madmom / BeatNet 等 beat tracker 产出的节拍/下拍/BPM 信息。
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BeatTrack:
    """节拍/下拍/BPM 跟踪结果。"""

    beats: list[float] = field(default_factory=list)
    """全部 beat 时间戳（秒）。"""

    downbeats: list[float] = field(default_factory=list)
    """下拍时间戳（秒）。"""

    bpm: float = 120.0
    """估计全局 BPM。"""

    tempo_confidence: float = 0.0
    """BPM 估计置信度 0-1。"""

    beat_count: int = 0
    """冗余字段：beats 数量。"""

    def __post_init__(self) -> None:
        if self.bpm <= 0:
            raise ValueError(f"bpm must be > 0, got {self.bpm}")
        if not 0.0 <= self.tempo_confidence <= 1.0:
            raise ValueError(
                f"tempo_confidence must be in [0,1], got {self.tempo_confidence}"
            )
        # 自动同步 beat_count
        object.__setattr__(self, "beat_count", len(self.beats))

    def duration(self) -> float:
        """最后 beat 时间戳；空时 0。"""
        if not self.beats:
            return 0.0
        return float(self.beats[-1])

    def to_dict(self) -> dict:
        return {
            "beats": list(self.beats),
            "downbeats": list(self.downbeats),
            "bpm": self.bpm,
            "tempo_confidence": self.tempo_confidence,
        }


__all__ = ["BeatTrack"]
