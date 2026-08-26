"""MIDI 数据模型：Note / Track / Project。

所有时间戳为绝对秒，不依赖拍号。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from mujik.time_signature.model import TimeSignatureSegment

StemName = Literal[
    "vocals",
    "drums",
    "bass",
    "other",
    "piano",   # 5/6-stem 模式
    "guitar",  # 6-stem 模式
]

VALID_STEM_NAMES: tuple[StemName, ...] = (
    "vocals", "drums", "bass", "other", "piano", "guitar",
)


@dataclass(frozen=True)
class Note:
    """MIDI 事件的最小单位。绝对时间戳（秒），不依赖拍号。"""

    start: float
    end: float
    pitch: int  # 0-127
    velocity: int  # 0-127
    channel: int = 0  # 0-15
    pitch_bend: tuple[float, ...] = ()  # 帧级弯音序列，-1..+1
    articulation: Literal[
        "", "slide", "bend", "hammer", "pull", "harmonic", "mute"
    ] = ""

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError(f"start must be >= 0, got {self.start}")
        if self.end < self.start:
            raise ValueError(
                f"end ({self.end}) must be >= start ({self.start})"
            )
        if not 0 <= self.pitch <= 127:
            raise ValueError(f"pitch must be in [0,127], got {self.pitch}")
        if not 0 <= self.velocity <= 127:
            raise ValueError(f"velocity must be in [0,127], got {self.velocity}")
        if not 0 <= self.channel <= 15:
            raise ValueError(f"channel must be in [0,15], got {self.channel}")
        for pb in self.pitch_bend:
            if not -1.0 <= pb <= 1.0:
                raise ValueError(f"pitch_bend values must be in [-1,1], got {pb}")

    def duration(self) -> float:
        return self.end - self.start


@dataclass
class Track:
    """一个 stem 转录后产出的 MIDI 轨。"""

    stem_name: StemName
    notes: list[Note] = field(default_factory=list)
    instrument: str = "Acoustic Grand Piano"  # MIDI program name
    channel: int = 0

    def add(self, note: Note) -> None:
        self.notes.append(note)

    def sort_by_start(self) -> None:
        self.notes.sort(key=lambda n: n.start)

    def duration(self) -> float:
        if not self.notes:
            return 0.0
        return max(n.end for n in self.notes)


@dataclass
class TempoSegment:
    """速度段，与 TimeSignatureSegment 模型一致。"""

    start_time: float
    end_time: float
    bpm: float
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if self.start_time >= self.end_time:
            raise ValueError("start must precede end")
        if self.bpm <= 0 or self.bpm > 500:
            raise ValueError(f"bpm out of range: {self.bpm}")


@dataclass
class ChordEvent:
    """和弦事件。"""

    start: float
    end: float
    root: str  # e.g. "C", "F#", "Bb"
    quality: str = ""  # e.g. "maj7", "m11"
    bass: str = ""  # slash chord bass


@dataclass
class Project:
    """整个项目。所有时间戳绝对秒。"""

    audio_path: str
    duration: float
    sample_rate: int
    time_signatures: list[TimeSignatureSegment]
    tempo_map: list[TempoSegment]
    tracks: dict[StemName, Track] = field(default_factory=dict)
    chord_track: list[ChordEvent] | None = None
    metadata: dict = field(default_factory=dict)

    def get_track(self, stem: StemName) -> Track:
        if stem not in self.tracks:
            self.tracks[stem] = Track(stem_name=stem)
        return self.tracks[stem]

    def total_notes(self) -> int:
        return sum(len(t.notes) for t in self.tracks.values())


__all__ = [
    "Note",
    "Track",
    "TempoSegment",
    "ChordEvent",
    "Project",
    "StemName",
    "VALID_STEM_NAMES",
]
