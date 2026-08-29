"""MIDI 数据模型：Note / Track / Project。

所有时间戳为绝对秒，不依赖拍号。
"""
from __future__ import annotations

import re
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


# ---------- ChordEvent 验证（v0.4.6 hardening）----------

# 合法 root 格式：单字母 A-G + 可选 #/b（大小写不敏感）
_CHORD_ROOT_RE = re.compile(r"^[A-Ga-g][#b]?$")

# 按 vocab 划分的合法 quality 集合
# - "root": 只有根音，quality 必须空
# - "root-quality": 根 + 大/小三和弦（""、maj、M、m、min、minor、-）
# - "extended": 在 root-quality 之上加 7、maj7、m7、dim、aug、sus（v0.4.6 范围）
# - "btc-extended": v0.4.8 新增；BTC-ISMIR19 large_voca 全 14 种 quality
#                   (min/maj/dim/aug/min6/maj6/min7/minmaj7/maj7/7/dim7/hdim7/sus2/sus4)
ALLOWED_QUALITIES_BY_VOCAB: dict[str, frozenset[str]] = {
    "root": frozenset({""}),
    "root-quality": frozenset({
        "", "maj", "major", "M",
        "m", "min", "minor", "-",
    }),
    "extended": frozenset({
        "", "maj", "major", "M",
        "m", "min", "minor", "-",
        "7", "dom", "dominant",
        "maj7", "M7", "major7",
        "m7", "min7", "minor7",
        "dim", "diminished",
        "aug", "augmented", "+",
        "sus", "sus2", "sus4",
    }),
    "btc-extended": frozenset({
        "", "maj", "major", "M",
        "m", "min", "minor", "-",
        "7", "dom", "dominant",
        "maj7", "M7", "major7",
        "m7", "min7", "minor7",
        "dim", "diminished",
        "aug", "augmented", "+",
        "sus", "sus2", "sus4",
        # BTC 独有：6ths, half-diminished, minor-major 7th
        "6", "maj6", "m6", "min6",
        "dim7", "hdim7",
        "mM7", "minmaj7",
    }),
}


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
    """和弦事件。

    v0.4.6 hardening：在 __post_init__ 验证字段。
    - root: 必须匹配 ^[A-Ga-g][#b]?$（如 "C", "F#", "Bb"）；空字符串拒绝
    - bass: 同 root 规则，但可为空字符串（表示非 slash chord）
    - quality: 由 vocab 决定允许集合（默认 extended）
    - start/end: start < end 且 start >= 0
    """

    start: float
    end: float
    root: str
    quality: str = ""
    bass: str = ""
    vocab: str = "extended"  # "root" | "root-quality" | "extended"

    def __post_init__(self) -> None:
        # start/end 时间合法性
        # - 允许 end == start（madmom adapter 内部 placeholder，下游 snap+defense 后 > 0）
        # - 拒绝 end < start
        if self.start < 0:
            raise ValueError(
                f"chord start must be >= 0, got {self.start} (root={self.root!r})"
            )
        if self.end < self.start:
            raise ValueError(
                f"chord end ({self.end}) must be >= start ({self.start}) "
                f"(root={self.root!r})"
            )
        # root 格式
        if not self.root or not _CHORD_ROOT_RE.match(self.root):
            raise ValueError(
                f"chord root must match [A-G][#b]? (case-insensitive), "
                f"got {self.root!r}"
            )
        # bass 格式（空字符串允许）
        if self.bass and not _CHORD_ROOT_RE.match(self.bass):
            raise ValueError(
                f"chord bass must match [A-G][#b]? or empty, got {self.bass!r}"
            )
        # quality vocab
        allowed = ALLOWED_QUALITIES_BY_VOCAB.get(self.vocab)
        if allowed is None:
            raise ValueError(
                f"unknown chord vocab {self.vocab!r}; "
                f"expected one of {list(ALLOWED_QUALITIES_BY_VOCAB)}"
            )
        if self.quality not in allowed:
            raise ValueError(
                f"chord quality {self.quality!r} not in vocab {self.vocab!r} "
                f"(allowed: {sorted(allowed)})"
            )


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
    "ALLOWED_QUALITIES_BY_VOCAB",
]
