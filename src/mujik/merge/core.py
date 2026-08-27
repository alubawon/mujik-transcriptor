"""Merge core：把多 stem 轨合并为单轨/总谱，按 MergeConfig.mode 派发。

模式（v0.2.3）：
  - "all"：所有非 drum / non-vocal 合并为 1 个 combined 轨，drum + vocal 保留
  - "piano_reduction"：启发式 piano 缩减 1 个 reduction 轨，drum + vocal 保留
  - "score"：no-op，所有 stem 保留为独立轨

密度过滤：max_simultaneous_notes 限制所有输出轨（除 drums）
preserve_drums / preserve_voice_separate 开关决定 drums / vocals 旁路
"""
from __future__ import annotations

from dataclasses import dataclass, field

from mujik.config.schema import MergeConfig
from mujik.merge.density import apply_density_filter
from mujik.merge.reduce import piano_reduce
from mujik.midi.model import Note, StemName, Track
from mujik.time_signature.model import TimeSignatureSegment


@dataclass
class MergeReport:
    """合并报告。"""

    mode: str
    output_tracks: list[StemName]
    notes_in: int = 0
    notes_out: int = 0
    dropped_by_density: int = 0
    dropped_by_reduction: int = 0

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "output_tracks": list(self.output_tracks),
            "notes_in": self.notes_in,
            "notes_out": self.notes_out,
            "dropped_by_density": self.dropped_by_density,
            "dropped_by_reduction": self.dropped_by_reduction,
        }


def _is_drum_track(track: Track) -> bool:
    """判断鼓轨：channel==9 或 stem_name=="drums"。"""
    if track.channel == 9:
        return True
    if track.stem_name == "drums":
        return True
    return False


def _is_vocal_track(track: Track) -> bool:
    return track.stem_name == "vocals"


def _combined_track(notes: list[Note], channel: int = 0) -> Track:
    """构造 combined Track。"""
    t = Track(stem_name="combined", channel=channel)
    for n in notes:
        t.add(n)
    t.sort_by_start()
    return t


def _piano_track(notes: list[Note]) -> Track:
    """构造 piano reduction Track。"""
    t = Track(stem_name="piano_reduction", channel=0)
    for n in notes:
        t.add(n)
    t.sort_by_start()
    return t


def _all_mode(
    tracks: dict[StemName, Track],
    config: MergeConfig,
) -> tuple[dict[StemName, Track], int, int]:
    """mode="all"：合并除 drums 外的所有轨为 combined 轨 + drums 保留。

    Returns: (output_tracks, dropped_by_density, dropped_total)
    """
    combined_notes: list[Note] = []
    output: dict[StemName, Track] = {}
    dropped_by_density = 0

    for stem, track in tracks.items():
        if _is_drum_track(track):
            # 鼓保留
            output[stem] = track
        elif _is_vocal_track(track) and config.preserve_voice_separate:
            # 人声保留
            output[stem] = track
        else:
            combined_notes.extend(track.notes)

    if combined_notes:
        if config.density_filter and config.max_simultaneous_notes > 0:
            combined_notes, dropped = apply_density_filter(
                combined_notes, config.max_simultaneous_notes,
            )
            dropped_by_density = dropped
        output["combined"] = _combined_track(combined_notes)

    return output, dropped_by_density, 0


def _piano_reduction_mode(
    tracks: dict[StemName, Track],
    config: MergeConfig,
) -> tuple[dict[StemName, Track], int, int]:
    """mode="piano_reduction"：启发式钢琴缩减 + drums/vocals 保留。"""
    all_notes: list[Note] = []
    output: dict[StemName, Track] = {}
    dropped_by_density = 0
    dropped_by_reduction = 0

    for stem, track in tracks.items():
        if _is_drum_track(track):
            output[stem] = track
        elif _is_vocal_track(track) and config.preserve_voice_separate:
            output[stem] = track
        else:
            all_notes.extend(track.notes)

    if all_notes:
        if config.density_filter and config.max_simultaneous_notes > 0:
            all_notes, dropped_by_density = apply_density_filter(
                all_notes, config.max_simultaneous_notes,
            )
        reduced, dropped = piano_reduce(all_notes, config.max_simultaneous_notes)
        dropped_by_reduction = dropped
        output["piano_reduction"] = _piano_track(reduced)
    else:
        output["piano_reduction"] = _piano_track([])

    return output, dropped_by_density, dropped_by_reduction


def _score_mode(
    tracks: dict[StemName, Track],
    config: MergeConfig,
) -> tuple[dict[StemName, Track], int, int]:
    """mode="score"：no-op，所有 stem 保留。density filter 仍可应用（per-track）。"""
    output: dict[StemName, Track] = {}
    dropped_by_density = 0

    for stem, track in tracks.items():
        notes = list(track.notes)
        if (
            config.density_filter
            and config.max_simultaneous_notes > 0
            and not _is_drum_track(track)
            and not (_is_vocal_track(track) and config.preserve_voice_separate)
        ):
            notes, dropped = apply_density_filter(notes, config.max_simultaneous_notes)
            dropped_by_density += dropped
        new_track = Track(
            stem_name=track.stem_name,
            notes=notes,
            instrument=track.instrument,
            channel=track.channel,
        )
        new_track.sort_by_start()
        output[stem] = new_track

    return output, dropped_by_density, 0


def merge_tracks(
    tracks: dict[StemName, Track],
    config: MergeConfig,
    time_signatures: list[TimeSignatureSegment] | None = None,
    bpm: float = 120.0,
) -> tuple[dict[StemName, Track], MergeReport]:
    """合并多轨为指定模式的输出。

    Args:
        tracks: 输入 stem → Track 映射
        config: MergeConfig
        time_signatures: 占位（v0.2.3 不用，留作 v0.4+ music-aware 接口）
        bpm: 占位（同上）

    Returns:
        (output_tracks_dict, report)
    """
    notes_in = sum(len(t.notes) for t in tracks.values())

    if config.mode == "all":
        output, d_density, d_reduce = _all_mode(tracks, config)
    elif config.mode == "piano_reduction":
        output, d_density, d_reduce = _piano_reduction_mode(tracks, config)
    elif config.mode == "score":
        output, d_density, d_reduce = _score_mode(tracks, config)
    else:
        raise ValueError(f"unknown merge mode: {config.mode!r}")

    notes_out = sum(len(t.notes) for t in output.values())
    report = MergeReport(
        mode=config.mode,
        output_tracks=list(output.keys()),
        notes_in=notes_in,
        notes_out=notes_out,
        dropped_by_density=d_density,
        dropped_by_reduction=d_reduce,
    )
    return output, report


__all__ = [
    "MergeReport",
    "merge_tracks",
]
