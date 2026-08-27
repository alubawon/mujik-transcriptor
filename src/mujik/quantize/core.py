"""Quantize core：把 track / project 里的 note 时间量化到 grid。

职责：
- quantize_track: 单轨量化，返回新 Track + stats
- quantize_project: 全工程量化，读 MIDI → 量化 → 写回
- QuantizeReport / TrackQuantizeStats: 报告 dataclass

约定（v0.2.3）：
- 量化只动 start / end，不动 pitch / velocity / pitch_bend / articulation
- strength=0 → 不动；strength=1 → 完全 snap；中间值线性插值
- tempo 用 BeatTrack.bpm 单值（rubato 留 v0.2.4）
- groove 模板生效后，grid 偏移（swing）加在 snap 后
- Note 是 frozen dataclass → 量化后必须返回新 Note 实例
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from mujik.config.schema import QuantizeConfig
from mujik.midi.io import read_midi_to_project, write_project_to_midi
from mujik.midi.model import Note, Project, StemName, Track
from mujik.quantize.grid import (
    beat_index_at_time,
    is_8th_offbeat_position,
    snap_to_grid,
)
from mujik.quantize.groove import groove_offset_seconds
from mujik.rhythm.model import BeatTrack
from mujik.time_signature.model import TimeSignatureSegment, find_segment_for_time


@dataclass
class TrackQuantizeStats:
    """单轨量化统计。"""

    stem_name: StemName
    notes_before: int
    notes_after: int
    mean_shift_ms: float
    max_shift_ms: float
    grid_resolution: int
    groove_template: str


@dataclass
class QuantizeReport:
    """全工程量化报告。"""

    per_track: dict[StemName, TrackQuantizeStats] = field(default_factory=dict)
    total_notes_before: int = 0
    total_notes_after: int = 0
    duration_sec: float = 0.0
    strength: float = 1.0
    grid_resolution: int = 16
    groove_template: str = "straight"

    def to_dict(self) -> dict:
        return {
            "total_notes_before": self.total_notes_before,
            "total_notes_after": self.total_notes_after,
            "duration_sec": self.duration_sec,
            "strength": self.strength,
            "grid_resolution": self.grid_resolution,
            "groove_template": self.groove_template,
            "per_track": {
                stem: {
                    "stem_name": stats.stem_name,
                    "notes_before": stats.notes_before,
                    "notes_after": stats.notes_after,
                    "mean_shift_ms": stats.mean_shift_ms,
                    "max_shift_ms": stats.max_shift_ms,
                    "grid_resolution": stats.grid_resolution,
                    "groove_template": stats.groove_template,
                }
                for stem, stats in self.per_track.items()
            },
        }


def _resolve_segment_or_default(
    t: float,
    time_signatures: list[TimeSignatureSegment],
    duration: float,
) -> TimeSignatureSegment:
    """找 t 所属段；找不到时用单段 4/4 覆盖 [0, duration] 兜底。"""
    seg = find_segment_for_time(time_signatures, t)
    if seg is not None:
        return seg
    # 兜底
    return TimeSignatureSegment(
        start_time=0.0,
        end_time=max(duration, 1.0),
        time_signature=(4, 4),
        confidence=0.0,
        source="default_4_4",
    )


def _snap_with_groove(
    t: float,
    segment: TimeSignatureSegment,
    bpm: float,
    grid_resolution: int,
    groove_template: str,
) -> float:
    """snap_to_grid + groove 偏移合成。"""
    snapped = snap_to_grid(t, segment, bpm, grid_resolution)
    if groove_template == "straight":
        return snapped

    # 计算 snapped 在 grid 中的位置
    beat_idx = beat_index_at_time(snapped, segment, bpm)
    grid_pos = round(beat_idx * grid_resolution)
    # 8th offbeat 路径：
    if is_8th_offbeat_position(beat_idx, grid_resolution):
        offset_sec = groove_offset_seconds(
            beat_position=beat_idx - int(beat_idx),
            grid_position=grid_pos,
            grid_resolution=grid_resolution,
            template=groove_template,
            bpm=bpm,
        )
        return snapped + offset_sec
    return snapped


def _apply_quantize_to_note(
    note: Note,
    time_signatures: list[TimeSignatureSegment],
    bpm: float,
    config: QuantizeConfig,
    duration: float,
) -> Note:
    """单 note 量化：返回新 Note。

    start 和 end 独立量化。
    """
    seg = _resolve_segment_or_default(note.start, time_signatures, duration)

    # 1. snap
    snapped_start = _snap_with_groove(
        note.start, seg, bpm, config.grid_resolution, config.groove_template,
    )
    snapped_end = _snap_with_groove(
        note.end, seg, bpm, config.grid_resolution, config.groove_template,
    )

    # 2. strength 混合
    new_start = note.start + config.strength * (snapped_start - note.start)
    new_end = note.end + config.strength * (snapped_end - note.end)

    # 3. 保证 start < end
    if new_end <= new_start:
        new_end = new_start + 1e-3

    # 4. clamp 非负
    new_start = max(0.0, new_start)
    new_end = max(new_start + 1e-3, new_end)

    return Note(
        start=new_start,
        end=new_end,
        pitch=note.pitch,
        velocity=note.velocity,
        channel=note.channel,
        pitch_bend=note.pitch_bend,
        articulation=note.articulation,
    )


def quantize_track(
    track: Track,
    beat_track: BeatTrack,
    time_signatures: list[TimeSignatureSegment],
    config: QuantizeConfig,
    duration: float,
) -> tuple[Track, TrackQuantizeStats]:
    """量化单轨。

    Args:
        track: 输入轨道
        beat_track: 节拍跟踪结果（v0.2.3 用 bpm 单值）
        time_signatures: 时间签名段
        config: 量化配置
        duration: 工程时长（秒）

    Returns:
        (新 Track, 统计)
    """
    bpm = float(beat_track.bpm) if beat_track.bpm > 0 else 120.0

    new_notes: list[Note] = []
    shifts_ms: list[float] = []
    for note in track.notes:
        new_note = _apply_quantize_to_note(
            note, time_signatures, bpm, config, duration,
        )
        new_notes.append(new_note)
        shift_ms = abs(new_note.start - note.start) * 1000.0
        shifts_ms.append(shift_ms)

    new_track = Track(
        stem_name=track.stem_name,
        notes=new_notes,
        instrument=track.instrument,
        channel=track.channel,
    )
    new_track.sort_by_start()

    mean_shift = (sum(shifts_ms) / len(shifts_ms)) if shifts_ms else 0.0
    max_shift = max(shifts_ms) if shifts_ms else 0.0

    stats = TrackQuantizeStats(
        stem_name=track.stem_name,
        notes_before=len(track.notes),
        notes_after=len(new_notes),
        mean_shift_ms=float(mean_shift),
        max_shift_ms=float(max_shift),
        grid_resolution=config.grid_resolution,
        groove_template=config.groove_template,
    )
    return new_track, stats


def quantize_project(
    project_midi_path: Path,
    beat_track: BeatTrack,
    time_signatures: list[TimeSignatureSegment],
    config: QuantizeConfig,
    output_midi_path: Path | None = None,
) -> tuple[Project, QuantizeReport]:
    """量化整个 project。

    Args:
        project_midi_path: 读入的 project.mid 路径
        beat_track: 节拍信息
        time_signatures: 时间签名段
        config: 量化配置
        output_midi_path: 写出的 project.mid 路径；None 时覆盖输入

    Returns:
        (新 Project, 报告)
    """
    project = read_midi_to_project(str(project_midi_path))
    duration = project.duration if project.duration > 0 else (
        max((n.end for t in project.tracks.values() for n in t.notes), default=0.0)
    )
    if duration <= 0:
        duration = 1.0  # 兜底

    if not config.enabled:
        # 不量化：直接写回 + 空 report
        out_path = Path(output_midi_path) if output_midi_path else project_midi_path
        write_project_to_midi(project, out_path)
        return project, QuantizeReport(
            total_notes_before=project.total_notes(),
            total_notes_after=project.total_notes(),
            duration_sec=float(duration),
            strength=config.strength,
            grid_resolution=config.grid_resolution,
            groove_template=config.groove_template,
        )

    # 跑量化
    new_project = Project(
        audio_path=project.audio_path,
        duration=project.duration,
        sample_rate=project.sample_rate,
        time_signatures=list(project.time_signatures),
        tempo_map=list(project.tempo_map),
        tracks={},
        chord_track=project.chord_track,
        metadata=dict(project.metadata),
    )
    report = QuantizeReport(
        strength=config.strength,
        grid_resolution=config.grid_resolution,
        groove_template=config.groove_template,
    )
    report.duration_sec = float(duration)

    for stem_name, track in project.tracks.items():
        new_track, stats = quantize_track(
            track, beat_track, time_signatures, config, duration,
        )
        new_project.tracks[stem_name] = new_track
        report.per_track[stem_name] = stats
        report.total_notes_before += stats.notes_before
        report.total_notes_after += stats.notes_after

    out_path = Path(output_midi_path) if output_midi_path else project_midi_path
    write_project_to_midi(new_project, out_path)

    return new_project, report


def write_quantize_report(
    report: QuantizeReport,
    path: str | Path,
) -> None:
    """写 quantize_report.json。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


def load_beat_track_from_json(path: str | Path) -> BeatTrack:
    """从 beats.json 读 BeatTrack。"""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return BeatTrack(
        beats=list(data.get("beats", [])),
        downbeats=list(data.get("downbeats", [])),
        bpm=float(data.get("bpm", 120.0)),
        tempo_confidence=float(data.get("tempo_confidence", 0.0)),
    )


__all__ = [
    "TrackQuantizeStats",
    "QuantizeReport",
    "quantize_track",
    "quantize_project",
    "write_quantize_report",
    "load_beat_track_from_json",
]
