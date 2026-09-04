"""MIDI I/O: Project ↔ .mid (pretty-midi).

负责把内部数据模型（Note / Track / Project）转换为标准 .mid 文件，
以及反向解析（用于回环测试 + 第三方 .mid 导入）。

约定：
- Drums 永远在 channel 9 (GM 标准)
- Pitched stems 按 Track 出现顺序分配 0-8, 10-15
- Time signatures 写入 `pm.time_signature_changes`
- Tempo 写入 `pm._tempo_changes`（pretty-midi 内部接口，标准用法）
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from loguru import logger

from mujik.midi.model import (
    Note,
    Project,
    StemName,
    TempoSegment,
    Track,
)
from mujik.time_signature.model import (
    TimeSignatureSegment,
    build_default_segments,
)

# Channel 9 is reserved for drums (General MIDI).
# Pitched stems get 0-8, 10-15 in the order they appear in project.tracks.
PITCHED_CHANNELS: tuple[int, ...] = (
    0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15,
)
DRUM_CHANNEL: int = 9

# GM program numbers by stem name (rough defaults; users can override)
_DEFAULT_PROGRAMS: dict[StemName, int] = {
    "vocals": 53,   # Choir Aahs
    "drums": 0,     # is_drum=True overrides program
    "bass": 33,     # Electric Bass (finger)
    "other": 24,    # Nylon Guitar (generic)
    "piano": 0,     # Acoustic Grand Piano
    "guitar": 25,   # Steel String Guitar
}

# Reverse mapping for read-back: program → stem (best effort)
_PROGRAM_TO_STEM: dict[int, StemName] = {
    53: "vocals",
    33: "bass",
    35: "bass",     # Fretless Bass
    24: "other",
    25: "guitar",
    0: "piano",
    # ... 大量 GM 映射留 v0.2.4 完善
}


def _stem_program(stem: StemName) -> int:
    """Stem → GM program number。"""
    return _DEFAULT_PROGRAMS.get(stem, 0)


def _stem_to_instrument_name(stem: StemName) -> str:
    return f"mujik/{stem}"


# v0.4.2: muscriptor 输出的标准 instrument name → stem 映射
# muscriptor 写多轨 MIDI 时使用标准名称（Electric Guitar, Drum Kit 等），
# 这里反查支持 muscriptor 输出（不破坏现有 mujik/<stem> 前缀匹配逻辑）
_MUSCRIPTOR_NAME_TO_STEM: dict[str, StemName] = {
    # Vocals
    "vocals": "vocals",
    "vocal": "vocals",
    "voice": "vocals",
    "singing": "vocals",
    "lead vocal": "vocals",
    "backing vocal": "vocals",
    # Drums
    "drum kit": "drums",
    "drums": "drums",
    "drum": "drums",
    "percussion": "drums",
    "drum set": "drums",
    # Bass
    "bass": "bass",
    "electric bass": "bass",
    "bass guitar": "bass",
    "fretless bass": "bass",
    "acoustic bass": "bass",
    "synth bass": "bass",
    # Piano
    "piano": "piano",
    "acoustic grand piano": "piano",
    "electric piano": "piano",
    "grand piano": "piano",
    "keyboard": "piano",
    # Guitar
    "guitar": "guitar",
    "electric guitar": "guitar",
    "acoustic guitar": "guitar",
    "classical guitar": "guitar",
    "steel guitar": "guitar",
    "nylon guitar": "guitar",
    "clean guitar": "guitar",
    "distorted guitar": "guitar",
    # Other (catch-all)
    "other": "other",
}


def _instrument_name_to_stem(name: str) -> StemName | None:
    """从 pretty_midi instrument name 反查 stem。

    v0.4.2 起支持 muscriptor 输出的标准 instrument name（如 "Electric Guitar"、
    "Drum Kit"）。原有 `mujik/<stem>` 前缀匹配仍优先。
    """
    # 1. mujik 内部写出的 prefix 形式（最高优先级）
    if name.startswith("mujik/"):
        candidate = name[len("mujik/"):]
        if candidate in _DEFAULT_PROGRAMS:
            return candidate  # type: ignore[return-value]
        return None
    # 2. muscriptor 标准 instrument name（大小写不敏感）
    lower = name.lower().strip()
    if lower in _MUSCRIPTOR_NAME_TO_STEM:
        return _MUSCRIPTOR_NAME_TO_STEM[lower]
    return None


def write_project_to_midi(
    project: Project,
    output_path: str | Path,
) -> Path:
    """Project → .mid 文件。

    Args:
        project: 内部 Project 模型
        output_path: 目标 .mid 路径（不存在则创建父目录）

    Returns:
        写入的 .mid 路径（绝对路径）

    Raises:
        ImportError: pretty-midi 未安装
        OSError: 写入失败
    """
    try:
        import pretty_midi
    except ImportError as e:
        raise ImportError(
            "pretty-midi not installed; install via "
            "`uv pip install mujik-transcriptor[midi]`"
        ) from e

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Tempo（v0.2.2 多段支持：用 pretty_midi._tick_scales 直接注入）
    initial_bpm = (
        float(project.tempo_map[0].bpm)
        if project.tempo_map and project.tempo_map[0].bpm > 0
        else 120.0
    )
    resolution = 220
    pm = pretty_midi.PrettyMIDI(
        initial_tempo=initial_bpm, resolution=resolution,
    )

    if project.tempo_map:
        # 多段 → 构造 _tick_scales
        # 简化：以首段 BPM 为基准 scale，后续段 tick 时间用首段 scale 换算
        # （完整多段 tick-scale 切换留 v0.2.4）
        bpm0 = float(project.tempo_map[0].bpm)
        scales: list[tuple[int, float]] = [
            (0, 60.0 / (bpm0 * resolution))
        ]
        for seg in project.tempo_map[1:]:
            tick = int(seg.start_time * resolution * (60.0 / bpm0 / resolution))
            bpm = float(seg.bpm)
            scales.append((tick, 60.0 / (bpm * resolution)))
        pm._tick_scales = scales

    # Time signatures
    if project.time_signatures:
        for seg in project.time_signatures:
            num, den = seg.time_signature
            pm.time_signature_changes.append(
                pretty_midi.TimeSignature(
                    numerator=int(num),
                    denominator=int(den),
                    time=float(seg.start_time),
                )
            )
    else:
        pm.time_signature_changes.append(
            pretty_midi.TimeSignature(4, 4, 0.0)
        )

    # Tracks: drums first (channel 9), then pitched in order
    pitched_tracks: list[Track] = []
    drum_track: Track | None = None
    for stem_name, track in project.tracks.items():
        if stem_name == "drums":
            drum_track = track
        else:
            pitched_tracks.append(track)

    channel_idx = 0
    if drum_track is not None:
        inst = pretty_midi.Instrument(
            program=0,
            is_drum=True,
            name=_stem_to_instrument_name("drums"),
        )
        for note in drum_track.notes:
            inst.notes.append(
                pretty_midi.Note(
                    velocity=int(note.velocity),
                    pitch=int(note.pitch),
                    start=float(note.start),
                    end=float(note.end),
                )
            )
        pm.instruments.append(inst)
        logger.debug(
            "midi write: drums → {n} notes (channel={ch})",
            n=len(inst.notes), ch=DRUM_CHANNEL,
        )

    for track in pitched_tracks:
        if channel_idx >= len(PITCHED_CHANNELS):
            logger.warning(
                "midi write: too many pitched tracks (>{}); "
                "extra tracks wrap to channel 0",
                len(PITCHED_CHANNELS),
            )
            channel_idx = 0
        channel = PITCHED_CHANNELS[channel_idx]
        channel_idx += 1
        inst = pretty_midi.Instrument(
            program=_stem_program(track.stem_name),
            is_drum=False,
            name=_stem_to_instrument_name(track.stem_name),
        )
        for note in track.notes:
            inst.notes.append(
                pretty_midi.Note(
                    velocity=int(note.velocity),
                    pitch=int(note.pitch),
                    start=float(note.start),
                    end=float(note.end),
                )
            )
        # v0.4.0: 注入 pitch_bend（per-frame → pretty_midi events）
        from mujik.postprocess.pitch_bend import inject_pitch_bends_to_pretty_midi
        n_bends = inject_pitch_bends_to_pretty_midi(inst, track.notes)
        if n_bends > 0:
            logger.debug(
                "midi write: {stem} → {n} pitch_bend events",
                stem=track.stem_name, n=n_bends,
            )
        pm.instruments.append(inst)
        logger.debug(
            "midi write: {stem} → {n} notes (channel={ch})",
            stem=track.stem_name, n=len(inst.notes), ch=channel,
        )

    pm.write(str(output_path))
    logger.info(
        "midi write: {n} tracks, {m} total notes → {path}",
        n=len(pm.instruments),
        m=sum(len(i.notes) for i in pm.instruments),
        path=output_path,
    )
    return output_path


def read_midi_to_project(
    midi_path: str | Path,
    audio_path: str = "",
    sample_rate: int = 44100,
) -> Project:
    """反向：.mid → Project（最佳努力）。

    - 包含 is_drum=True 的 instrument → 'drums' stem
    - 其他按 instrument.name 反查（"mujik/<stem>" 形式）
    - 缺字段给合理默认值

    Args:
        midi_path: .mid 文件路径
        audio_path: 关联的原始音频路径（写入 Project.audio_path）
        sample_rate: 关联的采样率（写入 Project.sample_rate）

    Returns:
        Project 实例
    """
    try:
        import pretty_midi
    except ImportError as e:
        raise ImportError(
            "pretty-midi not installed; install via "
            "`uv pip install mujik-transcriptor[midi]`"
        ) from e

    midi_path = Path(midi_path)
    if not midi_path.exists():
        raise FileNotFoundError(f"midi not found: {midi_path}")

    pm = pretty_midi.PrettyMIDI(str(midi_path))

    # Duration
    duration = max(
        (inst.get_end_time() for inst in pm.instruments),
        default=0.0,
    )

    # Tempo（pretty_midi 0.2.11+ 返回 (times_array, tempos_array) tuple）
    tempo_map: list[TempoSegment] = []
    try:
        result = pm.get_tempo_changes()
        # 兼容两种返回：tuple(2 arrays) 或单 array
        if isinstance(result, tuple) and len(result) == 2:
            _, tempos = result
        else:
            tempos = result
        if len(tempos) > 0:
            bpm = float(tempos[0])
            tempo_map.append(TempoSegment(
                start_time=0.0,
                end_time=float(duration) if duration > 0 else 1.0,
                bpm=bpm,
            ))
    except (ValueError, AttributeError, IndexError) as e:
        # 少于 2 个 note 时 estimate_tempo 失败，get_tempo_changes 可能也空
        # v0.5.2: 回退前至少留一条 debug 日志（docstring 声明的最佳努力路径）
        logger.debug("read_project_from_midi: tempo parse failed ({}), fallback 120 BPM", e)
    if not tempo_map:
        logger.debug("read_project_from_midi: no tempo events, fallback 120 BPM")
        tempo_map = [TempoSegment(0.0, duration if duration > 0 else 1.0, 120.0)]

    # Time signatures
    time_sigs: list[TimeSignatureSegment] = []
    fallback_end = max(duration, 1.0)  # 防止 start==end 校验失败
    for ts in pm.time_signature_changes:
        time_sigs.append(TimeSignatureSegment(
            start_time=float(ts.time),
            end_time=fallback_end if ts.time == 0.0 else float(ts.time) + 1e6,
            time_signature=(int(ts.numerator), int(ts.denominator)),
            confidence=1.0,
            source="auto_resnet18",
        ))
    if not time_sigs:
        time_sigs = build_default_segments(fallback_end)

    # Tracks
    project = Project(
        audio_path=audio_path,
        duration=duration,
        sample_rate=sample_rate,
        time_signatures=time_sigs,
        tempo_map=tempo_map,
    )
    for inst in pm.instruments:
        if inst.is_drum:
            stem: StemName = "drums"
        else:
            stem = _instrument_name_to_stem(inst.name) or _program_to_stem(inst.program) or "other"
        track = project.get_track(stem)
        notes_list: list[Note] = []
        for n in inst.notes:
            notes_list.append(Note(
                start=float(n.start),
                end=float(n.end),
                pitch=int(n.pitch),
                velocity=int(n.velocity),
                channel=DRUM_CHANNEL if inst.is_drum else 0,
            ))
        # v0.4.0: 反向提取 pitch_bend 事件 → Note.pitch_bend
        if not inst.is_drum and inst.pitch_bends:
            from mujik.postprocess.pitch_bend import extract_pitch_bends_from_pretty_midi
            notes_list = extract_pitch_bends_from_pretty_midi(inst, notes_list)
        for n in notes_list:
            track.add(n)

    logger.info(
        "midi read: {n} tracks, {m} notes from {path}",
        n=len(pm.instruments),
        m=sum(len(i.notes) for i in pm.instruments),
        path=midi_path,
    )
    return project


def _program_to_stem(program: int) -> StemName | None:
    return _PROGRAM_TO_STEM.get(program)


def _note_to_dict(note: Note) -> dict[str, Any]:
    return {
        "start": note.start,
        "end": note.end,
        "pitch": note.pitch,
        "velocity": note.velocity,
        "channel": note.channel,
        "articulation": note.articulation,
    }


__all__ = [
    "write_project_to_midi",
    "read_midi_to_project",
    "PITCHED_CHANNELS",
    "DRUM_CHANNEL",
]
