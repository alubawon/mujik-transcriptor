"""Pitch bend postprocessor（v0.4.0）。

把 `Note.pitch_bend`（per-frame 序列，范围 [-1, +1]）展平为 pretty_midi
`Instrument.pitch_bends`（per-event `(time, value)` 序列，范围 [0, 16383]）。

设计决策（v0.4.0）：
- 写：Note.pitch_bend tuple → Instrument.pitch_bends 事件序列
- 读：Instrument.pitch_bends → Note.pitch_bend tuple（按时间窗口归到对应 note）
- 不修改乐谱（Verovio 6.x `<bend>` 支持有限，留 v0.4.1+）
- 弯音仅对 pitched note 注入（drum channel 9 跳过）

约定：
- pretty_midi pitch_bend 中心 = 8192（无弯音）
- bend = +1 → pretty_pitch = 16383
- bend = -1 → pretty_pitch = 0
- 转换：`pretty_pitch = int((bend + 1) / 2 * 16383)` 钳制到 [0, 16383]
- 帧间隔：默认 100 fps（即每 10ms 一帧，匹配 nnnoiseless/MIDI 标准）
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    import pretty_midi
    from mujik.midi.model import Note

# pretty_midi 弯音中心值
PITCH_BEND_CENTER = 8192
PITCH_BEND_MAX = 16383

# 帧率：每 note 多少帧（用于把 tuple 摊到时间轴上）
DEFAULT_FRAME_RATE_HZ = 100


def bend_to_pretty_pitch(bend: float) -> int:
    """[-1, +1] → [0, 16383]。"""
    clamped = max(-1.0, min(1.0, bend))
    pretty = (clamped + 1.0) / 2.0 * PITCH_BEND_MAX
    return int(round(pretty))


def pretty_pitch_to_bend(pretty_pitch: int) -> float:
    """[0, 16383] → [-1, +1]。"""
    return (pretty_pitch / PITCH_BEND_MAX) * 2.0 - 1.0


def inject_pitch_bends_to_pretty_midi(
    pretty_instrument: "pretty_midi.Instrument",
    notes: list["Note"],
    frame_rate_hz: int = DEFAULT_FRAME_RATE_HZ,
) -> int:
    """把 `Note.pitch_bend` 序列展平写入 pretty_midi Instrument。

    Args:
        pretty_instrument: pretty_midi.Instrument 对象（in-place 修改 pitch_bends 列表）
        notes: 对应的 Note 列表
        frame_rate_hz: 帧率，用于把 tuple 摊到时间轴

    Returns:
        注入的 pitch_bend 事件数
    """
    if frame_rate_hz <= 0:
        raise ValueError(f"frame_rate_hz must be > 0, got {frame_rate_hz}")

    n_events = 0
    for note in notes:
        if not note.pitch_bend:
            continue  # 无弯音
        if note.end <= note.start:
            continue  # 零长 note

        n_frames = len(note.pitch_bend)
        duration = note.end - note.start
        dt = duration / max(n_frames, 1)

        for i, bend in enumerate(note.pitch_bend):
            t = note.start + i * dt
            if t < 0:
                continue
            pretty_pitch = bend_to_pretty_pitch(float(bend))
            # pretty_midi 的 PitchBend(time, pitch) 命名冲突，用 module ref
            import pretty_midi as _pm
            pretty_instrument.pitch_bends.append(_pm.PitchBend(time=t, pitch=pretty_pitch))
            n_events += 1

    if n_events > 0:
        # 按时间排序（pretty_midi 内部依赖）
        pretty_instrument.pitch_bends.sort(key=lambda pb: pb.time)
        logger.debug(
            "injected {n} pitch_bend events into {inst}",
            n=n_events, inst=pretty_instrument.name,
        )
    return n_events


def extract_pitch_bends_from_pretty_midi(
    pretty_instrument: "pretty_midi.Instrument",
    notes: list["Note"],
    frame_rate_hz: int = DEFAULT_FRAME_RATE_HZ,
) -> list["Note"]:
    """从 pretty_midi Instrument.pitch_bends 反向提取为 Note.pitch_bend tuple。

    Args:
        pretty_instrument: pretty_midi.Instrument
        notes: 原始 notes（用于确定时间窗口和通道）
        frame_rate_hz: 输出帧率

    Returns:
        新 Note 列表（带 pitch_bend 字段）
    """
    from mujik.midi.model import Note

    if not pretty_instrument.pitch_bends:
        return notes

    # 按时间窗口把 pitch_bend 事件归到对应 note
    out_notes: list[Note] = []
    for note in notes:
        # 找落在 [note.start, note.end] 范围内的事件
        relevant: list[tuple[float, int]] = []
        for pb in pretty_instrument.pitch_bends:
            if note.start - 1e-6 <= pb.time < note.end + 1e-6:
                relevant.append((pb.time, pb.pitch))
        if not relevant:
            out_notes.append(note)
            continue
        # 重新按帧率采：先按时间排序，再均匀重采样
        relevant.sort(key=lambda x: x[0])
        n_frames = max(
            1,
            int(round((note.end - note.start) * frame_rate_hz)),
        )
        # 实际采 n_frames 个时间点
        bend_values: list[float] = []
        for i in range(n_frames):
            t_target = note.start + i / frame_rate_hz
            # 找最近的 pitch_bend 事件
            best = min(relevant, key=lambda x: abs(x[0] - t_target))
            bend_values.append(pretty_pitch_to_bend(best[1]))
        new_note = Note(
            start=note.start,
            end=note.end,
            pitch=note.pitch,
            velocity=note.velocity,
            channel=note.channel,
            pitch_bend=tuple(bend_values),
            articulation=note.articulation,
        )
        out_notes.append(new_note)
    return out_notes


__all__ = [
    "PITCH_BEND_CENTER",
    "PITCH_BEND_MAX",
    "DEFAULT_FRAME_RATE_HZ",
    "bend_to_pretty_pitch",
    "pretty_pitch_to_bend",
    "inject_pitch_bends_to_pretty_midi",
    "extract_pitch_bends_from_pretty_midi",
]
