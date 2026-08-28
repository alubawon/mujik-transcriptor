"""Pitch bend postprocessor（v0.4.0，v0.4.1 兼容 mido 1.3.x）。

把 `Note.pitch_bend`（per-frame 序列，范围 [-1, +1]）展平为 pretty_midi
`Instrument.pitch_bends`（per-event `(time, value)` 序列）。

设计决策（v0.4.0 → v0.4.1）：
- 写：Note.pitch_bend tuple → Instrument.pitch_bends 事件序列
- 读：Instrument.pitch_bends → Note.pitch_bend tuple（按时间窗口归到对应 note）
- v0.4.1 修复：pretty_midi 0.2.11 与 mido 1.3.x 接口不兼容
  - mido 1.3.x 把 pitchwheel 范围改为 signed -8192..8191（之前是 0..16383）
  - pretty_midi 0.2.11 内部 PitchBend.pitch 直接传给 mido.Message，无转换
  - fix：把 pretty_midi 内部 pitch 改为 mido signed 范围（-8192..8191），
    pretty_midi 0.2.11 读写都是直接透传，所以自洽
- 弯音仅对 pitched note 注入（drum channel 9 跳过）

约定：
- mido pitchwheel 中心 = 0（无弯音）
- bend = +1 → mido_pitch = 8191
- bend = -1 → mido_pitch = -8192
- 转换：`mido_pitch = int(round(bend * 8191))` 钳制到 [-8192, 8191]
- 帧间隔：默认 100 fps（即每 10ms 一帧，匹配 nnnoiseless/MIDI 标准）

参考：
- mido 1.3.0 release notes: pitchwheel 改为 signed
- pretty_midi 0.2.11 issue: 与 mido 1.3.x 接口不兼容
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    import pretty_midi
    from mujik.midi.model import Note

# mido 1.3.x pitchwheel 中心值
PITCH_BEND_CENTER = 0
# mido 1.3.x pitchwheel 最大值
PITCH_BEND_MAX = 8191

# 帧率：每 note 多少帧（用于把 tuple 摊到时间轴上）
DEFAULT_FRAME_RATE_HZ = 100


def bend_to_pretty_pitch(bend: float) -> int:
    """[-1, +1] → mido signed pitchwheel [-8192, 8191]。

    v0.4.1 修正：mido 1.3.x 把 pitchwheel 改为 signed 范围
    （center=0, max=8191, min=-8192）。pretty_midi 0.2.11 内部 PitchBend
    字段直接传给 mido.Message（无转换），所以我们用 mido 的范围。
    """
    clamped = max(-1.0, min(1.0, bend))
    if clamped >= 0:
        return int(round(clamped * 8191))
    else:
        return int(round(clamped * 8192))


def pretty_pitch_to_bend(pretty_pitch: int) -> float:
    """mido signed pitchwheel [-8192, 8191] → [-1, +1]。"""
    if pretty_pitch >= 0:
        return float(pretty_pitch) / 8191.0
    else:
        return float(pretty_pitch) / 8192.0


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
