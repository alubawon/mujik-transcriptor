"""MusicXML <bend> 元素生成（v0.4.1）。

把 ``Note.pitch_bend``（basic-pitch 输出的 normalized [-1, +1] per-frame
序列）映射为 MusicXML 3.1 ``<bend>`` 元素，输出 ``<bend-alter>`` 半音整数。

设计决策（v0.4.1）：
- 简化：basic-pitch 的 pitch_bend 不是 semitone 整数。映射策略是
  "**最大绝对帧 × 2**"，钳制到 [-2, +2]。±1 = 半个音（MusicXML 半个音支持
  受限但 Verovio 接受整数 alter；±1 表示一个全音差，简化到 ±2 上限）。
  实际 v0.4.1 把 ±1 视为 ±1 个全音，原因是基本 pitch 最大摆动通常 < 1。
- 0 → 不发 ``<bend>`` 元素（视为无弯音）
- 不渲染连续 bend 曲线：只发单点最大 bend（形状 ``shape="line"``）
  曲线渲染留 v0.4.2+。
- MusicXML 3.1 规范（partwise.dtd）：
  ``<bend alter="..."><bend-alter>...</bend-alter></bend>``，
  ``alter`` 范围 [-2, +2]（Verovio 6.x 实测接受整数 alter；非整数将被 Verovio
  截断或忽略）。

参考：
- MusicXML 3.1 spec: bend 元素（partwise.dtd）
- Verovio 6.x 实测：``<bend-alter>`` 接受整数
"""
from __future__ import annotations


# MusicXML 3.1 <bend-alter> 合法范围（Verovio 6.x 接受整数）
BEND_ALTER_MIN = -2
BEND_ALTER_MAX = 2


def pitch_bend_to_alter(pitch_bend: tuple[float, ...]) -> int:
    """``Note.pitch_bend`` 序列 → MusicXML ``<bend-alter>`` 整数。

    聚合策略：取所有帧中**绝对值最大**的帧（保留符号），乘以 2 映射为半音，
    钳制到 ``[-2, +2]``。正向弯音返回正数，负向弯音返回负数。

    Args:
        pitch_bend: normalized [-1, +1] 的帧序列（basic-pitch 输出格式）

    Returns:
        ``<bend-alter>`` 整数值，范围 ``[-2, +2]``。0 表示无弯音，
        builder 应跳过 ``<bend>`` 渲染。
    """
    if not pitch_bend:
        return 0

    # 保留符号的最大幅度帧（取 max(|x|) 对应的 x，保持符号）
    peak_frame = max(pitch_bend, key=lambda b: abs(float(b)))
    # 基本音 ±1 在 MIDI 中通常对应一个全音，所以乘 2 把基本音摆动映射到 1 个全音
    alter = round(float(peak_frame) * 2)
    alter = max(BEND_ALTER_MIN, min(BEND_ALTER_MAX, alter))
    return int(alter)


def build_bend_element(alter: int) -> str:
    """生成 MusicXML ``<bend>`` 元素 XML 字符串。

    Args:
        alter: ``<bend-alter>`` 值，范围 ``[-2, +2]``

    Returns:
        形如 ``<bend alter="1"><bend-alter>1</bend-alter></bend>`` 的 XML 字符串

    Note:
        简化形状：``shape="line"``、无 ``<release>``。完整曲线留 v0.4.2+。
    """
    if not (BEND_ALTER_MIN <= alter <= BEND_ALTER_MAX):
        raise ValueError(
            f"alter must be in [{BEND_ALTER_MIN}, {BEND_ALTER_MAX}], got {alter}"
        )
    return (
        f'<bend alter="{alter}">'
        f'<bend-alter>{alter}</bend-alter>'
        f'</bend>'
    )


__all__ = [
    "BEND_ALTER_MIN",
    "BEND_ALTER_MAX",
    "pitch_bend_to_alter",
    "build_bend_element",
]
