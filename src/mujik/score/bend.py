"""MusicXML <bend> 元素生成（v0.4.1 引入，v0.4.3 扩展连续曲线）。

把 ``Note.pitch_bend``（basic-pitch 输出的 normalized [-1, +1] per-frame
序列）映射为 MusicXML 3.1 ``<bend>`` 元素。

设计决策（v0.4.1 → v0.4.3）：

v0.4.1（初代）：
- 简化：basic-pitch 的 pitch_bend 不是 semitone 整数。映射策略是
  "**最大绝对帧 × 2**"，钳制到 [-2, +2]。±1 = 半个音（MusicXML 半个音支持
  受限但 Verovio 接受整数 alter；±1 表示一个全音差，简化到 ±2 上限）。
- 0 → 不发 ``<bend>`` 元素（视为无弯音）
- 不渲染连续 bend 曲线：只发单点最大 bend
- MusicXML 3.1 规范（partwise.dtd）：
  ``<bend alter="..."><bend-alter>...</bend-alter></bend>``，
  ``alter`` 范围 [-2, +2]（Verovio 6.x 实测接受整数 alter；非整数将被 Verovio
  截断或忽略）。

v0.4.3（曲线渲染）：
- MusicXML 3.1 不支持 N 控制点 Bezier 曲线，但**支持同一 <note> 内多个
  <bend> 兄弟**（W3C PR #394，"bend + release" 双 bend 模式）
- 算法：检测 peak + 是否 return-to-0
  - 若 return-to-0 存在：发 2 个 <bend> 兄弟（positive alter + negative
    alter + ``<release/>`` marker）
  - 若无 return-to-0：发 1 个 <bend>（保留 v0.4.1 行为）+ 新增
    ``shape="curved"`` 属性（更接近 Guitar Pro / MuseScore 4 风格）
- ``BendPoint`` 数据类：``time_frac`` ∈ [0, 1] + ``alter`` ∈ [-2, +2]
- ``detect_bend_release(pitch_bend)`` → ``(peak_alter, has_release)``
- ``build_bend_elements(curve)`` → 多 <bend> 兄弟 XML
- Verovio 5.0+ 完整支持 multi-bend + release（5.4+ 支持 shape="straight/curved"）

参考：
- MusicXML 3.1 spec: bend 元素（partwise.dtd）
- W3C PR #394 (multiple bend per note): https://github.com/w3c-cg/musicxml/pull/394
- Verovio bend support: https://book.verovio.org/toolkit-reference/mei-support.html
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# MusicXML 3.1 <bend-alter> 合法范围（Verovio 5.x/6.x 接受整数）
BEND_ALTER_MIN = -2
BEND_ALTER_MAX = 2

# Release detection threshold：peak 后 |b| < 此值视为回到 0（normalized 单位）
BEND_RELEASE_THRESHOLD = 0.1

# shape 属性合法值
BendShape = Literal["straight", "curved", "hold"]


@dataclass(frozen=True)
class BendPoint:
    """Bend 曲线上的一个关键点。

    Attributes:
        time_frac: 该点时间位置在 note 时长内的比例，``[0, 1]``。
            0.0 = note 起始，1.0 = note 结束。
        alter: 该点对应的 ``<bend-alter>`` 值（整数半音），``[-2, +2]``。
            正数 = 上弯（pitch 提高），负数 = 下弯（pitch 降低）。
    """

    time_frac: float
    alter: int

    def __post_init__(self) -> None:
        if not 0.0 <= self.time_frac <= 1.0:
            raise ValueError(
                f"time_frac must be in [0, 1], got {self.time_frac}"
            )
        if not (BEND_ALTER_MIN <= self.alter <= BEND_ALTER_MAX):
            raise ValueError(
                f"alter must be in [{BEND_ALTER_MIN}, {BEND_ALTER_MAX}], "
                f"got {self.alter}"
            )


def pitch_bend_to_alter(pitch_bend: tuple[float, ...]) -> int:
    """``Note.pitch_bend`` 序列 → MusicXML ``<bend-alter>`` 整数。

    聚合策略：取所有帧中**绝对值最大**的帧（保留符号），乘以 2 映射为半音，
    钳制到 ``[-2, +2]``。正向弯音返回正数，负向弯音返回负数。

    v0.4.1 旧 API，保留向后兼容（builder.py 不再调用，但其他模块可能还在用）。

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


def detect_bend_release(
    pitch_bend: tuple[float, ...],
    release_threshold: float = BEND_RELEASE_THRESHOLD,
) -> tuple[int, bool]:
    """检测弯音曲线是否有 release 模式（peak 后返回 0）。

    算法：
        1. 找 peak（|b| 最大的帧）→ 映射 alter = round(peak * 2)，clamp [-2, +2]
        2. 取 peak 后所有帧；若 ``min(|b|) < release_threshold`` 视为有 release
        3. 全部为 0 或未到 peak → 无 release

    Args:
        pitch_bend: normalized [-1, +1] 帧序列（basic-pitch 输出）
        release_threshold: release 判定阈值（normalized 单位，默认 0.1）

    Returns:
        ``(peak_alter, has_release)`` 元组。``peak_alter=0`` 表示无明显弯音。
        ``has_release=True`` 表示 peak 后回到 0，builder 发 2 个 <bend> 兄弟。

    Examples:
        >>> detect_bend_release(())
        (0, False)
        >>> detect_bend_release((0.0, 0.0, 0.0))
        (0, False)
        >>> detect_bend_release((0.0, 0.3, 0.5, 0.3, 0.0))
        (1, True)
        >>> detect_bend_release((0.0, 0.3, 0.5, 0.5, 0.5))
        (1, False)
    """
    if not pitch_bend:
        return (0, False)

    # 1. 找 peak（保持符号）
    peak_idx = max(range(len(pitch_bend)), key=lambda i: abs(pitch_bend[i]))
    peak_b = float(pitch_bend[peak_idx])
    peak_alter = max(BEND_ALTER_MIN, min(BEND_ALTER_MAX, round(peak_b * 2)))
    if peak_alter == 0:
        return (0, False)

    # 2. 检查 peak 后是否回到 0
    post_peak = pitch_bend[peak_idx + 1:]
    if not post_peak:
        return (peak_alter, False)  # 无后续帧（peak 在最后）

    min_post_abs = min(abs(float(b)) for b in post_peak)
    has_release = min_post_abs < release_threshold

    return (peak_alter, has_release)


def build_bend_element(
    alter: int,
    shape: BendShape = "curved",
    release: bool = False,
) -> str:
    """生成 MusicXML ``<bend>`` 元素 XML 字符串。

    v0.4.3 扩展：可选 ``shape`` 属性 + ``<release/>`` marker。

    Args:
        alter: ``<bend-alter>`` 值，范围 ``[-2, +2]``
        shape: 曲线形状（默认 ``"curved"``；Verovio 5.4+ 支持）
        release: True 时附加 ``<release/>`` marker（标识"此 bend 为释放"）

    Returns:
        XML 字符串，例：
        - ``<bend shape="curved"><bend-alter>1</bend-alter></bend>``
        - ``<bend shape="curved"><bend-alter>-1</bend-alter><release/></bend>``

    Raises:
        ValueError: alter 超出 ``[-2, +2]`` 范围
    """
    if not (BEND_ALTER_MIN <= alter <= BEND_ALTER_MAX):
        raise ValueError(
            f"alter must be in [{BEND_ALTER_MIN}, {BEND_ALTER_MAX}], got {alter}"
        )
    release_xml = "<release/>" if release else ""
    return (
        f'<bend shape="{shape}">'
        f'<bend-alter>{alter}</bend-alter>'
        f'{release_xml}'
        f'</bend>'
    )


def build_bend_elements(
    curve: list[BendPoint],
    shape: BendShape = "curved",
) -> str:
    """从 ``BendPoint`` 列表生成多 ``<bend>`` 兄弟。

    简化策略（v0.4.3）：

    - **空列表**：返回空串（builder 不发任何 <bend>）
    - **1 个 BendPoint**：发 1 个 ``<bend>``（含 alter + shape）
    - **2 个 BendPoint 且 ``curve[0].alter > 0`` 且 ``curve[1].alter <= 0``**：
      发 2 个 ``<bend>`` 兄弟（"bend up + release"）—— 这就是 MusicXML 3.1
      spec 推荐的 bend+release 编码
    - **N 个 BendPoint**：发 N 个 ``<bend>``（每个独立 alter，无 release marker）

    Args:
        curve: ``BendPoint`` 列表
        shape: 曲线形状（应用到所有 <bend> 兄弟）

    Returns:
        XML 字符串（无换行）。builder 负责在前面加换行缩进。

    Examples:
        >>> from mujik.score.bend import BendPoint, build_bend_elements
        >>> xml = build_bend_elements([BendPoint(0.0, 1)])
        >>> xml
        '<bend shape="curved"><bend-alter>1</bend-alter></bend>'

        >>> xml = build_bend_elements([BendPoint(0.0, 1), BendPoint(1.0, 0)])
        >>> 'release' in xml and 'bend-alter>-1' in xml
        True
    """
    if not curve:
        return ""

    # 2-point bend + release（支持上弯后释放和下弯后释放两种模式）
    if (
        len(curve) == 2
        and curve[0].alter != 0
        and curve[1].alter == 0
    ):
        peak_alter = curve[0].alter
        release_alter = -peak_alter  # 与 peak 异号（回到 0）
        return (
            build_bend_element(peak_alter, shape=shape, release=False)
            + build_bend_element(release_alter, shape=shape, release=True)
        )

    # 1-point 或 N-point：每个 BendPoint 1 个 <bend>（无 release marker）
    return "".join(
        build_bend_element(p.alter, shape=shape, release=False)
        for p in curve
    )


__all__ = [
    "BEND_ALTER_MIN",
    "BEND_ALTER_MAX",
    "BEND_RELEASE_THRESHOLD",
    "BendShape",
    "BendPoint",
    "pitch_bend_to_alter",
    "detect_bend_release",
    "build_bend_element",
    "build_bend_elements",
]
