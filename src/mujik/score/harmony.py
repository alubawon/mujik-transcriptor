"""MusicXML <harmony> 元素生成（v0.4.1）。

把 ``ChordEvent``（root + quality + 可选 bass）映射为 MusicXML 3.1
``<harmony>`` 元素，输出 ``<root>`` / ``<kind>`` / 可选 ``<bass>``。

设计决策（v0.4.1）：
- 只支持 root + quality 简写（major / minor / 7 / maj7 / m7 / dim / aug）；
  爵士扩展（9 / 11 / 13 / alt）透传字符串到 ``<kind>``（Verovio 6.x 支持
  自定义 kind 字符串）。
- bass slash chord 走 ``<bass>`` 子元素。
- quality 字符串空 → ``major``（默认三和弦）
- root 解析规则：单字母 ``C/D/E/F/G/A/B`` + 可选 ``#/b`` 转 ``<root-step>``
  + ``<root-alter>``。例如 ``F#`` → step=F alter=1，``Bb`` → step=B alter=-1。
- 时间窗口查找：``find_chord_at_time`` 在 ``chord_track`` 列表里找覆盖
  ``t`` 的第一个 chord（chord 不重叠假设）。

MusicXML 3.1 格式（partwise.dtd）::

    <harmony>
      <root><root-step>C</root-step>        <!-- 必选 -->
             {<root-alter>1</root-alter>    <!-- 升降号可选 -->
      </root>
      <kind>major</kind>                    <!-- 必选 -->
      {<bass><bass-step>D</bass-step>...</bass>  <!-- slash chord 可选 -->
    </harmony>
"""
from __future__ import annotations

import re

from mujik.midi.model import ChordEvent


# quality → MusicXML <kind> 映射（MusicXML 3.1 标准值）
# 透传：未在表中时，原样写到 <kind>（Verovio 接受）
QUALITY_TO_KIND: dict[str, str] = {
    "": "major",
    "maj": "major",
    "major": "major",
    "M": "major",
    "m": "minor",
    "min": "minor",
    "minor": "minor",
    "-": "minor",
    "7": "dominant",
    "dom": "dominant",
    "dominant": "dominant",
    "maj7": "major-seventh",
    "M7": "major-seventh",
    "major7": "major-seventh",
    "m7": "minor-seventh",
    "min7": "minor-seventh",
    "minor7": "minor-seventh",
    "dim": "diminished",
    "diminished": "diminished",
    "aug": "augmented",
    "augmented": "augmented",
    "+": "augmented",
}

# 解析 root: 单字母 + 可选 #/b
_ROOT_RE = re.compile(r"^([A-Ga-g])(#|b)?$")


def _parse_root(root: str) -> tuple[str, int]:
    """解析 root 字符串 → (step, alter)。

    Args:
        root: 例如 "C", "F#", "Bb"（大小写不敏感）

    Returns:
        (step, alter): step ∈ {A..G} 大写，alter ∈ {-1, 0, +1}

    Raises:
        ValueError: 解析失败
    """
    m = _ROOT_RE.match(root.strip())
    if not m:
        raise ValueError(f"invalid root: {root!r}")
    step = m.group(1).upper()
    accidental = m.group(2)
    if accidental == "#":
        alter = 1
    elif accidental == "b":
        alter = -1
    else:
        alter = 0
    return step, alter


def build_harmony_element(chord: ChordEvent) -> str:
    """``ChordEvent`` → MusicXML ``<harmony>`` 元素 XML 字符串。

    Args:
        chord: ChordEvent（start, end, root, quality, bass）

    Returns:
        形如::

            <harmony>
              <root><root-step>C</root-step></root>
              <kind>major</kind>
            </harmony>
    """
    step, alter = _parse_root(chord.root)
    alter_xml = f"<root-alter>{alter}</root-alter>" if alter != 0 else ""
    kind = QUALITY_TO_KIND.get(chord.quality, chord.quality or "major")

    bass_xml = ""
    if chord.bass:
        b_step, b_alter = _parse_root(chord.bass)
        b_alter_xml = f"<bass-alter>{b_alter}</bass-alter>" if b_alter != 0 else ""
        bass_xml = (
            f"<bass>"
            f"<bass-step>{b_step}</bass-step>"
            f"{b_alter_xml}"
            f"</bass>"
        )

    return (
        f"<harmony>"
        f"<root>"
        f"<root-step>{step}</root-step>"
        f"{alter_xml}"
        f"</root>"
        f"<kind>{kind}</kind>"
        f"{bass_xml}"
        f"</harmony>"
    )


def find_chord_at_time(
    chord_track: list[ChordEvent] | None,
    t: float,
) -> ChordEvent | None:
    """在 ``chord_track`` 中找覆盖时间 ``t`` 的 chord。

    Args:
        chord_track: ChordEvent 列表（按 start 排序，无重叠假设）
        t: 目标时间（秒）

    Returns:
        第一个 start <= t < end 的 ChordEvent；找不到返回 None。
    """
    if not chord_track:
        return None
    for chord in chord_track:
        if chord.start <= t < chord.end:
            return chord
    return None


__all__ = [
    "QUALITY_TO_KIND",
    "build_harmony_element",
    "find_chord_at_time",
]
