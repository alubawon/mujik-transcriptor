"""MusicXML builder（v0.2.4 起 music21-free，v0.4.1 加 <bend>+<harmony>）。

输出格式：MusicXML 3.1 partwise（Verovio 标准输入）。

约定（v0.2.4 → v0.4.1）：
- music21-free：纯 Python + XML 字符串拼接
- 每 stem 一个 <part>，drums 单独 part
- layout="per_stem"：4-6 个独立 part
- layout="piano_reduction"：调 merge_tracks 拿 reduction 单 part + drums + vocals
- layout="score"：与 per_stem 等价（保留 switch 接口）
- PPQ 默认 480
- 跨 measure 的 note 用 <tie> 处理
- include_chord_symbols=True → 调 project.chord_track 渲染 <harmony>
  （v0.4.1 实现）
- Note.pitch_bend → 渲染 <bend>（v0.4.1 实现）
- include_lyrics：v0.2.4 no-op（Project 无 lyric 字段）
- 鼓轨（channel=9）默认 single-line percussion staff
"""
from __future__ import annotations

from typing import Literal

from loguru import logger

from mujik.config.schema import RenderConfig
from mujik.merge.core import merge_tracks
from mujik.midi.model import Note, Project, StemName, Track, TempoSegment
from mujik.score.bend import build_bend_element, pitch_bend_to_alter
from mujik.score.harmony import build_harmony_element, find_chord_at_time
from mujik.score.time_helpers import (
    bpm_at_time,
    measure_index_at_time,
    seconds_to_ticks,
    time_signature_at_time,
)
from mujik.time_signature.model import (
    TimeSignatureSegment,
    build_default_segments,
)


LayoutMode = Literal["per_stem", "piano_reduction", "score"]


class MusicXMLBuilderError(RuntimeError):
    pass


# MIDI pitch → (step, alter, octave)
def _pitch_to_xml_parts(pitch: int) -> tuple[str, int, int]:
    """MIDI pitch (0-127) → (step, alter, octave)。

    step ∈ {C, D, E, F, G, A, B}
    alter ∈ {-2, -1, 0, 1, 2} (sharps/flats)
    octave ∈ {0..9}
    """
    if not 0 <= pitch <= 127:
        raise ValueError(f"pitch out of range: {pitch}")
    # MIDI: C-1 = 0, C0 = 12, C4 = 60
    octave = (pitch // 12) - 1
    semitone_in_octave = pitch % 12
    # semitone 0=C, 1=C#, 2=D, 3=D#, 4=E, 5=F, 6=F#, 7=G, 8=G#, 9=A, 10=A#, 11=B
    mapping = [
        ("C", 0), ("C", 1), ("D", 0), ("D", 1), ("E", 0), ("F", 0),
        ("F", 1), ("G", 0), ("G", 1), ("A", 0), ("A", 1), ("B", 0),
    ]
    step, alter = mapping[semitone_in_octave]
    return step, alter, octave


def _is_drum_track(track: Track) -> bool:
    return track.channel == 9 or track.stem_name == "drums"


def _get_layout_tracks(
    project: Project,
    layout: LayoutMode,
    render_config: RenderConfig,
) -> dict[StemName, Track]:
    """根据 layout 决定 part 配置。"""
    if layout == "piano_reduction":
        from mujik.config.schema import MergeConfig
        merge_cfg = MergeConfig(
            mode="piano_reduction",
            density_filter=True,
            max_simultaneous_notes=12,
            preserve_drums=True,
            preserve_voice_separate=True,
        )
        out, _ = merge_tracks(project.tracks, merge_cfg, project.time_signatures)
        return out
    elif layout in ("per_stem", "score"):
        return project.tracks
    else:
        raise MusicXMLBuilderError(f"unknown layout: {layout}")


def _build_part_musicxml(
    part_id: str,
    part_name: str,
    track: Track,
    project: Project,
    ppq: int = 480,
    include_chord_symbols: bool = False,
) -> str:
    """构造单个 <part> 的 MusicXML XML 字符串。"""
    if _is_drum_track(track):
        return _build_drum_part_musicxml(part_id, part_name, track, project, ppq)

    notes_sorted = sorted(track.notes, key=lambda n: (n.start, -n.velocity))

    if not notes_sorted:
        # 空 part 也要写至少一个全音符休止
        return _build_empty_part_musicxml(part_id, part_name, project, ppq)

    # 切分到 measure
    measure_groups = _group_notes_by_measure(notes_sorted, project, ppq)

    measure_xmls: list[str] = []
    for mi, mnotes in enumerate(measure_groups):
        measure_xml = _build_measure_musicxml(
            mi + 1, mnotes, project, ppq, include_chord_symbols,
        )
        measure_xmls.append(measure_xml)

    measures_str = "\n    ".join(measure_xmls)
    return f"""  <part id="{part_id}">
    <measure number="1">
      <attributes>
        <divisions>{ppq}</divisions>
        <key><fifths>0</fifths></key>
        <time><beats>{project.time_signatures[0].time_signature[0] if project.time_signatures else 4}</beats><beat-type>{project.time_signatures[0].time_signature[1] if project.time_signatures else 4}</beat-type></time>
        <clef><sign>G</sign><line>2</line></clef>
      </attributes>
      {_tempo_directive_xml(project.tempo_map)}
    </measure>
    {measures_str}
  </part>"""


def _build_empty_part_musicxml(
    part_id: str, part_name: str, project: Project, ppq: int = 480,
) -> str:
    """空 part：单 measure 全音符休止。"""
    sig = project.time_signatures[0].time_signature if project.time_signatures else (4, 4)
    return f"""  <part id="{part_id}">
    <measure number="1">
      <attributes>
        <divisions>{ppq}</divisions>
        <key><fifths>0</fifths></key>
        <time><beats>{sig[0]}</beats><beat-type>{sig[1]}</beat-type></time>
        <clef><sign>G</sign><line>2</line></clef>
      </attributes>
      {_tempo_directive_xml(project.tempo_map)}
      <note>
        <rest/>
        <duration>{ppq * sig[0] * 4 // sig[1]}</duration>
        <type>whole</type>
      </note>
    </measure>
  </part>"""


def _build_drum_part_musicxml(
    part_id: str, part_name: str, track: Track, project: Project, ppq: int = 480,
) -> str:
    """鼓 part：percussion clef + 简化的 unpitched note。"""
    notes_sorted = sorted(track.notes, key=lambda n: (n.start, -n.velocity))
    if not notes_sorted:
        return _build_empty_part_musicxml(part_id, part_name, project, ppq)

    measure_groups = _group_notes_by_measure(notes_sorted, project, ppq)
    measure_xmls: list[str] = []
    for mi, mnotes in enumerate(measure_groups):
        note_xmls = []
        for note in mnotes:
            step, alter, octave = _pitch_to_xml_parts(note.pitch)
            dur = seconds_to_ticks(note.end - note.start, project.time_signatures[0], bpm_at_time(note.start, project.tempo_map), ppq)
            note_xmls.append(
                f"""        <note>
          <unpitched>
            <display-step>{step}</display-step>
            <display-octave>{octave}</display-octave>
          </unpitched>
          <duration>{max(1, dur)}</duration>
          <voice>1</voice>
          <type>quarter</type>
        </note>"""
            )
        notes_str = "\n".join(note_xmls)
        sig = project.time_signatures[0].time_signature if project.time_signatures else (4, 4)
        measure_xmls.append(
            f"""    <measure number="{mi + 1}">
      <attributes>
        <divisions>{ppq}</divisions>
        <key><fifths>0</fifths></key>
        <time><beats>{sig[0]}</beats><beat-type>{sig[1]}</beat-type></time>
        <clef><sign>percussion</sign><line>2</line></clef>
      </attributes>
      {_tempo_directive_xml(project.tempo_map)}
{notes_str}
    </measure>"""
        )
    measures_str = "\n".join(measure_xmls)
    return f"""  <part id="{part_id}">
    <measure number="1">
      <attributes>
        <divisions>{ppq}</divisions>
        <key><fifths>0</fifths></key>
        <time><beats>{project.time_signatures[0].time_signature[0] if project.time_signatures else 4}</beats><beat-type>{project.time_signatures[0].time_signature[1] if project.time_signatures else 4}</beat-type></time>
        <clef><sign>percussion</sign><line>2</line></clef>
      </attributes>
    </measure>
    {measures_str}
  </part>"""


def _group_notes_by_measure(
    notes: list[Note],
    project: Project,
    ppq: int,
) -> list[list[Note]]:
    """把 note 按所在 measure 分组。简化版：单段 4/4 单 bpm 假设。

    多段情况下，仍按 project.time_signatures[0] 的 bpm 走（v0.2.4 简化）。
    """
    if not project.time_signatures:
        # 兜底 4/4 10s
        seg = build_default_segments(10.0)[0]
    else:
        seg = project.time_signatures[0]
    bpm = bpm_at_time(0.0, project.tempo_map)

    bar_dur = seg.bar_duration_sec(bpm)
    if bar_dur <= 0:
        return [notes]

    total_bars = max(1, int(project.duration / bar_dur) + 1)
    measures: list[list[Note]] = [[] for _ in range(total_bars)]
    for note in notes:
        mi = measure_index_at_time(note.start, seg, bpm)
        mi = min(mi, total_bars - 1)
        measures[mi].append(note)

    # 丢弃空尾段
    while len(measures) > 1 and not measures[-1]:
        measures.pop()
    return measures


def _build_measure_musicxml(
    measure_num: int,
    notes: list[Note],
    project: Project,
    ppq: int,
    include_chord_symbols: bool,
) -> str:
    """构造单个 <measure> 的 XML。

    v0.4.1 起：
    - ``include_chord_symbols=True`` 时，若 ``project.chord_track`` 有
      覆盖该 measure 起始时间的 chord，在第一个 note 之前插入 ``<harmony>``
    - ``note.pitch_bend`` 非空且 alter != 0 时，在该 note ``<type>`` 之后
      插入 ``<bend>`` 元素
    """
    # v0.4.1: 找覆盖 measure 起始时间的 chord（measure 内仅发一次 harmony）
    harmony_xml = ""
    if include_chord_symbols and project.chord_track and notes:
        first_note_start = notes[0].start
        chord = find_chord_at_time(project.chord_track, first_note_start)
        if chord is not None:
            harmony_xml = "      " + build_harmony_element(chord) + "\n"

    note_xmls: list[str] = []
    for note in notes:
        step, alter, octave = _pitch_to_xml_parts(note.pitch)
        bpm = bpm_at_time(note.start, project.tempo_map)
        dur = seconds_to_ticks(note.end - note.start, project.time_signatures[0] if project.time_signatures else build_default_segments(10.0)[0], bpm, ppq)
        dur = max(1, dur)
        alter_xml = f"<alter>{alter}</alter>" if alter != 0 else ""
        # duration type (simplified)
        type_str = _ticks_to_type(dur, ppq)
        # v0.4.1: pitch_bend → <bend>
        bend_alter = pitch_bend_to_alter(note.pitch_bend)
        bend_xml = ""
        if bend_alter != 0:
            bend_xml = "\n          " + build_bend_element(bend_alter)
        note_xmls.append(
            f"""        <note>
          <pitch>
            <step>{step}</step>
            {alter_xml}
            <octave>{octave}</octave>
          </pitch>
          <duration>{dur}</duration>
          <voice>1</voice>
          <type>{type_str}</type>{bend_xml}
        </note>"""
        )
    notes_str = "\n".join(note_xmls) if note_xmls else _rest_xml(ppq)
    sig = project.time_signatures[0].time_signature if project.time_signatures else (4, 4)
    return f"""    <measure number="{measure_num}">
      <attributes>
        <divisions>{ppq}</divisions>
        <key><fifths>0</fifths></key>
        <time><beats>{sig[0]}</beats><beat-type>{sig[1]}</beat-type></time>
        <clef><sign>G</sign><line>2</line></clef>
      </attributes>
{harmony_xml}{notes_str}
    </measure>"""


def _rest_xml(ppq: int) -> str:
    return f"""        <note>
          <rest/>
          <duration>{ppq * 4}</duration>
          <type>whole</type>
        </note>"""


def _ticks_to_type(ticks: int, ppq: int) -> str:
    """简化：四分音符为 ppq，半音符=2*ppq，whole=4*ppq。"""
    quarter = ppq
    if ticks >= 4 * quarter:
        return "whole"
    if ticks >= 2 * quarter:
        return "half"
    if ticks >= quarter:
        return "quarter"
    if ticks >= quarter // 2:
        return "eighth"
    return "16th"


def _tempo_directive_xml(tempo_map: list[TempoSegment]) -> str:
    """生成 <sound tempo="..."/> 元素。"""
    if not tempo_map:
        return ""
    bpm = float(tempo_map[0].bpm)
    return f'      <sound tempo="{bpm:.1f}"/>\n'


def build_musicxml(
    project: Project,
    config: RenderConfig | None = None,
    layout: LayoutMode = "per_stem",
) -> str:
    """Project → MusicXML 字符串。

    Args:
        project: 输入 Project
        config: RenderConfig（当前未用，预留）
        layout: "per_stem" / "piano_reduction" / "score"
    """
    cfg = config or RenderConfig()
    tracks = _get_layout_tracks(project, layout, cfg)

    if not tracks:
        raise MusicXMLBuilderError("project has no tracks")

    # part-list
    part_list_entries: list[str] = []
    for i, (stem, track) in enumerate(tracks.items(), 1):
        part_id = f"P{i}"
        part_name = stem
        part_list_entries.append(
            f'    <score-part id="{part_id}"><part-name>{part_name}</part-name></score-part>'
        )
    part_list_str = "\n".join(part_list_entries)

    # parts
    part_xmls: list[str] = []
    for i, (stem, track) in enumerate(tracks.items(), 1):
        part_id = f"P{i}"
        part_xml = _build_part_musicxml(
            part_id, stem, track, project,
            ppq=480, include_chord_symbols=cfg.include_chord_symbols,
        )
        part_xmls.append(part_xml)
    parts_str = "\n".join(part_xmls)

    musicxml = f"""<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 3.1 Partwise//EN" "http://www.musicxml.org/dtds/partwise.dtd">
<score-partwise version="3.1">
  <part-list>
{part_list_str}
  </part-list>
{parts_str}
</score-partwise>
"""
    logger.info(
        "MusicXML built: layout={}, parts={}",
        layout, len(tracks),
    )
    return musicxml


__all__ = [
    "MusicXMLBuilderError",
    "LayoutMode",
    "build_musicxml",
    "_pitch_to_xml_parts",
]
