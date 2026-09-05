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

import re
from pathlib import Path
from typing import Literal
from xml.sax.saxutils import escape

from loguru import logger

from mujik.config.schema import RenderConfig
from mujik.merge.core import merge_tracks
from mujik.midi.model import Note, Project, StemName, TempoSegment, Track
from mujik.score.bend import (
    BendPoint,
    build_bend_elements,
    detect_bend_release,
)
from mujik.score.harmony import build_harmony_element, find_chord_at_time
from mujik.score.time_helpers import (
    bpm_at_time,
    measure_index_at_time,
    seconds_to_ticks,
)
from mujik.time_signature.model import (
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
            is_first=(mi == 0),
        )
        measure_xmls.append(measure_xml)

    measures_str = "\n    ".join(measure_xmls)
    return f"""  <part id="{part_id}">
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
      {_tempo_direction_xml(project.tempo_map)}
      <note>
        <rest/>
        <duration>{ppq * sig[0] * 4 // sig[1]}</duration>
        <type>whole</type>
      </note>
    </measure>
  </part>"""


# GM 鼓 note → 五线谱显示位置（percussion clef 简化谱位，保证乐器不重叠可读）。
# (display_step, display_octave, notehead)；cymbal 类用 x 头。
# 未列出的 pitch 走 _pitch_to_xml_parts 常规位置。
GM_DRUM_DISPLAY: dict[int, tuple[str, int, str | None]] = {
    36: ("F", 4, None),        # Kick
    38: ("C", 5, None),        # Snare
    41: ("F", 3, None),        # Low Floor Tom
    45: ("D", 4, None),        # Mid Tom
    48: ("B", 4, None),        # High Tom
    42: ("G", 5, "x"),         # Closed Hi-Hat
    46: ("G", 5, "circle-x"),  # Open Hi-Hat
    49: ("A", 5, "x"),         # Crash
    51: ("E", 5, "x"),         # Ride
}


def _drum_rest_xml(dur_ticks: int, ppq: int) -> str:
    """鼓小节内的补位 rest（duration 与 type 由 ticks 推导）。"""
    type_str = _ticks_to_type(dur_ticks, ppq)
    return f"""        <note>
          <rest/>
          <duration>{dur_ticks}</duration>
          <voice>1</voice>
          <type>{type_str}</type>
        </note>"""


def _drum_note_xml(note: Note, dur_ticks: int, ppq: int, is_chord: bool) -> str:
    """单个鼓 note（unpitched）；is_chord=True 时是同刻并击的后续乐器。"""
    disp = GM_DRUM_DISPLAY.get(note.pitch)
    if disp is not None:
        step, octave, head = disp
    else:
        step, _alter, octave = _pitch_to_xml_parts(note.pitch)
        head = None
    chord_xml = "\n          <chord/>" if is_chord else ""
    head_xml = f"\n          <notehead>{head}</notehead>" if head else ""
    type_str = _ticks_to_type(dur_ticks, ppq)
    return f"""        <note>{chord_xml}
          <unpitched>
            <display-step>{step}</display-step>
            <display-octave>{octave}</display-octave>
          </unpitched>
          <duration>{dur_ticks}</duration>
          <voice>1</voice>
          <type>{type_str}</type>{head_xml}
        </note>"""


def _build_drum_part_musicxml(
    part_id: str, part_name: str, track: Track, project: Project, ppq: int = 480,
) -> str:
    """鼓 part：percussion clef + unpitched note（v0.5.3 重写）。

    旧版硬伤（鼓谱面渲染崩坏的根因）：
    - 每 measure 重复完整 <attributes>（非法结构）
    - 固定 <type>quarter</type> 与实际 duration 自相矛盾
    - 小节不补 rest，时长与拍号规定不符
    - 同刻多乐器顺序堆叠（无 <chord/>），小节被撑爆
    - part 开头多一个空的 measure 1，与后续 measure 编号重复

    新版：attributes 仅在首小节；type 从 duration 推导；rest 补齐小节；
    同刻（<10ms）多乐器用 <chord/> 并击；谱位按 GM_DRUM_DISPLAY 映射。
    """
    notes_sorted = sorted(track.notes, key=lambda n: (n.start, n.pitch))
    if not notes_sorted:
        return _build_empty_part_musicxml(part_id, part_name, project, ppq)

    seg = (
        project.time_signatures[0]
        if project.time_signatures
        else build_default_segments(max(project.duration, 10.0))[0]
    )
    bpm = bpm_at_time(0.0, project.tempo_map)
    bar_dur_sec = seg.bar_duration_sec(bpm)
    bar_ticks = max(1, int(round(bar_dur_sec * bpm * ppq / 60.0)))
    sig = seg.time_signature

    measure_groups = _group_notes_by_measure(notes_sorted, project, ppq)
    measure_xmls: list[str] = []
    for mi, mnotes in enumerate(measure_groups):
        measure_start = seg.start_time + mi * bar_dur_sec

        # 同刻并击聚类（<10ms 视为同一击打的多乐器，如 kick+snare+hat）
        clusters: list[list[Note]] = []
        for note in mnotes:
            if clusters and note.start - clusters[-1][0].start < 0.01:
                clusters[-1].append(note)
            else:
                clusters.append([note])

        note_xmls: list[str] = []
        cursor = 0  # 本小节已填充的 ticks
        for cluster in clusters:
            lead = cluster[0]
            onset_ticks = seconds_to_ticks(
                max(0.0, lead.start - measure_start), seg, bpm, ppq,
            )
            onset_ticks = min(onset_ticks, bar_ticks)
            gap = onset_ticks - cursor
            if gap > 0:
                note_xmls.append(_drum_rest_xml(gap, ppq))
            # 击打时长：取 cluster 内最长者，截断到小节尾
            cluster_end = max(n.end for n in cluster)
            dur = seconds_to_ticks(
                max(0.0, min(cluster_end, measure_start + bar_dur_sec) - lead.start),
                seg, bpm, ppq,
            )
            dur = max(1, min(dur, bar_ticks - onset_ticks))
            for j, note in enumerate(cluster):
                note_xmls.append(_drum_note_xml(note, dur, ppq, is_chord=(j > 0)))
            cursor = onset_ticks + dur
        if bar_ticks - cursor > 0:
            note_xmls.append(_drum_rest_xml(bar_ticks - cursor, ppq))

        notes_str = "\n".join(note_xmls) if note_xmls else _drum_rest_xml(bar_ticks, ppq)
        # v0.5.3: <attributes> 仅首小节；旧版每小节重复且额外多一个空 measure 1
        head_xml = ""
        if mi == 0:
            head_xml = (
                f"      <attributes>\n"
                f"        <divisions>{ppq}</divisions>\n"
                f"        <key><fifths>0</fifths></key>\n"
                f"        <time><beats>{sig[0]}</beats><beat-type>{sig[1]}</beat-type></time>\n"
                f"        <clef><sign>percussion</sign><line>2</line></clef>\n"
                f"      </attributes>\n"
                f"{_tempo_direction_xml(project.tempo_map)}"
            )
        measure_xmls.append(
            f"""    <measure number="{mi + 1}">
{head_xml}{notes_str}
    </measure>"""
        )
    measures_str = "\n".join(measure_xmls)
    return f"""  <part id="{part_id}">
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
    is_first: bool = False,
) -> str:
    """构造单个 <measure> 的 XML。

    v0.4.1 起：
    - ``include_chord_symbols=True`` 时，若 ``project.chord_track`` 有
      覆盖该 measure 起始时间的 chord，在第一个 note 之前插入 ``<harmony>``
    - ``note.pitch_bend`` 非空且 alter != 0 时，在该 note ``<type>`` 之后
      插入 ``<bend>`` 元素
    - v0.5.3 修：``<attributes>``（divisions/key/time/clef）+ 速度记号只在
      首小节（is_first）发一次——旧版每小节重复完整 attributes 且 part
      开头多一个与后续编号重复的空 measure 1，verovio 小节线/编号全乱
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
        # v0.4.1: pitch_bend → <bend>；v0.4.3: 连续曲线 → bend+release 双 <bend>
        peak_alter, has_release = detect_bend_release(note.pitch_bend)
        bend_xml = ""
        if peak_alter != 0:
            if has_release:
                # bend up + release: 发 2 个 <bend> 兄弟
                curve = [BendPoint(0.0, peak_alter), BendPoint(1.0, 0)]
            else:
                # 单 bend（保留 v0.4.1 行为）
                curve = [BendPoint(0.0, peak_alter)]
            bend_xml = "\n          " + build_bend_elements(curve)
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
    head_xml = ""
    if is_first:
        sig = (
            project.time_signatures[0].time_signature
            if project.time_signatures else (4, 4)
        )
        head_xml = (
            f"      <attributes>\n"
            f"        <divisions>{ppq}</divisions>\n"
            f"        <key><fifths>0</fifths></key>\n"
            f"        <time><beats>{sig[0]}</beats><beat-type>{sig[1]}</beat-type></time>\n"
            f"        <clef><sign>G</sign><line>2</line></clef>\n"
            f"      </attributes>\n"
            f"{_tempo_direction_xml(project.tempo_map)}"
        )
    return f"""    <measure number="{measure_num}">
{head_xml}{harmony_xml}{notes_str}
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


def _tempo_direction_xml(tempo_map: list[TempoSegment]) -> str:
    """速度记号：<metronome> direction（verovio 才会画在谱面上）+ <sound tempo>。

    v0.5.3 修：旧版只发裸 <sound tempo="..."/>，verovio 不渲染它——
    用户打开 PDF 看不到任何 BPM 信息。beat-unit 固定 quarter（BPM 约定
    为四分音符/分钟）。
    """
    if not tempo_map:
        return ""
    bpm = float(tempo_map[0].bpm)
    return (
        '      <direction placement="above">\n'
        '        <direction-type>\n'
        '          <metronome parentheses="no">\n'
        '            <beat-unit>quarter</beat-unit>\n'
        f'            <per-minute>{bpm:.1f}</per-minute>\n'
        '          </metronome>\n'
        '        </direction-type>\n'
        f'        <sound tempo="{bpm:.1f}"/>\n'
        '      </direction>\n'
    )


def _score_title(project: Project) -> str:
    """从 audio_path 推谱面标题。

    去掉 ws 中间产物的命名前缀（loudnorm_/denoised_）和时长后缀
    （_189s），buhee 实际音频 loudnorm_buhee_189s.wav → "buhee"。
    """
    stem = Path(project.audio_path).stem if project.audio_path else ""
    stem = re.sub(r"^(loudnorm|denoised)_", "", stem)
    stem = re.sub(r"_\d+s$", "", stem)
    return escape(stem) if stem else "Untitled"


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
    for i, (stem, _track) in enumerate(tracks.items(), 1):
        part_id = f"P{i}"
        part_name = stem
        part_list_entries.append(
            f'    <score-part id="{part_id}"><part-name>{part_name}</part-name></score-part>'
        )
    part_list_str = "\n".join(part_list_entries)

    # v0.5.3: 谱面总体信息（标题）——旧版无 <work-title>，PDF 顶部空白
    title = _score_title(project)

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
  <work>
    <work-title>{title}</work-title>
  </work>
  <movement-title>{title}</movement-title>
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
