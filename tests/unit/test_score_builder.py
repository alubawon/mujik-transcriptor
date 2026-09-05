"""Tests for score/builder.py (MusicXML construction)."""
from __future__ import annotations

import pytest

from mujik.config.schema import RenderConfig
from mujik.midi.model import (
    ChordEvent,
    Note,
    Project,
    TempoSegment,
    Track,
)
from mujik.score.builder import (
    MusicXMLBuilderError,
    _pitch_to_xml_parts,
    build_musicxml,
)
from mujik.time_signature.model import TimeSignatureSegment


def _seg(sig=(4, 4), start=0.0, end=10.0) -> TimeSignatureSegment:
    return TimeSignatureSegment(
        start_time=start, end_time=end,
        time_signature=sig, confidence=1.0, source="manual",
    )


def _project(
    tracks: dict[str, Track],
    duration: float = 5.0,
    ts: list[TimeSignatureSegment] | None = None,
    tempo: list[TempoSegment] | None = None,
) -> Project:
    return Project(
        audio_path="song.wav",
        duration=duration,
        sample_rate=44100,
        time_signatures=ts or [_seg(end=duration)],
        tempo_map=tempo or [TempoSegment(0.0, duration, 120.0)],
        tracks=tracks,  # type: ignore[arg-type]
    )


def _track(notes: list[Note], stem: str, channel: int = 0) -> Track:
    t = Track(stem_name=stem, channel=channel)  # type: ignore[arg-type]
    for n in notes:
        t.add(n)
    return t


class TestPitchToXmlParts:
    def test_c4(self):
        # MIDI 60 = C4
        assert _pitch_to_xml_parts(60) == ("C", 0, 4)

    def test_c_sharp_4(self):
        # MIDI 61 = C#4
        assert _pitch_to_xml_parts(61) == ("C", 1, 4)

    def test_a4(self):
        # MIDI 69 = A4
        assert _pitch_to_xml_parts(69) == ("A", 0, 4)

    def test_c5(self):
        # MIDI 72 = C5
        assert _pitch_to_xml_parts(72) == ("C", 0, 5)

    def test_b3(self):
        # MIDI 59 = B3
        assert _pitch_to_xml_parts(59) == ("B", 0, 3)

    def test_invalid_pitch_raises(self):
        with pytest.raises(ValueError):
            _pitch_to_xml_parts(-1)
        with pytest.raises(ValueError):
            _pitch_to_xml_parts(128)


class TestBuildMusicxml:
    def test_basic_per_stem(self):
        proj = _project({
            "vocals": _track([
                Note(0.0, 0.5, 60, 100),
                Note(0.5, 1.0, 62, 90),
            ], "vocals"),
            "drums": _track([
                Note(0.0, 0.1, 36, 100, channel=9),
            ], "drums", channel=9),
        })
        xml = build_musicxml(proj, layout="per_stem")
        assert "<?xml" in xml
        assert "<score-partwise" in xml
        assert "P1" in xml  # part id
        assert "P2" in xml
        # part names
        assert "vocals" in xml
        assert "drums" in xml
        # note content
        assert "step>C</step>" in xml
        # percussion clef
        assert "percussion" in xml

    def test_empty_tracks_raises(self):
        proj = _project({})
        with pytest.raises(MusicXMLBuilderError, match="no tracks"):
            build_musicxml(proj)

    def test_single_track(self):
        proj = _project({
            "vocals": _track([Note(0.0, 1.0, 60, 100)], "vocals"),
        })
        xml = build_musicxml(proj)
        assert "<score-part" in xml
        assert "vocals" in xml

    def test_piano_reduction_layout(self):
        proj = _project({
            "bass": _track([Note(0.0, 1.0, 40, 100)], "bass"),
            "other": _track([Note(0.0, 1.0, 60, 100)], "other"),
            "vocals": _track([Note(0.0, 1.0, 64, 100)], "vocals"),
            "drums": _track([Note(0.0, 0.1, 36, 100, channel=9)], "drums", channel=9),
        })
        xml = build_musicxml(proj, layout="piano_reduction")
        # 应有 piano_reduction 轨
        assert "piano_reduction" in xml
        # vocals + drums 仍保留
        assert "vocals" in xml
        assert "drums" in xml

    def test_score_layout_same_as_per_stem(self):
        proj = _project({
            "vocals": _track([Note(0.0, 1.0, 60, 100)], "vocals"),
        })
        xml_score = build_musicxml(proj, layout="score")
        xml_per = build_musicxml(proj, layout="per_stem")
        # 应产生相同结构（part-list 一致）
        assert "P1" in xml_score
        assert "P1" in xml_per

    def test_unknown_layout_raises(self):
        proj = _project({
            "vocals": _track([Note(0.0, 1.0, 60, 100)], "vocals"),
        })
        with pytest.raises(MusicXMLBuilderError, match="unknown layout"):
            build_musicxml(proj, layout="bogus")  # type: ignore[arg-type]

    def test_pitch_alter_in_xml(self):
        """# (sharp) notes 应有 alter 元素。"""
        proj = _project({
            "vocals": _track([
                Note(0.0, 1.0, 61, 100),  # C#4
            ], "vocals"),
        })
        xml = build_musicxml(proj)
        assert "<alter>1</alter>" in xml

    def test_drum_part_uses_percussion_clef(self):
        proj = _project({
            "drums": _track([
                Note(0.0, 0.1, 36, 100, channel=9),
                Note(0.5, 0.6, 38, 100, channel=9),
            ], "drums", channel=9),
        })
        xml = build_musicxml(proj)
        # percussion part
        assert "percussion" in xml
        # unpitched display step (MIDI 36 = C2)
        assert "display-step" in xml or "step>C</step>" in xml

    def test_time_signature_in_xml(self):
        proj = _project(
            {
                "vocals": _track([Note(0.0, 1.0, 60, 100)], "vocals"),
            },
            ts=[_seg(sig=(3, 4))],
        )
        xml = build_musicxml(proj)
        assert "<beats>3</beats>" in xml
        assert "<beat-type>4</beat-type>" in xml

    def test_tempo_directive_in_xml(self):
        proj = _project(
            {
                "vocals": _track([Note(0.0, 1.0, 60, 100)], "vocals"),
            },
            tempo=[TempoSegment(0.0, 5.0, 144.0)],
        )
        xml = build_musicxml(proj)
        assert 'tempo="144.0"' in xml

    def test_empty_track_gets_whole_rest(self):
        proj = _project({
            "other": _track([], "other"),
        })
        xml = build_musicxml(proj)
        # empty part → 1 measure with whole rest
        assert "<rest/>" in xml

    def test_pitch_bend_renders_bend_element(self):
        """v0.4.1: Note.pitch_bend 非空时输出 <bend> 元素。"""
        proj = _project({
            "vocals": _track([
                Note(0.0, 1.0, 60, 100, pitch_bend=(0.0, 0.5, 0.5, 0.0)),
            ], "vocals"),
        })
        xml = build_musicxml(proj)
        assert "<bend" in xml
        assert "<bend-alter>1</bend-alter>" in xml  # 0.5 * 2 = 1

    def test_no_bend_skips_bend_element(self):
        """v0.4.1: pitch_bend 为空时不应出现 <bend> 元素。"""
        proj = _project({
            "vocals": _track([
                Note(0.0, 1.0, 60, 100),
            ], "vocals"),
        })
        xml = build_musicxml(proj)
        assert "<bend" not in xml
        assert "<bend-alter>" not in xml

    def test_bend_with_release_emits_two_siblings(self):
        """v0.4.3: 弯音曲线有 release → 发 2 个 <bend> 兄弟。"""
        # ramp up + 回到 0 → has_release
        proj = _project({
            "vocals": _track([
                Note(0.0, 1.0, 60, 100, pitch_bend=(0.0, 0.3, 0.5, 0.3, 0.0)),
            ], "vocals"),
        })
        xml = build_musicxml(proj)
        # 2 个 <bend> 兄弟（用 "<bend " 带空格避免匹配 <bend-alter>）
        assert xml.count("<bend ") == 2
        # bend up
        assert "<bend-alter>1</bend-alter>" in xml
        # release
        assert "<bend-alter>-1</bend-alter>" in xml
        assert "<release/>" in xml

    def test_plateau_bend_emits_single_bend(self):
        """v0.4.3: 平台弯音（无 release）→ 单 <bend>。"""
        # peak 在末尾，无 post-peak 返回 → 无 release
        proj = _project({
            "vocals": _track([
                Note(0.0, 1.0, 60, 100, pitch_bend=(0.0, 0.2, 0.4, 0.5, 0.5)),
            ], "vocals"),
        })
        xml = build_musicxml(proj)
        # 只有 1 个 <bend>
        assert xml.count("<bend ") == 1
        assert "<bend-alter>1</bend-alter>" in xml
        # 无 release marker
        assert "<release/>" not in xml

    def test_bend_curve_uses_curved_shape(self):
        """v0.4.3: <bend> 默认带 shape="curved" 属性。"""
        proj = _project({
            "vocals": _track([
                Note(0.0, 1.0, 60, 100, pitch_bend=(0.0, 0.2, 0.4, 0.5, 0.5)),
            ], "vocals"),
        })
        xml = build_musicxml(proj)
        assert 'shape="curved"' in xml

    def test_chord_symbols_render_harmony_element(self):
        """v0.4.1: chord_track + include_chord_symbols=True → <harmony>。"""
        proj = _project({
            "vocals": _track([
                Note(0.0, 0.5, 60, 100),
                Note(0.5, 1.0, 62, 90),
            ], "vocals"),
        })
        proj.chord_track = [
            ChordEvent(start=0.0, end=2.0, root="C", quality="maj7"),
        ]
        xml = build_musicxml(proj, config=RenderConfig(include_chord_symbols=True))
        assert "<harmony>" in xml
        assert "<root-step>C</root-step>" in xml
        assert "<kind>major-seventh</kind>" in xml

    def test_chord_symbols_disabled_skips_harmony(self):
        """v0.4.1: include_chord_symbols=False 时不输出 <harmony>（向后兼容）。"""
        proj = _project({
            "vocals": _track([
                Note(0.0, 0.5, 60, 100),
            ], "vocals"),
        })
        proj.chord_track = [
            ChordEvent(start=0.0, end=2.0, root="C", quality=""),
        ]
        xml = build_musicxml(proj, config=RenderConfig(include_chord_symbols=False))
        assert "<harmony>" not in xml

    def test_chord_symbols_no_chord_track_skips_harmony(self):
        """v0.4.1: chord_track 为 None 时不输出 <harmony>。"""
        proj = _project({
            "vocals": _track([
                Note(0.0, 0.5, 60, 100),
            ], "vocals"),
        })
        # chord_track=None (默认)
        xml = build_musicxml(proj, config=RenderConfig(include_chord_symbols=True))
        assert "<harmony>" not in xml


class TestPitchedPartStructure:
    """v0.5.3：有音高 part 结构修复——无重复 measure 编号、attributes 仅首小节、
    标题 + metronome 速度记号可见。"""

    def _xml(self, **kwargs) -> str:
        proj = _project({
            "vocals": _track([
                Note(0.0, 0.5, 60, 100),
                Note(0.5, 1.0, 62, 90),
            ], "vocals"),
        }, **kwargs)
        return build_musicxml(proj)

    def test_no_duplicate_measure_number_1(self):
        """旧版 part 开头多一个只含 attributes 的空 measure 1，与真实小节编号重复。"""

        xml = self._xml()
        assert xml.count('<measure number="1">') == 1

    def test_attributes_only_in_first_measure(self):

        xml = self._xml()
        assert xml.count("<attributes>") == 1

    def test_work_title_from_audio_path(self):
        xml = self._xml()
        assert "<work-title>song</work-title>" in xml
        assert "<movement-title>song</movement-title>" in xml

    def test_work_title_strips_ws_prefix_and_duration_tag(self):
        proj = _project({
            "vocals": _track([Note(0.0, 1.0, 60, 100)], "vocals"),
        })
        proj.audio_path = "/x/y/loudnorm_buhee_189s.wav"
        xml = build_musicxml(proj)
        assert "<work-title>buhee</work-title>" in xml

    def test_metronome_direction_renders_bpm(self):
        """裸 <sound tempo> verovio 不显示；必须带 <metronome> direction。"""
        proj = _project(
            {"vocals": _track([Note(0.0, 1.0, 60, 100)], "vocals")},
            tempo=[TempoSegment(0.0, 5.0, 144.0)],
        )
        xml = build_musicxml(proj)
        assert "<metronome" in xml
        assert "<per-minute>144.0</per-minute>" in xml
        # <sound tempo> 保留（MIDI 语义）
        assert 'tempo="144.0"' in xml

    def test_drum_part_also_has_metronome(self):
        proj = _project({
            "drums": _track([Note(0.0, 0.1, 36, 100, channel=9)], "drums", channel=9),
        })
        xml = build_musicxml(proj)
        assert "<per-minute>120.0</per-minute>" in xml

    def test_multi_measure_notes_span_measures(self):
        # 3 秒 @120BPM 4/4 = 1.5 小节 → 2 个小节且编号 1、2
        import re

        notes = [Note(float(i) * 0.25, float(i) * 0.25 + 0.2, 60 + i, 100) for i in range(12)]
        proj = _project({"vocals": _track(notes, "vocals")}, duration=3.0)
        xml = build_musicxml(proj)
        nums = re.findall(r'<measure number="(\d+)">', xml)
        assert nums == ["1", "2"]


class TestDrumPartMusicxml:
    """v0.5.3 重写后的鼓 part：结构合法 + 时值一致 + 并击 + rest 补齐。"""

    def _drum_xml(self, notes: list[Note], duration: float = 8.0) -> str:
        from mujik.score.builder import _build_drum_part_musicxml

        project = _project(
            {"drums": _track(notes, "drums", channel=9)},
            duration=duration,
        )
        return _build_drum_part_musicxml("P1", "drums", project.tracks["drums"], project)

    def test_attributes_only_in_first_measure(self):
        notes = [Note(0.0, 0.1, 36, 100, channel=9), Note(2.0, 2.1, 38, 100, channel=9)]
        xml = self._drum_xml(notes)
        assert xml.count("<attributes>") == 1
        assert "percussion" in xml

    def test_no_duplicate_measure_numbers(self):
        # 8 hits across 0–3.5s → 4/4 @120BPM (bar=2s) → measures 1–2 有内容，
        # 尾部空小节被丢弃 → 编号 1、2 各出现一次，无重复编号
        notes = [
            Note(0.0 + i * 0.5, 0.1 + i * 0.5, 36, 100, channel=9)
            for i in range(8)
        ]
        xml = self._drum_xml(notes, duration=8.0)
        for num in ('number="1"', 'number="2"'):
            assert xml.count(num) == 1

    def test_simultaneous_hits_use_chord(self):
        # kick + snare + hi-hat 同刻 → 首个普通 note + 2 个 <chord/>
        notes = [
            Note(0.0, 0.1, 36, 100, channel=9),
            Note(0.0, 0.1, 38, 100, channel=9),
            Note(0.0, 0.1, 42, 100, channel=9),
        ]
        xml = self._drum_xml(notes, duration=4.0)
        assert xml.count("<chord/>") == 2

    def test_rests_fill_measures(self):
        # 每小节 duration 总和 == bar ticks（4/4 @120BPM ppq480 → 1920）
        import re

        notes = [Note(0.0, 0.1, 36, 100, channel=9)]  # 只有第一拍一击
        xml = self._drum_xml(notes, duration=4.0)  # 尾部空小节丢弃 → 1 measure
        measures = re.findall(r"<measure .*?</measure>", xml, re.DOTALL)
        assert len(measures) == 1
        durs = [int(d) for d in re.findall(r"<duration>(\d+)</duration>", measures[0])]
        assert sum(durs) == 1920
        # 一击 + 补位 rest
        assert durs[0] < 1920
        assert len(durs) >= 2

    def test_notehead_x_for_cymbals(self):
        notes = [Note(0.0, 0.1, 42, 100, channel=9)]  # closed hi-hat
        xml = self._drum_xml(notes, duration=4.0)
        assert "<notehead>x</notehead>" in xml

    def test_display_positions_from_gm_map(self):
        # kick → F4；snare → C5（简化谱位映射）
        notes = [
            Note(0.0, 0.1, 36, 100, channel=9),
            Note(0.5, 0.6, 38, 100, channel=9),
        ]
        xml = self._drum_xml(notes, duration=4.0)
        assert "<display-step>F</display-step>" in xml
        assert "<display-octave>4</display-octave>" in xml
        assert "<display-step>C</display-step>" in xml

    def test_type_matches_duration_class(self):
        # 不再出现固定 quarter 与 50ms duration 矛盾：50ms @120BPM ppq480
        # ≈ 48 ticks < 240（eighth）→ type=16th
        import re

        notes = [Note(0.0, 0.05, 36, 100, channel=9)]
        xml = self._drum_xml(notes, duration=4.0)
        m = re.search(r"<unpitched>.*?<duration>(\d+)</duration>.*?<type>(\w+)</type>", xml, re.DOTALL)
        assert m is not None
        dur, type_str = int(m.group(1)), m.group(2)
        assert type_str == "16th"
        assert dur < 240

    def test_empty_drum_track_falls_back_to_empty_part(self):
        xml = self._drum_xml([], duration=4.0)
        assert "<rest/>" in xml
