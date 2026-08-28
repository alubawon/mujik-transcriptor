"""Tests for score/builder.py (MusicXML construction)."""
from __future__ import annotations

import pytest

from mujik.midi.model import Note, Project, TempoSegment, Track
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
