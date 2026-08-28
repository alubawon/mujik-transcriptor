"""Tests for score/harmony.py."""
from __future__ import annotations

import pytest

from mujik.midi.model import ChordEvent
from mujik.score.harmony import (
    QUALITY_TO_KIND,
    build_harmony_element,
    find_chord_at_time,
)


class TestBuildHarmonyElement:
    def test_major_chord(self):
        chord = ChordEvent(start=0.0, end=1.0, root="C", quality="")
        xml = build_harmony_element(chord)
        assert "<harmony>" in xml
        assert "</harmony>" in xml
        assert "<root-step>C</root-step>" in xml
        assert "<kind>major</kind>" in xml

    def test_minor_chord(self):
        chord = ChordEvent(start=0.0, end=1.0, root="A", quality="m")
        xml = build_harmony_element(chord)
        assert "<root-step>A</root-step>" in xml
        assert "<kind>minor</kind>" in xml

    def test_sharp_root(self):
        chord = ChordEvent(start=0.0, end=1.0, root="F#", quality="")
        xml = build_harmony_element(chord)
        assert "<root-step>F</root-step>" in xml
        assert "<root-alter>1</root-alter>" in xml

    def test_flat_root(self):
        chord = ChordEvent(start=0.0, end=1.0, root="Bb", quality="")
        xml = build_harmony_element(chord)
        assert "<root-step>B</root-step>" in xml
        assert "<root-alter>-1</root-alter>" in xml

    def test_dominant_seventh(self):
        chord = ChordEvent(start=0.0, end=1.0, root="G", quality="7")
        xml = build_harmony_element(chord)
        assert "<kind>dominant</kind>" in xml

    def test_major_seventh(self):
        chord = ChordEvent(start=0.0, end=1.0, root="C", quality="maj7")
        xml = build_harmony_element(chord)
        assert "<kind>major-seventh</kind>" in xml

    def test_minor_seventh(self):
        chord = ChordEvent(start=0.0, end=1.0, root="D", quality="m7")
        xml = build_harmony_element(chord)
        assert "<kind>minor-seventh</kind>" in xml

    def test_diminished(self):
        chord = ChordEvent(start=0.0, end=1.0, root="B", quality="dim")
        xml = build_harmony_element(chord)
        assert "<kind>diminished</kind>" in xml

    def test_augmented(self):
        chord = ChordEvent(start=0.0, end=1.0, root="C", quality="aug")
        xml = build_harmony_element(chord)
        assert "<kind>augmented</kind>" in xml

    def test_slash_chord_with_bass(self):
        chord = ChordEvent(start=0.0, end=1.0, root="C", quality="7", bass="D")
        xml = build_harmony_element(chord)
        assert "<bass>" in xml
        assert "<bass-step>D</bass-step>" in xml
        assert "</bass>" in xml

    def test_slash_chord_with_sharp_bass(self):
        chord = ChordEvent(start=0.0, end=1.0, root="C", quality="7", bass="F#")
        xml = build_harmony_element(chord)
        assert "<bass-step>F</bass-step>" in xml
        assert "<bass-alter>1</bass-alter>" in xml

    def test_unknown_quality_passes_through(self):
        """v0.4.6: extended vocab 内未知 kind 字符串透传到 <kind>（Verovio 接受）。

        v0.4.1 设计允许 9/11/13/alt 透传；v0.4.6 hardening 后，
        必须先用合法 quality 构造 ChordEvent，再透传到 <kind>。
        """
        # "sus" 在 extended vocab 内
        chord = ChordEvent(start=0.0, end=1.0, root="C", quality="sus")
        xml = build_harmony_element(chord)
        assert "<kind>sus</kind>" in xml

    def test_9_quality_rejected_at_construction(self):
        """v0.4.6: "9" 不在 extended vocab → ChordEvent 构造失败。"""
        # 9/11/13/alt 留 v0.4.8 BTC-HCQT 评估后再开
        with pytest.raises(ValueError, match="quality"):
            ChordEvent(start=0.0, end=1.0, root="C", quality="9")

    def test_lowercase_root_normalized(self):
        chord = ChordEvent(start=0.0, end=1.0, root="c", quality="m")
        xml = build_harmony_element(chord)
        assert "<root-step>C</root-step>" in xml

    def test_invalid_root_raises(self):
        """v0.4.6: H 不是合法 root → 在 ChordEvent 构造时拒绝（不再等到 build_harmony_element）。"""
        with pytest.raises(ValueError, match="root must match"):
            ChordEvent(start=0.0, end=1.0, root="H", quality="")


class TestFindChordAtTime:
    def test_empty_track(self):
        assert find_chord_at_time(None, 1.0) is None
        assert find_chord_at_time([], 1.0) is None

    def test_finds_chord_at_time(self):
        track = [
            ChordEvent(start=0.0, end=2.0, root="C", quality=""),
            ChordEvent(start=2.0, end=4.0, root="F", quality=""),
            ChordEvent(start=4.0, end=6.0, root="G", quality="7"),
        ]
        assert find_chord_at_time(track, 1.0).root == "C"
        assert find_chord_at_time(track, 2.5).root == "F"
        assert find_chord_at_time(track, 5.0).root == "G"

    def test_boundary_start_inclusive(self):
        """start ≤ t 视为包含。"""
        track = [ChordEvent(start=2.0, end=4.0, root="C", quality="")]
        chord = find_chord_at_time(track, 2.0)
        assert chord is not None
        assert chord.root == "C"

    def test_boundary_end_exclusive(self):
        """t == end 不算入当前 chord（下一 chord 开始）。"""
        track = [
            ChordEvent(start=0.0, end=2.0, root="C", quality=""),
            ChordEvent(start=2.0, end=4.0, root="F", quality=""),
        ]
        chord = find_chord_at_time(track, 2.0)
        # start inclusive end exclusive → 2.0 落在第二个 chord
        assert chord.root == "F"

    def test_outside_track(self):
        track = [ChordEvent(start=0.0, end=2.0, root="C", quality="")]
        assert find_chord_at_time(track, 3.0) is None
        assert find_chord_at_time(track, -0.5) is None


class TestQualityToKindMap:
    def test_all_standard_qualities_mapped(self):
        """至少 8 个标准 quality 都有映射。"""
        assert "" in QUALITY_TO_KIND
        assert "maj" in QUALITY_TO_KIND
        assert "m" in QUALITY_TO_KIND
        assert "7" in QUALITY_TO_KIND
        assert "maj7" in QUALITY_TO_KIND
        assert "m7" in QUALITY_TO_KIND
        assert "dim" in QUALITY_TO_KIND
        assert "aug" in QUALITY_TO_KIND

    def test_empty_quality_maps_to_major(self):
        assert QUALITY_TO_KIND[""] == "major"
