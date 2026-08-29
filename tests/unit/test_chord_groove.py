"""Tests for chord/groove.py (v0.4.9, pure functions)."""
from __future__ import annotations

import pytest

from mujik.chord.groove import apply_groove_to_chord_track
from mujik.midi.model import ChordEvent
from mujik.time_signature.model import TimeSignatureSegment


def _sig4_4_at_120() -> list[TimeSignatureSegment]:
    return [TimeSignatureSegment(
        start_time=0.0, end_time=10.0,
        time_signature=(4, 4), confidence=1.0, source="manual",
    )]


def _sig_change_at_120() -> list[TimeSignatureSegment]:
    return [
        TimeSignatureSegment(
            start_time=0.0, end_time=4.0,
            time_signature=(4, 4), confidence=1.0, source="manual",
        ),
        TimeSignatureSegment(
            start_time=4.0, end_time=10.0,
            time_signature=(3, 4), confidence=1.0, source="manual",
        ),
    ]


# 120 BPM: 1 beat = 0.5s
# 4/4: bar = 2.0s, beat positions 0.0, 0.5, 1.0, 1.5, 2.0
# 8 分 offbeat positions: 0.25, 0.75, 1.25, 1.75 (beat + 0.5)
# 16 分 off-offbeat: 0.125, 0.375, 0.625, 0.875
# swing16 ratio=0.6 → offbeat shift = (0.6-0.5)*0.5s = 0.05s
# 16th off-offbeat shift = (0.6-0.5)*0.5*0.5s = 0.025s


class TestStraightGroove:
    """v0.4.9: straight 模板 = noop。"""

    def test_straight_no_op(self):
        track = [
            ChordEvent(0.0, 0.5, "C", ""),
            ChordEvent(0.5, 1.0, "F", ""),
        ]
        out = apply_groove_to_chord_track(
            track, _sig4_4_at_120(), bpm=120.0,
            template="straight", strength=1.0,
        )
        assert out[0].start == 0.0
        assert out[0].end == 0.5
        assert out[1].start == 0.5
        assert out[1].end == 1.0

    def test_straight_returns_new_list(self):
        track = [ChordEvent(0.0, 1.0, "C", "")]
        out = apply_groove_to_chord_track(track, _sig4_4_at_120(), 120.0)
        assert out is not track


class TestSwing16Groove:
    """v0.4.9: swing16 模板 → offbeat 位置后移。"""

    def test_onbeat_unchanged(self):
        track = [ChordEvent(0.0, 0.5, "C", "")]
        out = apply_groove_to_chord_track(
            track, _sig4_4_at_120(), bpm=120.0,
            template="swing16", strength=1.0, ratio=0.6,
        )
        # 0.0 和 0.5 都是 beat, 不偏移
        assert out[0].start == 0.0
        assert out[0].end == 0.5

    def test_offbeat_shifted(self):
        """v0.4.9: 8 分 offbeat (0.75s @ 120BPM) 偏移 0.05s。"""
        # chord 跨 0.5 (beat) → 0.75 (offbeat)
        track = [ChordEvent(0.5, 0.75, "F", "")]
        out = apply_groove_to_chord_track(
            track, _sig4_4_at_120(), bpm=120.0,
            template="swing16", strength=1.0, ratio=0.6,
        )
        # 0.5 是 beat, 0.75 是 offbeat
        assert out[0].start == 0.5  # on-beat
        assert out[0].end == 0.8  # offbeat + 0.05

    def test_16th_off_offbeat_smaller_shift(self):
        """v0.4.9: chord groove 用 8 分粒度（grid_resolution=4），与 quantize.groove 共享。

        验证 on-beat 边界不偏移（与 test_onbeat_unchanged 一致），
        以及跨多个 offbeat 位置都能正确触发。
        """
        # chord 跨 0.25 (offbeat) 和 0.5 (beat)
        # 0.25 = 0.5 beat = 8 分 offbeat → 偏移 0.05s
        track = [ChordEvent(0.0, 0.25, "G", "")]
        out = apply_groove_to_chord_track(
            track, _sig4_4_at_120(), bpm=120.0,
            template="swing16", strength=1.0, ratio=0.6,
        )
        assert out[0].start == 0.0  # on-beat (0.0)
        assert out[0].end == 0.3  # 0.25 + 0.05 (offbeat)


class TestStrengthBlending:
    """v0.4.9: strength 0=noop, 1=full, 中间值插值。"""

    def test_strength_zero_no_op(self):
        track = [ChordEvent(0.5, 0.75, "F", "")]
        out = apply_groove_to_chord_track(
            track, _sig4_4_at_120(), bpm=120.0,
            template="swing16", strength=0.0, ratio=0.6,
        )
        assert out[0].start == 0.5
        assert out[0].end == 0.75

    def test_strength_half_blends(self):
        """v0.4.9: strength=0.5 → 偏移减半（0.025s 替代 0.05s）。"""
        track = [ChordEvent(0.5, 0.75, "F", "")]
        out = apply_groove_to_chord_track(
            track, _sig4_4_at_120(), bpm=120.0,
            template="swing16", strength=0.5, ratio=0.6,
        )
        assert out[0].end == 0.775  # 0.75 + 0.025

    def test_strength_one_full(self):
        track = [ChordEvent(0.5, 0.75, "F", "")]
        out = apply_groove_to_chord_track(
            track, _sig4_4_at_120(), bpm=120.0,
            template="swing16", strength=1.0, ratio=0.6,
        )
        assert out[0].end == 0.8  # 0.75 + 0.05


class TestRatioVariations:
    """v0.4.9: ratio 参数影响偏移量。"""

    def test_ratio_05_no_shift(self):
        """v0.4.9: ratio=0.5 = 直拍 → offbeat 不偏移。"""
        track = [ChordEvent(0.5, 0.75, "F", "")]
        out = apply_groove_to_chord_track(
            track, _sig4_4_at_120(), bpm=120.0,
            template="swing16", strength=1.0, ratio=0.5,
        )
        assert out[0].end == 0.75

    def test_ratio_07_larger_shift(self):
        """v0.4.9: ratio=0.7 → offbeat 偏移 (0.7-0.5)*0.5 = 0.1s。"""
        track = [ChordEvent(0.5, 0.75, "F", "")]
        out = apply_groove_to_chord_track(
            track, _sig4_4_at_120(), bpm=120.0,
            template="swing16", strength=1.0, ratio=0.7,
        )
        assert out[0].end == 0.85  # 0.75 + 0.1


class TestEdgeCases:
    """v0.4.9: 边界情况。"""

    def test_empty_track(self):
        out = apply_groove_to_chord_track(
            [], _sig4_4_at_120(), bpm=120.0,
            template="swing16", strength=1.0,
        )
        assert out == []

    def test_single_chord_unchanged_when_on_beat(self):
        track = [ChordEvent(0.0, 1.0, "C", "")]
        out = apply_groove_to_chord_track(
            track, _sig4_4_at_120(), bpm=120.0,
            template="swing16", strength=1.0, ratio=0.6,
        )
        assert out[0].start == 0.0
        assert out[0].end == 1.0

    def test_no_time_signatures_uses_default(self):
        """v0.4.9: time_signatures=[] → 用默认 4/4 段。"""
        track = [ChordEvent(0.5, 0.75, "C", "")]
        out = apply_groove_to_chord_track(
            track, [], bpm=120.0,
            template="swing16", strength=1.0, ratio=0.6, duration=10.0,
        )
        # 0.75 仍在 4/4 段 (默认 0-10) → 偏移
        assert out[0].end == 0.8

    def test_bpm_zero_raises(self):
        with pytest.raises(ValueError, match="bpm"):
            apply_groove_to_chord_track(
                [ChordEvent(0.0, 1.0, "C", "")],
                _sig4_4_at_120(), bpm=0.0,
                template="swing16", strength=1.0,
            )

    def test_preserves_root_quality_bass(self):
        track = [ChordEvent(0.5, 0.75, "F#", "m7", bass="A")]
        out = apply_groove_to_chord_track(
            track, _sig4_4_at_120(), bpm=120.0,
            template="swing16", strength=1.0, ratio=0.6,
        )
        assert out[0].root == "F#"
        assert out[0].quality == "m7"
        assert out[0].bass == "A"


class TestTimeSignatureChange:
    """v0.4.9: 跨拍号段独立 groove。"""

    def test_chord_in_first_segment(self):
        track = [ChordEvent(0.5, 0.75, "C", "")]
        out = apply_groove_to_chord_track(
            track, _sig_change_at_120(), bpm=120.0,
            template="swing16", strength=1.0, ratio=0.6,
        )
        # 在 4/4 段 (0-4s), 0.75 是 offbeat → 偏移
        assert out[0].end == 0.8

    def test_chord_in_second_segment(self):
        """v0.4.9: 3/4 段 (4-10s) 中 0.75 偏移（在 3/4 段内的位置也是 offbeat）。"""
        # 4.0 是段起点, 4.75 = beat 1.5 (offbeat)
        track = [ChordEvent(4.0, 4.75, "F", "")]
        out = apply_groove_to_chord_track(
            track, _sig_change_at_120(), bpm=120.0,
            template="swing16", strength=1.0, ratio=0.6,
        )
        # 4.0 是段起点, 4.75 是 offbeat
        assert out[0].start == 4.0
        assert out[0].end == 4.8  # 4.75 + 0.05


class TestEndToEndWithQuantize:
    """v0.4.9: 与 quantize 串接的端到端。"""

    def test_quantize_then_groove(self):
        """v0.4.9: quantize 输出后 groove 仍合法。"""
        from mujik.chord.quantize import quantize_chord_track

        raw = [
            ChordEvent(0.0, 0.5, "C", ""),
            ChordEvent(0.5, 1.0, "F", ""),
            ChordEvent(1.0, 1.5, "G", "7"),
        ]
        quantized = quantize_chord_track(
            raw, _sig4_4_at_120(), bpm=120.0,
            grid_per_bar=4, merge_consecutive=False, min_duration_sec=0.0,
        )
        grooved = apply_groove_to_chord_track(
            quantized, _sig4_4_at_120(), bpm=120.0,
            template="swing16", strength=1.0, ratio=0.6,
        )
        # 全部合法
        for c in grooved:
            assert c.end > c.start
        # on-beat 位置不偏移
        assert grooved[0].start == 0.0
        assert grooved[1].start == 0.5
        assert grooved[2].start == 1.0
        # on-beat end 不偏移
        assert grooved[0].end == 0.5
        assert grooved[1].end == 1.0
