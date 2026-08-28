"""Tests for chord/quantize.py (v0.4.5, pure functions)."""
from __future__ import annotations

from pathlib import Path

import pytest

from mujik.chord.quantize import (
    filter_short_chords,
    merge_consecutive_chords,
    quantize_chord_track,
    snap_chord_to_grid,
)
from mujik.midi.model import ChordEvent
from mujik.time_signature.model import TimeSignatureSegment


# ---------- helpers ----------

def _sig4_4_at_120() -> list[TimeSignatureSegment]:
    """4/4 拍号，120 BPM，一段覆盖 0-10s。"""
    return [
        TimeSignatureSegment(
            start_time=0.0,
            end_time=10.0,
            time_signature=(4, 4),
            confidence=1.0,
            source="manual",
        ),
    ]


def _sig_change_at_120() -> list[TimeSignatureSegment]:
    """变拍子：0-4s 4/4，4-10s 3/4。"""
    return [
        TimeSignatureSegment(
            start_time=0.0,
            end_time=4.0,
            time_signature=(4, 4),
            confidence=1.0,
            source="manual",
        ),
        TimeSignatureSegment(
            start_time=4.0,
            end_time=10.0,
            time_signature=(3, 4),
            confidence=1.0,
            source="manual",
        ),
    ]


# 120 BPM: 1 beat = 0.5s, 1 bar (4/4) = 2.0s
# grid_per_bar=4 → grid step = 0.5s (beat)
# grid_per_bar=2 → grid step = 1.0s (half-bar)
# grid_per_bar=1 → grid step = 2.0s (full bar)
# grid_per_bar=8 → grid step = 0.25s (8th)


class TestSnapToGrid:
    """v0.4.5: snap_chord_to_grid() 测试。"""

    def test_snap_to_beat_grid(self):
        """v0.4.5: grid_per_bar=4 (beat) 时，t=0.3s snap 到 0.5s。"""
        c = ChordEvent(start=0.1, end=0.7, root="C", quality="")
        out = snap_chord_to_grid(
            c, _sig4_4_at_120(), bpm=120.0, grid_per_bar=4, duration=10.0,
        )
        # grid step = 2.0/4 = 0.5s, grid points: 0.0, 0.5, 1.0, ...
        assert out.start == 0.0  # round(0.1/0.5) = 0
        assert out.end == 0.5  # round(0.7/0.5) = 1
        assert out.root == "C"
        assert out.quality == ""

    def test_snap_to_half_bar_grid(self):
        """v0.4.5: grid_per_bar=2 (half-bar) 时，t=0.4s snap 到 0.0s。"""
        c = ChordEvent(start=0.4, end=0.6, root="F", quality="")
        out = snap_chord_to_grid(
            c, _sig4_4_at_120(), bpm=120.0, grid_per_bar=2, duration=10.0,
        )
        # grid step = 2.0/2 = 1.0s, grid points: 0.0, 1.0, 2.0
        assert out.start == 0.0  # round(0.4/1.0) = 0
        assert out.end == 1.0  # round(0.6/1.0) = 1

    def test_snap_to_full_bar_grid(self):
        """v0.4.5: grid_per_bar=1 (bar) 时，t=0.4s snap 到 0.0s。"""
        c = ChordEvent(start=0.4, end=1.6, root="G", quality="m")
        out = snap_chord_to_grid(
            c, _sig4_4_at_120(), bpm=120.0, grid_per_bar=1, duration=10.0,
        )
        # grid step = 2.0s, grid points: 0.0, 2.0
        assert out.start == 0.0
        assert out.end == 2.0

    def test_snap_to_8th_grid(self):
        """v0.4.5: grid_per_bar=8 (8th) 时，t=0.4s snap 到 0.5s。"""
        c = ChordEvent(start=0.4, end=0.6, root="C", quality="")
        out = snap_chord_to_grid(
            c, _sig4_4_at_120(), bpm=120.0, grid_per_bar=8, duration=10.0,
        )
        # grid step = 2.0/8 = 0.25s, grid points: 0.0, 0.25, 0.5, 0.75
        assert out.start == 0.5  # round(0.4/0.25)=2 → 0.5
        # end=0.6 → round=2 → 0.5, 但 end==start → 防御: 扩展到 0.5+0.25=0.75
        assert out.end == 0.75
        assert out.end > out.start

    def test_snap_clamps_to_segment_start(self):
        """v0.4.5: t < segment.start → clamp 到 start。"""
        c = ChordEvent(start=-0.5, end=0.3, root="C", quality="")
        out = snap_chord_to_grid(
            c, _sig4_4_at_120(), bpm=120.0, grid_per_bar=4, duration=10.0,
        )
        assert out.start == 0.0
        assert out.end == 0.5

    def test_snap_clamps_to_segment_end(self):
        """v0.4.5: t > segment.end → clamp 到段内最后 grid 点。"""
        c = ChordEvent(start=8.0, end=12.0, root="C", quality="")
        out = snap_chord_to_grid(
            c, _sig4_4_at_120(), bpm=120.0, grid_per_bar=4, duration=10.0,
        )
        # t=8.0 正常 snap 到 8.0; t=12.0 超过段尾 10.0, clamp 到 10.0
        assert out.start == 8.0
        assert out.end == 10.0
        # 但 end==start? 实际 10.0 > 8.0, 不触发防御
        assert out.end > out.start

    def test_snap_invalid_bpm(self):
        c = ChordEvent(start=0.0, end=1.0, root="C", quality="")
        with pytest.raises(ValueError, match="bpm"):
            snap_chord_to_grid(c, _sig4_4_at_120(), bpm=0.0)

    def test_snap_invalid_grid_per_bar(self):
        c = ChordEvent(start=0.0, end=1.0, root="C", quality="")
        with pytest.raises(ValueError, match="grid_per_bar"):
            snap_chord_to_grid(c, _sig4_4_at_120(), bpm=120.0, grid_per_bar=3)

    def test_snap_uses_default_segment_when_empty(self):
        """v0.4.5: time_signatures=[] → 用兜底 4/4 段。"""
        c = ChordEvent(start=0.4, end=0.6, root="C", quality="")
        out = snap_chord_to_grid(c, [], bpm=120.0, grid_per_bar=4, duration=5.0)
        # 兜底段 (0, 5) 4/4, step=0.5
        # t=0.4 → 0.5; t=0.6 → 1.0
        assert out.start == 0.5
        assert out.end == 1.0

    def test_snap_preserves_root_quality_bass(self):
        """v0.4.5: snap 不改 root/quality/bass。"""
        c = ChordEvent(start=0.1, end=0.5, root="F#", quality="m", bass="A")
        out = snap_chord_to_grid(
            c, _sig4_4_at_120(), bpm=120.0, grid_per_bar=4, duration=10.0,
        )
        assert out.root == "F#"
        assert out.quality == "m"
        assert out.bass == "A"


class TestMergeConsecutive:
    """v0.4.5: merge_consecutive_chords() 测试。"""

    def test_merge_two_same_chords(self):
        """v0.4.5: 相邻 C → 合并为 C 0-4。"""
        track = [
            ChordEvent(0.0, 2.0, "C", ""),
            ChordEvent(2.0, 4.0, "C", ""),
        ]
        out = merge_consecutive_chords(track)
        assert len(out) == 1
        assert out[0] == ChordEvent(0.0, 4.0, "C", "")

    def test_merge_three_same_chords(self):
        track = [
            ChordEvent(0.0, 1.0, "C", ""),
            ChordEvent(1.0, 2.0, "C", ""),
            ChordEvent(2.0, 3.0, "C", ""),
        ]
        out = merge_consecutive_chords(track)
        assert len(out) == 1
        assert out[0] == ChordEvent(0.0, 3.0, "C", "")

    def test_no_merge_different_roots(self):
        track = [
            ChordEvent(0.0, 2.0, "C", ""),
            ChordEvent(2.0, 4.0, "F", ""),
        ]
        out = merge_consecutive_chords(track)
        assert len(out) == 2

    def test_no_merge_different_quality(self):
        track = [
            ChordEvent(0.0, 2.0, "C", ""),
            ChordEvent(2.0, 4.0, "C", "m"),
        ]
        out = merge_consecutive_chords(track)
        assert len(out) == 2

    def test_no_merge_different_bass(self):
        """v0.4.5: bass 不同不合并（slash chord 不同）。"""
        track = [
            ChordEvent(0.0, 2.0, "C", "", bass="E"),
            ChordEvent(2.0, 4.0, "C", "", bass="A"),
        ]
        out = merge_consecutive_chords(track)
        assert len(out) == 2

    def test_merge_with_gap_tolerance(self):
        """v0.4.5: 微小 gap (1e-6) 内仍合并。"""
        track = [
            ChordEvent(0.0, 2.0, "C", ""),
            ChordEvent(2.0 + 1e-7, 4.0, "C", ""),
        ]
        out = merge_consecutive_chords(track)
        assert len(out) == 1

    def test_no_merge_with_gap(self):
        track = [
            ChordEvent(0.0, 2.0, "C", ""),
            ChordEvent(2.1, 4.0, "C", ""),
        ]
        out = merge_consecutive_chords(track)
        # 2.0 < 2.1, 不连续，不合并
        assert len(out) == 2

    def test_merge_empty(self):
        assert merge_consecutive_chords([]) == []

    def test_merge_single(self):
        track = [ChordEvent(0.0, 2.0, "C", "")]
        out = merge_consecutive_chords(track)
        assert out == [ChordEvent(0.0, 2.0, "C", "")]


class TestFilterShort:
    """v0.4.5: filter_short_chords() 测试。"""

    def test_filter_below_threshold(self):
        track = [
            ChordEvent(0.0, 0.3, "C", ""),  # 0.3s < 0.5s, 过滤
            ChordEvent(0.5, 2.0, "F", ""),  # 1.5s >= 0.5s, 保留
        ]
        out = filter_short_chords(track, min_duration_sec=0.5)
        assert len(out) == 1
        assert out[0].root == "F"

    def test_filter_disabled_when_zero(self):
        """v0.4.5: min_duration_sec=0 时不过滤。"""
        track = [ChordEvent(0.0, 0.001, "C", "")]
        out = filter_short_chords(track, min_duration_sec=0.0)
        assert len(out) == 1

    def test_filter_at_exact_threshold(self):
        """v0.4.5: 持续时间 == min_duration_sec 时保留（>= 边界含）。"""
        track = [ChordEvent(0.0, 0.5, "C", "")]
        out = filter_short_chords(track, min_duration_sec=0.5)
        assert len(out) == 1

    def test_filter_all_short(self):
        track = [
            ChordEvent(0.0, 0.1, "C", ""),
            ChordEvent(0.2, 0.3, "F", ""),
        ]
        out = filter_short_chords(track, min_duration_sec=0.5)
        assert out == []


class TestTimeSignatureChange:
    """v0.4.5: 跨拍号段 snap。"""

    def test_snap_in_3_4_segment(self):
        """v0.4.5: 3/4 段内 grid = 0.5s × 3 = 1.5s/bar。"""
        c = ChordEvent(start=4.2, end=5.8, root="C", quality="")
        out = snap_chord_to_grid(
            c, _sig_change_at_120(), bpm=120.0, grid_per_bar=4, duration=10.0,
        )
        # 3/4 段在 4-10s, bar = 1.5s, grid step = 1.5/4 = 0.375s
        # 但 4.2 距离段起点 = 0.2, round(0.2/0.375) = 1 → 4.0 + 0.375 = 4.375
        # 实际允许一些浮点
        assert 4.0 <= out.start <= 4.5
        assert out.start >= 4.0  # 不跨段

    def test_snap_first_segment_then_second(self):
        """v0.4.5: 两个 chord 分别在 4/4 和 3/4 段，都正确 snap。"""
        c1 = ChordEvent(start=0.4, end=1.6, root="C", quality="")
        c2 = ChordEvent(start=5.0, end=6.0, root="F", quality="")
        out1 = snap_chord_to_grid(
            c1, _sig_change_at_120(), bpm=120.0, grid_per_bar=4, duration=10.0,
        )
        out2 = snap_chord_to_grid(
            c2, _sig_change_at_120(), bpm=120.0, grid_per_bar=4, duration=10.0,
        )
        # 4/4 段: grid step 0.5s
        assert out1.start == 0.5
        assert out1.end == 1.5
        # 3/4 段: grid step 0.375s
        assert 4.0 <= out2.start <= 5.5
        assert 4.0 <= out2.end <= 6.5


class TestQuantizeChordTrack:
    """v0.4.5: quantize_chord_track() 端到端。"""

    def test_empty_input(self):
        assert quantize_chord_track([], _sig4_4_at_120(), bpm=120.0) == []

    def test_snap_merge_filter_pipeline(self):
        """v0.4.5: snap → merge → filter 完整 pipeline。"""
        # 120 BPM, 4/4, grid_per_bar=4 → step=0.5s
        track = [
            ChordEvent(0.0, 1.5, "C", ""),     # 1.5s → snap (0.0, 1.5) → keep
            ChordEvent(1.5, 3.0, "C", ""),     # 1.5s → snap (1.5, 3.0) → merge with prev
            ChordEvent(3.0, 3.1, "F", ""),     # 0.1s → snap+defense (3.0, 3.5) → filter
        ]
        out = quantize_chord_track(
            track, _sig4_4_at_120(), bpm=120.0,
            grid_per_bar=4,
            merge_consecutive=True,
            min_duration_sec=0.6,  # 防御后 0.5s F 被过滤
            duration=10.0,
        )
        # C merged: (0.0, 3.0), F filtered
        assert len(out) == 1
        assert out[0].root == "C"
        assert out[0].start == 0.0
        assert out[0].end == 3.0

    def test_merge_after_snap(self):
        """v0.4.5: snap 后的相邻同 chord 合并。"""
        # 2 chord 在 0.0-0.4 和 0.4-0.8，grid=0.5
        # snap: (0.0, 0.5) 和 (0.5, 1.0) → 合并为 (0.0, 1.0)
        track = [
            ChordEvent(0.0, 0.4, "C", ""),
            ChordEvent(0.4, 0.8, "C", ""),
        ]
        out = quantize_chord_track(
            track, _sig4_4_at_120(), bpm=120.0,
            grid_per_bar=4, merge_consecutive=True, min_duration_sec=0.0,
        )
        assert len(out) == 1
        assert out[0] == ChordEvent(0.0, 1.0, "C", "")

    def test_no_merge_when_disabled(self):
        track = [
            ChordEvent(0.0, 2.0, "C", ""),
            ChordEvent(2.0, 4.0, "C", ""),
        ]
        out = quantize_chord_track(
            track, _sig4_4_at_120(), bpm=120.0,
            merge_consecutive=False, min_duration_sec=0.0,
        )
        assert len(out) == 2

    def test_no_filter_when_disabled(self):
        track = [ChordEvent(0.0, 0.1, "C", "")]
        out = quantize_chord_track(
            track, _sig4_4_at_120(), bpm=120.0,
            min_duration_sec=0.0,
        )
        assert len(out) == 1

    def test_realistic_madmom_output(self):
        """v0.4.5: 模拟 madmom 100ms 帧粒度输出。"""
        # madmom 输出 10fps 帧粒度: chord 边界不是整数秒
        track = [
            ChordEvent(0.1, 1.3, "C", ""),
            ChordEvent(1.3, 2.2, "F", ""),
            ChordEvent(2.2, 3.1, "C", ""),
        ]
        out = quantize_chord_track(
            track, _sig4_4_at_120(), bpm=120.0,
            grid_per_bar=4, merge_consecutive=False, min_duration_sec=0.0,
        )
        # snap 到 0.5/0.5 grid points
        # chord 1: start=0.1→0.0, end=1.3→1.5
        # chord 2: start=1.3→1.5, end=2.2→2.0
        # chord 3: start=2.2→2.0, end=3.1→3.0
        assert len(out) == 3
        assert out[0].start == 0.0
        assert out[0].end == 1.5
        assert out[1].start == 1.5
        assert out[1].end == 2.0
        assert out[2].start == 2.0
        assert out[2].end == 3.0


class TestEdgeCases:
    """v0.4.5: 边界情况。"""

    def test_bpm_zero_raises(self):
        with pytest.raises(ValueError, match="bpm"):
            quantize_chord_track(
                [ChordEvent(0.0, 1.0, "C", "")],
                _sig4_4_at_120(), bpm=0.0,
            )

    def test_single_chord(self):
        track = [ChordEvent(0.5, 1.5, "C", "")]
        out = quantize_chord_track(
            track, _sig4_4_at_120(), bpm=120.0,
            min_duration_sec=0.0, merge_consecutive=True,
        )
        assert len(out) == 1

    def test_all_chords_at_same_time(self):
        """v0.4.5: 所有 chord 重叠 → snap 后合并。"""
        track = [
            ChordEvent(0.0, 0.5, "C", ""),
            ChordEvent(0.0, 0.5, "F", ""),
            ChordEvent(0.0, 0.5, "G", ""),
        ]
        out = quantize_chord_track(
            track, _sig4_4_at_120(), bpm=120.0,
            min_duration_sec=0.0, merge_consecutive=False,
        )
        # 不同 root 不合并
        assert len(out) == 3
