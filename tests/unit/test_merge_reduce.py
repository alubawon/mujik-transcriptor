"""Tests for merge/reduce.py."""
from __future__ import annotations

import pytest

from mujik.merge.reduce import piano_reduce
from mujik.midi.model import Note


def _n(start, end, pitch, vel) -> Note:
    return Note(start=start, end=end, pitch=pitch, velocity=vel)


class TestPianoReduce:
    def test_empty_input(self):
        kept, dropped = piano_reduce([], max_simultaneous=6)
        assert kept == []
        assert dropped == 0

    def test_under_limit_kept(self):
        notes = [_n(0.0, 0.5, 60, 100), _n(0.0, 0.5, 64, 90)]
        kept, dropped = piano_reduce(notes, max_simultaneous=4)  # K=2
        assert dropped == 0
        assert len(kept) == 2

    def test_top_k_by_velocity(self):
        """8 notes 同 onset, max=12 → K=6, 保留 top 6 velocity。"""
        notes = [
            _n(0.0, 0.5, 60, 10),
            _n(0.0, 0.5, 62, 20),
            _n(0.0, 0.5, 64, 30),
            _n(0.0, 0.5, 65, 40),
            _n(0.0, 0.5, 67, 50),
            _n(0.0, 0.5, 69, 60),
            _n(0.0, 0.5, 71, 70),
            _n(0.0, 0.5, 72, 80),
        ]
        kept, dropped = piano_reduce(notes, max_simultaneous=12)  # K=6
        assert dropped == 2
        assert len(kept) == 6
        # 最低 vel=10, 20 应被丢
        kept_vels = sorted(n.velocity for n in kept)
        assert kept_vels[0] >= 30

    def test_held_note_dropped_when_low_velocity(self):
        """持续低 vel held note 应被丢。"""
        # 长 held note（vel=10） + 短 onset notes（vel 高）
        notes = [
            _n(0.0, 5.0, 36, 10),  # 长 held，低 vel
            _n(0.0, 0.1, 60, 100),  # 短 onset
            _n(0.0, 0.1, 64, 100),
        ]
        kept, dropped = piano_reduce(notes, max_simultaneous=4)  # K=2
        # 第一 onset: top 2 by vel = (60,100) (64,100)；held (36,10) 速度太低
        # 持续到下一 onset：median=100, 36 velocity<100 → 丢
        # 简化检查：长 held 应被丢
        held_kept = [n for n in kept if n.pitch == 36]
        assert held_kept == []  # 持续音被丢

    def test_single_note_kept(self):
        kept, dropped = piano_reduce([_n(0.0, 0.5, 60, 100)], max_simultaneous=6)
        assert len(kept) == 1
        assert dropped == 0

    def test_max_simultaneous_one_keeps_one_per_onset(self):
        """max=1 → K=1，每 onset 只留 1 个（最高 vel）。"""
        notes = [
            _n(0.0, 0.5, 60, 50),
            _n(0.0, 0.5, 62, 100),  # cluster 1 最高
            _n(1.0, 1.5, 64, 80),   # cluster 2 最高
            _n(1.0, 1.5, 65, 60),
        ]
        kept, dropped = piano_reduce(notes, max_simultaneous=1)
        # cluster 1: 保留 (62,100)，剩 (60,50) 丢
        # cluster 2: 保留 (64,80)，剩 (65,60) 丢
        # 持续音：(62,100) 在 cluster 2 触发 held check：median=100, 100>100 False → 丢
        # 总：2 kept, 3 dropped
        assert len(kept) == 2
        assert dropped == 3
        kept_pitches = sorted([n.pitch for n in kept])
        assert kept_pitches == [62, 64]

    def test_invalid_max_raises(self):
        with pytest.raises(ValueError):
            piano_reduce([_n(0.0, 0.5, 60, 100)], max_simultaneous=0)

    def test_preserves_timing_within_tolerance(self):
        notes = [_n(0.0, 0.5, 60, 100), _n(0.0001, 0.5, 62, 90)]
        kept, dropped = piano_reduce(notes, max_simultaneous=4)
        # 0.0001 差异 < tolerance (1e-4) 视为同 onset
        assert len(kept) == 2
