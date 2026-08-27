"""Tests for merge/density.py."""
from __future__ import annotations

import pytest

from mujik.merge.density import apply_density_filter
from mujik.midi.model import Note


def _n(start, end, pitch, vel, **kw) -> Note:
    return Note(start=start, end=end, pitch=pitch, velocity=vel, **kw)


class TestDensityFilter:
    def test_under_limit_kept(self):
        notes = [_n(0.0, 0.5, 60, 100), _n(0.0, 0.5, 64, 90), _n(0.0, 0.5, 67, 80)]
        kept, dropped = apply_density_filter(notes, max_simultaneous=4)
        assert dropped == 0
        assert len(kept) == 3

    def test_exact_limit_kept(self):
        notes = [_n(0.0, 0.5, 60, 100), _n(0.0, 0.5, 64, 90), _n(0.0, 0.5, 67, 80)]
        kept, dropped = apply_density_filter(notes, max_simultaneous=3)
        assert dropped == 0
        assert len(kept) == 3

    def test_over_limit_drops_lowest_velocity(self):
        """13 notes, max=12 → 最低 velocity 被丢弃。"""
        notes = [
            _n(0.0, 0.5, 60, 100),
            _n(0.0, 0.5, 62, 90),
            _n(0.0, 0.5, 64, 80),
            _n(0.0, 0.5, 65, 70),
            _n(0.0, 0.5, 67, 60),
            _n(0.0, 0.5, 69, 50),
            _n(0.0, 0.5, 71, 40),
            _n(0.0, 0.5, 72, 30),
            _n(0.0, 0.5, 74, 20),
            _n(0.0, 0.5, 76, 10),
            _n(0.0, 0.5, 77, 110),
            _n(0.0, 0.5, 79, 120),
            _n(0.0, 0.5, 81, 95),
        ]
        kept, dropped = apply_density_filter(notes, max_simultaneous=12)
        assert dropped == 1
        assert len(kept) == 12
        # 最低 vel=10 的 note 应被丢
        assert all(n.velocity != 10 for n in kept)

    def test_non_overlapping_no_drop(self):
        notes = [
            _n(0.0, 0.1, 60, 100),
            _n(0.0, 0.1, 62, 100),
            _n(0.0, 0.1, 64, 100),
            _n(0.0, 0.1, 65, 100),
            _n(0.0, 0.1, 67, 100),
            _n(0.0, 0.1, 69, 100),
            _n(0.0, 0.1, 71, 100),
            _n(0.0, 0.1, 72, 100),
            _n(0.0, 0.1, 74, 100),
            _n(0.0, 0.1, 76, 100),
            _n(0.0, 0.1, 77, 100),
            _n(0.0, 0.1, 79, 100),
            _n(0.0, 0.1, 81, 100),
            _n(0.0, 0.1, 83, 100),
            _n(0.0, 0.1, 84, 100),
        ]
        # 全在 0-0.1s 同时 → 超过 max=12
        kept, dropped = apply_density_filter(notes, max_simultaneous=12)
        assert dropped == 3
        assert len(kept) == 12

    def test_empty_input(self):
        kept, dropped = apply_density_filter([], max_simultaneous=12)
        assert kept == []
        assert dropped == 0

    def test_invalid_max_raises(self):
        with pytest.raises(ValueError):
            apply_density_filter([_n(0.0, 0.1, 60, 100)], max_simultaneous=0)
        with pytest.raises(ValueError):
            apply_density_filter([_n(0.0, 0.1, 60, 100)], max_simultaneous=-1)

    def test_sequential_onsets_kept(self):
        """按时间序列的 note，只要不同时发声就全保留。"""
        notes = [
            _n(0.0, 0.5, 60, 100),
            _n(0.5, 1.0, 62, 100),
            _n(1.0, 1.5, 64, 100),
            _n(1.5, 2.0, 65, 100),
        ]
        kept, dropped = apply_density_filter(notes, max_simultaneous=1)
        # 每个 note 在其 start 时 active 为 0（上一 note 0.5s 结束）→ 不超限
        assert dropped == 0
        assert len(kept) == 4

    def test_preserves_input_order(self):
        notes = [
            _n(0.0, 0.5, 60, 50),  # low
            _n(0.0, 0.5, 62, 100),  # high
        ]
        kept, dropped = apply_density_filter(notes, max_simultaneous=1)
        assert dropped == 1
        # 保留的应是 vel=100
        assert kept[0].velocity == 100
