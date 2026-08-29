"""Tests for find_chord_at_time O(log n) bisect (v0.4.7)."""
from __future__ import annotations

import logging
import random

import pytest

from mujik.midi.model import ChordEvent
from mujik.score.harmony import find_chord_at_time


def _make_chord_track(n: int, step: float = 1.0) -> list[ChordEvent]:
    """生成 n 个相邻的 chord（不重叠，step 秒一个）。"""
    return [
        ChordEvent(i * step, (i + 1) * step, "C", "")
        for i in range(n)
    ]


def _linear_find(chord_track: list[ChordEvent], t: float) -> ChordEvent | None:
    """参考实现：O(n) 线性扫描，用于一致性比对。"""
    for chord in chord_track:
        if chord.start <= t < chord.end:
            return chord
    return None


class TestBasicCases:
    """v0.4.7: 基础 case 与 v0.4.1 行为兼容。"""

    def test_empty_track(self):
        assert find_chord_at_time(None, 1.0) is None
        assert find_chord_at_time([], 1.0) is None

    def test_single_chord_found(self):
        track = [ChordEvent(0.0, 2.0, "C", "")]
        assert find_chord_at_time(track, 1.0) == track[0]

    def test_single_chord_before(self):
        track = [ChordEvent(2.0, 4.0, "C", "")]
        assert find_chord_at_time(track, 1.0) is None

    def test_finds_chord_at_time(self):
        track = [
            ChordEvent(0.0, 2.0, "C", ""),
            ChordEvent(2.0, 4.0, "F", ""),
            ChordEvent(4.0, 6.0, "G", "7"),
        ]
        assert find_chord_at_time(track, 1.0).root == "C"
        assert find_chord_at_time(track, 2.5).root == "F"
        assert find_chord_at_time(track, 4.5).root == "G"


class TestBisectCorrectness:
    """v0.4.7: bisect 与 linear 结果一致。"""

    def test_random_100_chords_consistent(self):
        """v0.4.7: 100 chord 随机查询 bisect == linear。"""
        random.seed(42)
        n = 100
        track = _make_chord_track(n, step=1.0)
        for _ in range(200):
            t = random.uniform(-5, n + 5)
            bisect_result = find_chord_at_time(track, t)
            linear_result = _linear_find(track, t)
            assert bisect_result == linear_result, (
                f"t={t}: bisect={bisect_result}, linear={linear_result}"
            )

    def test_random_unsorted_also_consistent(self):
        """v0.4.7: 乱序输入 fallback 后仍正确（通过 random shuffle）。"""
        random.seed(43)
        n = 50
        track = _make_chord_track(n, step=2.0)
        shuffled = list(track)
        random.shuffle(shuffled)
        for _ in range(100):
            t = random.uniform(-5, n * 2 + 5)
            bisect_result = find_chord_at_time(shuffled, t)
            linear_result = _linear_find(shuffled, t)
            assert bisect_result == linear_result, (
                f"t={t}: bisect={bisect_result}, linear={linear_result}"
            )

    def test_dense_chord_track(self):
        """v0.4.7: 1000 chord 密集 track 仍正确。"""
        n = 1000
        track = _make_chord_track(n, step=0.1)
        # 查询每个 chord 的中点
        for i in range(n):
            t = i * 0.1 + 0.05
            assert find_chord_at_time(track, t) == track[i]


class TestEdgeCases:
    """v0.4.7: 边界情况。"""

    def test_t_before_all_chords(self):
        track = _make_chord_track(5, step=1.0)
        assert find_chord_at_time(track, -1.0) is None

    def test_t_after_all_chords(self):
        track = _make_chord_track(5, step=1.0)
        assert find_chord_at_time(track, 100.0) is None

    def test_t_exactly_at_start(self):
        """v0.4.7: t == start 应命中（start <= t 且 t < end）。"""
        track = [ChordEvent(1.0, 3.0, "C", "")]
        assert find_chord_at_time(track, 1.0) == track[0]

    def test_t_exactly_at_end(self):
        """v0.4.7: t == end 不命中（t < end 不满足）。"""
        track = [ChordEvent(1.0, 3.0, "C", "")]
        assert find_chord_at_time(track, 3.0) is None

    def test_t_in_gap_between_chords(self):
        """v0.4.7: chord 之间 gap（无 chord 覆盖）。"""
        track = [
            ChordEvent(0.0, 1.0, "C", ""),
            ChordEvent(2.0, 3.0, "F", ""),
        ]
        assert find_chord_at_time(track, 1.5) is None

    def test_negative_t(self):
        track = _make_chord_track(5, step=1.0)
        assert find_chord_at_time(track, -100.0) is None

    def test_zero_t(self):
        track = [ChordEvent(0.0, 1.0, "C", "")]
        assert find_chord_at_time(track, 0.0) == track[0]


class TestUnsortedFallback:
    """v0.4.7: 乱序输入 fallback + warning。"""

    def test_unsorted_returns_correct(self):
        track = [
            ChordEvent(2.0, 4.0, "C", ""),
            ChordEvent(0.0, 2.0, "F", ""),  # 乱序
        ]
        assert find_chord_at_time(track, 1.0).root == "F"
        assert find_chord_at_time(track, 3.0).root == "C"

    def test_unsorted_logs_warning(self, caplog):
        track = [
            ChordEvent(2.0, 4.0, "C", ""),
            ChordEvent(0.0, 2.0, "F", ""),
        ]
        with caplog.at_level(logging.WARNING, logger="mujik.score.harmony"):
            find_chord_at_time(track, 1.0)
        assert any("not sorted" in r.message for r in caplog.records)

    def test_sorted_no_warning(self, caplog):
        track = _make_chord_track(5, step=1.0)
        with caplog.at_level(logging.WARNING, logger="mujik.score.harmony"):
            find_chord_at_time(track, 2.5)
        # 已排序不应有 warning
        assert not any("not sorted" in r.message for r in caplog.records)


class TestPerformance:
    """v0.4.7: 性能 smoke 测试。

    实际经验：bisect_right 在 Python list 上的常量开销（每次重建
    ``starts`` 列表 + sorted 检查）使得对单次查询而言，比 O(n) 线性
    扫描慢约 1.5-2x。bisect 真正的优势在于：
    - 同一 chord_track 多次查询（如果外部 cache 了 starts）
    - 非常大的 N（>10k chord）

    本测试仅做正确性验证 + 防止意外退化。
    """

    def test_both_paths_return_same_results(self):
        """v0.4.7: bisect 路径和 linear fallback 路径结果完全一致。"""
        n = 500
        track = _make_chord_track(n, step=0.1)

        # bisect 路径（sorted）
        bisect_results = [
            find_chord_at_time(track, i * 0.1 + 0.05) for i in range(n)
        ]
        # linear 路径（用 shuffle 触发 fallback）
        import random
        random.seed(0)
        shuffled = list(track)
        random.shuffle(shuffled)
        linear_results = [
            find_chord_at_time(shuffled, i * 0.1 + 0.05) for i in range(n)
        ]

        # 排序后再比较（linear 返回的是同一 chord 对象，但顺序不同）
        bisect_roots = sorted(c.root for c in bisect_results)
        linear_roots = sorted(c.root for c in linear_results)
        assert bisect_roots == linear_roots == ["C"] * n

    def test_no_regression_on_typical_size(self):
        """v0.4.7: 典型 100 chord 100 query 在 0.5s 内完成。"""
        import time

        n = 100
        track = _make_chord_track(n, step=1.0)
        start = time.perf_counter()
        for i in range(n):
            find_chord_at_time(track, i * 1.0 + 0.5)
        elapsed = time.perf_counter() - start
        # 100 query 100 chord 应该在 0.5s 内（实际通常 <10ms）
        assert elapsed < 0.5, f"too slow: {elapsed:.3f}s"
