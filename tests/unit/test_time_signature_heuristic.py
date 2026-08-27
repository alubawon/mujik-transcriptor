"""Tests for rhythm.time_signature (heuristic)."""
from __future__ import annotations

import pytest

from mujik.rhythm.time_signature import (
    VALID_NUMERATORS,
    infer_time_signature_from_downbeats,
)


def _make_4_4_beats(bpm: float = 120.0, bars: int = 4) -> tuple[list[float], list[float]]:
    """构造 4/4 节拍：每 bar 4 个 beat + 1 downbeat。"""
    beat_interval = 60.0 / bpm
    beats = []
    downbeats = []
    for bar in range(bars):
        t0 = bar * 4 * beat_interval
        downbeats.append(t0)
        for i in range(4):
            beats.append(t0 + i * beat_interval)
    return beats, downbeats


def _make_3_4_beats(bpm: float = 90.0, bars: int = 4) -> tuple[list[float], list[float]]:
    beat_interval = 60.0 / bpm
    beats = []
    downbeats = []
    for bar in range(bars):
        t0 = bar * 3 * beat_interval
        downbeats.append(t0)
        for i in range(3):
            beats.append(t0 + i * beat_interval)
    return beats, downbeats


def _make_5_4_beats(bpm: float = 80.0, bars: int = 4) -> tuple[list[float], list[float]]:
    beat_interval = 60.0 / bpm
    beats = []
    downbeats = []
    for bar in range(bars):
        t0 = bar * 5 * beat_interval
        downbeats.append(t0)
        for i in range(5):
            beats.append(t0 + i * beat_interval)
    return beats, downbeats


def _make_7_8_beats(bpm: float = 140.0, bars: int = 4) -> tuple[list[float], list[float]]:
    beat_interval = 60.0 / bpm
    beats = []
    downbeats = []
    for bar in range(bars):
        t0 = bar * 7 * beat_interval
        downbeats.append(t0)
        for i in range(7):
            beats.append(t0 + i * beat_interval)
    return beats, downbeats


def _make_meter_change(bpm: float = 120.0) -> tuple[list[float], list[float]]:
    """4/4 前 4 小节 → 3/4 后 4 小节（变拍子）。"""
    beat_interval = 60.0 / bpm
    beats = []
    downbeats = []
    # 4 bars of 4/4
    for bar in range(4):
        t0 = bar * 4 * beat_interval
        downbeats.append(t0)
        for i in range(4):
            beats.append(t0 + i * beat_interval)
    # 4 bars of 3/4
    t_44_end = 4 * 4 * beat_interval
    for bar in range(4):
        t0 = t_44_end + bar * 3 * beat_interval
        downbeats.append(t0)
        for i in range(3):
            beats.append(t0 + i * beat_interval)
    return beats, downbeats


class TestInference:
    def test_4_4(self):
        beats, downbeats = _make_4_4_beats()
        segs = infer_time_signature_from_downbeats(downbeats, beats, duration=8.0)
        assert len(segs) >= 1
        assert segs[0].time_signature == (4, 4)
        assert segs[0].confidence > 0.7

    def test_3_4(self):
        beats, downbeats = _make_3_4_beats()
        segs = infer_time_signature_from_downbeats(downbeats, beats, duration=8.0)
        assert segs[0].time_signature == (3, 4)
        assert segs[0].confidence > 0.7

    def test_5_4(self):
        beats, downbeats = _make_5_4_beats()
        segs = infer_time_signature_from_downbeats(downbeats, beats, duration=10.0)
        assert segs[0].time_signature == (5, 4)
        assert segs[0].confidence > 0.7

    def test_7_8(self):
        beats, downbeats = _make_7_8_beats()
        segs = infer_time_signature_from_downbeats(downbeats, beats, duration=12.0)
        assert segs[0].time_signature == (7, 4)  # denominator 固定 4
        assert segs[0].confidence > 0.7

    def test_meter_change_4_4_to_3_4(self):
        beats, downbeats = _make_meter_change()
        segs = infer_time_signature_from_downbeats(downbeats, beats, duration=14.0)
        # 至少 2 段
        assert len(segs) >= 2
        # 第一段是 4/4
        assert segs[0].time_signature[0] == 4
        # 后面有 3/4 段
        has_34 = any(s.time_signature[0] == 3 for s in segs[1:])
        assert has_34


class TestFallbacks:
    def test_empty_downbeats(self):
        segs = infer_time_signature_from_downbeats([], [], duration=10.0)
        assert len(segs) == 1
        assert segs[0].time_signature == (4, 4)
        assert segs[0].confidence == 0.3
        assert segs[0].source == "default_4_4"

    def test_insufficient_data(self):
        segs = infer_time_signature_from_downbeats([0.0], [0.0, 0.5], duration=2.0)
        assert segs[0].time_signature == (4, 4)
        assert segs[0].confidence == 0.3

    def test_custom_fallback(self):
        segs = infer_time_signature_from_downbeats(
            [], [], duration=5.0, fallback=(3, 4),
        )
        assert segs[0].time_signature == (3, 4)

    def test_estimated_count_from_interval(self):
        """downbeats 间无 beat（madmom 偶尔）→ 用间距/median_beat_interval 估算。"""
        # 构造 4/4 但 beats 不在 downbeats 之间（madmom 把 downbeat 排除在 beats 外）
        bpm = 120.0
        beat_interval = 60.0 / bpm
        downbeats = [i * 4 * beat_interval for i in range(4)]
        # beats 只在每个 downbeat 后 1 beat 间隔的位置（每小节 4 个）
        beats = []
        for d in downbeats:
            for i in range(4):
                beats.append(d + i * beat_interval)
        # 重新过滤：去掉 downbeat 自身
        db_set = set(round(d, 6) for d in downbeats)
        beats = [b for b in beats if round(b, 6) not in db_set]
        segs = infer_time_signature_from_downbeats(
            downbeats, beats, duration=8.0,
        )
        # 4/4 应被识别
        assert segs[0].time_signature[0] in (3, 4, 5)  # 估算容差


class TestValidNumerators:
    def test_contains_common(self):
        for n in (2, 3, 4, 5, 6, 7):
            assert n in VALID_NUMERATORS

    def test_does_not_contain_extreme(self):
        for n in (1, 8, 10, 11, 13, 16, 24):
            assert n not in VALID_NUMERATORS


class TestEdgeCases:
    def test_duration_zero(self):
        segs = infer_time_signature_from_downbeats([], [], duration=0.0)
        assert segs[0].end_time > 0  # 强制 >= 1.0

    def test_default_segments_helper(self):
        """ensure build_default_segments compatible."""
        from mujik.time_signature.model import build_default_segments
        default = build_default_segments(10.0)
        assert default[0].time_signature == (4, 4)
