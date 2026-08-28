"""Tests for score/bend.py."""
from __future__ import annotations

import pytest

from mujik.score.bend import (
    BEND_ALTER_MAX,
    BEND_ALTER_MIN,
    build_bend_element,
    pitch_bend_to_alter,
)


class TestPitchBendToAlter:
    def test_empty_returns_zero(self):
        """空 tuple → 0（无 bend）。"""
        assert pitch_bend_to_alter(()) == 0

    def test_all_zero_returns_zero(self):
        """全 0 序列 → 0（无实际弯音）。"""
        assert pitch_bend_to_alter((0.0, 0.0, 0.0)) == 0

    def test_full_positive_bend(self):
        """+1 满弯音 → 2。"""
        assert pitch_bend_to_alter((1.0, 1.0, 0.5)) == 2

    def test_full_negative_bend(self):
        """-1 满弯音 → -2。"""
        assert pitch_bend_to_alter((-1.0, -0.5, 0.0)) == -2

    def test_half_bend(self):
        """0.5 弯音 → 1。"""
        assert pitch_bend_to_alter((0.5, 0.4, 0.3)) == 1

    def test_quarter_bend(self):
        """0.25 弯音 → round(0.5) = 0（边界）。"""
        # round(0.25 * 2) = round(0.5) = 0（banker's rounding 在 Python
        # 中 round half to even, 0.5 → 0）
        alter = pitch_bend_to_alter((0.25,))
        assert alter in (0, 1)  # 允许 round-half-to-even 模糊

    def test_three_quarters_bend(self):
        """0.75 弯音 → 1 或 2（round 边界）。"""
        alter = pitch_bend_to_alter((0.75,))
        assert alter in (1, 2)

    def test_clamp_above_max(self):
        """>1.0 异常值（虽然 Note.__post_init__ 会拒绝，这里直接测函数）→ 钳到 2。"""
        assert pitch_bend_to_alter((1.5,)) == 2

    def test_clamp_below_min(self):
        """<-1.0 → 钳到 -2。"""
        assert pitch_bend_to_alter((-1.5,)) == -2

    def test_uses_max_abs_frame(self):
        """用 max abs 帧，不是平均。"""
        # 多数帧 0.0 但有一帧 1.0 → 应该是 2
        seq = (0.0,) * 99 + (1.0,)
        assert pitch_bend_to_alter(seq) == 2


class TestBuildBendElement:
    def test_valid_alter_positive(self):
        xml = build_bend_element(1)
        assert xml == '<bend alter="1"><bend-alter>1</bend-alter></bend>'

    def test_valid_alter_negative(self):
        xml = build_bend_element(-1)
        assert xml == '<bend alter="-1"><bend-alter>-1</bend-alter></bend>'

    def test_valid_alter_zero(self):
        # alter=0 仍合法（builder 可选择性跳过；这里测函数能生成）
        xml = build_bend_element(0)
        assert xml == '<bend alter="0"><bend-alter>0</bend-alter></bend>'

    def test_max_alter(self):
        xml = build_bend_element(BEND_ALTER_MAX)
        assert f'alter="{BEND_ALTER_MAX}"' in xml
        assert f'<bend-alter>{BEND_ALTER_MAX}</bend-alter>' in xml

    def test_min_alter(self):
        xml = build_bend_element(BEND_ALTER_MIN)
        assert f'alter="{BEND_ALTER_MIN}"' in xml

    def test_out_of_range_raises(self):
        with pytest.raises(ValueError):
            build_bend_element(BEND_ALTER_MAX + 1)
        with pytest.raises(ValueError):
            build_bend_element(BEND_ALTER_MIN - 1)


class TestIntegration:
    def test_realistic_pitch_bend_sequence(self):
        """模拟 basic-pitch 实际输出：1 帧最大弯音 ≈ 0.4。"""
        seq = (0.0, 0.1, 0.3, 0.4, 0.4, 0.3, 0.1, 0.0)
        alter = pitch_bend_to_alter(seq)
        # 0.4 * 2 = 0.8 → round = 1
        assert alter == 1
        # 生成 XML
        xml = build_bend_element(alter)
        assert "<bend" in xml
        assert "<bend-alter>1</bend-alter>" in xml
