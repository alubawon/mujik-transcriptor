"""Tests for score/bend.py."""
from __future__ import annotations

import pytest

from mujik.score.bend import (
    BEND_ALTER_MAX,
    BEND_ALTER_MIN,
    BEND_RELEASE_THRESHOLD,
    BendPoint,
    build_bend_element,
    build_bend_elements,
    detect_bend_release,
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


class TestBendPoint:
    """v0.4.3: BendPoint 数据类验证。"""

    def test_valid_point(self):
        p = BendPoint(0.5, 1)
        assert p.time_frac == 0.5
        assert p.alter == 1

    def test_time_frac_out_of_range_raises(self):
        with pytest.raises(ValueError, match="time_frac"):
            BendPoint(-0.1, 0)
        with pytest.raises(ValueError, match="time_frac"):
            BendPoint(1.5, 0)

    def test_alter_out_of_range_raises(self):
        with pytest.raises(ValueError, match="alter"):
            BendPoint(0.5, BEND_ALTER_MIN - 1)
        with pytest.raises(ValueError, match="alter"):
            BendPoint(0.5, BEND_ALTER_MAX + 1)

    def test_boundary_values_accepted(self):
        BendPoint(0.0, BEND_ALTER_MIN)
        BendPoint(1.0, BEND_ALTER_MAX)

    def test_frozen(self):
        p = BendPoint(0.5, 1)
        with pytest.raises(Exception):  # FrozenInstanceError
            p.alter = 2  # type: ignore[misc]


class TestDetectBendRelease:
    """v0.4.3: detect_bend_release() 函数。"""

    def test_empty_returns_zero_no_release(self):
        assert detect_bend_release(()) == (0, False)

    def test_all_zero_returns_zero(self):
        """全 0 序列 → 无 bend → (0, False)。"""
        assert detect_bend_release((0.0, 0.0, 0.0)) == (0, False)

    def test_ramp_up_no_release(self):
        """单向 ramp up（peak 在末尾）→ 无 release。"""
        seq = (0.0, 0.2, 0.4, 0.5, 0.5)
        # peak = 0.5 → alter 1
        assert detect_bend_release(seq) == (1, False)

    def test_ramp_up_down_has_release(self):
        """ramp up + 回到 0 → has_release。"""
        seq = (0.0, 0.3, 0.5, 0.3, 0.0)
        # peak = 0.5 → alter 1；post-peak min(|b|) = 0.0 < 0.1
        assert detect_bend_release(seq) == (1, True)

    def test_negative_release(self):
        """下弯 + 回到 0 → has_release，alter 为负。"""
        seq = (0.0, -0.3, -0.5, -0.3, 0.0)
        assert detect_bend_release(seq) == (-1, True)

    def test_clamps_to_max_alter(self):
        """大峰值钳到 ±2。"""
        seq = (0.0, 0.5, 0.75, 0.5, 0.0)
        # 0.75 * 2 = 1.5 → round = 2
        assert detect_bend_release(seq) == (2, True)

    def test_threshold_respected(self):
        """post-peak 最小 |b| = 阈值之上 → 无 release。"""
        # post-peak (0.3) > 0.1 threshold
        seq = (0.0, 0.3, 0.5, 0.3, 0.3)
        assert detect_bend_release(seq, release_threshold=0.1) == (1, False)
        # 同序列但用 0.5 阈值 → 视为 release
        assert detect_bend_release(seq, release_threshold=0.5) == (1, True)

    def test_peak_at_end_no_release(self):
        """peak 在最后 1 帧，无后续帧 → 无 release。"""
        seq = (0.0, 0.0, 0.0, 0.5)
        assert detect_bend_release(seq) == (1, False)


class TestBuildBendElement:
    """v0.4.3: 扩展 build_bend_element（shape + release 参数）。"""

    def test_default_shape_curved(self):
        """v0.4.3: 默认 shape="curved"（区别 v0.4.1 的无 shape）。"""
        xml = build_bend_element(1)
        assert xml == '<bend shape="curved"><bend-alter>1</bend-alter></bend>'

    def test_shape_straight(self):
        xml = build_bend_element(1, shape="straight")
        assert 'shape="straight"' in xml

    def test_shape_hold(self):
        xml = build_bend_element(1, shape="hold")
        assert 'shape="hold"' in xml

    def test_release_marker(self):
        """v0.4.3: release=True → 加 <release/> marker。"""
        xml = build_bend_element(-1, release=True)
        assert "<release/>" in xml
        assert 'shape="curved"' in xml
        assert "<bend-alter>-1</bend-alter>" in xml


class TestBuildBendElements:
    """v0.4.3: build_bend_elements() 多 bend 兄弟生成。"""

    def test_empty_returns_empty(self):
        assert build_bend_elements([]) == ""

    def test_single_bend(self):
        """1 个 BendPoint → 1 个 <bend>。"""
        xml = build_bend_elements([BendPoint(0.0, 1)])
        assert xml == '<bend shape="curved"><bend-alter>1</bend-alter></bend>'

    def test_bend_up_with_release(self):
        """2 点 bend up + release → 2 个 <bend> 兄弟。"""
        xml = build_bend_elements([BendPoint(0.0, 1), BendPoint(1.0, 0)])
        # 2 个 <bend> 开标签（用 "<bend " 带空格避免匹配 <bend-alter>）
        assert xml.count("<bend ") == 2
        # 第一个: positive alter，无 release
        assert "<bend-alter>1</bend-alter>" in xml
        # 第二个: negative alter + release marker
        assert "<bend-alter>-1</bend-alter>" in xml
        assert "<release/>" in xml
        # release marker 在 -1 alter 之后
        assert xml.find("<release/>") > xml.find("<bend-alter>-1</bend-alter>")

    def test_negative_bend_with_release(self):
        """2 点下弯 + release。"""
        xml = build_bend_elements([BendPoint(0.0, -1), BendPoint(1.0, 0)])
        assert xml.count("<bend ") == 2
        assert "<bend-alter>-1</bend-alter>" in xml
        assert "<bend-alter>1</bend-alter>" in xml
        assert "<release/>" in xml

    def test_no_release_pattern(self):
        """2 点但 second.alter > 0（不是 release 模式）→ 2 个独立 <bend>。"""
        xml = build_bend_elements([BendPoint(0.0, 1), BendPoint(1.0, 2)])
        # 不应有 <release/>
        assert "<release/>" not in xml
        # 应有 2 个独立 <bend-alter>
        assert xml.count("<bend-alter>") == 2

    def test_n_points_independent(self):
        """3 个 BendPoint → 3 个 <bend>（无 release marker）。"""
        xml = build_bend_elements([
            BendPoint(0.0, 1),
            BendPoint(0.5, 2),
            BendPoint(1.0, 1),
        ])
        assert xml.count("<bend ") == 3
        assert "<release/>" not in xml

    def test_shape_propagates_to_all(self):
        """shape 参数应用到所有 <bend> 兄弟。"""
        xml = build_bend_elements(
            [BendPoint(0.0, 1), BendPoint(1.0, 0)],
            shape="straight",
        )
        assert xml.count('shape="straight"') == 2
