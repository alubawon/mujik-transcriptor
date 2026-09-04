"""Tests for mujik.rhythm.tempo（v0.5.3 BPM reconciliation）。"""
from __future__ import annotations

import pytest

from mujik.rhythm.tempo import (
    SOURCE_BEATS_DERIVED,
    SOURCE_ESTIMATE,
    SOURCE_OCTAVE_CORRECTED,
    derive_bpm_from_beats,
    reconcile_bpm,
)


def _beats_from_bpm(bpm: float, n: int = 40, start: float = 0.0) -> list[float]:
    step = 60.0 / bpm
    return [round(start + i * step, 4) for i in range(n)]


class TestDeriveBpmFromBeats:
    def test_uniform_grid(self):
        # 0.48s IOI → 125 BPM（buhee 实测值）
        assert derive_bpm_from_beats(_beats_from_bpm(125.0)) == 125.0

    def test_missing_beats_median_robust(self):
        # 漏拍（IOI 加倍）不应污染 median——mean 会被拉低，median 稳定
        beats = _beats_from_bpm(125.0, n=40)
        beats[10:20] = [t for t in beats[10:20] if (beats.index(t)) % 2 == 0]
        derived = derive_bpm_from_beats(beats)
        assert derived is not None
        assert abs(derived - 125.0) / 125.0 < 0.08

    def test_too_few_beats_returns_none(self):
        assert derive_bpm_from_beats([0.0, 0.5, 1.0]) is None
        assert derive_bpm_from_beats([]) is None
        assert derive_bpm_from_beats(None) is None

    def test_degenerate_ioi_returns_none(self):
        assert derive_bpm_from_beats([1.0, 1.0, 1.0, 1.0, 1.0]) is None

    def test_out_of_sane_range_returns_none(self):
        # 10ms IOI → 6000 BPM，超出 sanity 范围
        beats = [round(i * 0.01, 4) for i in range(50)]
        assert derive_bpm_from_beats(beats) is None


class TestReconcileBpm:
    def test_estimate_agrees_with_beats(self):
        # 一致时仍返回拍点推导值（网格真实位置）
        bpm, source = reconcile_bpm(_beats_from_bpm(120.0), 120.0)
        assert bpm == 120.0
        assert source == SOURCE_ESTIMATE

    def test_half_tempo_estimate_octave_corrected(self):
        # buhee/moon/dança 实测形态：估计 62.5，拍点 125
        bpm, source = reconcile_bpm(_beats_from_bpm(125.0), 62.5)
        assert bpm == 125.0
        assert source == SOURCE_OCTAVE_CORRECTED

    def test_octave_corrected_returns_beat_value_not_estimate(self):
        # 不返回 62.5×2=123.7（估计的 bin 量化误差），而返回拍点 125.0
        bpm, source = reconcile_bpm(_beats_from_bpm(125.0), 61.855670103092784)
        assert bpm == 125.0
        assert source == SOURCE_OCTAVE_CORRECTED

    def test_double_tempo_estimate_octave_corrected(self):
        # 反向：估计倍速，拍点 62.5
        bpm, source = reconcile_bpm(_beats_from_bpm(62.5), 125.0)
        assert bpm == 62.5
        assert source == SOURCE_OCTAVE_CORRECTED

    def test_no_octave_relation_trusts_beats(self):
        # 估计 60，拍点 150 —— 60/120/240 都对不上 → 信拍点
        bpm, source = reconcile_bpm(_beats_from_bpm(150.0), 60.0)
        assert bpm == 150.0
        assert source == SOURCE_BEATS_DERIVED

    def test_no_beats_trusts_estimate(self):
        bpm, source = reconcile_bpm([], 97.3)
        assert bpm == 97.3
        assert source == SOURCE_ESTIMATE

    def test_invalid_estimate_falls_back_to_120(self):
        bpm, source = reconcile_bpm([], 0.0)
        assert bpm == 120.0
        assert source == SOURCE_ESTIMATE

    def test_tolerance_band(self):
        # ±8% 内视为一致（dança 拍点有 rubato 抖动）
        source = reconcile_bpm(_beats_from_bpm(120.0), 114.0)[1]
        assert source == SOURCE_ESTIMATE
        source = reconcile_bpm(_beats_from_bpm(120.0), 100.0)[1]
        assert source != SOURCE_ESTIMATE


class TestDemoRegressions:
    """demo 三曲实测数字的回归测试（2026-09 baseline）。"""

    @pytest.mark.parametrize(
        ("estimated", "expected_bpm"),
        [
            (62.5, 125.0),     # buhee
            (61.855670103092784, 125.0),  # moon
            (53.097345132743364, 107.1),  # dança
        ],
    )
    def test_demo_half_tempo_fixed(self, estimated, expected_bpm):
        beats = _beats_from_bpm(expected_bpm, n=100)
        bpm, source = reconcile_bpm(beats, estimated)
        assert source == SOURCE_OCTAVE_CORRECTED
        assert abs(bpm - expected_bpm) < 0.2
