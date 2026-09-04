"""BPM reconciliation（v0.5.3）：拍点数组推导 + 全局 tempo 估计校正。

背景（2026-09 demo 三曲实测）：madmom TempoEstimationProcessor 的全局
BPM 估计在三首 demo 曲上全部报出 ≈半速值（62.5 vs 拍点 125.0）——
tempo 估计的半速/倍速混淆是经典坑；而 DBN 拍点数组本身自洽
（median IOI 方差 <2%）。此前 pipeline 直接信 tempo 估计、弃用拍点数组，
导致 score/quantize/chord 全链路 BPM 错一位。

本模块提供纯函数：
- derive_bpm_from_beats: median IOI → BPM
- reconcile_bpm: 拍点推导值与全局估计做一致性校验 + 倍频校正
  （×2 / ÷2 取与拍点接近者；仍不匹配 → 信拍点数组）

约定：拍点数组是节拍位置的 ground truth，全局估计只是参考。
"""
from __future__ import annotations

import statistics

from loguru import logger

# 拍点推导 BPM 的合理范围（DBN 跟踪的是 beat 而非 tatum，正常不会出界）
_BPM_SANITY_MIN = 20.0
_BPM_SANITY_MAX = 300.0

# 估计值与拍点推导值的相对偏差容差
DEFAULT_TOLERANCE = 0.08

# 倍频校正尝试的因子（半速/倍速）
_OCTAVE_FACTORS: tuple[float, ...] = (2.0, 0.5)

# reconcile_bpm 的 source 取值
SOURCE_ESTIMATE = "estimate"
SOURCE_OCTAVE_CORRECTED = "octave-corrected"
SOURCE_BEATS_DERIVED = "beats-derived"


def derive_bpm_from_beats(beats: list[float] | None) -> float | None:
    """从 beat 时间戳序列推导 BPM（median IOI 的倒数）。

    用 median 而非 mean：DBN 偶发漏拍会拉长个别 IOI，median 抗 outlier
    （moon 实测：mean 0.449s vs median 0.480s，漏拍污染 mean 但 median 稳定）。

    Returns:
        BPM（round 2 位）；beat < 4 个或 IOI 退化（≤0）时 None。
    """
    if not beats or len(beats) < 4:
        return None
    seq = sorted(float(b) for b in beats if b == b)  # 滤 NaN
    iois = [
        seq[i + 1] - seq[i]
        for i in range(len(seq) - 1)
        if seq[i + 1] - seq[i] > 1e-3
    ]
    if not iois:
        return None
    bpm = 60.0 / statistics.median(iois)
    if not (_BPM_SANITY_MIN <= bpm <= _BPM_SANITY_MAX):
        logger.warning(
            "derive_bpm_from_beats: derived bpm={:.1f} outside sane range "
            "[{}, {}] — beat array suspect",
            bpm, _BPM_SANITY_MIN, _BPM_SANITY_MAX,
        )
        return None
    return round(bpm, 2)


def reconcile_bpm(
    beats: list[float] | None,
    estimated_bpm: float,
    tolerance: float = DEFAULT_TOLERANCE,
) -> tuple[float, str]:
    """用拍点数组校正全局 tempo 估计。

    策略（拍点数组是位置 ground truth，估计只是直方图峰值）：
      1. 拍点不足 → 信估计（无更优信息源）
      2. 拍点充足时**永远返回拍点推导值**；全局估计只用来判定
         provenance：
         - 与估计一致（±tolerance）→ "estimate"（两源互证）
         - 与估计 ×2 / ÷2 一致 → "octave-corrected"（估计半速/倍速混淆）
         - 都不一致 → "beats-derived"（估计离谱，直接弃用）
         不返回校正后的估计值：madmom tempo 直方图有 bin 量化误差
         （实测 62.5×2=123.7 vs 拍点 125.0），均匀网格 BPM 偏 1% 会在
         100s 内漂移 ~1.5s，拍点推导值才是网格真实位置。

    Returns:
        (bpm, source)；source ∈ {"estimate", "octave-corrected", "beats-derived"}，
        由调用方写入 beats.json 作 provenance 并决定 warning 级别。
    """
    bpm_beats = derive_bpm_from_beats(beats)
    if bpm_beats is None or estimated_bpm <= 0:
        return (float(estimated_bpm) if estimated_bpm > 0 else 120.0,
                SOURCE_ESTIMATE)

    def _close(a: float, b: float) -> bool:
        return b > 0 and abs(a - b) / b <= tolerance

    if _close(estimated_bpm, bpm_beats):
        return (bpm_beats, SOURCE_ESTIMATE)

    for factor in _OCTAVE_FACTORS:
        if _close(estimated_bpm * factor, bpm_beats):
            logger.warning(
                "reconcile_bpm: tempo estimate {:.1f} BPM is {}x off the "
                "beat grid ({:.1f} BPM) — using beat-derived BPM",
                estimated_bpm, int(round(1 / factor)), bpm_beats,
            )
            return (bpm_beats, SOURCE_OCTAVE_CORRECTED)

    logger.warning(
        "reconcile_bpm: tempo estimate {:.1f} BPM inconsistent with beat grid "
        "({:.1f} BPM, no octave relation) — trusting beat grid",
        estimated_bpm, bpm_beats,
    )
    return (bpm_beats, SOURCE_BEATS_DERIVED)


__all__ = [
    "derive_bpm_from_beats",
    "reconcile_bpm",
    "SOURCE_ESTIMATE",
    "SOURCE_OCTAVE_CORRECTED",
    "SOURCE_BEATS_DERIVED",
    "DEFAULT_TOLERANCE",
]
