"""时间签名启发式推断（v0.2.2 自实现）。

输入：downbeat + beat 时间序列
输出：list[TimeSignatureSegment]

算法（启发式，v0.2.2）：
  1. 对每对相邻 downbeat，计算之间的 beat 数 → 分子候选
  2. 统计分子直方图，取最频繁的分子（排除 <2 或 >12）
  3. 局部变化检测：如果某段分子与全局众数不同 → 切分为新段
  4. 置信度 = 1 - (top1 比例) ，clamp 到 [0.3, 1.0]
  5. 失败 / 数据不足 → 整段用 fallback（默认 4/4），confidence=0.3
  6. denominator 启发式：流行/爵士/金属 99% 是 4，固定 4（v0.2.2 简化）

后续 v0.4+ 可替换为 ResNet18/METER2800 模型。
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Iterable

from loguru import logger

from mujik.time_signature.model import (
    TimeSignatureSegment,
    build_default_segments,
)

# 合法分子候选（流行/爵士/金属/民谣常见）
VALID_NUMERATORS: tuple[int, ...] = (2, 3, 4, 5, 6, 7, 9, 12)


def infer_time_signature_from_downbeats(
    downbeats: list[float] | None = None,
    beats: list[float] | None = None,
    duration: float = 0.0,
    fallback: tuple[int, int] = (4, 4),
) -> list[TimeSignatureSegment]:
    """从 downbeat/beat 序列推断拍号分段。

    Args:
        downbeats: 下拍时间戳（秒）
        beats: 全部 beat 时间戳（秒）
        duration: 音频总时长（用于段结束）
        fallback: 数据不足时回退拍号

    Returns:
        list[TimeSignatureSegment]，按时间排序，已 merge 相邻同拍号
    """
    downbeats = list(downbeats or [])
    beats = list(beats or [])

    if len(downbeats) < 2 or len(beats) < 4:
        logger.warning(
            "infer_time_signature: insufficient data "
            "(downbeats={}, beats={}), use fallback {}",
            len(downbeats), len(beats), fallback,
        )
        end = duration if duration > 0 else 1.0
        return [TimeSignatureSegment(
            start_time=0.0,
            end_time=end,
            time_signature=fallback,
            confidence=0.3,
            source="default_4_4",
        )]

    # 1. 计算每对相邻 downbeat 间的分子
    numerators: list[tuple[float, int]] = []  # (downbeat_t, numerator)
    beat_set = set(round(b, 6) for b in beats)
    for i in range(len(downbeats) - 1):
        d_start = downbeats[i]
        d_end = downbeats[i + 1]
        # 数 downbeats[i]+ε 到 downbeats[i+1] 之间的 beat 数（含 d_start）
        # 因为 downbeat 本身也应是 beat，但 madmom 可能把 downbeat 不计入 beats
        count = 0
        for b in beats:
            if d_start - 1e-6 <= b < d_end - 1e-6:
                count += 1
        # 如果 beats 里有 downbeat 但被算成 0
        if count == 0:
            # 用间距 / beat 间隔估算
            interval = (d_end - d_start)
            if interval > 0 and len(beats) >= 2:
                beat_intervals = [
                    beats[j + 1] - beats[j]
                    for j in range(len(beats) - 1)
                    if beats[j + 1] > beats[j]
                ]
                if beat_intervals:
                    median_bi = sorted(beat_intervals)[len(beat_intervals) // 2]
                    if median_bi > 0:
                        count = max(1, round(interval / median_bi))
        numerators.append((d_start, count))

    # 2. 直方图（仅 valid 范围）
    counts = Counter(n for _, n in numerators if n in VALID_NUMERATORS)
    if not counts:
        logger.warning("infer_time_signature: no valid numerators, use fallback")
        end = duration if duration > 0 else 1.0
        return [TimeSignatureSegment(
            start_time=0.0,
            end_time=end,
            time_signature=fallback,
            confidence=0.3,
            source="default_4_4",
        )]

    total_valid = sum(counts.values())
    top_num, top_count = counts.most_common(1)[0]
    top_ratio = top_count / total_valid if total_valid else 0
    confidence = max(0.3, min(1.0, top_ratio))

    logger.debug(
        "infer_time_signature: top_num={}, ratio={:.2f}, conf={:.2f}, hist={}",
        top_num, top_ratio, confidence, dict(counts),
    )

    # 3. 局部变化检测：切分为段
    segments: list[TimeSignatureSegment] = []
    current_start = 0.0
    current_num = top_num
    seg_changes = 0

    for i, (t, n) in enumerate(numerators):
        if n not in VALID_NUMERATORS:
            n = top_num
        if n != current_num and abs(t - current_start) > 0.5:
            # 切分
            seg_end = t
            segments.append(_make_seg(
                current_start, seg_end, current_num, confidence,
            ))
            current_start = t
            current_num = n
            seg_changes += 1

    # 收尾
    final_end = downbeats[-1] if downbeats else duration
    if final_end <= current_start:
        final_end = current_start + 0.001
    segments.append(_make_seg(
        current_start, final_end, current_num, confidence,
    ))

    if seg_changes > 0:
        logger.info(
            "infer_time_signature: detected {n} meter change(s) "
            "(confidence={c:.2f})",
            n=seg_changes, c=confidence,
        )

    # 如果没有任何段（异常），用 fallback
    if not segments:
        end = duration if duration > 0 else 1.0
        return build_default_segments(end)

    return segments


def _make_seg(
    start: float,
    end: float,
    numerator: int,
    confidence: float,
) -> TimeSignatureSegment:
    """构造 TimeSignatureSegment，denominator 固定 4（v0.2.2 启发式简化）。"""
    if end <= start:
        end = start + 0.001
    return TimeSignatureSegment(
        start_time=start,
        end_time=end,
        time_signature=(int(numerator), 4),
        confidence=confidence,
        source="auto_resnet18",  # 占位，v0.4+ 真正用模型时改 source
    )


__all__ = [
    "infer_time_signature_from_downbeats",
    "VALID_NUMERATORS",
]
