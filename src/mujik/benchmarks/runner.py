"""Benchmark runner（v0.5.0）。

BenchmarkRunner.run(dataset, pipeline_func) → BenchmarkReport

- dataset: 任何实现 DatasetAdapter 接口的对象
- pipeline_func: callable(audio_path) → dict，包含 predicted 字段
  （note_transcription / beat_tracking / chord_recognition）

设计：
- 顺序跑（不并行，避免 GPU 争用）
- 失败样本不中断整体（用 try/except + warning）
- 按 genre 聚合 + 总体分
"""
from __future__ import annotations

import logging
import traceback
from collections import defaultdict
from typing import Callable

from mujik.benchmarks import (
    BenchmarkMetrics,
    BenchmarkReport,
    DatasetAdapter,
)

logger = logging.getLogger(__name__)


# 3 个 metric 的 key（在 metrics dict 中）
_METRIC_KEYS: tuple[str, ...] = (
    "note_transcription", "beat_tracking", "chord_recognition",
)


# 简化的 metric 聚合（每个 metric 取主分）
_AGGREGATE_PRIMARY: dict[str, str] = {
    "note_transcription": "f1",
    "beat_tracking": "cmlt",
    "chord_recognition": "majmin",
}


def _aggregate_metrics(per_sample: list[BenchmarkMetrics]) -> dict:
    """聚合所有样本的 metrics。"""
    # 按 metric_name 聚合 primary score
    totals: dict[str, list[float]] = defaultdict(list)
    for sm in per_sample:
        for metric_name, score in sm.metrics.items():
            if metric_name in _AGGREGATE_PRIMARY:
                primary_key = _AGGREGATE_PRIMARY[metric_name]
                # score 是 dict, 取 primary
                if isinstance(score, dict) and primary_key in score:
                    totals[metric_name].append(score[primary_key])
                elif isinstance(score, (int, float)):
                    totals[metric_name].append(float(score))

    overall = {}
    for metric_name, scores in totals.items():
        if scores:
            overall[metric_name] = round(sum(scores) / len(scores), 4)
    return overall


def _aggregate_by_genre(per_sample: list[BenchmarkMetrics]) -> dict[str, dict[str, float]]:
    """按 genre 聚合 metrics。"""
    by_genre: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for sm in per_sample:
        for metric_name, score in sm.metrics.items():
            if metric_name in _AGGREGATE_PRIMARY:
                primary_key = _AGGREGATE_PRIMARY[metric_name]
                if isinstance(score, dict) and primary_key in score:
                    by_genre[sm.genre][metric_name].append(score[primary_key])
                elif isinstance(score, (int, float)):
                    by_genre[sm.genre][metric_name].append(float(score))

    out: dict[str, dict[str, float]] = {}
    for genre, metric_scores in by_genre.items():
        out[genre] = {
            metric: round(sum(scores) / len(scores), 4)
            for metric, scores in metric_scores.items()
        }
    return out


class BenchmarkRunner:
    """Benchmark 编排器。"""

    def __init__(
        self,
        version: str = "0.5.0",
        metric_calculators: dict[str, Callable] | None = None,
    ):
        self.version = version
        self.metric_calculators = metric_calculators or {}

    def run(
        self,
        dataset: DatasetAdapter,
        pipeline_func: Callable,
    ) -> BenchmarkReport:
        """对 dataset 的每个样本跑 pipeline_func + 计算 metrics。

        Args:
            dataset: 实现 DatasetAdapter 接口的对象
            pipeline_func: callable(audio_path: str) → dict
                返回 dict 应包含 'predicted' 键，nested dict with
                'note_transcription' / 'beat_tracking' / 'chord_recognition' 字段

        Returns:
            BenchmarkReport
        """
        samples = dataset.list_samples()
        per_sample: list[BenchmarkMetrics] = []

        for sample in samples:
            try:
                # 1. 跑 pipeline
                predicted = pipeline_func(sample.audio_path)

                # 2. 提取 predicted 子字段
                pred_nt = predicted.get("note_transcription", {})
                pred_bt = predicted.get("beat_tracking", {})
                pred_cr = predicted.get("chord_recognition", {})

                # 3. 构造 ground_truth dict
                gt = {
                    "notes": sample.gt_notes,
                    "beats": sample.gt_beats,
                    "chords": sample.gt_chords,
                }

                # 4. 计算 metrics
                metrics: dict[str, dict] = {}
                for metric_name, calc in self.metric_calculators.items():
                    pred_dict = {
                        "note_transcription": pred_nt,
                        "beat_tracking": pred_bt,
                        "chord_recognition": pred_cr,
                    }[metric_name]
                    gt_dict = {
                        "note_transcription": {"notes": gt["notes"]},
                        "beat_tracking": {"beats": gt["beats"]},
                        "chord_recognition": {"chords": gt["chords"]},
                    }[metric_name]
                    metrics[metric_name] = calc.compute(pred_dict, gt_dict)

                per_sample.append(BenchmarkMetrics(
                    sample_id=sample.sample_id,
                    genre=sample.genre,
                    metrics=metrics,
                ))
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "benchmark: sample %s failed: %s\n%s",
                    sample.sample_id, e, traceback.format_exc(),
                )
                # 失败样本记为 0 分
                per_sample.append(BenchmarkMetrics(
                    sample_id=sample.sample_id,
                    genre=sample.genre,
                    metrics={m: {"f1": 0.0, "cmlt": 0.0, "majmin": 0.0,
                                "n_pred": 0, "n_gt": 0} for m in _METRIC_KEYS},
                ))

        # 5. 聚合
        per_genre = _aggregate_by_genre(per_sample)
        overall = _aggregate_metrics(per_sample)

        return BenchmarkReport(
            version=self.version,
            dataset_name=dataset.name,
            n_samples=len(samples),
            per_sample=per_sample,
            per_genre=per_genre,
            overall=overall,
        )


__all__ = ["BenchmarkRunner"]
