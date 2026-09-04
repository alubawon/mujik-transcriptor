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
from pathlib import Path
from typing import Callable

from mujik.benchmarks import (
    BenchmarkMetrics,
    BenchmarkReport,
    DatasetAdapter,
)
from mujik.benchmarks.datasets.local import LocalBenchmarkDataset
from mujik.benchmarks.datasets.synthetic import SyntheticBenchmarkDataset
from mujik.benchmarks.pipeline_adapter import (
    PipelineBenchmarkAdapter,
    _build_default_metric_calculators,
)
from mujik.benchmarks.report import render_json, render_markdown

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


def main(argv: list[str] | None = None) -> int:
    """CLI 入口：python -m mujik.benchmarks.runner [options]

    - --dataset synthetic: 内置 5 genre × 3 file synthetic baseline（CI smoke）
    - --dataset local: 自家曲库（需 --data-dir + manifest.json，v0.5.2）
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m mujik.benchmarks.runner",
        description="mujik 5-genre benchmark（synthetic baseline / 本地真实曲库）",
    )
    parser.add_argument(
        "--dataset", choices=["synthetic", "local"], default="synthetic",
        help="synthetic=内置 baseline；local=用户曲库（需 --data-dir）",
    )
    parser.add_argument(
        "--data-dir", default=None,
        help="local 数据集目录（含 manifest.json + 音频）",
    )
    parser.add_argument(
        "--preset", choices=["pop", "jazz", "metal"], default="pop",
        help="benchmark 跑的管线 preset（默认 pop）",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="只跑前 N 个样本（调试/快速验证用）",
    )
    parser.add_argument(
        "--work-dir", default="bench_work",
        help="管线中间产物根目录（默认 ./bench_work）",
    )
    parser.add_argument("--no-chords", action="store_true",
                        help="关闭 chord 检测（默认强制开，供 chord 指标）")
    parser.add_argument("--output", "-o", default="bench.md",
                        help="markdown 报告输出路径（默认 bench.md）")
    parser.add_argument("--json", default="bench.json",
                        help="JSON 报告输出路径（默认 bench.json，空串跳过）")
    args = parser.parse_args(argv)

    if args.dataset == "local":
        if not args.data_dir:
            parser.error("--dataset local 需要 --data-dir")
        dataset = LocalBenchmarkDataset(args.data_dir)
    else:
        dataset = SyntheticBenchmarkDataset()

    if args.limit is not None and args.limit > 0:
        full = dataset.list_samples()
        limited = full[: args.limit]
        if not limited:
            parser.error("--limit N 必须 > 0")
        dataset = _SubsetDataset(dataset, limited)

    pipeline_func = PipelineBenchmarkAdapter(
        preset=args.preset,
        work_dir=args.work_dir,
        enable_chords=not args.no_chords,
    )

    runner = BenchmarkRunner(
        version="0.5.2",
        metric_calculators=_build_default_metric_calculators(),
    )
    report = runner.run(dataset, pipeline_func)

    markdown = render_markdown(report)
    print(markdown)
    Path(args.output).write_text(markdown, encoding="utf-8")
    if args.json:
        Path(args.json).write_text(render_json(report), encoding="utf-8")
    print(f"\nreport → {args.output}" + (f" + {args.json}" if args.json else ""))

    # v0.5.2: 全部样本失败 → 非零退出码（此前全 0 分报告也 exit 0，
    # "管线全崩"和"管线跑通但分数低"无法从自动化流程区分）
    n_ok = sum(1 for sm in report.per_sample
               if sm.metrics.get("note_transcription", {}).get("n_pred", 0) > 0)
    if report.n_samples > 0 and n_ok == 0:
        print("\n❌ all samples failed to produce predictions")
        return 1
    return 0


class _SubsetDataset:
    """--limit 的轻量包装（截取前 N 个样本，不复制数据集实现）。"""

    def __init__(self, base, samples: list):
        self._base = base
        self._samples = samples

    @property
    def name(self) -> str:
        return f"{self._base.name}[limit:{len(self._samples)}]"

    def list_samples(self):
        return self._samples


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["BenchmarkRunner", "main"]
