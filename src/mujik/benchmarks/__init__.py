"""5-genre benchmark 框架（v0.5.0）。

设计目标：
- 统一评估 v0.4 全栈（分离/转录/节拍/和弦）在 5 个 genre 上的表现
- 离线运行，输出 markdown 报告
- 支持 synthetic baseline（CI smoke）+ 真实数据集（按 PR 引入）

架构：
- DatasetAdapter: list_files + load_audio + load_ground_truth
- MetricCalculator: compute (predicted, ground_truth) → dict
- BenchmarkRunner: 编排 pipeline + 收集 metrics
- BenchmarkReport: 渲染 markdown 表格

5 genre：
- pop（流行）
- jazz（爵士）
- metal（金属）
- rnb（R&B/放克）
- classical（古典/管弦）
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

# 5 个 genre 分类
BENCHMARK_GENRES: tuple[str, ...] = ("pop", "jazz", "metal", "rnb", "classical")


@dataclass
class BenchmarkSample:
    """单个 benchmark 样本。"""

    sample_id: str
    genre: str
    audio_path: str
    duration: float
    # ground truth（各 task 可选）
    gt_notes: list = field(default_factory=list)  # [(pitch, onset, offset), ...]
    gt_beats: list[float] = field(default_factory=list)
    gt_chords: list = field(default_factory=list)  # [(start, end, root, quality), ...]


@dataclass
class BenchmarkMetrics:
    """单个样本的 metrics 集合。"""

    sample_id: str
    genre: str
    metrics: dict[str, float] = field(default_factory=dict)
    # raw: 不平均的 metric（如 per-frame scores）
    raw: dict = field(default_factory=dict)


@dataclass
class BenchmarkReport:
    """完整 benchmark 报告。"""

    version: str
    dataset_name: str
    n_samples: int
    per_sample: list[BenchmarkMetrics]
    per_genre: dict[str, dict[str, float]] = field(default_factory=dict)
    overall: dict[str, float] = field(default_factory=dict)


class DatasetAdapter(Protocol):
    """数据集 adapter 接口。"""

    @property
    def name(self) -> str: ...

    def list_samples(self) -> list[BenchmarkSample]: ...


class MetricCalculator(Protocol):
    """指标计算器接口。"""

    @property
    def name(self) -> str: ...

    def compute(
        self,
        predicted: dict,
        ground_truth: dict,
    ) -> dict[str, float]: ...


__all__ = [
    "BENCHMARK_GENRES",
    "BenchmarkSample",
    "BenchmarkMetrics",
    "BenchmarkReport",
    "DatasetAdapter",
    "MetricCalculator",
]
