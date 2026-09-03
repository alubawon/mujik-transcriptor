"""Benchmark datasets 集合（v0.5.0）。

内置 synthetic baseline（5 genre × 3 file）用于 CI 验证 framework。
v0.5.2：LocalBenchmarkDataset —— manifest 驱动的本地真实曲库
（仓库不携带数据，版权干净）。标准数据集（MUSDB18/MAPS 等）按 PR 增量引入。
"""
from mujik.benchmarks.datasets.local import LocalBenchmarkDataset
from mujik.benchmarks.datasets.synthetic import SyntheticBenchmarkDataset

__all__ = [
    "LocalBenchmarkDataset",
    "SyntheticBenchmarkDataset",
]
