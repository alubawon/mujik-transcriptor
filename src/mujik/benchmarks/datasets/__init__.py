"""Benchmark datasets 集合（v0.5.0）。

内置 synthetic baseline（5 genre × 3 file）用于 CI 验证 framework。
真实数据集（如 MusicNet CC-BY、MAPS）按 PR 增量引入。
"""
from mujik.benchmarks.datasets.synthetic import SyntheticBenchmarkDataset

__all__ = [
    "SyntheticBenchmarkDataset",
]
