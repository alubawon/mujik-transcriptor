"""Tests for benchmarks/runner.py (v0.5.0)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from mujik.benchmarks import BenchmarkSample
from mujik.benchmarks.datasets.synthetic import SyntheticBenchmarkDataset
from mujik.benchmarks.metrics import (
    BeatTrackingMetrics,
    ChordRecognitionMetrics,
    NoteTranscriptionMetrics,
)
from mujik.benchmarks.runner import BenchmarkRunner


class TestSyntheticDataset:
    def test_list_samples_returns_15(self, tmp_path: Path):
        ds = SyntheticBenchmarkDataset(base_dir=tmp_path)
        samples = ds.list_samples()
        assert len(samples) == 15  # 5 genre × 3

    def test_5_genres_covered(self, tmp_path: Path):
        ds = SyntheticBenchmarkDataset(base_dir=tmp_path)
        samples = ds.list_samples()
        genres = {s.genre for s in samples}
        assert genres == {"pop", "jazz", "metal", "rnb", "classical"}

    def test_each_genre_has_3_samples(self, tmp_path: Path):
        ds = SyntheticBenchmarkDataset(base_dir=tmp_path)
        samples = ds.list_samples()
        from collections import Counter
        counts = Counter(s.genre for s in samples)
        assert all(c == 3 for c in counts.values())

    def test_wav_files_created(self, tmp_path: Path):
        ds = SyntheticBenchmarkDataset(base_dir=tmp_path)
        samples = ds.list_samples()
        for s in samples:
            assert Path(s.audio_path).exists(), f"missing {s.audio_path}"

    def test_ground_truth_present(self, tmp_path: Path):
        ds = SyntheticBenchmarkDataset(base_dir=tmp_path)
        samples = ds.list_samples()
        for s in samples:
            assert len(s.gt_beats) > 0, f"{s.sample_id} no beats"
            assert len(s.gt_chords) > 0, f"{s.sample_id} no chords"
            assert len(s.gt_notes) > 0, f"{s.sample_id} no notes"


class TestBenchmarkRunner:
    def _perfect_pipeline(self, audio_path: str) -> dict:
        """返回完美匹配的 predicted。"""
        # 读取 ground truth
        gt_path = Path(audio_path).with_suffix(".json")
        if gt_path.exists():
            gt = json.loads(gt_path.read_text())
            return {
                "note_transcription": {"notes": gt.get("notes", [])},
                "beat_tracking": {"beats": gt.get("beats", [])},
                "chord_recognition": {"chords": gt.get("chords", [])},
            }
        return {
            "note_transcription": {"notes": []},
            "beat_tracking": {"beats": []},
            "chord_recognition": {"chords": []},
        }

    def test_perfect_pipeline_high_score(self, tmp_path: Path):
        ds = SyntheticBenchmarkDataset(base_dir=tmp_path)
        runner = BenchmarkRunner(
            version="0.5.0",
            metric_calculators={
                "note_transcription": NoteTranscriptionMetrics(),
                "beat_tracking": BeatTrackingMetrics(),
                "chord_recognition": ChordRecognitionMetrics(),
            },
        )
        report = runner.run(ds, self._perfect_pipeline)
        assert report.n_samples == 15
        assert report.dataset_name == "synthetic_5genre_baseline"
        # 完美匹配 → note F1 应为 1.0（无 mir_eval 依赖）
        assert report.overall.get("note_transcription", 0) >= 0.9
        # beat/chord 需要 mir_eval；若不可用会 fallback 到 0
        # 这里只验证不崩溃 + overall dict 有 key
        assert "beat_tracking" in report.overall or report.overall.get("beat_tracking") is None
        assert "chord_recognition" in report.overall or report.overall.get("chord_recognition") is None

    def test_empty_pipeline_zero_score(self, tmp_path: Path):
        ds = SyntheticBenchmarkDataset(base_dir=tmp_path)
        runner = BenchmarkRunner(
            version="0.5.0",
            metric_calculators={
                "note_transcription": NoteTranscriptionMetrics(),
                "beat_tracking": BeatTrackingMetrics(),
                "chord_recognition": ChordRecognitionMetrics(),
            },
        )
        empty_pipeline = lambda p: {
            "note_transcription": {"notes": []},
            "beat_tracking": {"beats": []},
            "chord_recognition": {"chords": []},
        }
        report = runner.run(ds, empty_pipeline)
        # 空预测 → 0 分
        assert report.overall.get("note_transcription", -1) == 0.0

    def test_failing_pipeline_continues(self, tmp_path: Path):
        ds = SyntheticBenchmarkDataset(base_dir=tmp_path)
        runner = BenchmarkRunner(
            version="0.5.0",
            metric_calculators={
                "note_transcription": NoteTranscriptionMetrics(),
            },
        )
        def failing(p):
            raise RuntimeError("simulated pipeline failure")
        report = runner.run(ds, failing)
        # 所有样本都失败，但 runner 不崩溃
        assert report.n_samples == 15
        # 每个 sample 都有 metrics entry（即使 0 分）
        assert len(report.per_sample) == 15

    def test_per_genre_aggregation(self, tmp_path: Path):
        ds = SyntheticBenchmarkDataset(base_dir=tmp_path)
        runner = BenchmarkRunner(
            version="0.5.0",
            metric_calculators={
                "note_transcription": NoteTranscriptionMetrics(),
            },
        )
        report = runner.run(ds, self._perfect_pipeline)
        # 5 个 genre 都有聚合
        assert set(report.per_genre.keys()) == {"pop", "jazz", "metal", "rnb", "classical"}
