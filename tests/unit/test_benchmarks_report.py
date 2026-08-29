"""Tests for benchmarks/report.py (v0.5.0)."""
from __future__ import annotations

import json

import pytest

from mujik.benchmarks import BenchmarkMetrics, BenchmarkReport
from mujik.benchmarks.report import render_json, render_markdown


def _make_report() -> BenchmarkReport:
    per_sample = [
        BenchmarkMetrics(
            sample_id="pop_01", genre="pop",
            metrics={"note_transcription": {"f1": 0.9, "precision": 0.9, "recall": 0.9}},
        ),
        BenchmarkMetrics(
            sample_id="jazz_01", genre="jazz",
            metrics={"note_transcription": {"f1": 0.8, "precision": 0.8, "recall": 0.8}},
        ),
    ]
    return BenchmarkReport(
        version="0.5.0",
        dataset_name="synthetic_5genre_baseline",
        n_samples=2,
        per_sample=per_sample,
        per_genre={"pop": {"note_transcription": 0.9}, "jazz": {"note_transcription": 0.8}},
        overall={"note_transcription": 0.85},
    )


class TestRenderMarkdown:
    def test_contains_version(self):
        md = render_markdown(_make_report())
        assert "v0.5.0" in md
        assert "synthetic_5genre_baseline" in md

    def test_contains_overall(self):
        md = render_markdown(_make_report())
        assert "Overall" in md
        assert "0.85" in md

    def test_contains_per_genre(self):
        md = render_markdown(_make_report())
        assert "Per-Genre" in md
        assert "pop" in md
        assert "jazz" in md

    def test_table_format(self):
        md = render_markdown(_make_report())
        # markdown 表格应含 | 分隔符
        assert md.count("|") >= 5


class TestRenderJSON:
    def test_json_structure(self):
        json_str = render_json(_make_report())
        data = json.loads(json_str)
        assert data["version"] == "0.5.0"
        assert data["dataset_name"] == "synthetic_5genre_baseline"
        assert data["n_samples"] == 2
        assert "overall" in data
        assert "per_genre" in data
        assert "per_sample" in data
        assert len(data["per_sample"]) == 2

    def test_json_per_sample(self):
        json_str = render_json(_make_report())
        data = json.loads(json_str)
        sample = data["per_sample"][0]
        assert sample["sample_id"] == "pop_01"
        assert sample["genre"] == "pop"
        assert "metrics" in sample
