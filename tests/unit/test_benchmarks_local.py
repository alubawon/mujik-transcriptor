"""Tests for benchmarks/datasets/local.py + runner CLI（v0.5.2 真实数据 benchmark）。"""

from __future__ import annotations

import json
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

from mujik.benchmarks.datasets.local import (
    LocalBenchmarkDataset,
    LocalDatasetError,
)
from mujik.benchmarks.pipeline_adapter import PipelineBenchmarkAdapter
from mujik.benchmarks.runner import BenchmarkRunner, main

REPO_ROOT = Path(__file__).resolve().parents[2]


def _make_wav(path: Path, duration: float = 1.0, sr: int = 8000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.linspace(0, duration, int(duration * sr), endpoint=False)
    audio = (0.3 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(np.int16(audio * 32767).tobytes())


def _write_manifest(data_dir: Path, entries: list[dict]) -> None:
    (data_dir / "manifest.json").write_text(
        json.dumps(entries),
        encoding="utf-8",
    )


class TestLocalDatasetHappyPath:
    def test_load_full_entry(self, tmp_path: Path):
        audio = tmp_path / "audio" / "song1.wav"
        _make_wav(audio)
        _write_manifest(
            tmp_path,
            [
                {
                    "sample_id": "song1",
                    "genre": "jazz",
                    "audio": "audio/song1.wav",
                    "duration": 1.0,
                    "notes": [[60, 0.0, 0.5], [62, 0.5, 1.0]],
                    "beats": [0.0, 0.5, 1.0],
                    "chords": [[0.0, 1.0, "C", "maj7"]],
                }
            ],
        )
        samples = LocalBenchmarkDataset(tmp_path).list_samples()
        assert len(samples) == 1
        s = samples[0]
        assert s.sample_id == "song1"
        assert s.genre == "jazz"
        assert s.audio_path == str(audio)
        assert s.gt_notes == [(60, 0.0, 0.5), (62, 0.5, 1.0)]
        assert s.gt_beats == [0.0, 0.5, 1.0]
        assert s.gt_chords == [(0.0, 1.0, "C", "maj7")]

    def test_wrapped_samples_format(self, tmp_path: Path):
        """{"samples": [...]} 顶层包装也接受。"""
        audio = tmp_path / "a.wav"
        _make_wav(audio)
        (tmp_path / "manifest.json").write_text(
            json.dumps(
                {
                    "samples": [{"sample_id": "a", "genre": "pop", "audio": "a.wav"}],
                }
            )
        )
        samples = LocalBenchmarkDataset(tmp_path).list_samples()
        assert len(samples) == 1
        assert samples[0].gt_notes == []

    def test_dataset_name(self, tmp_path: Path):
        audio = tmp_path / "a.wav"
        _make_wav(audio)
        _write_manifest(
            tmp_path,
            [
                {"sample_id": "a", "genre": "pop", "audio": "a.wav"},
            ],
        )
        assert LocalBenchmarkDataset(tmp_path).name == f"local:{tmp_path.name}"

    def test_caching(self, tmp_path: Path):
        audio = tmp_path / "a.wav"
        _make_wav(audio)
        _write_manifest(
            tmp_path,
            [
                {"sample_id": "a", "genre": "pop", "audio": "a.wav"},
            ],
        )
        ds = LocalBenchmarkDataset(tmp_path)
        assert ds.list_samples() is ds.list_samples()


class TestLocalDatasetFailLoud:
    def test_missing_data_dir(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="data_dir not found"):
            LocalBenchmarkDataset(tmp_path / "nope")

    def test_missing_manifest(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="manifest not found"):
            LocalBenchmarkDataset(tmp_path)

    def test_empty_manifest(self, tmp_path: Path):
        (tmp_path / "manifest.json").write_text("[]")
        with pytest.raises(LocalDatasetError, match="非空 list"):
            LocalBenchmarkDataset(tmp_path).list_samples()

    def test_missing_required_field(self, tmp_path: Path):
        (tmp_path / "manifest.json").write_text(
            json.dumps([{"sample_id": "a", "audio": "a.wav"}])  # 缺 genre
        )
        (tmp_path / "a.wav").write_bytes(b"RIFF")
        with pytest.raises(LocalDatasetError, match="genre"):
            LocalBenchmarkDataset(tmp_path).list_samples()

    def test_missing_audio_file(self, tmp_path: Path):
        _write_manifest(
            tmp_path,
            [
                {"sample_id": "a", "genre": "pop", "audio": "audio/missing.wav"},
            ],
        )
        with pytest.raises(FileNotFoundError, match="missing.wav"):
            LocalBenchmarkDataset(tmp_path).list_samples()

    def test_bad_notes_format(self, tmp_path: Path):
        (tmp_path / "a.wav").write_bytes(b"RIFF")
        _write_manifest(
            tmp_path,
            [
                {
                    "sample_id": "a",
                    "genre": "pop",
                    "audio": "a.wav",
                    "notes": [[60, 0.0]],  # 少 offset
                }
            ],
        )
        with pytest.raises(LocalDatasetError, match=r"notes\[0\]"):
            LocalBenchmarkDataset(tmp_path).list_samples()

    def test_bad_notes_pitch_range(self, tmp_path: Path):
        (tmp_path / "a.wav").write_bytes(b"RIFF")
        _write_manifest(
            tmp_path,
            [
                {
                    "sample_id": "a",
                    "genre": "pop",
                    "audio": "a.wav",
                    "notes": [[200, 0.0, 1.0]],
                }
            ],
        )
        with pytest.raises(LocalDatasetError, match="pitch"):
            LocalBenchmarkDataset(tmp_path).list_samples()

    def test_bad_beats_format(self, tmp_path: Path):
        (tmp_path / "a.wav").write_bytes(b"RIFF")
        _write_manifest(
            tmp_path,
            [
                {
                    "sample_id": "a",
                    "genre": "pop",
                    "audio": "a.wav",
                    "beats": ["x"],
                }
            ],
        )
        with pytest.raises(LocalDatasetError, match=r"beats\[0\]"):
            LocalBenchmarkDataset(tmp_path).list_samples()

    def test_bad_chords_format(self, tmp_path: Path):
        (tmp_path / "a.wav").write_bytes(b"RIFF")
        _write_manifest(
            tmp_path,
            [
                {
                    "sample_id": "a",
                    "genre": "pop",
                    "audio": "a.wav",
                    "chords": [[0.0, 1.0, "C"]],  # 缺 quality
                }
            ],
        )
        with pytest.raises(LocalDatasetError, match=r"chords\[0\]"):
            LocalBenchmarkDataset(tmp_path).list_samples()

    def test_chord_root_empty(self, tmp_path: Path):
        (tmp_path / "a.wav").write_bytes(b"RIFF")
        _write_manifest(
            tmp_path,
            [
                {
                    "sample_id": "a",
                    "genre": "pop",
                    "audio": "a.wav",
                    "chords": [[0.0, 1.0, "", "maj"]],
                }
            ],
        )
        with pytest.raises(LocalDatasetError, match="root"):
            LocalBenchmarkDataset(tmp_path).list_samples()

    def test_entry_not_object(self, tmp_path: Path):
        (tmp_path / "manifest.json").write_text(json.dumps(["not-a-dict"]))
        with pytest.raises(LocalDatasetError, match="object"):
            LocalBenchmarkDataset(tmp_path).list_samples()


class TestRunnerMain:
    """python -m mujik.benchmarks.runner CLI（stub pipeline，不跑真管线）。"""

    def _stub_adapter(self, audio_path):
        return {
            "note_transcription": {"notes": [(60, 0.0, 0.5)]},
            "beat_tracking": {"beats": [0.0, 0.5]},
            "chord_recognition": {"chords": [(0.0, 1.0, "C", "maj")]},
        }

    def _stub_class(self, pred=None):
        """替代 PipelineBenchmarkAdapter 的 stub 类（接受 init kwargs）。"""
        pred = pred or {
            "note_transcription": {"notes": [(60, 0.0, 0.5)]},
            "beat_tracking": {"beats": [0.0, 0.5]},
            "chord_recognition": {"chords": [(0.0, 1.0, "C", "maj")]},
        }

        class _Stub:
            def __init__(self, **kwargs):
                pass

            def __call__(self, audio_path):
                return pred

        return _Stub

    def test_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "mujik.benchmarks.runner", "--help"],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=REPO_ROOT,
        )
        assert result.returncode == 0
        assert "--dataset" in result.stdout

    def test_local_requires_data_dir(self):
        result = subprocess.run(
            [sys.executable, "-m", "mujik.benchmarks.runner", "--dataset", "local"],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=REPO_ROOT,
        )
        assert result.returncode != 0
        assert "--data-dir" in (result.stderr + result.stdout)

    def test_synthetic_run_with_stub_pipeline(self, tmp_path, monkeypatch):
        """--limit 1 + stub pipeline → 报告落盘。"""
        monkeypatch.setattr(
            "mujik.benchmarks.runner.PipelineBenchmarkAdapter",
            self._stub_class(),
        )
        out_md = tmp_path / "bench.md"
        out_json = tmp_path / "bench.json"
        rc = main(
            [
                "--dataset",
                "synthetic",
                "--limit",
                "1",
                "--output",
                str(out_md),
                "--json",
                str(out_json),
            ]
        )
        assert rc == 0
        md = out_md.read_text()
        assert "Benchmark Report" in md
        assert "synthetic_5genre_baseline[limit:1]" in md
        data = json.loads(out_json.read_text())
        assert data["n_samples"] == 1

    def test_synthetic_run_reports_scores(self, tmp_path, monkeypatch):
        """stub predicted 与 GT 完全一致时 note f1 应为 1。"""
        import json as _json

        def echo_gt(audio_path):
            gt = _json.loads(Path(audio_path).with_suffix(".json").read_text())
            return {
                "note_transcription": {"notes": [tuple(n) for n in gt["notes"]]},
                "beat_tracking": {"beats": gt["beats"]},
                "chord_recognition": {"chords": [tuple(c) for c in gt["chords"]]},
            }

        stub = type(
            "_Echo",
            (),
            {
                "__init__": lambda self, **kw: None,
                "__call__": staticmethod(echo_gt),
            },
        )
        monkeypatch.setattr(
            "mujik.benchmarks.runner.PipelineBenchmarkAdapter",
            stub,
        )
        out_md = tmp_path / "bench.md"
        rc = main(
            [
                "--dataset",
                "synthetic",
                "--limit",
                "1",
                "--preset",
                "pop",
                "--work-dir",
                str(tmp_path / "w"),
                "--output",
                str(out_md),
                "--json",
                "",
            ]
        )
        assert rc == 0
        # note F1 = 1.0（4/4 个 GT note 精确命中）
        assert "| 1.0 |" in out_md.read_text()

    def test_runner_runs_with_local_dataset(self, tmp_path, monkeypatch):
        """local dataset 端到端：manifest → runner → report。"""
        audio = tmp_path / "audio" / "s.wav"
        _make_wav(audio)
        _write_manifest(
            tmp_path,
            [
                {
                    "sample_id": "s",
                    "genre": "rock",
                    "audio": "audio/s.wav",
                    "notes": [[60, 0.0, 0.5]],
                    "beats": [0.0, 0.5],
                },
            ],
        )
        monkeypatch.setattr(
            "mujik.benchmarks.runner.PipelineBenchmarkAdapter",
            self._stub_class(),
        )
        out_md = tmp_path / "bench.md"
        rc = main(
            [
                "--dataset",
                "local",
                "--data-dir",
                str(tmp_path),
                "--output",
                str(out_md),
                "--json",
                "",
            ]
        )
        assert rc == 0
        assert f"local:{tmp_path.name}" in out_md.read_text()


class TestPipelineAdapterUnit:
    """PipelineBenchmarkAdapter 配置/提取逻辑（不跑真管线）。"""

    def test_invalid_preset_rejected(self):
        with pytest.raises(ValueError, match="preset"):
            PipelineBenchmarkAdapter(preset="nope")

    def test_contract_matches_benchmark_runner(self):
        """pipeline_func 契约：callable(audio_path) → dict（带 3 个 task key）。"""
        adapter = PipelineBenchmarkAdapter(preset="pop", work_dir="x")
        assert callable(adapter)
        # 契约 key 与 runner 的 _METRIC_KEYS 对齐
        from mujik.benchmarks.runner import _METRIC_KEYS

        assert set(_METRIC_KEYS) == {
            "note_transcription",
            "beat_tracking",
            "chord_recognition",
        }

    def test_real_runner_end_to_end_with_fake_pipeline(self, tmp_path: Path):
        """BenchmarkRunner + LocalBenchmarkDataset 全链路（pipeline 恒失败 → 0 分不崩）。"""
        audio = tmp_path / "a.wav"
        _make_wav(audio)
        _write_manifest(
            tmp_path,
            [
                {"sample_id": "a", "genre": "metal", "audio": "a.wav", "notes": [[60, 0.0, 0.5]]},
            ],
        )
        ds = LocalBenchmarkDataset(tmp_path)

        def failing_pipeline(audio_path):
            raise RuntimeError("boom")

        report = BenchmarkRunner(version="test").run(ds, failing_pipeline)
        assert report.n_samples == 1
        # 失败样本记 0 分而非抛异常
        assert report.overall["note_transcription"] == 0.0
