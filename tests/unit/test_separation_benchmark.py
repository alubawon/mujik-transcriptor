"""Tests for benchmarks/separation.py（v0.5.2 MUSDB18 + museval）。

musdb.DB 与 demucs 分离全程 mock；_evaluate_stems 用真实 museval 数学
（无 museval 环境 skip）。
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from mujik.benchmarks.separation import (
    SEPARATION_STEMS,
    SeparationBenchmarkError,
    _evaluate_stems,
    main,
    render_markdown,
    run_separation_benchmark,
)


def _skip_without_deps():
    """museval→musdb→stempeg import 链在无 ffmpeg 时抛 RuntimeError → skip。

    Returns:
        museval 模块（供 _evaluate_stems 使用）。
    """
    try:
        import museval

        pytest.importorskip("museval")
        pytest.importorskip("musdb")
    except RuntimeError as e:  # stempeg ffmpeg check（museval import 链即触发）
        pytest.skip(f"deps unavailable: {e}")
    return museval


SR = 8000


def _stereo(freq: float, seconds: float = 1.0) -> np.ndarray:
    t = np.linspace(0, seconds, int(SR * seconds), endpoint=False)
    sig = 0.3 * np.sin(2 * np.pi * freq * t)
    return np.stack([sig, sig], axis=-1)


class TestEvaluateStems:
    def test_identical_signals_high_sdr(self):
        """ref == est → SDR 极高（∞ 风险，断言 > 10dB）。"""
        museval = _skip_without_deps()
        ref = {s: _stereo(220 + 30 * i) for i, s in enumerate(SEPARATION_STEMS)}
        scores = _evaluate_stems(ref, {s: v.copy() for s, v in ref.items()}, museval)
        for stem in SEPARATION_STEMS:
            assert scores[stem]["SDR"] > 10, scores

    def test_noise_estimates_low_sdr(self):
        museval = _skip_without_deps()
        rng = np.random.default_rng(42)
        ref = {s: _stereo(220 + 30 * i) for i, s in enumerate(SEPARATION_STEMS)}
        est = {s: rng.standard_normal(ref[s].shape) * 0.1 for s in SEPARATION_STEMS}
        scores = _evaluate_stems(ref, est, museval)
        for stem in SEPARATION_STEMS:
            assert scores[stem]["SDR"] < 10

    def test_no_common_stems_raises(self):
        museval = _skip_without_deps()
        with pytest.raises(SeparationBenchmarkError, match="no common stems"):
            _evaluate_stems({}, {}, museval)


class _FakeTarget:
    def __init__(self, audio: np.ndarray):
        self.audio = audio


class _FakeTrack:
    def __init__(self, idx: int):
        self.artist = f"Artist{idx}"
        self.title = f"Track{idx}"
        self.rate = SR
        self.audio = _stereo(110 + idx, 2.0)
        self.targets = {
            s: _FakeTarget(_stereo(220 + 30 * i, 2.0)) for i, s in enumerate(SEPARATION_STEMS)
        }


def _fake_separate(mix_path, out_dir, config=None):
    """伪造 separate_audio：给每个 stem 写 wav + 返回 Stems-like。"""
    import soundfile as sf

    from mujik.separate.model import Stem, Stems

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stems = Stems()
    for i, stem_name in enumerate(SEPARATION_STEMS):
        p = out_dir / f"{stem_name}.wav"
        sf.write(p, _stereo(220 + 30 * i, 2.0), SR)
        stems.add(
            Stem(
                name=stem_name,
                audio_path=p,
                sample_rate=SR,
                duration=2.0,
                source_model="demucs/test",
            )
        )
    return stems


class TestRunSeparationBenchmark:
    def test_end_to_end_mocked(self, tmp_path: Path):
        _skip_without_deps()
        pytest.importorskip("soundfile")
        fake_db = [_FakeTrack(0), _FakeTrack(1)]

        with (
            patch("mujik.benchmarks.separation.load_musdb", return_value=fake_db),
            patch("mujik.separate.router.separate_audio", side_effect=_fake_separate),
        ):
            report = run_separation_benchmark(
                musdb_root=tmp_path,
                limit=2,
                work_dir=tmp_path / "w",
            )
        assert report["n_tracks"] == 2
        assert report["variant"] == "htdemucs_ft"  # 默认
        assert set(report["per_stem_median"].keys()) == set(SEPARATION_STEMS)
        # est == ref → 各 stem SDR 都该 > 10dB
        for stem, m in report["per_stem_median"].items():
            assert m["SDR"] > 10, (stem, m)
        assert report["mean_sdr"] > 10

    def test_limit_and_markdown(self, tmp_path: Path):
        _skip_without_deps()
        pytest.importorskip("soundfile")
        fake_db = [_FakeTrack(i) for i in range(3)]
        with (
            patch("mujik.benchmarks.separation.load_musdb", return_value=fake_db),
            patch("mujik.separate.router.separate_audio", side_effect=_fake_separate),
        ):
            report = run_separation_benchmark(
                musdb_root=tmp_path,
                limit=2,
                work_dir=tmp_path / "w",
            )
        assert report["n_tracks"] == 2
        md = render_markdown(report)
        assert "Separation Benchmark" in md
        assert "| Stem | SDR | SIR | SAR |" in md

    def test_empty_subset_raises(self, tmp_path: Path):
        _skip_without_deps()  # _require_deps 在 load_musdb 之前
        with (
            patch("mujik.benchmarks.separation.load_musdb", return_value=[]),
            pytest.raises(SeparationBenchmarkError, match="no tracks"),
        ):
            run_separation_benchmark(musdb_root=tmp_path)


class TestLoadMusdbFailLoud:
    def test_missing_root(self, tmp_path: Path):
        _skip_without_deps()
        from mujik.benchmarks.separation import load_musdb

        with pytest.raises(FileNotFoundError, match="musdb root not found"):
            load_musdb(tmp_path / "nope")

    def test_missing_deps_message(self):
        """musdb/museval 缺失时指向安装命令。"""
        import mujik.benchmarks.separation as sep

        def fake_import(name, *args, **kwargs):
            raise ImportError(f"No module named {name!r}")

        with (
            patch("builtins.__import__", side_effect=fake_import),
            pytest.raises(SeparationBenchmarkError, match="separation-bench"),
        ):
            sep.load_musdb("/nonexistent")


class TestCli:
    def test_help(self):
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "-m", "mujik.benchmarks.separation", "--help"],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=Path(__file__).resolve().parents[2],
        )
        assert result.returncode == 0
        assert "--musdb-root" in result.stdout

    def test_main_writes_reports(self, tmp_path: Path, monkeypatch):
        _skip_without_deps()
        pytest.importorskip("soundfile")
        fake_db = [_FakeTrack(0)]
        monkeypatch.setattr(
            "mujik.benchmarks.separation.load_musdb",
            lambda *a, **k: fake_db,
        )
        monkeypatch.setattr(
            "mujik.separate.router.separate_audio",
            _fake_separate,
        )
        out_md = tmp_path / "sep.md"
        out_json = tmp_path / "sep.json"
        rc = main(
            [
                "--musdb-root",
                str(tmp_path / "dummy"),
                "--limit",
                "1",
                "--device",
                "cpu",
                "--output",
                str(out_md),
                "--json",
                str(out_json),
            ]
        )
        assert rc == 0
        assert "Separation Benchmark" in out_md.read_text()
        data = json.loads(out_json.read_text())
        assert data["n_tracks"] == 1
        assert data["mean_sdr"] > 10
