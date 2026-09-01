"""Tests for scripts/_demo_report.py (v0.5.1 + 修 5 曲名目录布局)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

# 直接 import _demo_report 模块
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))
import _demo_report  # type: ignore[import-not-found]


class TestSummarizeRun:
    def _make_run(self, run_dir: Path, *, preset: str = "pop", ws_dir: Path | None = None) -> None:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "project.json").write_text(json.dumps({
            "mujik_version": "0.5.1",
            "preset": preset,
            "separator": "demucs/htdemucs_ft",
            "transcribe_mode": "per_stem",
            "rhythm_enabled": True,
            "chord_enabled": True,
            "chord_backend": "madmom",
            "chord_quantize_enabled": False,
            "score_features": ["bend", "harmony"],
        }))
        ws = ws_dir if ws_dir is not None else run_dir / "ws"
        ws.mkdir(parents=True, exist_ok=True)
        (ws / "beats.json").write_text(json.dumps({
            "bpm": 120.0,
            "beats": [0.0, 0.5, 1.0, 1.5, 2.0],
            "downbeats": [0.0, 2.0],
        }))
        (ws / "chords.json").write_text(json.dumps([
            {"start": 0.0, "end": 1.0, "root": "C", "quality": "", "bass": None},
            {"start": 1.0, "end": 2.0, "root": "F", "quality": "", "bass": None},
        ]))
        (ws / "time_signatures.json").write_text(json.dumps([
            {"start": 0.0, "end": 2.0, "sig": [4, 4], "confidence": 1.0, "source": "heuristic"},
        ]))
        (run_dir / "project.mid").write_bytes(b"MThd" + b"\x00" * 100)

    def test_summary_fields_default_ws(self, tmp_path: Path):
        """单 preset 布局：曲名/project.json + 曲名/ws/。"""
        self._make_run(tmp_path / "buhee")
        s = _demo_report._summarize_run(tmp_path / "buhee", tmp_path)
        assert s["run"] == "buhee"
        assert s["preset"] == "pop"
        assert s["version"] == "0.5.1"
        assert s["bpm"] == 120.0
        assert s["n_beats"] == 5
        assert s["n_chords"] == 2
        assert s["n_time_sigs"] == 1
        assert s["has_mid"] is True
        assert s["has_musicxml"] is False
        assert s["has_pdf"] is False

    def test_summary_fields_shared_ws(self, tmp_path: Path):
        """多 preset 布局：曲名/{preset}/ + 共享 曲名/ws/。"""
        ws = tmp_path / "buhee" / "ws"
        ws.mkdir(parents=True)
        self._make_run(tmp_path / "buhee" / "pop", preset="pop", ws_dir=ws)
        s = _demo_report._summarize_run(tmp_path / "buhee" / "pop", tmp_path)
        assert s["run"] == "buhee/pop"
        assert s["bpm"] == 120.0  # 从共享 ws 读到 beats
        assert s["n_chords"] == 2

    def test_summary_chord_disabled(self, tmp_path: Path):
        d = tmp_path / "song" / "metal"
        d.mkdir(parents=True)
        ws = d / "ws"
        ws.mkdir()
        (d / "project.json").write_text(json.dumps({
            "mujik_version": "0.5.1",
            "preset": "metal",
            "chord_enabled": False,
        }))
        (ws / "beats.json").write_text(json.dumps({"bpm": 90.0, "beats": [], "downbeats": []}))
        s = _demo_report._summarize_run(d, tmp_path)
        assert s["chord_enabled"] is False
        assert s["n_chords"] == 0

    def test_ws_dir_skipped_in_discovery(self, tmp_path: Path):
        """ws/ 里的 project.json（不存在于真实流程）不当作 run。"""
        d = tmp_path / "song"
        self._make_run(d)
        runs = _demo_report._find_run_dirs(tmp_path)
        assert runs == [d]


class TestRenderMarkdown:
    def test_full_report_song_layout(self, tmp_path: Path):
        for song in ("buhee", "my_song"):
            d = tmp_path / song
            d.mkdir()
            (d / "project.json").write_text(json.dumps({
                "mujik_version": "0.5.1",
                "preset": "pop",
                "separator": "demucs/htdemucs_ft",
                "transcribe_mode": "per_stem",
                "rhythm_enabled": True,
                "chord_enabled": False,
                "chord_backend": "madmom",
                "chord_quantize_enabled": False,
                "score_features": ["bend", "harmony"],
            }))
            ws = d / "ws"
            ws.mkdir()
            (ws / "beats.json").write_text(json.dumps({
                "bpm": 120.0, "beats": [0.0], "downbeats": [0.0],
            }))
            (d / "project.mid").write_bytes(b"x")

        md = _demo_report.render_markdown(tmp_path)
        assert "# mujik-transcriptor demo report" in md
        assert "**buhee**" in md
        assert "**my_song**" in md
        assert "## Summary" in md
        assert "## Feature flags" in md
        assert "## Artifacts" in md
        assert "## Per-run detail" in md

    def test_empty_output(self, tmp_path: Path):
        md = _demo_report.render_markdown(tmp_path)
        assert "_No outputs found" in md


class TestMainCLI:
    def test_main_no_args(self):
        r = subprocess.run(
            [sys.executable, str(Path(__file__).resolve().parent.parent.parent / "scripts" / "_demo_report.py")],
            capture_output=True, text=True,
        )
        assert r.returncode == 2
        assert "usage" in r.stderr

    def test_main_writes_markdown(self, tmp_path: Path):
        d = tmp_path / "buhee"
        d.mkdir()
        (d / "project.json").write_text(json.dumps({
            "mujik_version": "0.5.1",
            "preset": "pop",
        }))
        ws = d / "ws"
        ws.mkdir()
        (ws / "beats.json").write_text(json.dumps({"bpm": 120.0, "beats": [], "downbeats": []}))
        r = subprocess.run(
            [sys.executable, str(Path(__file__).resolve().parent.parent.parent / "scripts" / "_demo_report.py"), str(tmp_path)],
            capture_output=True, text=True,
        )
        assert r.returncode == 0
        assert "demo report" in r.stdout
