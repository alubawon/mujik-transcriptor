"""Tests for scripts/_demo_report.py (v0.5.1)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

# 直接 import _demo_report 模块
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))
import _demo_report  # type: ignore[import-not-found]


class TestSummarizePreset:
    def _make_preset(self, root: Path, name: str) -> None:
        d = root / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "project.json").write_text(json.dumps({
            "mujik_version": "0.5.1",
            "preset": name,
            "separator": "demucs/htdemucs_ft",
            "transcribe_mode": "per_stem",
            "rhythm_enabled": True,
            "chord_enabled": True,
            "chord_backend": "madmom",
            "chord_quantize_enabled": False,
            "score_features": ["bend", "harmony"],
        }))
        (d / "beats.json").write_text(json.dumps({
            "bpm": 120.0,
            "beats": [0.0, 0.5, 1.0, 1.5, 2.0],
            "downbeats": [0.0, 2.0],
        }))
        (d / "chords.json").write_text(json.dumps([
            {"start": 0.0, "end": 1.0, "root": "C", "quality": "", "bass": None},
            {"start": 1.0, "end": 2.0, "root": "F", "quality": "", "bass": None},
        ]))
        (d / "time_signatures.json").write_text(json.dumps([
            {"start": 0.0, "end": 2.0, "sig": [4, 4], "confidence": 1.0, "source": "heuristic"},
        ]))
        (d / "project.mid").write_bytes(b"MThd" + b"\x00" * 100)

    def test_summary_fields(self, tmp_path: Path):
        self._make_preset(tmp_path, "pop")
        summary = _demo_report._summarize_preset(tmp_path / "pop")
        assert summary["preset"] == "pop"
        assert summary["version"] == "0.5.1"
        assert summary["bpm"] == 120.0
        assert summary["n_beats"] == 5
        assert summary["n_chords"] == 2
        assert summary["n_time_sigs"] == 1
        assert summary["has_mid"] is True
        assert summary["has_musicxml"] is False
        assert summary["has_pdf"] is False

    def test_summary_chord_disabled(self, tmp_path: Path):
        d = tmp_path / "metal"
        d.mkdir()
        (d / "project.json").write_text(json.dumps({
            "mujik_version": "0.5.1",
            "preset": "metal",
            "chord_enabled": False,
        }))
        (d / "beats.json").write_text(json.dumps({"bpm": 90.0, "beats": [], "downbeats": []}))
        s = _demo_report._summarize_preset(d)
        assert s["chord_enabled"] is False
        assert s["n_chords"] == 0


class TestRenderMarkdown:
    def test_full_report(self, tmp_path: Path):
        for name in ("pop", "jazz", "metal"):
            d = tmp_path / name
            d.mkdir()
            (d / "project.json").write_text(json.dumps({
                "mujik_version": "0.5.1",
                "preset": name,
                "separator": "demucs/htdemucs_ft",
                "transcribe_mode": "per_stem",
                "rhythm_enabled": True,
                "chord_enabled": name == "jazz",
                "chord_backend": "madmom",
                "chord_quantize_enabled": name == "jazz",
                "score_features": ["bend", "harmony"],
            }))
            (d / "beats.json").write_text(json.dumps({
                "bpm": 120.0, "beats": [0.0], "downbeats": [0.0],
            }))
            (d / "project.mid").write_bytes(b"x")

        md = _demo_report.render_markdown(tmp_path)
        assert "# mujik-transcriptor demo report" in md
        assert "**pop**" in md
        assert "**jazz**" in md
        assert "**metal**" in md
        assert "## Summary" in md
        assert "## Feature flags" in md
        assert "## Artifacts" in md
        assert "## Per-preset detail" in md

    def test_empty_output(self, tmp_path: Path):
        md = _demo_report.render_markdown(tmp_path)
        assert "_No preset outputs found._" in md


class TestMainCLI:
    def test_main_no_args(self):
        r = subprocess.run(
            [sys.executable, str(Path(__file__).resolve().parent.parent.parent / "scripts" / "_demo_report.py")],
            capture_output=True, text=True,
        )
        assert r.returncode == 2
        assert "usage" in r.stderr

    def test_main_writes_markdown(self, tmp_path: Path):
        d = tmp_path / "pop"
        d.mkdir()
        (d / "project.json").write_text(json.dumps({
            "mujik_version": "0.5.1",
            "preset": "pop",
        }))
        (d / "beats.json").write_text(json.dumps({"bpm": 120.0, "beats": [], "downbeats": []}))
        r = subprocess.run(
            [sys.executable, str(Path(__file__).resolve().parent.parent.parent / "scripts" / "_demo_report.py"), str(tmp_path)],
            capture_output=True, text=True,
        )
        assert r.returncode == 0
        assert "demo report" in r.stdout
