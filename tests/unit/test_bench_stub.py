"""Tests for benchmarks/ scaffold scripts."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def _write_dummy_wav(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"RIFF" * 100)


class TestRunSeparation:
    def test_help(self):
        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / "benchmarks" / "run_separation.py"), "--help"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0
        assert "Demucs" in result.stdout or "separation" in result.stdout.lower()

    def test_basic(self, tmp_path: Path):
        in_wav = tmp_path / "song.wav"
        _write_dummy_wav(in_wav)
        out_json = tmp_path / "report.json"

        result = subprocess.run(
            [
                sys.executable, str(REPO_ROOT / "benchmarks" / "run_separation.py"),
                "--input", str(in_wav),
                "--variant", "htdemucs_ft",
                "--out", str(out_json),
            ],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0
        assert out_json.exists()
        data = json.loads(out_json.read_text())
        assert data["variant"] == "htdemucs_ft"
        assert "elapsed_sec" in data
        assert data["sdr_db"] is None  # v0.4.0 scaffold

    def test_input_not_found(self, tmp_path: Path):
        result = subprocess.run(
            [
                sys.executable, str(REPO_ROOT / "benchmarks" / "run_separation.py"),
                "--input", str(tmp_path / "missing.wav"),
            ],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 1


class TestRunTranscription:
    def test_help(self):
        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / "benchmarks" / "run_transcription.py"), "--help"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0
        assert "adapter" in result.stdout.lower()

    def test_basic(self, tmp_path: Path):
        in_wav = tmp_path / "song.wav"
        _write_dummy_wav(in_wav)
        out_json = tmp_path / "report.json"

        result = subprocess.run(
            [
                sys.executable, str(REPO_ROOT / "benchmarks" / "run_transcription.py"),
                "--input", str(in_wav),
                "--adapter", "basic-pitch",
                "--out", str(out_json),
            ],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0
        assert out_json.exists()
        data = json.loads(out_json.read_text())
        assert data["adapter"] == "basic-pitch"

    def test_input_not_found(self, tmp_path: Path):
        result = subprocess.run(
            [
                sys.executable, str(REPO_ROOT / "benchmarks" / "run_transcription.py"),
                "--input", str(tmp_path / "missing.wav"),
            ],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 1
