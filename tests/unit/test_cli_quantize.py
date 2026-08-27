"""Tests for mujik quantize CLI subcommand."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from mujik.cli import main
from mujik.midi.model import Note, Project, TempoSegment
from mujik.midi.io import write_project_to_midi
from mujik.time_signature.io import write_time_signatures_json
from mujik.time_signature.model import TimeSignatureSegment


def _setup_artifact_dir(project_dir: Path, bpm: float = 120.0) -> None:
    """构造一个最小可量化的 artifact 目录。"""
    project_dir.mkdir(parents=True, exist_ok=True)

    # beats.json
    (project_dir / "beats.json").write_text(json.dumps({
        "beats": [0.0, 0.5, 1.0, 1.5, 2.0],
        "downbeats": [0.0, 1.0, 2.0],
        "bpm": bpm,
        "tempo_confidence": 0.9,
    }))

    # time_signatures.json
    write_time_signatures_json(
        [TimeSignatureSegment(
            start_time=0.0, end_time=10.0,
            time_signature=(4, 4), confidence=1.0, source="default_4_4",
        )],
        project_dir / "time_signatures.json",
    )

    # project.mid
    proj = Project(
        audio_path="song.wav",
        duration=10.0,
        sample_rate=44100,
        time_signatures=[TimeSignatureSegment(
            start_time=0.0, end_time=10.0,
            time_signature=(4, 4), confidence=1.0, source="default_4_4",
        )],
        tempo_map=[TempoSegment(0.0, 10.0, bpm)],
    )
    proj.get_track("vocals").add(Note(0.123, 0.5, 60, 100))
    proj.get_track("vocals").add(Note(0.987, 1.5, 62, 90))
    write_project_to_midi(proj, project_dir / "project.mid")


class TestQuantizeCLI:
    def test_writes_report_and_midi(self, tmp_path: Path):
        project_dir = tmp_path / "p"
        _setup_artifact_dir(project_dir)

        rc = main([
            "quantize",
            "--project-dir", str(project_dir),
        ])

        assert rc == 0
        assert (project_dir / "quantize_report.json").exists()
        assert (project_dir / "project.mid").exists()
        data = json.loads((project_dir / "quantize_report.json").read_text())
        assert data["total_notes_before"] == 2
        assert data["total_notes_after"] == 2

    def test_with_config_json(self, tmp_path: Path):
        project_dir = tmp_path / "p"
        _setup_artifact_dir(project_dir)

        cfg_path = tmp_path / "cfg.json"
        cfg_path.write_text(json.dumps({
            "grid_resolution": 32,
            "strength": 0.5,
            "groove_template": "swing16",
        }))

        rc = main([
            "quantize",
            "--project-dir", str(project_dir),
            "--config-yaml", str(cfg_path),
        ])

        assert rc == 0
        data = json.loads((project_dir / "quantize_report.json").read_text())
        assert data["grid_resolution"] == 32
        assert data["strength"] == 0.5
        assert data["groove_template"] == "swing16"

    def test_missing_beats_json_errors(self, tmp_path: Path):
        project_dir = tmp_path / "p"
        project_dir.mkdir()
        (project_dir / "project.mid").write_bytes(b"x")
        # beats.json 缺失

        rc = main([
            "quantize",
            "--project-dir", str(project_dir),
        ])

        assert rc == 2  # missing beats.json

    def test_missing_project_mid_errors(self, tmp_path: Path):
        project_dir = tmp_path / "p"
        project_dir.mkdir()

        rc = main([
            "quantize",
            "--project-dir", str(project_dir),
        ])

        assert rc == 1  # missing project.mid

    def test_out_dir_creates_new_directory(self, tmp_path: Path):
        project_dir = tmp_path / "p"
        _setup_artifact_dir(project_dir)
        out_dir = tmp_path / "out"

        rc = main([
            "quantize",
            "--project-dir", str(project_dir),
            "--out-dir", str(out_dir),
        ])

        assert rc == 0
        assert (out_dir / "project.mid").exists()
        assert (out_dir / "quantize_report.json").exists()
        # 原 project_dir 不应有 quantize_report.json
        assert not (project_dir / "quantize_report.json").exists()

    def test_no_write_report_skips_report(self, tmp_path: Path):
        project_dir = tmp_path / "p"
        _setup_artifact_dir(project_dir)

        rc = main([
            "quantize",
            "--project-dir", str(project_dir),
            "--no-write-report",
        ])

        assert rc == 0
        assert not (project_dir / "quantize_report.json").exists()
