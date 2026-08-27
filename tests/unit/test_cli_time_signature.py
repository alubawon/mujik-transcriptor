"""Tests for mujik time-signature change CLI subcommand."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from mujik.cli import main
from mujik.time_signature.io import (
    read_time_signatures_json,
    write_time_signatures_json,
)
from mujik.time_signature.model import TimeSignatureSegment


def _setup_ts_json(project_dir: Path, sigs: list[TimeSignatureSegment] | None = None) -> None:
    project_dir.mkdir(parents=True, exist_ok=True)
    if sigs is None:
        sigs = [TimeSignatureSegment(
            start_time=0.0, end_time=20.0,
            time_signature=(4, 4), confidence=1.0, source="default_4_4",
        )]
    write_time_signatures_json(sigs, project_dir / "time_signatures.json")


class TestTimeSignatureChangeCLI:
    def test_mode_a_redraws_bars(self, tmp_path: Path):
        project_dir = tmp_path / "p"
        _setup_ts_json(project_dir)

        rc = main([
            "time-signature", "change",
            "--project-dir", str(project_dir),
            "--at", "5.0",
            "--new", "7/8",
            "--mode", "A",
        ])
        assert rc == 0

        loaded = read_time_signatures_json(project_dir / "time_signatures.json")
        # 应有新拍号段
        new_sigs = [seg.time_signature for seg in loaded]
        assert (7, 8) in new_sigs

    def test_mode_b_preserve_time(self, tmp_path: Path):
        project_dir = tmp_path / "p"
        _setup_ts_json(project_dir)

        rc = main([
            "time-signature", "change",
            "--project-dir", str(project_dir),
            "--at", "8.0",
            "--new", "3/4",
            "--mode", "B",
        ])
        assert rc == 0

        loaded = read_time_signatures_json(project_dir / "time_signatures.json")
        # 8.0s 之后应为 3/4
        post = [s for s in loaded if s.start_time >= 8.0]
        assert any(s.time_signature == (3, 4) for s in post)

    def test_mode_b_regrid(self, tmp_path: Path):
        project_dir = tmp_path / "p"
        _setup_ts_json(project_dir)

        rc = main([
            "time-signature", "change",
            "--project-dir", str(project_dir),
            "--at", "4.0",
            "--new", "6/8",
            "--mode", "B",
            "--regrid",
        ])
        assert rc == 0
        loaded = read_time_signatures_json(project_dir / "time_signatures.json")
        assert any(s.time_signature == (6, 8) for s in loaded)

    def test_mm_ss_time_format(self, tmp_path: Path):
        project_dir = tmp_path / "p"
        _setup_ts_json(project_dir)

        rc = main([
            "time-signature", "change",
            "--project-dir", str(project_dir),
            "--at", "0:05.500",
            "--new", "3/4",
            "--mode", "A",
        ])
        assert rc == 0
        loaded = read_time_signatures_json(project_dir / "time_signatures.json")
        assert (3, 4) in [s.time_signature for s in loaded]

    def test_out_dir(self, tmp_path: Path):
        project_dir = tmp_path / "p"
        out_dir = tmp_path / "out"
        _setup_ts_json(project_dir)

        rc = main([
            "time-signature", "change",
            "--project-dir", str(project_dir),
            "--out-dir", str(out_dir),
            "--at", "5.0",
            "--new", "5/4",
            "--mode", "A",
        ])
        assert rc == 0
        assert (out_dir / "time_signatures.json").exists()
        # 原 project_dir 文件不动
        assert (project_dir / "time_signatures.json").exists()

    def test_invalid_sig_errors(self, tmp_path: Path):
        project_dir = tmp_path / "p"
        _setup_ts_json(project_dir)

        with pytest.raises(SystemExit):
            main([
                "time-signature", "change",
                "--project-dir", str(project_dir),
                "--at", "5.0",
                "--new", "5/3",  # 无效分母
                "--mode", "A",
            ])

    def test_missing_file_errors(self, tmp_path: Path):
        project_dir = tmp_path / "p"
        project_dir.mkdir()
        # time_signatures.json 不存在

        rc = main([
            "time-signature", "change",
            "--project-dir", str(project_dir),
            "--at", "5.0",
            "--new", "3/4",
            "--mode", "A",
        ])
        assert rc == 1

    def test_unknown_mode_errors(self, tmp_path: Path):
        project_dir = tmp_path / "p"
        _setup_ts_json(project_dir)

        with pytest.raises(SystemExit):
            main([
                "time-signature", "change",
                "--project-dir", str(project_dir),
                "--at", "5.0",
                "--new", "3/4",
                "--mode", "Z",
            ])

    def test_time_signature_help(self, tmp_path: Path):
        """`mujik time-signature`（无子命令）应打印 help 且返回非 0。"""
        rc = main(["time-signature"])
        assert rc == 1
