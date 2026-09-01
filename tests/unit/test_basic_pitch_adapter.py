"""Tests for transcribe.basic_pitch_adapter (mocked subprocess)."""
from __future__ import annotations

import csv
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from mujik.transcribe.basic_pitch_adapter import (
    BASIC_PITCH_CLI,
    BasicPitchAdapterError,
    check_basic_pitch_available,
    transcribe_with_basic_pitch,
)
from mujik.config.schema import BasicPitchConfig


def _write_bp_csv(
    csv_path: Path,
    rows: list[tuple[float, float, int, int]],
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        # v0.5.1: basic-pitch ≥0.3 列名（velocity；pitch_bend 逐帧续行）
        writer.writerow([
            "start_time_s", "end_time_s", "pitch_midi", "velocity", "pitch_bend"
        ])
        for s, e, p, v in rows:
            writer.writerow([s, e, p, v])


class TestCheckAvailable:
    def test_available_true(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            assert check_basic_pitch_available() is True

    def test_available_false(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1)
            assert check_basic_pitch_available() is False

    def test_not_found(self):
        with patch(
            "subprocess.run",
            side_effect=FileNotFoundError("basic-pitch"),
        ):
            assert check_basic_pitch_available() is False


class TestTranscribe:
    def test_basic(self, tmp_path: Path):
        audio = tmp_path / "song.wav"
        audio.write_bytes(b"RIFF")

        def fake_run(cmd, *args, **kwargs):
            out_dir = Path(cmd[2])
            _write_bp_csv(out_dir / f"{audio.stem}_basic_pitch.csv", [
                (0.0, 1.0, 60, 100),   # C4
                (1.0, 2.0, 62, 90),    # D4
                (2.0, 3.0, 64, 80),    # E4
            ])
            return MagicMock(returncode=0, stderr="", stdout="ok")

        with patch("subprocess.run", side_effect=fake_run):
            notes = transcribe_with_basic_pitch(audio, out_dir=tmp_path)

        assert len(notes) == 3
        assert notes[0].pitch == 60
        assert notes[0].velocity == 100
        assert notes[0].start == 0.0
        assert notes[0].end == 1.0
        # channel 0 for pitched stem
        assert notes[0].channel == 0

    def test_cmd_passes_thresholds(self, tmp_path: Path):
        audio = tmp_path / "song.wav"
        audio.write_bytes(b"RIFF")

        def fake_run(cmd, *args, **kwargs):
            out_dir = Path(cmd[2])
            _write_bp_csv(out_dir / f"{audio.stem}_basic_pitch.csv", [])
            return MagicMock(returncode=0, stderr="", stdout="ok")

        cfg = BasicPitchConfig(
            onset_threshold=0.7, frame_threshold=0.4, min_note_length_ms=80.0,
        )
        with patch("subprocess.run", side_effect=fake_run) as mock_run:
            transcribe_with_basic_pitch(audio, config=cfg, out_dir=tmp_path)

        cmd = mock_run.call_args[0][0]
        # 验证 CLI + 阈值（v0.5.1: cmd[1]=--save-note-events，位置参数后移）
        assert cmd[0] == BASIC_PITCH_CLI
        assert "--save-note-events" in cmd
        assert str(tmp_path) in cmd[2]
        assert str(audio) in cmd[3]
        # --onset-threshold 0.7
        idx = cmd.index("--onset-threshold")
        assert cmd[idx + 1] == "0.7"
        idx = cmd.index("--frame-threshold")
        assert cmd[idx + 1] == "0.4"
        idx = cmd.index("--minimum-note-length")
        assert cmd[idx + 1] == "80"

    def test_uses_input_stem_in_output(self, tmp_path: Path):
        audio = tmp_path / "my_song.wav"
        audio.write_bytes(b"RIFF")
        out_dir = tmp_path / "out"

        def fake_run(cmd, *args, **kwargs):
            out = Path(cmd[2])
            expected = out / "my_song_basic_pitch.csv"
            _write_bp_csv(expected, [(0.0, 0.5, 60, 80)])
            return MagicMock(returncode=0, stderr="", stdout="ok")

        with patch("subprocess.run", side_effect=fake_run):
            notes = transcribe_with_basic_pitch(audio, out_dir=out_dir)
        assert len(notes) == 1

    def test_velocity_clamped(self, tmp_path: Path):
        audio = tmp_path / "song.wav"
        audio.write_bytes(b"RIFF")

        def fake_run(cmd, *args, **kwargs):
            out_dir = Path(cmd[2])
            _write_bp_csv(out_dir / f"{audio.stem}_basic_pitch.csv", [
                (0.0, 0.5, 60, 200),  # 超出 127
                (0.5, 1.0, 60, -10),  # 负数
            ])
            return MagicMock(returncode=0, stderr="", stdout="ok")

        with patch("subprocess.run", side_effect=fake_run):
            notes = transcribe_with_basic_pitch(audio, out_dir=tmp_path)

        assert notes[0].velocity == 127
        assert notes[1].velocity == 0

    def test_out_of_range_pitch_skipped(self, tmp_path: Path):
        audio = tmp_path / "song.wav"
        audio.write_bytes(b"RIFF")

        def fake_run(cmd, *args, **kwargs):
            out_dir = Path(cmd[2])
            _write_bp_csv(out_dir / f"{audio.stem}_basic_pitch.csv", [
                (0.0, 0.5, 60, 80),
                (0.5, 1.0, 200, 80),  # 超 MIDI 范围
                (1.0, 1.5, -1, 80),   # 负
            ])
            return MagicMock(returncode=0, stderr="", stdout="ok")

        with patch("subprocess.run", side_effect=fake_run):
            notes = transcribe_with_basic_pitch(audio, out_dir=tmp_path)

        assert len(notes) == 1
        assert notes[0].pitch == 60

    def test_notes_sorted_by_start(self, tmp_path: Path):
        audio = tmp_path / "song.wav"
        audio.write_bytes(b"RIFF")

        def fake_run(cmd, *args, **kwargs):
            out_dir = Path(cmd[2])
            # 故意乱序写入
            _write_bp_csv(out_dir / f"{audio.stem}_basic_pitch.csv", [
                (2.0, 3.0, 64, 80),
                (0.0, 1.0, 60, 100),
                (1.0, 2.0, 62, 90),
            ])
            return MagicMock(returncode=0, stderr="", stdout="ok")

        with patch("subprocess.run", side_effect=fake_run):
            notes = transcribe_with_basic_pitch(audio, out_dir=tmp_path)

        assert [n.pitch for n in notes] == [60, 62, 64]

    def test_malformed_row_skipped(self, tmp_path: Path):
        audio = tmp_path / "song.wav"
        audio.write_bytes(b"RIFF")

        def fake_run(cmd, *args, **kwargs):
            out_dir = Path(cmd[2])
            csv_path = out_dir / f"{audio.stem}_basic_pitch.csv"
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            with open(csv_path, "w") as f:
                f.write("start_time_s,end_time_s,pitch_midi,velocity,pitch_bend\n")
                f.write("0.0,1.0,60,100\n")
                f.write("not_a_number,1.0,62,90\n")  # 坏行
                f.write("2.0,3.0,64,80\n")
            return MagicMock(returncode=0, stderr="", stdout="ok")

        with patch("subprocess.run", side_effect=fake_run):
            notes = transcribe_with_basic_pitch(audio, out_dir=tmp_path)

        assert len(notes) == 2
        assert notes[0].pitch == 60
        assert notes[1].pitch == 64

    def test_pitch_bend_semitones_normalized(self, tmp_path: Path):
        """v0.5.1: basic-pitch ≥0.3 逐帧 semitone bend → mujik [-1,+1]（÷2）。"""
        audio = tmp_path / "song.wav"
        audio.write_bytes(b"RIFF")

        def fake_run(cmd, *args, **kwargs):
            out_dir = Path(cmd[2])
            csv_path = out_dir / f"{audio.stem}_basic_pitch.csv"
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            with open(csv_path, "w") as f:
                f.write("start_time_s,end_time_s,pitch_midi,velocity,pitch_bend\n")
                f.write("0.0,1.0,60,100,1,2,0\n")
            return MagicMock(returncode=0, stderr="", stdout="ok")

        with patch("subprocess.run", side_effect=fake_run):
            notes = transcribe_with_basic_pitch(audio, out_dir=tmp_path)

        assert len(notes) == 1
        assert notes[0].pitch_bend == (0.5, 1.0, 0.0)


class TestErrorPaths:
    def test_input_not_found(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            transcribe_with_basic_pitch(tmp_path / "missing.wav")

    def test_subprocess_failure(self, tmp_path: Path):
        audio = tmp_path / "song.wav"
        audio.write_bytes(b"RIFF")

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stderr="tf crashed")
            with pytest.raises(BasicPitchAdapterError, match="basic-pitch failed"):
                transcribe_with_basic_pitch(audio, out_dir=tmp_path)

    def test_subprocess_timeout(self, tmp_path: Path):
        audio = tmp_path / "song.wav"
        audio.write_bytes(b"RIFF")

        import subprocess
        with patch(
            "subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="x", timeout=1800),
        ):
            with pytest.raises(BasicPitchAdapterError, match="timeout"):
                transcribe_with_basic_pitch(audio, out_dir=tmp_path)

    def test_missing_output_csv(self, tmp_path: Path):
        audio = tmp_path / "song.wav"
        audio.write_bytes(b"RIFF")

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            with pytest.raises(BasicPitchAdapterError, match="output csv not found"):
                transcribe_with_basic_pitch(audio, out_dir=tmp_path)
