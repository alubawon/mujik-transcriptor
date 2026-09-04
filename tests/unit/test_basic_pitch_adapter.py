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
    denoise_bend,
    resolve_basic_pitch_cli,
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


class TestResolveCli:
    """v0.5.2：CLI 路径解析——venv 下直接跑 .venv/bin/mujik 时子进程
    PATH 里未必有 .venv/bin，需优先解析解释器同目录的 console script。"""

    def test_sibling_of_executable_wins(self, tmp_path: Path):
        fake_py = tmp_path / "python"
        fake_py.write_text("")  # is_file() 需要真实文件
        sibling = tmp_path / BASIC_PITCH_CLI
        sibling.write_text("")
        with patch("mujik.transcribe.basic_pitch_adapter.sys") as mock_sys:
            mock_sys.executable = str(fake_py)
            assert resolve_basic_pitch_cli() == str(sibling)

    def test_fallback_to_path_name(self, tmp_path: Path):
        fake_py = tmp_path / "python"
        fake_py.write_text("")
        with patch("mujik.transcribe.basic_pitch_adapter.sys") as mock_sys:
            mock_sys.executable = str(fake_py)
            assert resolve_basic_pitch_cli() == BASIC_PITCH_CLI

    def test_cmd_uses_resolved_cli(self, tmp_path: Path):
        """transcribe_with_basic_pitch 的 cmd[0] 应为解析结果而非裸名。"""
        resolved = "/some/venv/bin/basic-pitch"
        with (
            patch(
                "mujik.transcribe.basic_pitch_adapter.resolve_basic_pitch_cli",
                return_value=resolved,
            ),
            patch("subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(returncode=0)
            (tmp_path / "a.wav").write_bytes(b"RIFF")  # 存在性检查
            csv_path = tmp_path / "a_basic_pitch.csv"
            _write_bp_csv(csv_path, [(0.0, 1.0, 60, 100)])
            transcribe_with_basic_pitch(tmp_path / "a.wav", BasicPitchConfig(),
                                        out_dir=tmp_path)
            assert mock_run.call_args[0][0][0] == resolved


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
        # v0.5.2: cmd[0] 是解析后的 CLI（venv 同目录优先），与裸名等价即可
        assert cmd[0] in (BASIC_PITCH_CLI, str(Path(BASIC_PITCH_CLI).parent / BASIC_PITCH_CLI)) or cmd[0].endswith(BASIC_PITCH_CLI)
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
                # 前导 0（无 bend）→ +1 semitone 4 帧 → +2 semitone 4 帧
                f.write("0.0,1.0,60,100," + ",".join(["0", "0"] + ["1"] * 4 + ["2"] * 4) + "\n")
            return MagicMock(returncode=0, stderr="", stdout="ok")

        with patch("subprocess.run", side_effect=fake_run):
            notes = transcribe_with_basic_pitch(audio, out_dir=tmp_path)

        assert len(notes) == 1
        assert notes[0].pitch_bend == (0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0)


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


class TestDenoiseBend:
    """v0.5.3: 逐帧 bend 去噪。"""

    def test_strips_leading_trailing_zeros(self):
        # 首尾 0 段剥掉后剩 +0.5×3 / −0.5×3 两段，均达段长下限
        assert denoise_bend([0, 0.5, 0.5, 0.5, -0.5, -0.5, -0.5, 0]) == (
            0.5, 0.5, 0.5, -0.5, -0.5, -0.5,
        )

    def test_zero_only_after_strip_leaves_constant(self):
        # 剥掉首尾 0 后只剩恒定段 → 按"音高偏差"规则丢弃
        assert denoise_bend([0, 0, 0.5, 0.5, 0, 0]) == ()

    def test_drops_short_jitter_segments(self):
        # 1 帧 +1 / 2 帧 -1 的抖动段全部 < 3 帧下限 → 丢弃
        assert denoise_bend([0.5, -0.5, -0.5, 0.5]) == ()

    def test_keeps_long_vibrato(self):
        # +1/−1 各 4 帧：合法颤音保留
        values = [0.5] * 4 + [-0.5] * 4
        assert denoise_bend(values) == tuple(values)

    def test_drops_constant_nonzero(self):
        # 全程同一值 → 音高识别偏差而非 bend
        assert denoise_bend([1.0] * 10) == ()

    def test_drops_all_zero(self):
        assert denoise_bend([0.0] * 10) == ()

    def test_drops_below_total_frames(self):
        # 两段各 2 帧：段长达标与否取决于 min_segment_frames，总帧数可控丢弃
        values = [0.5, 0.5, 1.0, 1.0]
        assert denoise_bend(values, min_segment_frames=2, min_total_frames=4) == (0.5, 0.5, 1.0, 1.0)
        assert denoise_bend(values, min_segment_frames=2, min_total_frames=5) == ()

    def test_empty(self):
        assert denoise_bend([]) == ()
        assert denoise_bend(()) == ()
