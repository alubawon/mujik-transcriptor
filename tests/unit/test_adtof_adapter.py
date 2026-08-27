"""Tests for transcribe.adtof_adapter (mocked subprocess)."""
from __future__ import annotations

import csv
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from mujik.transcribe.adtof_adapter import (
    AdtofAdapterError,
    GM_DRUM_MAP_5CLASS,
    GM_DRUM_MAP_9CLASS,
    check_adtof_available,
    transcribe_drums_with_adtof,
)
from mujik.config.schema import AdtofConfig
from mujik.midi.io import DRUM_CHANNEL


def _write_adtof_csv(
    csv_path: Path,
    rows: list[tuple[float, int, float]],
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time_s", "class_id", "velocity"])
        for t, c, v in rows:
            writer.writerow([t, c, v])


class TestCheckAvailable:
    def test_available_true(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            assert check_adtof_available() is True

    def test_available_false_import_error(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stderr="no adtof")
            assert check_adtof_available() is False

    def test_timeout(self):
        import subprocess
        with patch(
            "subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="x", timeout=30),
        ):
            assert check_adtof_available() is False


class TestTranscribe5Class:
    def test_basic_5class(self, tmp_path: Path):
        """subprocess mock：CSV 写入，验证 Note 列表。"""
        audio = tmp_path / "song.wav"
        audio.write_bytes(b"RIFF")
        out_dir = tmp_path / "out"

        # 写假 CSV（subprocess 的"输出"）
        def fake_run(cmd, *args, **kwargs):
            csv_path = Path(cmd[3])  # cmd[2] is output csv
            _write_adtof_csv(csv_path, [
                (0.0, 0, 0.9),    # kick
                (0.5, 1, 0.85),   # snare
                (1.0, 2, 0.7),    # closed hh
                (1.5, 3, 0.6),    # open hh
                (2.0, 4, 0.5),    # crash
            ])
            return MagicMock(returncode=0, stderr="", stdout="ok")

        with patch("subprocess.run", side_effect=fake_run):
            notes = transcribe_drums_with_adtof(audio, out_dir=out_dir)

        assert len(notes) == 5
        # 验证 GM note 映射
        assert notes[0].pitch == GM_DRUM_MAP_5CLASS[0]  # 36
        assert notes[0].channel == DRUM_CHANNEL  # 9
        assert notes[0].velocity == round(0.9 * 127)
        assert notes[1].pitch == 38  # snare
        assert notes[2].pitch == 42  # closed hh
        assert notes[3].pitch == 46  # open hh
        assert notes[4].pitch == 49  # crash

    def test_9class_mapping(self, tmp_path: Path):
        audio = tmp_path / "song.wav"
        audio.write_bytes(b"RIFF")
        out_dir = tmp_path / "out"

        def fake_run(cmd, *args, **kwargs):
            csv_path = Path(cmd[3])
            _write_adtof_csv(csv_path, [
                (0.0, 4, 0.9),   # tom-hi → 50
                (0.5, 5, 0.8),   # tom-mid → 47
                (1.0, 6, 0.7),   # tom-low → 45
                (1.5, 7, 0.6),   # crash → 49
                (2.0, 8, 0.5),   # ride → 51
            ])
            return MagicMock(returncode=0, stderr="", stdout="ok")

        cfg = AdtofConfig(model="adtof-9class")
        with patch("subprocess.run", side_effect=fake_run):
            notes = transcribe_drums_with_adtof(audio, config=cfg, out_dir=out_dir)

        assert len(notes) == 5
        assert notes[0].pitch == 50  # tom-hi
        assert notes[3].pitch == 49  # crash
        assert notes[4].pitch == 51  # ride

    def test_velocity_clamped(self, tmp_path: Path):
        audio = tmp_path / "song.wav"
        audio.write_bytes(b"RIFF")
        out_dir = tmp_path / "out"

        def fake_run(cmd, *args, **kwargs):
            _write_adtof_csv(Path(cmd[3]), [
                (0.0, 0, 1.5),   # 超出 1.0 → clamp 到 127
                (0.5, 0, -0.1),  # 负数 → clamp 到 0
                (1.0, 0, 0.0),   # 0 → 跳过（velocity < 1）
            ])
            return MagicMock(returncode=0, stderr="", stdout="ok")

        with patch("subprocess.run", side_effect=fake_run):
            notes = transcribe_drums_with_adtof(audio, out_dir=out_dir)

        # 1.5 → 127，-0.1 → 0（被跳过），0.0 → 跳过
        assert len(notes) == 1
        assert notes[0].velocity == 127

    def test_threshold_filtering(self, tmp_path: Path):
        audio = tmp_path / "song.wav"
        audio.write_bytes(b"RIFF")
        out_dir = tmp_path / "out"

        # CSV 里全写低 velocity（subprocess 端由 threshold 过滤，但 wrapper
        # 的 threshold 是 adtof_config.onset_threshold；这里我们假设
        # subprocess 已经把低于 0.5 的过滤掉）
        def fake_run(cmd, *args, **kwargs):
            _write_adtof_csv(Path(cmd[3]), [
                (0.0, 0, 0.9),  # 通过
                (0.5, 1, 0.4),  # 低于 0.5 → 应被 wrapper 过滤（mock 不真过滤）
            ])
            return MagicMock(returncode=0, stderr="", stdout="ok")

        # 验证 cmd 行的 threshold 参数
        cfg = AdtofConfig(onset_threshold=0.6)
        with patch("subprocess.run", side_effect=fake_run) as mock_run:
            transcribe_drums_with_adtof(audio, config=cfg, out_dir=out_dir)

        cmd = mock_run.call_args[0][0]
        # cmd 顺序: [python, wrapper, input, output_csv, model, device, threshold]
        # cmd[6] = "0.6"
        assert cmd[6] == "0.6"


class TestErrorPaths:
    def test_input_not_found(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            transcribe_drums_with_adtof(tmp_path / "missing.wav")

    def test_subprocess_failure(self, tmp_path: Path):
        audio = tmp_path / "song.wav"
        audio.write_bytes(b"RIFF")

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stderr="crashed")
            with pytest.raises(AdtofAdapterError, match="adtof failed"):
                transcribe_drums_with_adtof(audio, out_dir=tmp_path / "out")

    def test_subprocess_timeout(self, tmp_path: Path):
        audio = tmp_path / "song.wav"
        audio.write_bytes(b"RIFF")

        import subprocess
        with patch(
            "subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="x", timeout=1800),
        ):
            with pytest.raises(AdtofAdapterError, match="timeout"):
                transcribe_drums_with_adtof(audio, out_dir=tmp_path / "out")

    def test_missing_output_csv(self, tmp_path: Path):
        audio = tmp_path / "song.wav"
        audio.write_bytes(b"RIFF")

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            with pytest.raises(AdtofAdapterError, match="output csv not found"):
                transcribe_drums_with_adtof(audio, out_dir=tmp_path / "out")


class TestGMDrumMaps:
    def test_5class_complete(self):
        """5 类映射全有。"""
        assert set(GM_DRUM_MAP_5CLASS.keys()) == {0, 1, 2, 3, 4}
        for c, p in GM_DRUM_MAP_5CLASS.items():
            assert 0 <= p <= 127

    def test_9class_complete(self):
        assert set(GM_DRUM_MAP_9CLASS.keys()) == {0, 1, 2, 3, 4, 5, 6, 7, 8}
        for c, p in GM_DRUM_MAP_9CLASS.items():
            assert 0 <= p <= 127

    def test_crash_in_5class_is_49(self):
        """5 类中的"cymbal" → 49 (Crash 1)"""
        assert GM_DRUM_MAP_5CLASS[4] == 49


class TestChannelAssignment:
    def test_all_notes_channel_9(self, tmp_path: Path):
        audio = tmp_path / "song.wav"
        audio.write_bytes(b"RIFF")

        def fake_run(cmd, *args, **kwargs):
            _write_adtof_csv(Path(cmd[3]), [
                (0.0, 0, 0.9),
                (0.5, 1, 0.8),
            ])
            return MagicMock(returncode=0, stderr="", stdout="ok")

        with patch("subprocess.run", side_effect=fake_run):
            notes = transcribe_drums_with_adtof(audio, out_dir=tmp_path / "out")

        for n in notes:
            assert n.channel == DRUM_CHANNEL
