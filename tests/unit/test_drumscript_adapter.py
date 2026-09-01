"""Tests for transcribe.drumscript_adapter (mocked subprocess)."""
from __future__ import annotations

import csv
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mujik.config.schema import DrumScriptConfig
from mujik.midi.io import DRUM_CHANNEL
from mujik.transcribe.drumscript_adapter import (
    GM_DRUM_MAP_DRUMSCRIPT,
    DrumScriptAdapterError,
    check_drumscript_available,
    transcribe_drums_with_drumscript,
)


def _write_ds_csv(
    csv_path: Path,
    rows: list[tuple[float, str]],
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time_s", "instrument", "velocity"])
        for t, inst in rows:
            writer.writerow([t, inst, 1.0])


class TestCheckAvailable:
    def test_available_true(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            assert check_drumscript_available() is True

    def test_available_false_import_error(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stderr="no drumscript")
            assert check_drumscript_available() is False

    def test_timeout(self):
        with patch(
            "subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="x", timeout=30),
        ):
            assert check_drumscript_available() is False


class TestTranscribe:
    def test_basic_mapping(self, tmp_path: Path):
        """subprocess mock：CSV 写入，验证 GM 映射 + channel 9。"""
        audio = tmp_path / "song.wav"
        audio.write_bytes(b"RIFF")
        out_dir = tmp_path / "out"

        def fake_run(cmd, *args, **kwargs):
            csv_path = Path(cmd[3])  # cmd = [python, wrapper, input, csv]
            _write_ds_csv(csv_path, [
                (0.0, "kick"),
                (0.5, "snare"),
                (1.0, "hi_hat_closed"),
                (1.5, "hi_hat_open"),
                (2.0, "crash"),
                (2.5, "ride"),
            ])
            return MagicMock(returncode=0, stderr="", stdout="ok")

        with patch("subprocess.run", side_effect=fake_run):
            notes = transcribe_drums_with_drumscript(audio, out_dir=out_dir)

        assert len(notes) == 6
        assert notes[0].pitch == 36  # kick
        assert notes[1].pitch == 38  # snare
        assert notes[2].pitch == 42  # closed hh
        assert notes[3].pitch == 46  # open hh
        assert notes[4].pitch == 49  # crash
        assert notes[5].pitch == 51  # ride
        for n in notes:
            assert n.channel == DRUM_CHANNEL
            assert n.velocity == 100  # default_velocity

    def test_tom_mapping(self, tmp_path: Path):
        audio = tmp_path / "song.wav"
        audio.write_bytes(b"RIFF")

        def fake_run(cmd, *args, **kwargs):
            _write_ds_csv(Path(cmd[3]), [
                (0.0, "high_tom"),
                (0.5, "mid_tom"),
                (1.0, "low_tom"),
            ])
            return MagicMock(returncode=0, stderr="", stdout="ok")

        with patch("subprocess.run", side_effect=fake_run):
            notes = transcribe_drums_with_drumscript(audio, out_dir=tmp_path / "out")

        assert [n.pitch for n in notes] == [48, 45, 41]

    def test_kick_clicky_maps_to_kick(self, tmp_path: Path):
        audio = tmp_path / "song.wav"
        audio.write_bytes(b"RIFF")

        def fake_run(cmd, *args, **kwargs):
            _write_ds_csv(Path(cmd[3]), [(0.0, "kick_clicky")])
            return MagicMock(returncode=0, stderr="", stdout="ok")

        with patch("subprocess.run", side_effect=fake_run):
            notes = transcribe_drums_with_drumscript(audio, out_dir=tmp_path / "out")

        assert notes[0].pitch == 36

    def test_unknown_instrument_skipped(self, tmp_path: Path):
        audio = tmp_path / "song.wav"
        audio.write_bytes(b"RIFF")

        def fake_run(cmd, *args, **kwargs):
            _write_ds_csv(Path(cmd[3]), [
                (0.0, "kick"),
                (0.5, "unknown"),
            ])
            return MagicMock(returncode=0, stderr="", stdout="ok")

        with patch("subprocess.run", side_effect=fake_run):
            notes = transcribe_drums_with_drumscript(audio, out_dir=tmp_path / "out")

        assert len(notes) == 1
        assert notes[0].pitch == 36

    def test_default_velocity_configurable(self, tmp_path: Path):
        audio = tmp_path / "song.wav"
        audio.write_bytes(b"RIFF")

        def fake_run(cmd, *args, **kwargs):
            _write_ds_csv(Path(cmd[3]), [(0.0, "kick")])
            return MagicMock(returncode=0, stderr="", stdout="ok")

        cfg = DrumScriptConfig(default_velocity=64)
        with patch("subprocess.run", side_effect=fake_run):
            notes = transcribe_drums_with_drumscript(audio, config=cfg, out_dir=tmp_path / "out")

        assert notes[0].velocity == 64

    def test_duration_from_min_note_length(self, tmp_path: Path):
        audio = tmp_path / "song.wav"
        audio.write_bytes(b"RIFF")

        def fake_run(cmd, *args, **kwargs):
            _write_ds_csv(Path(cmd[3]), [(0.0, "kick")])
            return MagicMock(returncode=0, stderr="", stdout="ok")

        cfg = DrumScriptConfig(min_note_length_ms=120.0)
        with patch("subprocess.run", side_effect=fake_run):
            notes = transcribe_drums_with_drumscript(audio, config=cfg, out_dir=tmp_path / "out")

        assert notes[0].start == 0.0
        assert notes[0].end == pytest.approx(0.12)


class TestDedup:
    def test_same_pitch_within_interval_deduped(self, tmp_path: Path):
        """同一 instrument 40ms 内的相邻 onset 视为同一击打。"""
        audio = tmp_path / "song.wav"
        audio.write_bytes(b"RIFF")

        def fake_run(cmd, *args, **kwargs):
            _write_ds_csv(Path(cmd[3]), [
                (0.0, "kick"),
                (0.02, "kick"),   # 20ms 后 → 去重
                (0.03, "kick"),   # 30ms 后 → 去重
                (0.5, "kick"),    # 距上次保留点 > 40ms → 保留
            ])
            return MagicMock(returncode=0, stderr="", stdout="ok")

        with patch("subprocess.run", side_effect=fake_run):
            notes = transcribe_drums_with_drumscript(audio, out_dir=tmp_path / "out")

        assert [n.start for n in notes] == pytest.approx([0.0, 0.5])

    def test_different_pitch_same_time_kept(self, tmp_path: Path):
        """同刻不同 instrument（如 snare+hh 同时击打）不去重。"""
        audio = tmp_path / "song.wav"
        audio.write_bytes(b"RIFF")

        def fake_run(cmd, *args, **kwargs):
            _write_ds_csv(Path(cmd[3]), [
                (0.0, "snare"),
                (0.0, "hi_hat_closed"),
            ])
            return MagicMock(returncode=0, stderr="", stdout="ok")

        with patch("subprocess.run", side_effect=fake_run):
            notes = transcribe_drums_with_drumscript(audio, out_dir=tmp_path / "out")

        assert {n.pitch for n in notes} == {38, 42}

    def test_dedup_interval_configurable(self, tmp_path: Path):
        audio = tmp_path / "song.wav"
        audio.write_bytes(b"RIFF")

        def fake_run(cmd, *args, **kwargs):
            _write_ds_csv(Path(cmd[3]), [
                (0.0, "kick"),
                (0.06, "kick"),  # 60ms：默认 40ms 会保留；设 100ms 则去重
            ])
            return MagicMock(returncode=0, stderr="", stdout="ok")

        cfg = DrumScriptConfig(min_onset_interval_ms=100.0)
        with patch("subprocess.run", side_effect=fake_run):
            notes = transcribe_drums_with_drumscript(audio, config=cfg, out_dir=tmp_path / "out")

        assert len(notes) == 1

    def test_notes_sorted_by_start(self, tmp_path: Path):
        audio = tmp_path / "song.wav"
        audio.write_bytes(b"RIFF")

        def fake_run(cmd, *args, **kwargs):
            # CSV 顺序故意乱序（同 onset 多 instrument 交错）
            _write_ds_csv(Path(cmd[3]), [
                (1.0, "kick"),
                (0.0, "snare"),
                (0.5, "hi_hat_closed"),
            ])
            return MagicMock(returncode=0, stderr="", stdout="ok")

        with patch("subprocess.run", side_effect=fake_run):
            notes = transcribe_drums_with_drumscript(audio, out_dir=tmp_path / "out")

        starts = [n.start for n in notes]
        assert starts == sorted(starts)


class TestErrorPaths:
    def test_input_not_found(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            transcribe_drums_with_drumscript(tmp_path / "missing.wav")

    def test_subprocess_failure(self, tmp_path: Path):
        audio = tmp_path / "song.wav"
        audio.write_bytes(b"RIFF")

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stderr="crashed")
            with pytest.raises(DrumScriptAdapterError, match="drumscript failed"):
                transcribe_drums_with_drumscript(audio, out_dir=tmp_path / "out")

    def test_subprocess_timeout(self, tmp_path: Path):
        audio = tmp_path / "song.wav"
        audio.write_bytes(b"RIFF")

        with patch(
            "subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="x", timeout=1800),
        ):
            with pytest.raises(DrumScriptAdapterError, match="timeout"):
                transcribe_drums_with_drumscript(audio, out_dir=tmp_path / "out")

    def test_missing_output_csv(self, tmp_path: Path):
        audio = tmp_path / "song.wav"
        audio.write_bytes(b"RIFF")

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            with pytest.raises(DrumScriptAdapterError, match="output csv not found"):
                transcribe_drums_with_drumscript(audio, out_dir=tmp_path / "out")


class TestGMDrumMap:
    def test_covers_all_drumscript_classes(self):
        """DrumScript 全部具名 instrument 都有映射（unknown 除外）。"""
        assert set(GM_DRUM_MAP_DRUMSCRIPT) == {
            "kick", "kick_clicky", "snare", "hi_hat_closed", "hi_hat_open",
            "low_tom", "mid_tom", "high_tom", "crash", "ride",
        }

    def test_all_pitches_valid(self):
        for pitch in GM_DRUM_MAP_DRUMSCRIPT.values():
            assert 0 <= pitch <= 127

    def test_gm_conventions(self):
        """GM 标准鼓位：kick 36 / snare 38 / closed hh 42 / open hh 46 / crash 49 / ride 51。"""
        assert GM_DRUM_MAP_DRUMSCRIPT["kick"] == 36
        assert GM_DRUM_MAP_DRUMSCRIPT["snare"] == 38
        assert GM_DRUM_MAP_DRUMSCRIPT["hi_hat_closed"] == 42
        assert GM_DRUM_MAP_DRUMSCRIPT["hi_hat_open"] == 46
        assert GM_DRUM_MAP_DRUMSCRIPT["crash"] == 49
        assert GM_DRUM_MAP_DRUMSCRIPT["ride"] == 51
