"""Tests for chord/madmom_adapter.py (v0.4.4, mocked subprocess + label parsing)."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from mujik.midi.model import ChordEvent
from mujik.chord.madmom_adapter import (
    MADMOM_CHORD_TIMEOUT_DEFAULT,
    MadmomChordAdapterError,
    _parse_madmom_chord_label,
    check_madmom_chord_available,
    detect_chords_with_madmom,
)


def _write_fake_chord_json(
    json_path: Path,
    entries: list[dict] | None = None,
) -> None:
    """写 madmom chord 输出 JSON。"""
    if entries is None:
        entries = [
            {"start": 0.0, "end": 2.0, "label": "C:maj"},
            {"start": 2.0, "end": 4.0, "label": "F:maj"},
            {"start": 4.0, "end": 4.5, "label": "N"},
            {"start": 4.5, "end": 5.0, "label": "G:min"},
        ]
    json_path.write_text(json.dumps(entries), encoding="utf-8")


class TestCheckAvailable:
    def test_madmom_importable(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            assert check_madmom_chord_available() is True

    def test_madmom_missing(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1)
            assert check_madmom_chord_available() is False

    def test_subprocess_timeout(self):
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="x", timeout=30)):
            assert check_madmom_chord_available() is False

    def test_filenotfound(self):
        with patch("subprocess.run", side_effect=FileNotFoundError):
            assert check_madmom_chord_available() is False


class TestParseLabel:
    """v0.4.4: _parse_madmom_chord_label() 解析逻辑。"""

    def test_major_chord(self):
        c = _parse_madmom_chord_label("C:maj")
        assert c is not None
        assert c.root == "C"
        assert c.quality == ""
        assert c.start == 0.0
        assert c.end == 0.0

    def test_minor_chord(self):
        c = _parse_madmom_chord_label("F#:min")
        assert c is not None
        assert c.root == "F#"
        assert c.quality == "m"

    def test_sharp_root(self):
        c = _parse_madmom_chord_label("F#:maj")
        assert c is not None
        assert c.root == "F#"
        assert c.quality == ""

    def test_flat_root(self):
        c = _parse_madmom_chord_label("Bb:maj")
        assert c is not None
        assert c.root == "Bb"
        assert c.quality == ""

    def test_skip_no_chord(self):
        """v0.4.4: 'N' → None（无和弦）"""
        assert _parse_madmom_chord_label("N") is None

    def test_skip_unknown(self):
        """v0.4.4: 'X' → None（未知）"""
        assert _parse_madmom_chord_label("X") is None

    def test_skip_malformed_no_colon(self):
        assert _parse_madmom_chord_label("invalid") is None

    def test_skip_empty(self):
        assert _parse_madmom_chord_label("") is None

    def test_whitespace_stripped(self):
        c = _parse_madmom_chord_label("  C : maj  ")
        assert c is not None
        assert c.root == "C"
        assert c.quality == ""

    def test_quality_normalization(self):
        """maj → ''，min → 'm'，其他 → 透传。"""
        assert _parse_madmom_chord_label("C:maj").quality == ""
        assert _parse_madmom_chord_label("C:M").quality == ""
        assert _parse_madmom_chord_label("C:major").quality == ""
        assert _parse_madmom_chord_label("C:min").quality == "m"
        assert _parse_madmom_chord_label("C:m").quality == "m"
        assert _parse_madmom_chord_label("C:minor").quality == "m"
        # 其他 quality 透传（v0.4.4 不会触发，CRNN 模型只输出 maj/min）
        c = _parse_madmom_chord_label("C:7")
        assert c is not None
        assert c.quality == "7"


class TestDetectChords:
    """v0.4.4: detect_chords_with_madmom() subprocess 集成。"""

    def test_default_timeout_constant(self):
        """v0.4.4: 默认 30 分钟超时。"""
        assert MADMOM_CHORD_TIMEOUT_DEFAULT == 1800

    def test_audio_not_found(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="audio not found"):
            detect_chords_with_madmom(tmp_path / "missing.wav")

    def test_successful_detection(self, tmp_path: Path):
        """v0.4.4: 成功路径，mock subprocess。"""
        audio = tmp_path / "song.wav"
        audio.write_bytes(b"RIFF" * 100)
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        json_path = out_dir / f"chords_{audio.stem}.json"
        _write_fake_chord_json(json_path)

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="")
            chord_track = detect_chords_with_madmom(audio, out_dir=out_dir)

        # 4 entries → 3 ChordEvent（N 已过滤）
        assert len(chord_track) == 3
        assert chord_track[0] == ChordEvent(0.0, 2.0, "C", "")
        assert chord_track[1] == ChordEvent(2.0, 4.0, "F", "")
        assert chord_track[2] == ChordEvent(4.5, 5.0, "G", "m")

    def test_filters_out_no_chord(self, tmp_path: Path):
        """v0.4.4: 全部 N/X → 空列表（不抛错）。"""
        audio = tmp_path / "song.wav"
        audio.write_bytes(b"RIFF" * 100)
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        json_path = out_dir / f"chords_{audio.stem}.json"
        _write_fake_chord_json(json_path, entries=[
            {"start": 0.0, "end": 1.0, "label": "N"},
            {"start": 1.0, "end": 2.0, "label": "X"},
        ])

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="")
            chord_track = detect_chords_with_madmom(audio, out_dir=out_dir)

        assert chord_track == []

    def test_subprocess_failure_raises(self, tmp_path: Path):
        audio = tmp_path / "song.wav"
        audio.write_bytes(b"RIFF" * 100)

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=3, stderr="madmom not installed",
            )
            with pytest.raises(MadmomChordAdapterError, match="madmom chord failed"):
                detect_chords_with_madmom(audio)

    def test_subprocess_timeout_raises(self, tmp_path: Path):
        audio = tmp_path / "song.wav"
        audio.write_bytes(b"RIFF" * 100)

        with patch("subprocess.run",
                   side_effect=subprocess.TimeoutExpired(cmd="x", timeout=1800)):
            with pytest.raises(MadmomChordAdapterError, match="timeout"):
                detect_chords_with_madmom(audio, out_dir=tmp_path / "out")

    def test_missing_output_json_raises(self, tmp_path: Path):
        audio = tmp_path / "song.wav"
        audio.write_bytes(b"RIFF" * 100)
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        # 不写 chords JSON

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="")
            with pytest.raises(MadmomChordAdapterError, match="output json not found"):
                detect_chords_with_madmom(audio, out_dir=out_dir)

    def test_invalid_json_raises(self, tmp_path: Path):
        audio = tmp_path / "song.wav"
        audio.write_bytes(b"RIFF" * 100)
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        json_path = out_dir / f"chords_{audio.stem}.json"
        json_path.write_text("not valid json", encoding="utf-8")

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="")
            with pytest.raises(MadmomChordAdapterError, match="parse madmom chord"):
                detect_chords_with_madmom(audio, out_dir=out_dir)

    def test_malformed_entry_skipped(self, tmp_path: Path):
        """v0.4.4: 个别 entry 字段缺失 → 跳过该项，不抛错。"""
        audio = tmp_path / "song.wav"
        audio.write_bytes(b"RIFF" * 100)
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        json_path = out_dir / f"chords_{audio.stem}.json"
        # 一条正常 + 一条缺 label
        _write_fake_chord_json(json_path, entries=[
            {"start": 0.0, "end": 2.0, "label": "C:maj"},
            {"start": 2.0, "end": 4.0},  # missing 'label'
        ])

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="")
            chord_track = detect_chords_with_madmom(audio, out_dir=out_dir)

        # 只有第 1 条被解析
        assert len(chord_track) == 1
        assert chord_track[0].root == "C"

    def test_default_out_dir_creates_tmp(self, tmp_path: Path):
        """v0.4.4: out_dir=None → 写到 tmpdir。"""
        audio = tmp_path / "song.wav"
        audio.write_bytes(b"RIFF" * 100)

        # mock tempfile.mkdtemp + 直接控制 wrapper 路径
        tmp_chord_dir = tmp_path / "chord_tmp"
        tmp_chord_dir.mkdir()

        def fake_run(cmd, **kwargs):
            # 第 4 个 arg 是 output json path
            json_path = Path(cmd[3])
            json_path.parent.mkdir(parents=True, exist_ok=True)
            _write_fake_chord_json(json_path)
            return MagicMock(returncode=0, stderr="")

        with patch("subprocess.run", side_effect=fake_run), \
             patch("mujik.chord.madmom_adapter.tempfile.mkdtemp",
                   return_value=str(tmp_chord_dir)):
            chord_track = detect_chords_with_madmom(audio)
        assert len(chord_track) == 3
