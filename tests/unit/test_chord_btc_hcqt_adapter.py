"""Tests for chord/btc_hcqt_adapter.py (v0.4.8, mocked subprocess + label parsing)."""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

import mujik.chord.btc_hcqt_adapter

from mujik.midi.model import ChordEvent
from mujik.chord.btc_hcqt_adapter import (
    BTC_HCQT_TIMEOUT_DEFAULT,
    BtcHcqtAdapterError,
    _parse_btc_chord_label,
    check_btc_hcqt_available,
    detect_chords_with_btc,
)


def _write_fake_btc_json(
    json_path: Path,
    entries: list[dict] | None = None,
) -> None:
    """写 BTC 输出 JSON。"""
    if entries is None:
        entries = [
            {"start": 0.0, "end": 2.0, "label": "C"},
            {"start": 2.0, "end": 4.0, "label": "F:min7"},
            {"start": 4.0, "end": 4.5, "label": "N"},
            {"start": 4.5, "end": 5.0, "label": "G:maj7"},
        ]
    json_path.write_text(json.dumps(entries), encoding="utf-8")


class TestCheckAvailable:
    def test_available(self):
        with patch.dict("sys.modules", {
            "torch": MagicMock(), "librosa": MagicMock(),
        }):
            assert check_btc_hcqt_available() is True

    def test_torch_missing(self):
        with patch.dict("sys.modules", {
            "torch": None, "librosa": MagicMock(),
        }):
            assert check_btc_hcqt_available() is False

    def test_librosa_missing(self):
        with patch.dict("sys.modules", {
            "torch": MagicMock(), "librosa": None,
        }):
            assert check_btc_hcqt_available() is False


class TestParseLabel:
    """v0.4.8: _parse_btc_chord_label() 解析逻辑。"""

    def test_bare_root_means_major(self):
        """v0.4.8: BTC bare root（无 :）表示 maj。"""
        c = _parse_btc_chord_label("C")
        assert c is not None
        assert c.root == "C"
        assert c.quality == ""

    def test_min(self):
        c = _parse_btc_chord_label("C:min")
        assert c.root == "C"
        assert c.quality == "m"

    def test_maj7(self):
        c = _parse_btc_chord_label("C:maj7")
        assert c.root == "C"
        assert c.quality == "maj7"

    def test_dominant_7(self):
        c = _parse_btc_chord_label("C:7")
        assert c.root == "C"
        assert c.quality == "7"

    def test_min7(self):
        c = _parse_btc_chord_label("F#:min7")
        assert c.root == "F#"
        assert c.quality == "m7"

    def test_dim(self):
        # v0.4.8: BTC 用 # 不用 b；flat 用 enharmonic 替代
        c = _parse_btc_chord_label("A#:dim")
        assert c.root == "A#"
        assert c.quality == "dim"

    def test_aug(self):
        c = _parse_btc_chord_label("C:aug")
        assert c.quality == "aug"

    def test_sus2(self):
        c = _parse_btc_chord_label("C:sus2")
        assert c.quality == "sus2"

    def test_sus4(self):
        c = _parse_btc_chord_label("C:sus4")
        assert c.quality == "sus4"

    def test_dim7(self):
        """v0.4.8: BTC 独有 dim7。"""
        c = _parse_btc_chord_label("C:dim7")
        assert c.quality == "dim7"

    def test_hdim7(self):
        """v0.4.8: BTC 独有 hdim7（half-diminished）。"""
        c = _parse_btc_chord_label("B:hdim7")
        assert c.root == "B"
        assert c.quality == "hdim7"

    def test_minmaj7(self):
        """v0.4.8: BTC minmaj7 → mujik mM7。"""
        c = _parse_btc_chord_label("C:minmaj7")
        assert c.quality == "mM7"

    def test_min6_maj6(self):
        """v0.4.8: BTC min6/maj6 → m6/maj6。"""
        assert _parse_btc_chord_label("C:min6").quality == "m6"
        assert _parse_btc_chord_label("C:maj6").quality == "maj6"

    def test_skip_N_X(self):
        assert _parse_btc_chord_label("N") is None
        assert _parse_btc_chord_label("X") is None

    def test_skip_empty(self):
        assert _parse_btc_chord_label("") is None

    def test_skip_invalid_root_with_b(self):
        """v0.4.8: BTC 不用 b（flat），用 # 不用 b。"""
        # "Db" 不是 BTC 合法 root（BTC 用 C# 不用 Db）
        assert _parse_btc_chord_label("Db:min") is None

    def test_skip_malformed_no_colon_invalid_root(self):
        assert _parse_btc_chord_label("Xfoo") is None

    def test_whitespace_stripped(self):
        c = _parse_btc_chord_label("  C : maj7  ")
        assert c.root == "C"
        assert c.quality == "maj7"


class TestDetectChords:
    """v0.4.8: detect_chords_with_btc() subprocess 集成。"""

    def test_default_timeout_constant(self):
        assert BTC_HCQT_TIMEOUT_DEFAULT == 1800

    def test_audio_not_found(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="audio not found"):
            detect_chords_with_btc(tmp_path / "missing.wav")

    def test_successful_detection_large_voca(self, tmp_path: Path):
        """v0.4.8: 成功路径，mock subprocess + 4 entries → 3 ChordEvent。"""
        audio = tmp_path / "song.wav"
        audio.write_bytes(b"RIFF" * 100)
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        json_path = out_dir / f"btc_chords_{audio.stem}.json"
        _write_fake_btc_json(json_path)

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="")
            chord_track = detect_chords_with_btc(audio, out_dir=out_dir)

        # 4 entries → 3 ChordEvent (N 过滤)
        assert len(chord_track) == 3
        # v0.4.8: BTC 输出用 btc-extended vocab
        assert chord_track[0] == ChordEvent(0.0, 2.0, "C", "", vocab="btc-extended")
        assert chord_track[1] == ChordEvent(2.0, 4.0, "F", "m7", vocab="btc-extended")
        assert chord_track[2] == ChordEvent(4.5, 5.0, "G", "maj7", vocab="btc-extended")

    def test_filters_out_N_X(self, tmp_path: Path):
        audio = tmp_path / "song.wav"
        audio.write_bytes(b"RIFF" * 100)
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        json_path = out_dir / f"btc_chords_{audio.stem}.json"
        _write_fake_btc_json(json_path, entries=[
            {"start": 0.0, "end": 1.0, "label": "N"},
            {"start": 1.0, "end": 2.0, "label": "X"},
        ])

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="")
            chord_track = detect_chords_with_btc(audio, out_dir=out_dir)
        assert chord_track == []

    def test_subprocess_failure_raises(self, tmp_path: Path):
        audio = tmp_path / "song.wav"
        audio.write_bytes(b"RIFF" * 100)

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=3, stderr="torch not installed",
            )
            with pytest.raises(BtcHcqtAdapterError, match="btc-hcqt chord failed"):
                detect_chords_with_btc(audio)

    def test_subprocess_timeout_raises(self, tmp_path: Path):
        audio = tmp_path / "song.wav"
        audio.write_bytes(b"RIFF" * 100)

        with patch("subprocess.run",
                   side_effect=subprocess.TimeoutExpired(cmd="x", timeout=1800)):
            with pytest.raises(BtcHcqtAdapterError, match="timeout"):
                detect_chords_with_btc(audio, out_dir=tmp_path / "out")

    def test_missing_output_json_raises(self, tmp_path: Path):
        audio = tmp_path / "song.wav"
        audio.write_bytes(b"RIFF" * 100)
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="")
            with pytest.raises(BtcHcqtAdapterError, match="output json not found"):
                detect_chords_with_btc(audio, out_dir=out_dir)

    def test_invalid_json_raises(self, tmp_path: Path):
        audio = tmp_path / "song.wav"
        audio.write_bytes(b"RIFF" * 100)
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        json_path = out_dir / f"btc_chords_{audio.stem}.json"
        json_path.write_text("not valid json", encoding="utf-8")

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="")
            with pytest.raises(BtcHcqtAdapterError, match="parse btc-hcqt"):
                detect_chords_with_btc(audio, out_dir=out_dir)

    def test_malformed_entry_skipped(self, tmp_path: Path):
        audio = tmp_path / "song.wav"
        audio.write_bytes(b"RIFF" * 100)
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        json_path = out_dir / f"btc_chords_{audio.stem}.json"
        _write_fake_btc_json(json_path, entries=[
            {"start": 0.0, "end": 2.0, "label": "C:maj7"},
            {"start": 2.0, "end": 4.0},  # missing 'label'
        ])

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="")
            chord_track = detect_chords_with_btc(audio, out_dir=out_dir)
        assert len(chord_track) == 1
        assert chord_track[0].quality == "maj7"

    def test_default_out_dir_creates_tmp(self, tmp_path: Path):
        """v0.4.8: out_dir=None → 写到 tmpdir。"""
        audio = tmp_path / "song.wav"
        audio.write_bytes(b"RIFF" * 100)

        tmp_chord_dir = tmp_path / "btc_chord_tmp"
        tmp_chord_dir.mkdir()

        def fake_run(cmd, **kwargs):
            # cmd layout: [python, wrapper, audio, json, model, voca]
            json_path = Path(cmd[3])
            json_path.parent.mkdir(parents=True, exist_ok=True)
            _write_fake_btc_json(json_path)
            return MagicMock(returncode=0, stderr="")

        with patch("subprocess.run", side_effect=fake_run), \
             patch("mujik.chord.btc_hcqt_adapter.tempfile.mkdtemp",
                   return_value=str(tmp_chord_dir)):
            chord_track = detect_chords_with_btc(audio)
        assert len(chord_track) == 3

    def test_pass_model_path_and_voca(self, tmp_path: Path):
        """v0.4.8: model_path + voca 参数透传到 wrapper。"""
        audio = tmp_path / "song.wav"
        audio.write_bytes(b"RIFF" * 100)
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        class FakeConfig:
            btc_model_path = "/models/btc_large.pt"
            btc_voca = "large"
            btc_timeout_sec = 600

        json_path = out_dir / f"btc_chords_{audio.stem}.json"
        _write_fake_btc_json(json_path)

        captured_cmd = []
        def fake_run(cmd, **kwargs):
            captured_cmd.extend(cmd)
            return MagicMock(returncode=0, stderr="")

        with patch("subprocess.run", side_effect=fake_run):
            detect_chords_with_btc(audio, config=FakeConfig(), out_dir=out_dir)

        # cmd[4] should be model_path, cmd[5] should be voca
        assert captured_cmd[4] == "/models/btc_large.pt"
        assert captured_cmd[5] == "large"


class TestVendorDirAndModelPath:
    """v0.5.2: vendored _btc/ env 注入 + 权重路径解析顺序。"""

    @pytest.fixture(autouse=True)
    def _clean_env(self):
        """隔离 MUJIK_BTC_MODEL / BTC_ISMIR19_PATH，避免宿主机环境污染。"""
        with patch.dict(os.environ):
            os.environ.pop("MUJIK_BTC_MODEL", None)
            os.environ.pop("BTC_ISMIR19_PATH", None)
            yield

    def _make_audio(self, tmp_path: Path) -> Path:
        audio = tmp_path / "song.wav"
        audio.write_bytes(b"RIFF" * 100)
        return audio

    @staticmethod
    def _fake_run_capture(cmd_log: list, env_log: dict):
        """fake subprocess.run：记录 cmd/env，并在 json 输出路径写合法结果。"""
        def fake_run(cmd, **kwargs):
            cmd_log.extend(cmd)
            env_log.update(kwargs.get("env") or {})
            json_path = Path(cmd[3])
            json_path.parent.mkdir(parents=True, exist_ok=True)
            _write_fake_btc_json(json_path)
            return MagicMock(returncode=0, stderr="")
        return fake_run

    def test_vendor_dir_injected_when_exists(self, tmp_path: Path):
        """包内 _btc/ 存在时，BTC_ISMIR19_PATH 注入 subprocess env。"""
        audio = self._make_audio(tmp_path)

        vendor_dir = Path(mujik.chord.btc_hcqt_adapter.__file__).resolve().parent / "_btc"

        captured_cmd, captured_env = [], {}
        with patch("subprocess.run",
                   side_effect=self._fake_run_capture(captured_cmd, captured_env)):
            detect_chords_with_btc(audio)

        if vendor_dir.is_dir():
            assert captured_env.get("BTC_ISMIR19_PATH") == str(vendor_dir)
        else:
            # 未 vendor 时不注入（但也不报错，交给 wrapper exit 4 fail-loud）
            assert "BTC_ISMIR19_PATH" not in captured_env

    def test_user_env_not_overridden(self, tmp_path: Path):
        """用户显式设置的 BTC_ISMIR19_PATH 优先，不被 vendor 目录覆盖。"""
        audio = self._make_audio(tmp_path)
        os.environ["BTC_ISMIR19_PATH"] = "/custom/btc"

        captured_cmd, captured_env = [], {}
        with patch("subprocess.run",
                   side_effect=self._fake_run_capture(captured_cmd, captured_env)):
            detect_chords_with_btc(audio)

        assert captured_env.get("BTC_ISMIR19_PATH") == "/custom/btc"

    def test_model_path_from_env(self, tmp_path: Path):
        """config.btc_model_path 缺省时回退 env MUJIK_BTC_MODEL。"""
        audio = self._make_audio(tmp_path)
        os.environ["MUJIK_BTC_MODEL"] = "/app/models/btc_model_large_voca.pt"

        captured_cmd, captured_env = [], {}
        with patch("subprocess.run",
                   side_effect=self._fake_run_capture(captured_cmd, captured_env)):
            detect_chords_with_btc(audio)

        assert captured_cmd[4] == "/app/models/btc_model_large_voca.pt"

    def test_config_model_path_wins_over_env(self, tmp_path: Path):
        """显式 config.btc_model_path 优先于 env MUJIK_BTC_MODEL。"""
        audio = self._make_audio(tmp_path)
        os.environ["MUJIK_BTC_MODEL"] = "/app/models/btc_model_large_voca.pt"

        class FakeConfig:
            btc_model_path = "/explicit/model.pt"
            btc_voca = "large"
            btc_timeout_sec = 600

        captured_cmd, captured_env = [], {}
        with patch("subprocess.run",
                   side_effect=self._fake_run_capture(captured_cmd, captured_env)):
            detect_chords_with_btc(audio, config=FakeConfig())

        assert captured_cmd[4] == "/explicit/model.pt"

    def test_no_model_path_no_env_passes_empty(self, tmp_path: Path):
        """两者都没有 → cmd 传空串（wrapper exit 5 fail-loud）。"""
        audio = self._make_audio(tmp_path)

        captured_cmd, captured_env = [], {}
        with patch("subprocess.run",
                   side_effect=self._fake_run_capture(captured_cmd, captured_env)):
            detect_chords_with_btc(audio)

        assert captured_cmd[4] == ""


class TestQualityMapping:
    """v0.4.8: BTC quality → mujik quality 映射完整性。"""

    def test_all_14_btc_qualities_mapped(self):
        """v0.4.8: BTC 14 种 quality 全部有映射（即使 passthrough）。"""
        for btc_q, _mujik_q in [
            ("maj", ""), ("min", "m"),
            ("dim", "dim"), ("aug", "aug"),
            ("min6", "m6"), ("maj6", "maj6"),
            ("min7", "m7"), ("maj7", "maj7"),
            ("minmaj7", "mM7"),
            ("7", "7"),
            ("dim7", "dim7"), ("hdim7", "hdim7"),
            ("sus2", "sus2"), ("sus4", "sus4"),
        ]:
            c = _parse_btc_chord_label(f"C:{btc_q}")
            assert c is not None, f"BTC {btc_q} parse failed"
            assert c.root == "C"
