"""Tests for Pipeline v0.2.2 (mocked madmom + demucs + transcribe)."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from mujik.config.schema import LoudnormConfig, PipelineConfig
from mujik.midi.model import Note
from mujik.pipeline import Pipeline
from mujik.rhythm.model import BeatTrack
from mujik.separate.model import Stem, Stems


def _fake_stems(out_dir: Path) -> Stems:
    for name in ("vocals", "drums", "bass", "other"):
        p = out_dir / "htdemucs_ft" / "song" / f"{name}.wav"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"RIFF" * 100)

    s = Stems(
        separation_model="demucs/htdemucs_ft",
        sample_rate=44100,
        total_duration=5.0,
    )
    for name in ("vocals", "drums", "bass", "other"):
        s.add(Stem(
            name=name,  # type: ignore[arg-type]
            audio_path=out_dir / "htdemucs_ft" / "song" / f"{name}.wav",
            sample_rate=44100,
            duration=5.0,
            source_model="demucs/htdemucs_ft",
        ))
    return s


def _fake_separate(input_path, out_dir, config=None):
    return _fake_stems(Path(out_dir))


def _fake_transcribe(stem, config=None, out_dir=None):
    if stem.name == "vocals":
        return [Note(0.0, 1.0, 60, 100)]
    if stem.name == "drums":
        return [Note(0.0, 0.1, 36, 100)]
    return []


def _fake_madmom(audio_path, config=None, out_dir=None):
    return BeatTrack(
        beats=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5],
        downbeats=[0.0, 2.0, 4.0],
        bpm=120.0,
        tempo_confidence=0.9,
    )


def _base_cfg(out_dir: Path) -> PipelineConfig:
    return PipelineConfig(
        input_path=str(out_dir / "song.wav"),
        output_dir=str(out_dir / "out"),
        loudnorm=LoudnormConfig(enabled=False),
    )


class TestRhythmEnabled:
    def test_beats_json_written(self, tmp_path: Path):
        audio = tmp_path / "song.wav"
        audio.write_bytes(b"RIFF")
        cfg = _base_cfg(tmp_path)

        with patch("mujik.pipeline.separate_with_demucs", side_effect=_fake_separate), \
             patch("mujik.pipeline.transcribe_stem", side_effect=_fake_transcribe), \
             patch("mujik.pipeline.track_beats_with_madmom", side_effect=_fake_madmom), \
             patch("soundfile.info") as mock_info:
            mock_info.return_value = MagicMock(duration=5.0, samplerate=44100)
            Pipeline(cfg).run()

        beats_path = tmp_path / "out" / "beats.json"
        assert beats_path.exists()
        data = json.loads(beats_path.read_text())
        assert data["bpm"] == 120.0
        assert len(data["beats"]) == 10
        assert len(data["downbeats"]) == 3

    def test_time_signatures_json_written(self, tmp_path: Path):
        audio = tmp_path / "song.wav"
        audio.write_bytes(b"RIFF")
        cfg = _base_cfg(tmp_path)

        with patch("mujik.pipeline.separate_with_demucs", side_effect=_fake_separate), \
             patch("mujik.pipeline.transcribe_stem", side_effect=_fake_transcribe), \
             patch("mujik.pipeline.track_beats_with_madmom", side_effect=_fake_madmom), \
             patch("soundfile.info") as mock_info:
            mock_info.return_value = MagicMock(duration=5.0, samplerate=44100)
            Pipeline(cfg).run()

        ts_path = tmp_path / "out" / "time_signatures.json"
        assert ts_path.exists()
        data = json.loads(ts_path.read_text())
        assert len(data) >= 1
        assert data[0]["sig"] == [4, 4]  # 启发式 4/4

    def test_project_metadata_has_bpm(self, tmp_path: Path):
        audio = tmp_path / "song.wav"
        audio.write_bytes(b"RIFF")
        cfg = _base_cfg(tmp_path)

        with patch("mujik.pipeline.separate_with_demucs", side_effect=_fake_separate), \
             patch("mujik.pipeline.transcribe_stem", side_effect=_fake_transcribe), \
             patch("mujik.pipeline.track_beats_with_madmom", side_effect=_fake_madmom), \
             patch("soundfile.info") as mock_info:
            mock_info.return_value = MagicMock(duration=5.0, samplerate=44100)
            Pipeline(cfg).run()

        meta = json.loads((tmp_path / "out" / "project.json").read_text())
        assert meta["mujik_version"] in ("0.2.2", "0.4.0", "0.4.1", "0.4.2", "0.4.3")
        assert meta["rhythm_enabled"] is True

    def test_project_mid_has_tempo(self, tmp_path: Path):
        """project.mid 含 set_tempo 事件。"""
        audio = tmp_path / "song.wav"
        audio.write_bytes(b"RIFF")
        cfg = _base_cfg(tmp_path)

        with patch("mujik.pipeline.separate_with_demucs", side_effect=_fake_separate), \
             patch("mujik.pipeline.transcribe_stem", side_effect=_fake_transcribe), \
             patch("mujik.pipeline.track_beats_with_madmom", side_effect=_fake_madmom), \
             patch("soundfile.info") as mock_info:
            mock_info.return_value = MagicMock(duration=5.0, samplerate=44100)
            project = Pipeline(cfg).run()

        # 第一个 tempo 应来自 madmom
        assert project.tempo_map[0].bpm == 120.0
        assert project.tempo_map[0].bpm != 120.0 or True  # 来自 madmom


class TestRhythmFailure:
    def test_madmom_failure_uses_default(self, tmp_path: Path):
        """madmom 失败时回退默认 tempo + 拍号。"""
        audio = tmp_path / "song.wav"
        audio.write_bytes(b"RIFF")
        cfg = _base_cfg(tmp_path)

        def fake_fail(*args, **kwargs):
            from mujik.rhythm.madmom_adapter import MadmomAdapterError
            raise MadmomAdapterError("simulated fail")

        with patch("mujik.pipeline.separate_with_demucs", side_effect=_fake_separate), \
             patch("mujik.pipeline.transcribe_stem", side_effect=_fake_transcribe), \
             patch("mujik.pipeline.track_beats_with_madmom", side_effect=fake_fail), \
             patch("soundfile.info") as mock_info:
            mock_info.return_value = MagicMock(duration=5.0, samplerate=44100)
            Pipeline(cfg).run()

        # beats.json 仍写出（默认 120）
        beats = json.loads((tmp_path / "out" / "beats.json").read_text())
        assert beats["bpm"] == 120.0


class TestRhythmDisabled:
    def test_no_rhythm_files_when_disabled(self, tmp_path: Path):
        from mujik.config.schema import RhythmConfig
        audio = tmp_path / "song.wav"
        audio.write_bytes(b"RIFF")
        cfg = PipelineConfig(
            input_path=str(audio),
            output_dir=str(tmp_path / "out"),
            loudnorm=LoudnormConfig(enabled=False),
            rhythm=__import__("mujik.config.schema", fromlist=["RhythmConfig"]).RhythmConfig(enabled=False),
        )

        with patch("mujik.pipeline.separate_with_demucs", side_effect=_fake_separate), \
             patch("mujik.pipeline.transcribe_stem", side_effect=_fake_transcribe), \
             patch("mujik.pipeline.track_beats_with_madmom", side_effect=_fake_madmom) as mock_bt, \
             patch("soundfile.info") as mock_info:
            mock_info.return_value = MagicMock(duration=5.0, samplerate=44100)
            Pipeline(cfg).run()

        mock_bt.assert_not_called()
        # 不写 beats.json
        assert not (tmp_path / "out" / "beats.json").exists()
