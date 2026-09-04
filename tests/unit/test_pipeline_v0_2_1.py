"""Tests for Pipeline v0.2.1 (mocked all heavy adapters)."""
from __future__ import annotations

import csv
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from mujik.config.schema import LoudnormConfig, PipelineConfig
from mujik.midi.model import Note
from mujik.pipeline import Pipeline
from mujik.separate.model import Stem, Stems


def _fake_stems(out_dir: Path, input_stem: str = "song") -> Stems:
    """构造 4-stem Stems 对象，audio_path 是占位文件。"""
    for name in ("vocals", "drums", "bass", "other"):
        p = out_dir / "htdemucs_ft" / input_stem / f"{name}.wav"
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
            audio_path=out_dir / "htdemucs_ft" / input_stem / f"{name}.wav",
            sample_rate=44100,
            duration=5.0,
            source_model="demucs/htdemucs_ft",
        ))
    return s


def _mock_separate_with_demucs(input_path, out_dir, config=None):
    return _fake_stems(Path(out_dir), input_path.stem)


class TestPipelineWiring:
    def test_input_not_found(self, tmp_path: Path):
        cfg = PipelineConfig(
            input_path=str(tmp_path / "missing.wav"),
            output_dir=str(tmp_path / "out"),
        )
        pipeline = Pipeline(cfg)
        with pytest.raises(FileNotFoundError):
            pipeline.run()

    def test_creates_output_dir(self, tmp_path: Path):
        audio = tmp_path / "song.wav"
        audio.write_bytes(b"RIFF" * 100)
        cfg = PipelineConfig(
            input_path=str(audio),
            output_dir=str(tmp_path / "deep" / "nest" / "out"),
            loudnorm=LoudnormConfig(enabled=False),
        )

        with patch("mujik.separate.demucs_adapter.separate_with_demucs",
                   side_effect=_mock_separate_with_demucs), \
             patch("mujik.pipeline.transcribe_stem", return_value=[]), \
             patch("soundfile.info") as mock_info:
            mock_info.return_value = MagicMock(duration=5.0, samplerate=44100)
            pipeline = Pipeline(cfg)
            project = pipeline.run()

        assert (tmp_path / "deep" / "nest" / "out").exists()
        assert (tmp_path / "deep" / "nest" / "out" / "project.mid").exists()
        assert (tmp_path / "deep" / "nest" / "out" / "project.json").exists()

    def test_loudnorm_called_when_enabled(self, tmp_path: Path):
        audio = tmp_path / "song.wav"
        audio.write_bytes(b"RIFF" * 100)
        cfg = PipelineConfig(
            input_path=str(audio),
            output_dir=str(tmp_path / "out"),
        )

        with patch("mujik.pipeline.normalize_loudness") as mock_ln, \
             patch("mujik.separate.demucs_adapter.separate_with_demucs",
                   side_effect=_mock_separate_with_demucs), \
             patch("mujik.pipeline.transcribe_stem", return_value=[]), \
             patch("soundfile.info") as mock_info:
            mock_info.return_value = MagicMock(duration=5.0, samplerate=44100)
            mock_ln.return_value = audio  # loudnorm 返回原路径（mock）
            Pipeline(cfg).run()

        assert mock_ln.called

    def test_loudnorm_skipped_when_disabled(self, tmp_path: Path):
        audio = tmp_path / "song.wav"
        audio.write_bytes(b"RIFF" * 100)
        cfg = PipelineConfig(
            input_path=str(audio),
            output_dir=str(tmp_path / "out"),
            loudnorm=LoudnormConfig(enabled=False),
        )

        with patch("mujik.pipeline.normalize_loudness") as mock_ln, \
             patch("mujik.separate.demucs_adapter.separate_with_demucs",
                   side_effect=_mock_separate_with_demucs), \
             patch("mujik.pipeline.transcribe_stem", return_value=[]), \
             patch("soundfile.info") as mock_info:
            mock_info.return_value = MagicMock(duration=5.0, samplerate=44100)
            Pipeline(cfg).run()

        mock_ln.assert_not_called()


class TestPipelineTranscribe:
    def test_notes_flow_into_project(self, tmp_path: Path):
        audio = tmp_path / "song.wav"
        audio.write_bytes(b"RIFF" * 100)
        cfg = PipelineConfig(
            input_path=str(audio),
            output_dir=str(tmp_path / "out"),
            loudnorm=LoudnormConfig(enabled=False),
        )

        # 转录返回不同 stem 不同 note
        def fake_transcribe_stem(stem, config=None, out_dir=None):
            if stem.name == "vocals":
                return [Note(0.0, 1.0, 60, 100), Note(1.0, 2.0, 62, 80)]
            if stem.name == "drums":
                return [Note(0.0, 0.1, 36, 100)]
            return []

        with patch("mujik.separate.demucs_adapter.separate_with_demucs",
                   side_effect=_mock_separate_with_demucs), \
             patch("mujik.pipeline.transcribe_stem",
                   side_effect=fake_transcribe_stem), \
             patch("soundfile.info") as mock_info:
            mock_info.return_value = MagicMock(duration=5.0, samplerate=44100)
            project = Pipeline(cfg).run()

        # vocals: 2 notes, drums: 1
        assert len(project.tracks["vocals"].notes) == 2
        assert len(project.tracks["drums"].notes) == 1
        assert project.total_notes() == 3

    def test_transcribe_failure_does_not_stop_pipeline(self, tmp_path: Path):
        audio = tmp_path / "song.wav"
        audio.write_bytes(b"RIFF" * 100)
        cfg = PipelineConfig(
            input_path=str(audio),
            output_dir=str(tmp_path / "out"),
            loudnorm=LoudnormConfig(enabled=False),
        )

        def fake_transcribe(stem, config=None, out_dir=None):
            if stem.name == "vocals":
                raise RuntimeError("vocals failed")
            if stem.name == "drums":
                return [Note(0.0, 0.1, 36, 100)]
            return []

        with patch("mujik.separate.demucs_adapter.separate_with_demucs",
                   side_effect=_mock_separate_with_demucs), \
             patch("mujik.pipeline.transcribe_stem",
                   side_effect=fake_transcribe), \
             patch("soundfile.info") as mock_info:
            mock_info.return_value = MagicMock(duration=5.0, samplerate=44100)
            project = Pipeline(cfg).run()

        # vocals 失败被跳过，drums 还在
        assert "drums" in project.tracks
        assert len(project.tracks["drums"].notes) == 1
        # vocals 不在 project.tracks
        assert "vocals" not in project.tracks or len(
            project.tracks["vocals"].notes
        ) == 0


class TestProjectMetadata:
    def test_metadata_includes_version_and_separator(self, tmp_path: Path):
        audio = tmp_path / "song.wav"
        audio.write_bytes(b"RIFF" * 100)
        cfg = PipelineConfig(
            input_path=str(audio),
            output_dir=str(tmp_path / "out"),
            loudnorm=LoudnormConfig(enabled=False),
            preset="pop",
        )

        with patch("mujik.separate.demucs_adapter.separate_with_demucs",
                   side_effect=_mock_separate_with_demucs), \
             patch("mujik.pipeline.transcribe_stem", return_value=[]), \
             patch("soundfile.info") as mock_info:
            mock_info.return_value = MagicMock(duration=5.0, samplerate=44100)
            Pipeline(cfg).run()

        import json
        meta = json.loads((tmp_path / "out" / "project.json").read_text())
        assert meta["mujik_version"] in ("0.2.1", "0.2.2", "0.4.0", "0.4.1", "0.4.2", "0.4.3", "0.4.4", "0.4.5", "0.4.6", "0.4.7", "0.4.8", "0.4.9", "0.5.0", "0.5.1")
        assert meta["preset"] == "pop"
        assert "demucs" in meta["separator"]
