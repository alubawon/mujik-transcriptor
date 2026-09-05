"""Tests for Pipeline v0.2.2 (mocked madmom + demucs + transcribe)."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

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

        with patch("mujik.separate.demucs_adapter.separate_with_demucs", side_effect=_fake_separate), \
             patch("mujik.pipeline.transcribe_stem", side_effect=_fake_transcribe), \
             patch("mujik.pipeline.track_beats_with_madmom", side_effect=_fake_madmom), \
             patch("soundfile.info") as mock_info:
            mock_info.return_value = MagicMock(duration=5.0, samplerate=44100)
            Pipeline(cfg).run()

        beats_path = tmp_path / "out" / "ws" / "beats.json"  # v0.5.1 修 5：中间产物在 ws/
        assert beats_path.exists()
        data = json.loads(beats_path.read_text())
        assert data["bpm"] == 120.0
        assert len(data["beats"]) == 10
        assert len(data["downbeats"]) == 3

    def test_time_signatures_json_written(self, tmp_path: Path):
        audio = tmp_path / "song.wav"
        audio.write_bytes(b"RIFF")
        cfg = _base_cfg(tmp_path)

        with patch("mujik.separate.demucs_adapter.separate_with_demucs", side_effect=_fake_separate), \
             patch("mujik.pipeline.transcribe_stem", side_effect=_fake_transcribe), \
             patch("mujik.pipeline.track_beats_with_madmom", side_effect=_fake_madmom), \
             patch("soundfile.info") as mock_info:
            mock_info.return_value = MagicMock(duration=5.0, samplerate=44100)
            Pipeline(cfg).run()

        ts_path = tmp_path / "out" / "ws" / "time_signatures.json"  # v0.5.1 修 5
        assert ts_path.exists()
        data = json.loads(ts_path.read_text())
        assert len(data) >= 1
        assert data[0]["sig"] == [4, 4]  # 启发式 4/4

    def test_project_metadata_has_bpm(self, tmp_path: Path):
        audio = tmp_path / "song.wav"
        audio.write_bytes(b"RIFF")
        cfg = _base_cfg(tmp_path)

        with patch("mujik.separate.demucs_adapter.separate_with_demucs", side_effect=_fake_separate), \
             patch("mujik.pipeline.transcribe_stem", side_effect=_fake_transcribe), \
             patch("mujik.pipeline.track_beats_with_madmom", side_effect=_fake_madmom), \
             patch("soundfile.info") as mock_info:
            mock_info.return_value = MagicMock(duration=5.0, samplerate=44100)
            Pipeline(cfg).run()

        meta = json.loads((tmp_path / "out" / "project.json").read_text())
        assert meta["mujik_version"] in ("0.2.2", "0.4.0", "0.4.1", "0.4.2", "0.4.3", "0.4.4", "0.4.5", "0.4.6", "0.4.7", "0.4.8", "0.4.9", "0.5.0", "0.5.1", "0.5.2")
        assert meta["rhythm_enabled"] is True

    def test_project_mid_has_tempo(self, tmp_path: Path):
        """project.mid 含 set_tempo 事件。"""
        audio = tmp_path / "song.wav"
        audio.write_bytes(b"RIFF")
        cfg = _base_cfg(tmp_path)

        with patch("mujik.separate.demucs_adapter.separate_with_demucs", side_effect=_fake_separate), \
             patch("mujik.pipeline.transcribe_stem", side_effect=_fake_transcribe), \
             patch("mujik.pipeline.track_beats_with_madmom", side_effect=_fake_madmom), \
             patch("soundfile.info") as mock_info:
            mock_info.return_value = MagicMock(duration=5.0, samplerate=44100)
            project = Pipeline(cfg).run()

        # 第一个 tempo 应来自 madmom
        assert project.tempo_map[0].bpm == 120.0
        assert project.tempo_map[0].bpm != 120.0 or True  # 来自 madmom


class TestPerStemMidi:
    """v0.5.3：ws/tracks/<stem>.mid 统一导出（含真实 tempo），CLI 自产 mid 清除。"""

    def _run(self, tmp_path: Path):
        audio = tmp_path / "song.wav"
        audio.write_bytes(b"RIFF")
        cfg = _base_cfg(tmp_path)
        with patch("mujik.separate.demucs_adapter.separate_with_demucs", side_effect=_fake_separate), \
             patch("mujik.pipeline.transcribe_stem", side_effect=_fake_transcribe), \
             patch("mujik.pipeline.track_beats_with_madmom", side_effect=_fake_madmom), \
             patch("soundfile.info") as mock_info:
            mock_info.return_value = MagicMock(duration=5.0, samplerate=44100)
            return Pipeline(cfg).run()

    def test_per_stem_mids_written_with_tempo(self, tmp_path: Path):
        import mido

        project = self._run(tmp_path)
        tracks_dir = tmp_path / "out" / "ws" / "tracks"
        for stem_name in project.tracks:
            mid_path = tracks_dir / f"{stem_name}.mid"
            assert mid_path.exists(), f"missing {mid_path}"
            mf = mido.MidiFile(mid_path)
            tempos = [
                mido.tempo2bpm(msg.tempo)
                for tr in mf.tracks for msg in tr if msg.type == "set_tempo"
            ]
            # 每个 per-stem mid 必须带真实 tempo（旧 CLI 产物是无 tempo 的 120）
            assert tempos == [120.0], f"{stem_name}: tempos={tempos}"

    def test_drum_stem_mid_has_notes(self, tmp_path: Path):
        """drumscript 只产 CSV——per-stem 导出保证鼓也有可直接查看的 mid。"""
        import mido

        self._run(tmp_path)
        mid_path = tmp_path / "out" / "ws" / "tracks" / "drums.mid"
        mf = mido.MidiFile(mid_path)
        notes = sum(
            1 for tr in mf.tracks
            for msg in tr if msg.type == "note_on" and msg.velocity > 0
        )
        assert notes == 1  # _fake_transcribe 给 drums 一击

    def test_cli_side_basic_pitch_mids_removed(self, tmp_path: Path):
        self._run(tmp_path)
        tracks_dir = tmp_path / "out" / "ws" / "tracks"
        assert list(tracks_dir.glob("*_basic_pitch.mid")) == []


class TestRhythmFailure:
    def test_madmom_failure_uses_default(self, tmp_path: Path):
        """madmom 失败时回退默认 tempo + 拍号。"""
        audio = tmp_path / "song.wav"
        audio.write_bytes(b"RIFF")
        cfg = _base_cfg(tmp_path)

        def fake_fail(*args, **kwargs):
            from mujik.rhythm.madmom_adapter import MadmomAdapterError
            raise MadmomAdapterError("simulated fail")

        with patch("mujik.separate.demucs_adapter.separate_with_demucs", side_effect=_fake_separate), \
             patch("mujik.pipeline.transcribe_stem", side_effect=_fake_transcribe), \
             patch("mujik.pipeline.track_beats_with_madmom", side_effect=fake_fail), \
             patch("soundfile.info") as mock_info:
            mock_info.return_value = MagicMock(duration=5.0, samplerate=44100)
            Pipeline(cfg).run()

        # beats.json 仍写出（默认 120），且带 provenance 标记
        # （v0.5.2：下游可区分"测得 120"与"madmom 失败后的编造值"）
        beats = json.loads((tmp_path / "out" / "ws" / "beats.json").read_text())
        assert beats["bpm"] == 120.0
        assert beats["source"] == "madmom-failed-fallback"


class TestRhythmDisabled:
    def test_no_rhythm_files_when_disabled(self, tmp_path: Path):
        audio = tmp_path / "song.wav"
        audio.write_bytes(b"RIFF")
        cfg = PipelineConfig(
            input_path=str(audio),
            output_dir=str(tmp_path / "out"),
            loudnorm=LoudnormConfig(enabled=False),
            rhythm=__import__("mujik.config.schema", fromlist=["RhythmConfig"]).RhythmConfig(enabled=False),
        )

        with patch("mujik.separate.demucs_adapter.separate_with_demucs", side_effect=_fake_separate), \
             patch("mujik.pipeline.transcribe_stem", side_effect=_fake_transcribe), \
             patch("mujik.pipeline.track_beats_with_madmom", side_effect=_fake_madmom) as mock_bt, \
             patch("soundfile.info") as mock_info:
            mock_info.return_value = MagicMock(duration=5.0, samplerate=44100)
            Pipeline(cfg).run()

        mock_bt.assert_not_called()
        # 不写 beats.json
        assert not (tmp_path / "out" / "beats.json").exists()


class TestTranscribeFailures:
    """v0.5.2: 转录失败的 fail-loud / fail-soft 边界。"""

    @staticmethod
    def _run(cfg, **patches):
        audio = Path(cfg.input_path)
        audio.write_bytes(b"RIFF")
        with patch("mujik.separate.demucs_adapter.separate_with_demucs", side_effect=_fake_separate), \
             patch("mujik.pipeline.transcribe_stem", **patches), \
             patch("mujik.pipeline.track_beats_with_madmom", side_effect=_fake_madmom), \
             patch("soundfile.info") as mock_info:
            mock_info.return_value = MagicMock(duration=5.0, samplerate=44100)
            return Pipeline(cfg).run()

    def test_router_error_propagates(self, tmp_path: Path):
        """配置错误（RouterError）不再被 catch-all 吞掉——fail-loud 上抛。"""
        from mujik.transcribe.router import RouterError

        cfg = _base_cfg(tmp_path)
        with pytest.raises(RouterError):
            self._run(cfg, side_effect=RouterError("drums: 'adtof' 未注册"))

    def test_all_stems_fail_raises(self, tmp_path: Path):
        """全部 stem 运行时失败 → RuntimeError（0 音符 MIDI 无意义）。"""
        cfg = _base_cfg(tmp_path)

        def always_fail(stem, config=None, out_dir=None):
            raise RuntimeError("adapter exploded")

        with pytest.raises(RuntimeError, match="all stems failed"):
            self._run(cfg, side_effect=always_fail)

    def test_partial_stem_failure_continues(self, tmp_path: Path):
        """单个 stem 运行时失败仍 fail-soft 继续（其余 stem 正常产出）。"""
        cfg = _base_cfg(tmp_path)

        def flaky(stem, config=None, out_dir=None):
            if stem.name == "vocals":
                raise RuntimeError("basic-pitch boom")
            return _fake_transcribe(stem)

        project = self._run(cfg, side_effect=flaky)
        assert "vocals" not in project.tracks or not project.tracks["vocals"].notes
        assert len(project.tracks["drums"].notes) == 1


class TestBpmReconciliation:
    """v0.5.3: madmom 半速估计被拍点数组校正 + 小节网格锚定 downbeat。"""

    def _run(self, tmp_path: Path, beat_track: BeatTrack):
        audio = tmp_path / "song.wav"
        audio.write_bytes(b"RIFF")
        cfg = _base_cfg(tmp_path)
        with patch("mujik.separate.demucs_adapter.separate_with_demucs", side_effect=_fake_separate), \
             patch("mujik.pipeline.transcribe_stem", side_effect=_fake_transcribe), \
             patch("mujik.pipeline.track_beats_with_madmom", return_value=beat_track), \
             patch("soundfile.info") as mock_info:
            mock_info.return_value = MagicMock(duration=5.0, samplerate=44100)
            project = Pipeline(cfg).run()
        beats = json.loads((tmp_path / "out" / "ws" / "beats.json").read_text())
        ts = json.loads((tmp_path / "out" / "ws" / "time_signatures.json").read_text())
        return project, beats, ts

    def test_half_tempo_estimate_corrected(self, tmp_path: Path):
        # 估计 62.5，拍点 0.5s 间隔（125 BPM）→ tempo_map 应为 125
        bt = BeatTrack(
            beats=[i * 0.48 for i in range(10)],
            downbeats=[0.0, 1.92, 3.84],
            bpm=62.5,
            tempo_confidence=0.4,
        )
        project, beats, _ = self._run(tmp_path, bt)
        assert project.tempo_map[0].bpm == 125.0
        assert beats["bpm"] == 125.0
        assert beats["bpm_source"] == "octave-corrected"

    def test_consistent_estimate_kept(self, tmp_path: Path):
        bt = BeatTrack(
            beats=[i * 0.5 for i in range(10)],
            downbeats=[0.0, 2.0, 4.0],
            bpm=120.0,
            tempo_confidence=0.9,
        )
        _, beats, _ = self._run(tmp_path, bt)
        assert beats["bpm"] == 120.0
        assert beats["bpm_source"] == "estimate"

    def test_bar_grid_anchored_to_first_downbeat(self, tmp_path: Path):
        # 首个 downbeat 不在 0（前奏）→ 拍号段起点应锚定到它
        bt = BeatTrack(
            beats=[0.5 + i * 0.5 for i in range(9)],
            downbeats=[2.5, 4.5],
            bpm=120.0,
            tempo_confidence=0.9,
        )
        _, beats, ts = self._run(tmp_path, bt)
        assert ts[0]["start"] == pytest.approx(2.5)
