"""Pipeline 主入口（v0.2.2 完整 7 步）。

v0.2.2 完整流程：
  1. 加载音频元数据
  2. 响度归一（pyloudnorm）
  2.5 节拍/下拍/BPM 跟踪（madmom）→ beats.json
  2.6 时间签名推断（启发式）→ time_signatures.json
  3. Demucs v4 4-stem 源分离
  4. 按 stem 路由到转录 adapter（basic-pitch / adtof）
  5. 写入 Project.tracks
  6. 写出 out/project.mid（含真实 tempo + time_signature 事件）
  7. 写 out/project.json 元数据
"""
from __future__ import annotations

import json
from pathlib import Path

from loguru import logger

from mujik.config.schema import PipelineConfig
from mujik.midi.io import write_project_to_midi
from mujik.midi.model import Project, TempoSegment
from mujik.preprocess.loudnorm import normalize_loudness
from mujik.rhythm.madmom_adapter import track_beats_with_madmom
from mujik.rhythm.time_signature import infer_time_signature_from_downbeats
from mujik.separate.demucs_adapter import separate_with_demucs
from mujik.separate.model import Stems
from mujik.time_signature.model import build_default_segments
from mujik.transcribe.router import transcribe_stem


class Pipeline:
    """端到端管线主类。"""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        logger.info("Pipeline initialized with preset={}", config.preset)

    def run(self) -> Project:
        """执行 v0.2.2 完整管线。"""
        cfg = self.config
        audio_path = Path(cfg.input_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"input not found: {audio_path}")

        out_dir = Path(cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # ---- 1. 加载音频元数据 ----
        duration, sample_rate = _probe_audio(audio_path)
        logger.info(
            "pipeline[1/7]: input={path}, duration={dur:.2f}s, sr={sr}",
            path=audio_path, dur=duration, sr=sample_rate,
        )

        # ---- 2. 响度归一 ----
        if cfg.loudnorm.enabled:
            norm_path = normalize_loudness(audio_path, config=cfg.loudnorm)
            sep_input = norm_path
            logger.info("pipeline[2/7]: loudnorm done → {}", norm_path)
        else:
            sep_input = audio_path
            logger.info("pipeline[2/7]: loudnorm skipped (disabled)")

        # ---- 2.5/2.6 节拍 + 时间签名（rhythm 层，v0.2.2 新增）----
        if cfg.rhythm.enabled:
            try:
                beat_track = track_beats_with_madmom(
                    sep_input, config=cfg.rhythm, out_dir=out_dir,
                )
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "pipeline[2.5/7]: madmom failed: {err}, use defaults",
                    err=e,
                )
                beat_track = None

            if beat_track is not None:
                # tempo
                tempo = TempoSegment(
                    start_time=0.0,
                    end_time=duration if duration > 0 else 1.0,
                    bpm=beat_track.bpm,
                )
                # time signature 启发式
                time_sigs = infer_time_signature_from_downbeats(
                    downbeats=beat_track.downbeats,
                    beats=beat_track.beats,
                    duration=duration if duration > 0 else 1.0,
                    fallback=cfg.rhythm.time_signature_fallback,
                )
                logger.info(
                    "pipeline[2.5/7]: rhythm: bpm={bpm:.1f}, {n} beats, {d} downbeats, "
                    "{ts} time-sig segment(s)",
                    bpm=beat_track.bpm,
                    n=len(beat_track.beats),
                    d=len(beat_track.downbeats),
                    ts=len(time_sigs),
                )
            else:
                tempo = TempoSegment(
                    start_time=0.0,
                    end_time=duration if duration > 0 else 1.0,
                    bpm=120.0,
                )
                time_sigs = build_default_segments(duration if duration > 0 else 1.0)

            # 写 beats.json
            (out_dir / "beats.json").write_text(json.dumps(
                beat_track.to_dict() if beat_track else {"bpm": 120.0},
                ensure_ascii=False, indent=2,
            ))
            (out_dir / "time_signatures.json").write_text(json.dumps([
                {
                    "start": s.start_time,
                    "end": s.end_time,
                    "sig": list(s.time_signature),
                    "confidence": s.confidence,
                    "source": s.source,
                }
                for s in time_sigs
            ], ensure_ascii=False, indent=2))
        else:
            tempo = TempoSegment(
                start_time=0.0,
                end_time=duration if duration > 0 else 1.0,
                bpm=120.0,
            )
            time_sigs = build_default_segments(duration if duration > 0 else 1.0)
            logger.info("pipeline[2.5/7]: rhythm skipped (disabled)")

        # ---- 3. Demucs 4-stem 分离 ----
        stems_out_dir = out_dir / "stems"
        stems: Stems = separate_with_demucs(
            sep_input, stems_out_dir, config=cfg.source_separation,
        )
        logger.info(
            "pipeline[3/7]: demucs done, {n} stems ({names})",
            n=stems.stem_count, names=list(stems.names),
        )

        # ---- 4. 初始化 Project ----
        project = Project(
            audio_path=str(audio_path),
            duration=duration,
            sample_rate=sample_rate,
            time_signatures=time_sigs,
            tempo_map=[tempo],
            metadata={
                "mujik_version": "0.2.2",
                "preset": cfg.preset,
                "loudnorm_enabled": cfg.loudnorm.enabled,
                "rhythm_enabled": cfg.rhythm.enabled,
                "separator": stems.separation_model,
            },
        )

        # ---- 5. 按 stem 转录 ----
        for stem in stems.primary_stems():
            try:
                notes = transcribe_stem(
                    stem, config=cfg.transcribe, out_dir=str(out_dir / "tracks"),
                )
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "pipeline[5/7]: transcribe {stem} failed: {err}",
                    stem=stem.name, err=e,
                )
                continue

            track = project.get_track(stem.name)  # type: ignore[arg-type]
            for note in notes:
                track.add(note)
            track.sort_by_start()
            logger.info(
                "pipeline[5/7]: {stem} → {n} notes",
                stem=stem.name, n=len(track.notes),
            )

        # ---- 6. 写 MIDI ----
        midi_path = out_dir / "project.mid"
        write_project_to_midi(project, midi_path)
        logger.info(
            "pipeline[6/7]: wrote MIDI → {path} ({tracks} tracks, {notes} notes)",
            path=midi_path, tracks=len(project.tracks),
            notes=project.total_notes(),
        )

        # ---- 7. 写 metadata sidecar ----
        meta_path = out_dir / "project.json"
        meta_path.write_text(json.dumps(
            project.metadata, ensure_ascii=False, indent=2,
        ))

        return project


def _probe_audio(audio_path: Path) -> tuple[float, int]:
    """探测音频时长与采样率。"""
    try:
        import soundfile as sf
        info = sf.info(str(audio_path))
        return float(info.duration), int(info.samplerate)
    except (ImportError, Exception) as e:  # noqa: BLE001
        logger.warning("could not probe audio ({}), fallback to default", e)
        return 0.0, 44100


__all__ = ["Pipeline"]
