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

        # ---- 1.5 去噪（v0.4.0 新增）----
        if cfg.preprocess.denoise_enabled:
            try:
                from mujik.preprocess.denoise import denoise
                denoised_path = denoise(
                    audio_path, config=cfg.preprocess,
                    out_path=out_dir / f"denoised_{audio_path.name}",
                )
                # 后续步骤用去噪后的文件
                audio_path_for_sep = denoised_path
                logger.info("pipeline[1.5/7]: denoise done → {}", denoised_path)
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "pipeline[1.5/7]: denoise failed: {err}, use original",
                    err=e,
                )
                audio_path_for_sep = audio_path
        else:
            audio_path_for_sep = audio_path
            logger.info("pipeline[1.5/7]: denoise skipped (disabled)")

        # ---- 2. 响度归一 ----
        if cfg.loudnorm.enabled:
            norm_path = normalize_loudness(audio_path_for_sep, config=cfg.loudnorm)
            sep_input = norm_path
            logger.info("pipeline[2/7]: loudnorm done → {}", norm_path)
        else:
            sep_input = audio_path_for_sep
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

        # ---- 2.7/7 和弦识别（chord 层，v0.4.4 新增）----
        chord_track: list = []
        if cfg.chord.enabled:
            try:
                from mujik.chord.madmom_adapter import detect_chords_with_madmom
                chord_track = detect_chords_with_madmom(
                    sep_input, config=cfg.chord, out_dir=out_dir,
                )
                # 写 out/chords.json
                (out_dir / "chords.json").write_text(json.dumps(
                    [
                        {
                            "start": c.start,
                            "end": c.end,
                            "root": c.root,
                            "quality": c.quality,
                            "bass": c.bass,
                        }
                        for c in chord_track
                    ],
                    ensure_ascii=False, indent=2,
                ))
                logger.info(
                    "pipeline[2.7/7]: chord: {n} chords detected",
                    n=len(chord_track),
                )
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "pipeline[2.7/7]: madmom chord failed: {err}, "
                    "skipping (chord_track=[])",
                    err=e,
                )
                chord_track = []
        else:
            logger.info("pipeline[2.7/7]: chord skipped (disabled)")

        # ---- 2.8/7 chord quantize（v0.4.5 新增）----
        quantized_chord_track: list = chord_track
        if cfg.chord.enabled and cfg.chord.quantize_enabled and chord_track:
            try:
                from mujik.chord.quantize import quantize_chord_track
                quantized_chord_track = quantize_chord_track(
                    chord_track,
                    time_sigs,
                    tempo.bpm,
                    grid_per_bar=cfg.chord.grid_per_bar,
                    merge_consecutive=cfg.chord.merge_consecutive,
                    min_duration_sec=cfg.chord.min_duration_sec,
                    duration=duration,
                )
                # 写 out/chords_quantized.json（保留原始 + 量化两份）
                (out_dir / "chords_quantized.json").write_text(json.dumps(
                    [
                        {
                            "start": c.start,
                            "end": c.end,
                            "root": c.root,
                            "quality": c.quality,
                            "bass": c.bass,
                        }
                        for c in quantized_chord_track
                    ],
                    ensure_ascii=False, indent=2,
                ))
                logger.info(
                    "pipeline[2.8/7]: chord quantize: {n_in} → {n_out} (grid={g}/bar)",
                    n_in=len(chord_track),
                    n_out=len(quantized_chord_track),
                    g=cfg.chord.grid_per_bar,
                )
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "pipeline[2.8/7]: chord quantize failed: {err}, "
                    "using raw chord_track",
                    err=e,
                )
                quantized_chord_track = chord_track
        else:
            logger.info("pipeline[2.8/7]: chord quantize skipped")

        # ---- 3. 源分离 OR muscriptor multitrack 模式分支（v0.4.2）----
        if cfg.transcribe.mode == "multitrack":
            # v0.4.2 multitrack 模式：跳过源分离，直接 muscriptor 多乐器转录
            from mujik.transcribe.muscriptor_adapter import transcribe_multitrack
            project = transcribe_multitrack(
                sep_input,
                config=cfg.transcribe,
                out_dir=out_dir / "muscriptor",
                model=cfg.transcribe.muscriptor_model,
            )
            # muscriptor 输出的 audio_path 改回原始 audio_path（而非去噪后）
            project.audio_path = str(audio_path)
            project.duration = duration
            project.sample_rate = sample_rate
            project.time_signatures = time_sigs
            project.tempo_map = [tempo]
            project.chord_track = quantized_chord_track  # v0.4.5
            project.metadata.update({
                "mujik_version": "0.4.5",
                "preset": cfg.preset,
                "loudnorm_enabled": cfg.loudnorm.enabled,
                "rhythm_enabled": cfg.rhythm.enabled,
                "chord_enabled": cfg.chord.enabled,
                "chord_quantize_enabled": cfg.chord.quantize_enabled,
                "denoise_enabled": cfg.preprocess.denoise_enabled,
                "denoise_backend": cfg.preprocess.denoise_backend,
                "transcribe_mode": "multitrack",
                "muscriptor_model": cfg.transcribe.muscriptor_model,
                "score_features": ["bend", "harmony"],
            })
            logger.info(
                "pipeline[3'/7]: muscriptor multitrack done, {n} tracks",
                n=len(project.tracks),
            )
            return project

        # ---- 3. Demucs 4-stem 分离（per_stem 模式）----
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
            chord_track=quantized_chord_track,  # v0.4.5
            metadata={
                "mujik_version": "0.4.5",
                "preset": cfg.preset,
                "loudnorm_enabled": cfg.loudnorm.enabled,
                "rhythm_enabled": cfg.rhythm.enabled,
                "chord_enabled": cfg.chord.enabled,
                "chord_quantize_enabled": cfg.chord.quantize_enabled,
                "denoise_enabled": cfg.preprocess.denoise_enabled,
                "denoise_backend": cfg.preprocess.denoise_backend,
                "separator": stems.separation_model,
                "transcribe_mode": "per_stem",
                # v0.4.1 新增：score 渲染支持 bend/harmony
                "score_features": ["bend", "harmony"],
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
