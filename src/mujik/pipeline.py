"""Pipeline 主入口（v0.2.2 完整 7 步）。

v0.2.2 完整流程：
  1. 加载音频元数据
  2. 响度归一（pyloudnorm）
  2.5 节拍/下拍/BPM 跟踪（madmom）→ beats.json
  2.6 时间签名推断（启发式）→ time_signatures.json
  3. Demucs v4 4-stem 源分离
  4. 按 stem 路由到转录 adapter（basic-pitch / drumscript）
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
from mujik.pipeline_progress import PipelineProgress
from mujik.preprocess.loudnorm import normalize_loudness
from mujik.rhythm.madmom_adapter import track_beats_with_madmom
from mujik.rhythm.time_signature import infer_time_signature_from_downbeats
from mujik.separate.demucs_adapter import separate_with_demucs
from mujik.separate.model import Stems
from mujik.time_signature.model import build_default_segments
from mujik.transcribe.router import transcribe_stem

# Pipeline 固定阶段数（per_stem 模式：denoise + loudnorm + rhythm + chord + quantize
# + groove + demucs + per_stem transcribe + write + multitrack 分支另算）
PIPELINE_TOTAL_STEPS_PERSTEM = 10
PIPELINE_TOTAL_STEPS_MULTITRACK = 6


class Pipeline:
    """端到端管线主类。"""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        logger.info("Pipeline initialized with preset={}", config.preset)

    def run(self) -> Project:
        """执行 v0.2.2 完整管线（v0.5.1 加进度条）。"""
        cfg = self.config
        audio_path = Path(cfg.input_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"input not found: {audio_path}")

        out_dir = Path(cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        # v0.5.1 修 5：中间产物目录（stems/tracks/beats.json 等）；
        # output_dir 只放最终产物（project.mid/score.musicxml/project.json）
        ws_dir = Path(cfg.workspace_dir) if cfg.workspace_dir else out_dir / "ws"
        ws_dir.mkdir(parents=True, exist_ok=True)

        # ---- 0. 顶层进度条（v0.5.1：自动 no-op on non-TTY / no tqdm）----
        # 不包裹整套代码（避免大段重缩进）；用 prog 变量，全程可用，
        # 退出前手动 close。
        initial_total = (
            PIPELINE_TOTAL_STEPS_MULTITRACK
            if cfg.transcribe.mode == "multitrack"
            else PIPELINE_TOTAL_STEPS_PERSTEM
        )
        prog = PipelineProgress(
            total=initial_total, title=f"mujik run ({cfg.preset})",
        ).__enter__()

        # ---- 1. 加载音频元数据 ----
        duration, sample_rate = _probe_audio(audio_path)
        prog.advance("probe", extra=f"{duration:.1f}s")
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
                    out_path=ws_dir / f"denoised_{audio_path.name}",
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
        prog.advance("denoise")

        # ---- 2. 响度归一 ----
        if cfg.loudnorm.enabled:
            # v0.5.1 修 5：确定性文件名（含时长，避免不同裁剪长度串味），
            # 落 ws/ 而非系统 tempfile；同曲重跑不会在 /tmp 堆积随机临时文件
            _dur_tag = f"{int(duration)}s" if duration and duration > 0 else "x"
            norm_path = normalize_loudness(
                audio_path_for_sep, config=cfg.loudnorm,
                out_path=ws_dir / f"loudnorm_{audio_path.stem}_{_dur_tag}.wav",
            )
            sep_input = norm_path
            logger.info("pipeline[2/7]: loudnorm done → {}", norm_path)
        else:
            sep_input = audio_path_for_sep
            logger.info("pipeline[2/7]: loudnorm skipped (disabled)")
        prog.advance("loudnorm")

        # ---- 2.5/2.6 节拍 + 时间签名（rhythm 层，v0.2.2 新增）----
        if cfg.rhythm.enabled:
            try:
                beat_track = track_beats_with_madmom(
                    sep_input, config=cfg.rhythm, out_dir=ws_dir,
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
            (ws_dir / "beats.json").write_text(json.dumps(
                beat_track.to_dict() if beat_track else {"bpm": 120.0},
                ensure_ascii=False, indent=2,
            ))
            (ws_dir / "time_signatures.json").write_text(json.dumps([
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
        prog.advance("rhythm", extra=f"{tempo.bpm:.0f} BPM")

        # ---- 2.7/7 和弦识别（chord 层，v0.4.4 madmom + v0.4.8 BTC-HCQT）----
        chord_track: list = []
        if cfg.chord.enabled:
            try:
                if cfg.chord.backend == "btc-hcqt":
                    from mujik.chord.btc_hcqt_adapter import detect_chords_with_btc
                    chord_track = detect_chords_with_btc(
                        sep_input, config=cfg.chord, out_dir=ws_dir,
                    )
                else:  # "madmom" (default fallback for v0.4.4 compat)
                    from mujik.chord.madmom_adapter import detect_chords_with_madmom
                    chord_track = detect_chords_with_madmom(
                        sep_input, config=cfg.chord, out_dir=ws_dir,
                    )
                # 写 out/chords.json
                (ws_dir / "chords.json").write_text(json.dumps(
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
                    "pipeline[2.7/7]: chord ({backend}): {n} chords detected",
                    backend=cfg.chord.backend, n=len(chord_track),
                )
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "pipeline[2.7/7]: {backend} chord failed: {err}, "
                    "skipping (chord_track=[])",
                    backend=cfg.chord.backend, err=e,
                )
                chord_track = []
        else:
            logger.info("pipeline[2.7/7]: chord skipped (disabled)")
        prog.advance("chord", extra=f"{len(chord_track)} chords")

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
                (ws_dir / "chords_quantized.json").write_text(json.dumps(
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
        prog.advance("chord-quantize")

        # ---- 2.85/7 chord groove 联动（v0.4.9 新增，默认关闭）----
        grooved_chord_track: list = quantized_chord_track
        if (
            cfg.chord.enabled
            and cfg.chord.quantize_enabled
            and cfg.chord.apply_groove
            and quantized_chord_track
        ):
            try:
                from mujik.chord.groove import apply_groove_to_chord_track
                grooved_chord_track = apply_groove_to_chord_track(
                    quantized_chord_track,
                    time_sigs,
                    tempo.bpm,
                    template=cfg.chord.chord_groove_template,
                    strength=cfg.chord.chord_groove_strength,
                    ratio=cfg.chord.chord_groove_ratio,
                    duration=duration,
                )
                # 写 out/chords_grooved.json（第三份 artifact）
                (ws_dir / "chords_grooved.json").write_text(json.dumps(
                    [
                        {
                            "start": c.start,
                            "end": c.end,
                            "root": c.root,
                            "quality": c.quality,
                            "bass": c.bass,
                        }
                        for c in grooved_chord_track
                    ],
                    ensure_ascii=False, indent=2,
                ))
                logger.info(
                    "pipeline[2.85/7]: chord groove: {tpl} strength={s} "
                    "({n} chords shifted)",
                    tpl=cfg.chord.chord_groove_template,
                    s=cfg.chord.chord_groove_strength,
                    n=len(grooved_chord_track),
                )
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "pipeline[2.85/7]: chord groove failed: {err}, "
                    "using quantized chord_track",
                    err=e,
                )
                grooved_chord_track = quantized_chord_track
        else:
            logger.info("pipeline[2.85/7]: chord groove skipped (disabled)")
        prog.advance("chord-groove")

        # ---- 3. 源分离 OR muscriptor multitrack 模式分支（v0.4.2）----
        if cfg.transcribe.mode == "multitrack":
            # v0.4.2 multitrack 模式：跳过源分离，直接 muscriptor 多乐器转录
            from mujik.transcribe.muscriptor_adapter import transcribe_multitrack
            project = transcribe_multitrack(
                sep_input,
                config=cfg.transcribe,
                out_dir=ws_dir / "muscriptor",
                model=cfg.transcribe.muscriptor_model,
            )
            # muscriptor 输出的 audio_path 改回原始 audio_path（而非去噪后）
            project.audio_path = str(audio_path)
            project.duration = duration
            project.sample_rate = sample_rate
            project.time_signatures = time_sigs
            project.tempo_map = [tempo]
            project.chord_track = grooved_chord_track  # v0.4.9
            project.metadata.update({
                "mujik_version": "0.5.1",
                "preset": cfg.preset,
                "loudnorm_enabled": cfg.loudnorm.enabled,
                "rhythm_enabled": cfg.rhythm.enabled,
                "chord_enabled": cfg.chord.enabled,
                "chord_backend": cfg.chord.backend,
                "chord_quantize_enabled": cfg.chord.quantize_enabled,
                "chord_groove_enabled": cfg.chord.apply_groove,
                "chord_groove_template": cfg.chord.chord_groove_template,
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
            prog.advance("muscriptor", extra=f"{len(project.tracks)} tracks")
            prog.advance("midi-write", extra=f"{project.total_notes()} notes")
            prog.advance("metadata")
            prog.__exit__(None, None, None)
            return project

        # ---- 3. Demucs 4-stem 分离（per_stem 模式）----
        stems_out_dir = ws_dir / "stems"
        stems: Stems = separate_with_demucs(
            sep_input, stems_out_dir, config=cfg.source_separation,
        )
        logger.info(
            "pipeline[3/7]: demucs done, {n} stems ({names})",
            n=stems.stem_count, names=list(stems.names),
        )
        prog.advance("demucs", extra=f"{stems.stem_count} stems")
        # per-stem 阶段：动态增加总步数
        prog.update_total(prog.step_idx + stems.stem_count + 2)

        # ---- 4. 初始化 Project ----
        project = Project(
            audio_path=str(audio_path),
            duration=duration,
            sample_rate=sample_rate,
            time_signatures=time_sigs,
            tempo_map=[tempo],
            chord_track=grooved_chord_track,  # v0.4.9
            metadata={
                "mujik_version": "0.5.1",
                "preset": cfg.preset,
                "loudnorm_enabled": cfg.loudnorm.enabled,
                "rhythm_enabled": cfg.rhythm.enabled,
                "chord_enabled": cfg.chord.enabled,
                "chord_backend": cfg.chord.backend,
                "chord_groove_enabled": cfg.chord.apply_groove,
                "chord_groove_template": cfg.chord.chord_groove_template,
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
                    stem, config=cfg.transcribe, out_dir=str(ws_dir / "tracks"),
                )
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "pipeline[5/7]: transcribe {stem} failed: {err}",
                    stem=stem.name, err=e,
                )
                prog.advance(f"transcribe:{stem.name}", extra="failed")
                continue

            track = project.get_track(stem.name)  # type: ignore[arg-type]
            for note in notes:
                track.add(note)
            track.sort_by_start()
            logger.info(
                "pipeline[5/7]: {stem} → {n} notes",
                stem=stem.name, n=len(track.notes),
            )
            prog.advance(f"transcribe:{stem.name}", extra=f"{len(track.notes)} notes")

        # ---- 6. 写 MIDI ----
        midi_path = out_dir / "project.mid"
        write_project_to_midi(project, midi_path)
        logger.info(
            "pipeline[6/7]: wrote MIDI → {path} ({tracks} tracks, {notes} notes)",
            path=midi_path, tracks=len(project.tracks),
            notes=project.total_notes(),
        )
        prog.advance("midi-write", extra=f"{project.total_notes()} notes")

        # ---- 7. 写 metadata sidecar ----
        meta_path = out_dir / "project.json"
        meta_path.write_text(json.dumps(
            project.metadata, ensure_ascii=False, indent=2,
        ))
        prog.advance("metadata")
        prog.__exit__(None, None, None)

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
