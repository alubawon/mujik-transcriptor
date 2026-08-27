"""Pipeline 主入口（v0.2.1 端到端最小垂直切片）。

v0.2.1 完整流程：
  1. 加载音频元数据
  2. 响度归一（pyloudnorm）
  3. Demucs v4 4-stem 源分离
  4. 按 stem 路由到转录 adapter（basic-pitch / adtof）
  5. 写入 Project.tracks
  6. 写出 out/project.mid
"""
from __future__ import annotations

import json
from pathlib import Path

from loguru import logger

from mujik.config.schema import PipelineConfig
from mujik.midi.io import write_project_to_midi
from mujik.midi.model import Project, TempoSegment
from mujik.preprocess.loudnorm import normalize_loudness
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
        """执行 v0.2.1 完整管线。"""
        cfg = self.config
        audio_path = Path(cfg.input_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"input not found: {audio_path}")

        out_dir = Path(cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # ---- 1. 加载音频元数据 ----
        duration, sample_rate = _probe_audio(audio_path)
        logger.info(
            "pipeline[1/6]: input={path}, duration={dur:.2f}s, sr={sr}",
            path=audio_path, dur=duration, sr=sample_rate,
        )

        # ---- 2. 响度归一 ----
        if cfg.loudnorm.enabled:
            norm_path = normalize_loudness(audio_path, config=cfg.loudnorm)
            sep_input = norm_path
            logger.info("pipeline[2/6]: loudnorm done → {}", norm_path)
        else:
            sep_input = audio_path
            logger.info("pipeline[2/6]: loudnorm skipped (disabled)")

        # ---- 3. Demucs 4-stem 分离 ----
        stems_out_dir = out_dir / "stems"
        stems: Stems = separate_with_demucs(
            sep_input, stems_out_dir, config=cfg.source_separation,
        )
        logger.info(
            "pipeline[3/6]: demucs done, {n} stems ({names})",
            n=stems.stem_count, names=list(stems.names),
        )

        # ---- 4. 初始化 Project ----
        project = Project(
            audio_path=str(audio_path),
            duration=duration,
            sample_rate=sample_rate,
            time_signatures=build_default_segments(duration if duration > 0 else 1.0),
            tempo_map=[TempoSegment(0.0, duration if duration > 0 else 1.0, 120.0)],
            metadata={
                "mujik_version": "0.2.1",
                "preset": cfg.preset,
                "loudnorm_enabled": cfg.loudnorm.enabled,
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
                # 单 stem 失败不阻塞整管线（Demucs 失败除外）
                logger.warning(
                    "pipeline[5/6]: transcribe {stem} failed: {err}",
                    stem=stem.name, err=e,
                )
                continue

            track = project.get_track(stem.name)  # type: ignore[arg-type]
            for note in notes:
                track.add(note)
            track.sort_by_start()
            logger.info(
                "pipeline[5/6]: {stem} → {n} notes",
                stem=stem.name, n=len(track.notes),
            )

        # ---- 6. 写 MIDI ----
        midi_path = out_dir / "project.mid"
        write_project_to_midi(project, midi_path)
        logger.info(
            "pipeline[6/6]: wrote MIDI → {path} ({tracks} tracks, {notes} notes)",
            path=midi_path, tracks=len(project.tracks),
            notes=project.total_notes(),
        )

        # 写 metadata sidecar
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
