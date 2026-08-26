"""Pipeline 主入口（v0.1 占位）。

v0.1 阶段：构造 + 框架；实际环节接入在 v0.2。
"""
from __future__ import annotations

from pathlib import Path

from loguru import logger

from mujik.config.schema import PipelineConfig
from mujik.midi.model import Project, TempoSegment
from mujik.time_signature.model import build_default_segments


class Pipeline:
    """端到端管线主类。"""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        logger.info("Pipeline initialized with preset={}", config.preset)

    def run(self) -> Project:
        """执行完整管线。"""
        cfg = self.config
        logger.info("Pipeline.run: input={}, output={}", cfg.input_path, cfg.output_dir)

        # 1. 加载音频元数据（v0.1 占位：从文件名推断；v0.2 用 soundfile）
        audio_path = Path(cfg.input_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"input not found: {audio_path}")

        # v0.1: 创建空 Project；v0.2+ 接入各环节
        project = Project(
            audio_path=str(audio_path),
            duration=0.0,
            sample_rate=44100,
            time_signatures=build_default_segments(0.0),
            tempo_map=[TempoSegment(0.0, 0.0, 120.0)],
        )

        out_dir = Path(cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # v0.1 占位：仅做项目骨架
        logger.info("v0.1 pipeline: skeleton only; see docs/design.md for v0.2 plan")
        return project


__all__ = ["Pipeline"]
