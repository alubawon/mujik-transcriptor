"""Stem → 转录 adapter 派发表。

v0.4.1 路由：
  vocals → basic_pitch_adapter
  bass   → basic_pitch_adapter
  other  → basic_pitch_adapter
  drums  → drumscript_adapter (v0.5.2；此前 adtof 因死链+CC-BY-NC-SA 权重弃用)
  piano  → bytedance-piano adapter（v0.4.0 已实现）
  guitar → NotImplementedError（v0.5+ Apollo，仓库 TBD）
"""
from __future__ import annotations

from loguru import logger

from mujik.config.schema import DrumScriptConfig, TranscribeConfig
from mujik.midi.model import Note, StemName
from mujik.separate.model import Stem


class RouterError(NotImplementedError):
    pass


def transcribe_stem(
    stem: Stem,
    config: TranscribeConfig | None = None,
    out_dir: str | None = None,
) -> list[Note]:
    """按 stem.name 派发到对应 adapter。

    Args:
        stem: 源分离产出的单个 stem
        config: 转录配置
        out_dir: 子 adapter 输出目录

    Returns:
        list[Note]（已按 start 排序）

    Raises:
        RouterError: unknown stem 或 adapter 未实现
    """
    cfg = config or TranscribeConfig()
    adapter_name = _route(stem.name, cfg)

    logger.info(
        "router: {stem} → {adapter}",
        stem=stem.name, adapter=adapter_name,
    )

    if adapter_name == "basic-pitch":
        from mujik.transcribe.basic_pitch_adapter import transcribe_with_basic_pitch
        # v0.5.3: per-stem 配置（stem_basic_pitch）优先，未列出的 stem 用
        # 全局 basic_pitch。min_note_length 仍取 TranscribeConfig 的全局值
        # （drumscript 与 basic-pitch 共用同一语义）。
        # v0.5.3 修：删除 `onset_threshold=onset_interval_min_ms / 100`——
        # 毫秒时间参数除以 100 冒充概率阈值，是类型说谎（50→0.5 纯属撞对）
        bp_cfg = cfg.stem_basic_pitch.get(stem.name) or cfg.basic_pitch
        bp_cfg = bp_cfg.model_copy(update={"min_note_length_ms": cfg.min_note_length_ms})
        return transcribe_with_basic_pitch(
            stem.audio_path,
            config=bp_cfg,
            out_dir=out_dir,
        )
    if adapter_name == "drumscript":
        from mujik.transcribe.drumscript_adapter import transcribe_drums_with_drumscript
        return transcribe_drums_with_drumscript(
            stem.audio_path,
            config=DrumScriptConfig(
                min_note_length_ms=cfg.min_note_length_ms,
            ),
            out_dir=out_dir,
        )
    if adapter_name == "bytedance-piano":
        from mujik.transcribe.bytedance_piano_adapter import transcribe_piano_bytedance
        return transcribe_piano_bytedance(
            stem.audio_path,
            config=cfg,
            out_dir=out_dir,
        )
    if adapter_name == "apollo":
        raise RouterError(
            f"guitar stem 暂未实现 (v0.5+ Apollo, repo TBD): {stem.name}"
        )
    raise RouterError(f"unknown adapter '{adapter_name}' for stem '{stem.name}'")


def _route(stem_name: StemName, config: TranscribeConfig) -> str:
    """从 config 中按 stem 取 adapter 名。"""
    mapping: dict[StemName, str] = {
        "vocals": config.vocals,
        "bass": config.bass,
        "drums": config.drums,
        "other": config.other,
        "piano": config.piano,
        "guitar": config.guitar,
    }
    if stem_name not in mapping:
        raise RouterError(f"unknown stem name: {stem_name}")
    return mapping[stem_name]


__all__ = [
    "transcribe_stem",
    "RouterError",
]
