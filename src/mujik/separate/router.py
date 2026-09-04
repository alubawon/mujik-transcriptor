"""分离后端路由（v0.5.2）。

统一入口 `separate_audio()`：按 SourceSeparationConfig.model/variant 派发：

- demucs + htdemucs_6s  → htdemucs_6s adapter（6-stem，含 piano/guitar）
- demucs + 其余 variant → demucs 4-stem adapter
- mdx23c / bsroformer / melbandroformer → **fail-loud**：
  Roformer 家族尚未实现（未评估，见 docs/research.md），配置里写了
  必须报错，绝不静默降级到 demucs（v0.5.2 前的 bug：jazz preset 的
  mdx23c 被静默忽略，用户以为在用 Roformer 实际跑的是 demucs）。
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from mujik.config.schema import SourceSeparationConfig
from mujik.separate.model import Stems

# 尚未实现的模型（schema 允许配置，但后端不存在）
ROFORMER_MODELS: tuple[str, ...] = (
    "mdx23c",
    "bsroformer",
    "melbandroformer",
)


class SeparationBackendError(RuntimeError):
    """分离后端未实现或不可用。"""


def separate_audio(
    input_path: str | Path,
    out_dir: str | Path,
    config: SourceSeparationConfig | None = None,
) -> Stems:
    """按配置路由到具体分离 adapter。

    Args:
        input_path: 输入音频
        out_dir: 输出目录
        config: 分离配置；None 用默认（demucs/htdemucs_ft 4-stem）

    Returns:
        Stems 容器

    Raises:
        SeparationBackendError: model 指向未实现的后端（Roformer 家族）
    """

    cfg = config or SourceSeparationConfig()
    input_path = Path(input_path)
    out_dir = Path(out_dir)

    if cfg.model in ROFORMER_MODELS:
        raise SeparationBackendError(
            f"separation model '{cfg.model}' 尚未实现（Roformer 家族未评估，"
            f"见 docs/research.md）。当前可用：model=demucs "
            f"(variant=htdemucs_ft 4-stem / htdemucs_6s 6-stem)。"
            f"请改配置，不要期待静默降级。"
        )

    if cfg.model != "demucs":  # pragma: no cover — schema Literal 兜底
        raise SeparationBackendError(f"unknown separation model: {cfg.model!r}")

    if cfg.variant == "htdemucs_6s":
        from mujik.separate.htdemucs_6s_adapter import separate_with_htdemucs_6s

        if cfg.stem_count != 6:
            logger.warning(
                "stem_count={} 与 htdemucs_6s 的 6-stem 输出不匹配，以实际产出为准",
                cfg.stem_count,
            )
        return separate_with_htdemucs_6s(input_path, out_dir, config=cfg)

    from mujik.separate.demucs_adapter import separate_with_demucs

    if cfg.stem_count != 4:
        logger.warning(
            "stem_count={} 只有 htdemucs_6s 支持；variant={} 固定产出 4-stem"
            "（如需 piano/guitar 请设 variant=htdemucs_6s）",
            cfg.stem_count,
            cfg.variant,
        )
    return separate_with_demucs(input_path, out_dir, config=cfg)


__all__ = ["separate_audio", "SeparationBackendError", "ROFORMER_MODELS"]
