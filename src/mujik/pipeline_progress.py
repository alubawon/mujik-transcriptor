"""pipeline_progress.py — 轻量级阶段进度条（v0.5.1）。

设计目标：
- 极简依赖（只用 tqdm，core-io 已有）
- 优雅降级：无 tqdm / 非 TTY → 静默 no-op
- 不污染 loguru 日志（独立 stderr 流）
- 支持 context manager + 手动 advance

用法（pipeline.py）：
    from mujik.pipeline_progress import PipelineProgress, silent_progress

    with PipelineProgress(total=8, title="mujik pipeline") as prog:
        prog.advance("loudnorm")
        ...
        prog.advance("transcribe:vocals", extra="12 notes")
        ...
"""
from __future__ import annotations

import os
import sys
from typing import Any


def _is_tty() -> bool:
    """是否在交互终端（CI 自动 no）。"""
    return sys.stderr.isatty() and "CI" not in os.environ


class _NullProgress:
    """降级 no-op（无 tqdm / 非 TTY / disable 时）。"""

    def __init__(self) -> None:
        self._step_idx = 0

    def __enter__(self) -> "_NullProgress":
        return self

    def __exit__(self, *exc: Any) -> None:
        pass

    def advance(self, step_name: str, extra: str = "") -> None:
        self._step_idx += 1

    def update_total(self, total: int) -> None:
        pass

    @property
    def step_idx(self) -> int:
        return self._step_idx


class PipelineProgress:
    """顶层管线进度条管理器。

    内部用 tqdm；每个阶段 `advance(step_name)` 推进一格 + 写出阶段名。
    非 TTY 或 tqdm 不可用时自动降级到 no-op，调用方无感。
    """

    def __init__(self, total: int, title: str = "mujik pipeline", enabled: bool = True) -> None:
        self._total = total
        self._title = title
        self._enabled = enabled and _is_tty()
        self._bar: Any = None
        self._step_idx = 0
        self._impl: Any = self

    def __enter__(self) -> "PipelineProgress | _NullProgress":
        if not self._enabled:
            return _NullProgress()
        try:
            from tqdm import tqdm  # type: ignore[import-not-found]
        except ImportError:
            return _NullProgress()
        self._bar = tqdm(
            total=self._total,
            desc=self._title,
            unit="step",
            file=sys.stderr,
            dynamic_ncols=True,
            mininterval=0.2,
        )
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._bar is not None:
            self._bar.close()

    def advance(self, step_name: str, extra: str = "") -> None:
        """推进一格 + 阶段名显示。

        Args:
            step_name: 阶段名（如 "loudnorm", "demucs", "transcribe:vocals"）
            extra: 附加信息（如 "5 chords"）
        """
        self._step_idx += 1
        if self._bar is None:
            return
        postfix = f"→ {step_name}"
        if extra:
            postfix += f" ({extra})"
        self._bar.set_postfix_str(postfix, refresh=True)
        self._bar.update(1)

    def update_total(self, total: int) -> None:
        """运行时调整总步数（适用于多 stem 转录）。"""
        if self._bar is None:
            return
        if total > self._bar.total:
            self._bar.total = total
            self._bar.refresh()

    @property
    def step_idx(self) -> int:
        return self._step_idx


def silent_progress() -> _NullProgress:
    """显式 no-op 进度条（用于测试 / dry-run / 单步 CLI）。"""
    return _NullProgress()


__all__ = ["PipelineProgress", "silent_progress"]
