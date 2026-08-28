"""MuScriptor multitrack adapter (v0.4.2).

subprocess 模式：调 `uvx muscriptor transcribe` → 解析输出的多轨 MIDI → Project。

设计决策（v0.4.2）：
- 进程隔离：muscriptor 包（含 CC-BY-NC 4.0 权重）作为独立子进程
  运行，主线**不直接 import muscriptor**，避免触发 liccheck 警告
- subprocess 模式（参考 `bytedance_piano_adapter.py` 同模式）
- 输出多轨 MIDI（vocals/drums/bass/piano/guitar 各 1 track），
  muscriptor instrument name 标准化（"Electric Guitar", "Drum Kit" 等）
  通过 `_program_to_stem` + `_instrument_name_to_stem` 反查
- HF_TOKEN 必需：muscriptor 权重 CC-BY-NC 4.0，需在 HuggingFace
  https://huggingface.co/MuScriptor/muscriptor-{small,medium,large}
  接受 license 后才能下载权重
- 模型尺寸：small (103M, CPU 友好) / medium (307M, 默认) / large (1.4B, GPU 推荐)

约定：
- 输入：wav 路径（完整音频，未分轨）
- 输出：Project（含 muscriptor 检测到的多 Track）
- 默认调用：`uvx muscriptor transcribe <audio> --output <out_dir> --model <small|medium|large>`
- muscriptor 输出文件名约定：`<audio_stem>.mid`（默认模式）
- 错误码：
  - 缺 uvx → FileNotFoundError 提示装 uv
  - 缺/无效 HF_TOKEN → MuscriptorAdapterError 带明确指引
  - subprocess timeout → MuscriptorAdapterError
  - 其他非零退出码 → MuscriptorAdapterError 带 stderr 摘要

参考：
- muscriptor PyPI: https://pypi.org/project/muscriptor/
- muscriptor GitHub: https://github.com/muscriptor/muscriptor
- muscriptor HF: https://huggingface.co/MuScriptor
- muscriptor 论文: arXiv:2607.08168
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from loguru import logger

from mujik.midi.model import Project

if TYPE_CHECKING:
    from mujik.config.schema import TranscribeConfig

# muscriptor 进程默认超时（30 分钟，足以处理 5 分钟音频）
MUSCRIPTOR_TIMEOUT_DEFAULT = 1800

# muscriptor 模型尺寸
MuscriptorModel = Literal["small", "medium", "large"]
VALID_MUSCRIPTOR_MODELS: tuple[MuscriptorModel, ...] = ("small", "medium", "large")


class MuscriptorAdapterError(RuntimeError):
    pass


def check_muscriptor_available() -> bool:
    """检查 `uvx` 工具是否可调用（muscriptor 通过 `uvx muscriptor` 间接调用）。"""
    return shutil.which("uvx") is not None


def _parse_error(stderr: str) -> str:
    """从 muscriptor stderr 提取关键错误信息。

    muscriptor 错误常见关键词：
    - "huggingface" / "HF_TOKEN" / "gated" → token 问题
    - "Repository Not Found" / "401" / "403" → 权重未接受或无权限
    - "OutOfMemory" / "CUDA" → 显存问题
    """
    lower = stderr.lower()
    if "401" in stderr or "403" in stderr or "gated" in lower:
        return (
            "HuggingFace authentication failed. Please:\n"
            "  1. Visit https://huggingface.co/MuScriptor/muscriptor-{small,medium,large}\n"
            "  2. Accept the CC BY-NC 4.0 model license\n"
            "  3. Set HF_TOKEN environment variable (https://huggingface.co/settings/tokens)"
        )
    if "huggingface" in lower or "hf_token" in lower or "repository" in lower:
        return (
            "HuggingFace error. Please verify:\n"
            "  1. HF_TOKEN is set in your environment\n"
            "  2. You have accepted the model license at "
            "https://huggingface.co/MuScriptor"
        )
    if "outofmemory" in lower or "cuda" in lower or "mps" in lower:
        return (
            "GPU/memory error. Try:\n"
            "  - Use a smaller model: --model small (103M params, CPU friendly)\n"
            "  - Shorter audio: split input into < 2 min segments\n"
            "  - On Windows: pass --torch-backend=cu128 to use GPU"
        )
    # 兜底：返回 stderr 前 500 字符
    return stderr[:500]


def transcribe_multitrack(
    audio_path: Path | str,
    config: "TranscribeConfig | None" = None,
    out_dir: Path | str | None = None,
    model: MuscriptorModel = "medium",
    timeout_sec: int = MUSCRIPTOR_TIMEOUT_DEFAULT,
    uvx_path: str = "uvx",
) -> Project:
    """用 muscriptor 一次性转写多乐器音频为 Project。

    Args:
        audio_path: 输入 wav 路径（**完整音频**，未分轨）
        config: TranscribeConfig（暂未用，预留 muscriptor 专属配置）
        out_dir: 输出目录；None 时写到 tmp
        model: muscriptor 模型尺寸（small/medium/large）
        timeout_sec: subprocess 超时（秒）
        uvx_path: uvx 可执行文件路径（默认 PATH 查找）

    Returns:
        Project：含 muscriptor 检测到的多 Track（vocals/drums/bass/piano/guitar 等）

    Raises:
        FileNotFoundError: 音频文件不存在，或 uvx 未安装
        MuscriptorAdapterError: HF_TOKEN 缺失/无效、subprocess 失败、超时
    """
    audio_path = Path(audio_path)

    if not audio_path.exists():
        raise FileNotFoundError(f"audio not found: {audio_path}")

    if model not in VALID_MUSCRIPTOR_MODELS:
        raise MuscriptorAdapterError(
            f"invalid muscriptor model: {model!r}; "
            f"must be one of {VALID_MUSCRIPTOR_MODELS}"
        )

    if shutil.which(uvx_path) is None:
        raise FileNotFoundError(
            f"`uvx` not found at {uvx_path!r}. "
            f"Install uv: https://docs.astral.sh/uv/getting-started/installation/"
        )

    # HF_TOKEN 检查：muscriptor 必需
    if not os.environ.get("HF_TOKEN"):
        logger.warning(
            "HF_TOKEN not set; muscriptor may fail to download model weights"
        )

    if out_dir is None:
        out_dir = Path(tempfile.mkdtemp(prefix="mujik_muscriptor_"))
    else:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    # muscriptor CLI: uvx muscriptor transcribe <audio> --output <dir> --model <size>
    cmd = [
        uvx_path, "muscriptor", "transcribe",
        str(audio_path),
        "--output", str(out_dir),
        "--model", model,
    ]
    logger.info(
        "muscriptor: cmd={cmd}, model={model}, audio={audio}",
        cmd=" ".join(cmd), model=model, audio=audio_path.name,
    )

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired as e:
        raise MuscriptorAdapterError(
            f"muscriptor timeout ({timeout_sec}s). "
            f"Try a smaller model (--model small) or split the audio."
        ) from e

    if result.returncode != 0:
        stderr_hint = _parse_error(result.stderr)
        raise MuscriptorAdapterError(
            f"muscriptor failed (exit={result.returncode}):\n{stderr_hint}"
        )

    # muscriptor 默认输出文件名约定：<audio_stem>.mid
    output_midi = out_dir / f"{audio_path.stem}.mid"
    if not output_midi.exists():
        # 兜底：在 out_dir 中找任何 .mid 文件
        candidates = list(out_dir.glob("*.mid"))
        if not candidates:
            raise MuscriptorAdapterError(
                f"muscriptor produced no MIDI file in {out_dir}. "
                f"Check stdout/stderr above."
            )
        output_midi = candidates[0]
        logger.warning(
            "muscriptor: expected {expected}, found {actual}",
            expected=output_midi.name, actual=candidates[0].name,
        )

    # 用 pretty_midi 解析多轨 MIDI
    from mujik.midi.io import read_midi_to_project
    try:
        project = read_midi_to_project(
            str(output_midi),
            audio_path=str(audio_path),
        )
    except Exception as e:
        raise MuscriptorAdapterError(
            f"failed to parse muscriptor MIDI output: {e}"
        ) from e

    logger.info(
        "muscriptor: {tracks} tracks, {notes} notes from {audio} (model={model})",
        tracks=len(project.tracks),
        notes=project.total_notes(),
        audio=audio_path.name,
        model=model,
    )
    return project


__all__ = [
    "MuscriptorAdapterError",
    "MuscriptorModel",
    "VALID_MUSCRIPTOR_MODELS",
    "MUSCRIPTOR_TIMEOUT_DEFAULT",
    "check_muscriptor_available",
    "transcribe_multitrack",
]
