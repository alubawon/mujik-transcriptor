"""Demucs v4 源分离 adapter（4-stem）。

通过 subprocess 隔离 demucs 的重量级依赖（PyTorch），
保证主进程不会被 ML 栈污染。

调用方式：
    adapter = DemucsAdapter(Stems, model="htdemucs_ft", device="cuda")
    stems = adapter.separate("input.wav", out_dir="stems/")

返回值：Stems 容器，包含 vocals/drums/bass/other 4 个 stem 的 wav 路径。
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Literal

from loguru import logger

from mujik.config.schema import SourceSeparationConfig
from mujik.separate.model import Stem, Stems

DemucsVariant = Literal[
    "htdemucs", "htdemucs_ft", "htdemucs_6s", "mdx_q", "mdx_extra_q"
]


class DemucsAdapterError(RuntimeError):
    pass


def check_demucs_available() -> bool:
    """检查 demucs CLI 是否在 PATH 中可用。"""
    try:
        result = subprocess.run(
            ["python", "-m", "demucs", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def _resolve_device(device: str) -> str:
    """校验 device 可用性；cuda 请求但环境无 CUDA 时显式回退 cpu。

    默认配置 device="cuda" 是给 GPU 生产环境的；CPU-only 机器（如 macOS
    容器）会直接报 "no CUDA GPUs available"。这里不静默：打 warning 日志。
    """
    if device != "cuda":
        return device
    try:
        result = subprocess.run(
            [sys.executable, "-c",
             "import torch; print(int(torch.cuda.is_available()))"],
            capture_output=True, text=True, timeout=60,
        )
        cuda_ok = result.returncode == 0 and result.stdout.strip() == "1"
    except (subprocess.TimeoutExpired, FileNotFoundError):
        cuda_ok = False
    if not cuda_ok:
        logger.warning(
            "device=cuda requested but CUDA is not available in this "
            "environment; falling back to cpu"
        )
        return "cpu"
    return "cuda"


def separate_with_demucs(
    input_path: str | Path,
    out_dir: str | Path,
    config: SourceSeparationConfig | None = None,
) -> Stems:
    """调用 Demucs 分离输入音频为 4-stem。

    Args:
        input_path: 输入音频（WAV/FLAC/MP3）
        out_dir: 输出目录
        config: 源分离配置；None 时使用默认

    Returns:
        Stems 容器，含 vocals/drums/bass/other 4 个 stem

    Raises:
        DemucsAdapterError: Demucs 调用失败
        FileNotFoundError: 输入文件不存在
    """
    cfg = config or SourceSeparationConfig()
    device = _resolve_device(cfg.device)
    input_path = Path(input_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"input not found: {input_path}")

    logger.info(
        "Demucs separating: input={input}, model={model}/{variant}, device={device}, precision={precision}",
        input=input_path,
        model=cfg.model,
        variant=cfg.variant,
        device=device,
        precision=cfg.precision,
    )

    # demucs CLI: python -m demucs -n htdemucs_ft --device cuda --out out_dir input.wav
    # 注意：demucs CLI 的 --segment 只接受 int（v0.5.1 修：7.5 → 7）
    cmd = [
        sys.executable, "-m", "demucs",
        "-n", cfg.variant,
        "--device", device,
        "--segment", str(int(cfg.segment_length)),
        "--overlap", str(cfg.overlap),
        "--jobs", str(cfg.jobs),
        "--out", str(out_dir),
    ]
    if cfg.out_format != "wav":
        cmd.extend(["--out-format", cfg.out_format])
    cmd.append(str(input_path))

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,
        )
    except subprocess.TimeoutExpired as e:
        raise DemucsAdapterError("Demucs timeout after 1h") from e

    if result.returncode != 0:
        # 尾部截取：traceback 的真实异常在最后几行，头部多为无关 warning
        raise DemucsAdapterError(
            f"Demucs failed (exit={result.returncode}): {result.stderr[-2000:]}"
        )

    # Demucs 输出结构：<out_dir>/<model_name>/<input_stem>/{vocals,drums,bass,other}.{wav,mp3,...}
    model_dir = out_dir / cfg.variant
    input_stem = input_path.stem
    track_dir = model_dir / input_stem
    if not track_dir.exists():
        raise DemucsAdapterError(
            f"Demucs output not found at {track_dir}; full output tree:\n"
            f"{list(model_dir.glob('**/*')) if model_dir.exists() else 'no model dir'}"
        )

    # 探测采样率（可选依赖 soundfile）
    sample_rate = 44100
    duration = 0.0
    try:
        import soundfile as sf
        for name in ("vocals", "drums", "bass", "other"):
            audio_path = track_dir / f"{name}.{cfg.out_format}"
            if audio_path.exists():
                info = sf.info(str(audio_path))
                sample_rate = info.samplerate
                duration = info.duration
                break
    except (ImportError, Exception) as e:  # noqa: BLE001
        # 包含 LibsndfileError（声文件无法读取）等所有异常
        logger.debug("could not probe sample rate/duration from stems: {}", e)

    stems = Stems(
        separation_model=f"demucs/{cfg.variant}",
        sample_rate=sample_rate,
        total_duration=duration,
    )
    for name in ("vocals", "drums", "bass", "other"):
        audio_path = track_dir / f"{name}.{cfg.out_format}"
        if not audio_path.exists():
            raise DemucsAdapterError(f"missing stem: {audio_path}")
        stems.add(Stem(
            name=name,  # type: ignore[arg-type]
            audio_path=audio_path,
            sample_rate=sample_rate,
            duration=duration,
            source_model=f"demucs/{cfg.variant}",
            metadata={"format": cfg.out_format, "bitrate": cfg.out_bitrate},
        ))

    logger.info(
        "Demucs done: {n} stems, duration={dur:.1f}s",
        n=stems.stem_count, dur=duration,
    )
    return stems


__all__ = [
    "separate_with_demucs",
    "check_demucs_available",
    "DemucsAdapterError",
    "DemucsVariant",
]
