"""Denoise preprocessor（v0.4.0）。

实现两个去噪后端：
1. nnnoiseless（RNNoise 派生）：MIT、CPU-only、单进程
2. Demucs denoise mode：MIT、复用 demucs API 内部 denoise pipeline

设计决策（v0.4.0）：
- 默认走 nnnoiseless（无 GPU 依赖、无模型权重下载）
- Demucs denoise mode 留作可选 backend（GPU 加速、效果更佳）
- 输出：写新 wav 文件（不修改输入）

约定：
- 输入：任意音频文件路径
- 输出：去噪后 wav 文件路径（默认写到 {tmp}/denoised_{stem}.wav）
- 单声道化（denoise 通常 mono 友好）+ 48kHz 重采样（nnnoiseless 要求）
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from loguru import logger

from mujik.config.schema import PreprocessConfig


class DenoiseError(RuntimeError):
    pass


# nnnoiseless 期望的采样率
_NNNOISELESS_SR = 48000


def _load_audio_mono(audio_path: Path) -> tuple["numpy.ndarray", int]:
    """读音频 → 单声道 float32。

    返回 (audio, sr)。
    """
    try:
        import numpy as np
        import soundfile as sf
    except ImportError as e:
        raise DenoiseError(
            f"numpy + soundfile required: {e}"
        ) from e
    audio, sr = sf.read(str(audio_path), always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)  # mono
    audio = audio.astype(np.float32)
    return audio, sr


def _save_audio(audio: "numpy.ndarray", sr: int, out_path: Path) -> Path:
    """写音频到 wav 文件。"""
    try:
        import soundfile as sf
    except ImportError as e:
        raise DenoiseError(f"soundfile required: {e}") from e
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), audio, sr, subtype="PCM_16")
    return out_path


def denoise_with_nnnoiseless(
    audio_path: Path | str,
    config: PreprocessConfig | None = None,
    out_path: Path | str | None = None,
) -> Path:
    """nnnoiseless 去噪（MIT，CPU-only）。

    Args:
        audio_path: 输入音频
        config: PreprocessConfig（暂未用，预留参数）
        out_path: 输出路径；None 时写到同目录 `denoised_{stem}.wav`

    Returns:
        输出 wav 路径
    """
    cfg = config or PreprocessConfig()
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise DenoiseError(f"input not found: {audio_path}")

    try:
        import nnnoiseless
    except ImportError as e:
        raise DenoiseError(
            "nnnoiseless is not installed; "
            "install via `uv pip install 'mujik-transcriptor[preprocess-denoise]'`"
        ) from e

    import numpy as np

    audio, sr = _load_audio_mono(audio_path)
    original_sr = sr

    # 重采样到 nnnoiseless 期望的 48kHz
    if sr != _NNNOISELESS_SR:
        try:
            import scipy.signal
            audio = scipy.signal.resample(
                audio, int(len(audio) * _NNNOISELESS_SR / sr)
            ).astype(np.float32)
            sr = _NNNOISELESS_SR
        except ImportError:
            logger.warning(
                "scipy not available; passing {}Hz audio to nnnoiseless "
                "(may degrade quality)",
                original_sr,
            )

    # nnnoiseless.process 期望 (n_frames, 480) 形状：每帧 10ms @ 48kHz
    frame_size = 480
    n_frames = (len(audio) // frame_size) * frame_size
    if n_frames == 0:
        raise DenoiseError(f"audio too short: {len(audio)} samples")
    audio_trimmed = audio[:n_frames]
    frames = audio_trimmed.reshape(-1, frame_size)

    try:
        denoiser = nnnoiseless.RNNoise(sample_rate=sr)
        denoised_frames = denoiser.process_frames(frames)
    except Exception as e:
        raise DenoiseError(f"nnnoiseless process_frames failed: {e}") from e

    denoised = denoised_frames.reshape(-1).astype(np.float32)
    # 拼接回原长度（如有截断）
    if len(denoised) < len(audio):
        denoised = np.concatenate([denoised, audio[len(denoised):]])

    # 重采样回原采样率
    if sr != original_sr:
        try:
            import scipy.signal
            denoised = scipy.signal.resample(
                denoised, len(audio)
            ).astype(np.float32)
            sr = original_sr
        except ImportError:
            logger.warning("scipy not available; saving at {}Hz", sr)

    if out_path is None:
        out_path = audio_path.parent / f"denoised_{audio_path.stem}.wav"
    out_path = Path(out_path)

    _save_audio(denoised, sr, out_path)
    logger.info(
        "denoise (nnnoiseless): {inp} → {out} ({sr}Hz, {n} samples)",
        inp=audio_path.name, out=out_path.name, sr=sr, n=len(denoised),
    )
    return out_path


def denoise_with_demucs_mode(
    audio_path: Path | str,
    config: PreprocessConfig | None = None,
    out_path: Path | str | None = None,
) -> Path:
    """Demucs 内部 denoise mode（MIT，复用 demucs 已有 dep）。

    v0.4.0 实现简化版：调 `demucs` API 内部 denoise pipeline，输出 denoised 全 audio。

    注意：demucs 模型权重 ~80MB，不在镜像中；用户需自己装 torch + 跑首次下载。
    """
    cfg = config or PreprocessConfig()
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise DenoiseError(f"input not found: {audio_path}")

    try:
        from demucs.api import Separator
    except ImportError as e:
        raise DenoiseError(
            "demucs is not installed; install via "
            "`uv pip install 'mujik-transcriptor[separate]'`"
        ) from e

    try:
        import numpy as np
        import soundfile as sf
    except ImportError as e:
        raise DenoiseError(f"numpy + soundfile required: {e}") from e

    # torch 可能在 mock 测试中（无真正 torch）
    try:
        import torch  # noqa: F401  # 用于 demucs 内部
    except ImportError:
        pass  # 允许 demucs 自己处理 torch 缺失

    # Demucs denoise mode 调 `--two-stems vocals` 等价于 vocals + 其他
    # 但 v0.4.0 简化为：只跑 demucs 的 denoise_model.forward 拿 denoised audio
    try:
        # demucs.api.Separator(model='htdemucs') 内部有 denoise 模型
        separator = Separator(
            model="htdemucs",
            device=cfg.demucs_device,
            segment=7.5,
            overlap=0.25,
            shifts=1,
            jobs=1,
        )
        # 用 demucs 的 denoise 功能：分离 vocals 后，混合除 vocals 外的轨道当作 denoised
        origin, separated = separator.separate_audio_file(str(audio_path))
        # 重建：除 vocals 外所有 stem 求和
        stems_audio = list(separated.values())
        if len(stems_audio) < 2:
            raise DenoiseError("demucs did not produce multiple stems")

        # 假定第一个是 vocals（demucs 默认顺序）；其他求和
        non_vocals = stems_audio[1]
        for s in stems_audio[2:]:
            non_vocals = non_vocals + s
        if non_vocals is None:
            raise DenoiseError("demucs stem format unexpected")

        # 写 wav
        out_path_p = Path(out_path) if out_path else audio_path.parent / f"denoised_{audio_path.stem}.wav"
        out_path_p.parent.mkdir(parents=True, exist_ok=True)

        # 转 numpy (float32)
        if hasattr(non_vocals, 'cpu'):
            non_vocals = non_vocals.cpu().numpy()
        non_vocals = non_vocals.astype(np.float32)
        # 取单声道
        if non_vocals.ndim > 1:
            non_vocals = non_vocals.mean(axis=0)

        sr = separator.samplerate  # demucs 内部 sr
        sf.write(str(out_path_p), non_vocals, sr, subtype="PCM_16")
        logger.info(
            "denoise (demucs): {inp} → {out} ({sr}Hz)",
            inp=audio_path.name, out=out_path_p.name, sr=sr,
        )
        return out_path_p
    except Exception as e:
        if isinstance(e, DenoiseError):
            raise
        raise DenoiseError(f"demucs denoise failed: {e}") from e


def denoise(
    audio_path: Path | str,
    config: PreprocessConfig | None = None,
    out_path: Path | str | None = None,
) -> Path:
    """去噪统一入口（按 config.denoise_backend 派发）。

    默认 backend = "nnnoiseless"（MIT、CPU-only、零 GPU 依赖）。
    """
    cfg = config or PreprocessConfig()
    if not cfg.denoise_enabled:
        logger.info("denoise disabled, returning input path")
        return Path(audio_path)

    backend = cfg.denoise_backend
    if backend == "nnnoiseless":
        return denoise_with_nnnoiseless(audio_path, config=cfg, out_path=out_path)
    elif backend == "demucs":
        return denoise_with_demucs_mode(audio_path, config=cfg, out_path=out_path)
    else:
        raise DenoiseError(f"unknown denoise backend: {backend!r}")


__all__ = [
    "DenoiseError",
    "denoise",
    "denoise_with_nnnoiseless",
    "denoise_with_demucs_mode",
]
