"""BTC-HCQT (a.k.a. BTC-ISMIR19) chord recognition adapter (v0.4.8).

参考：
- Paper: Park et al., "A Bi-Directional Transformer for Musical Chord Recognition", ISMIR 2019
- Code (MIT): https://github.com/jayg996/BTC-ISMIR19

设计动机（v0.4.8）：
- v0.4.4 madmom CRNN 只支持 major/minor（25 类）
- BTC large_voca 支持 14 种 quality × 12 root + N + X = 170 类
- 含 7/maj7/m7/dim7/hdim7/sus2/sus4/min6/maj6/minmaj7 等延伸和弦

集成方式（与 madmom 一致）：
- 进程隔离：subprocess 调 `python _btc_predict_wrapper.py`
- 用户需提供 pretrained model 文件路径（.pt 格式）
- 用户需安装 PyTorch + librosa（通过 [chord-btc] extra）
- 用户需提供 BTC-ISMIR19 代码路径或 vendor copy

输出格式（与 madmom 一致）：
- JSON: [{start, end, label, ...}, ...]
- label 形如 "C:maj7", "F#:min7", "Bb:dim", "N"
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from mujik.midi.model import ChordEvent

logger = logging.getLogger(__name__)


BTC_HCQT_TIMEOUT_DEFAULT = 1800  # 30 分钟


class BtcHcqtAdapterError(RuntimeError):
    """BTC-HCQT 适配器错误。"""


# BTC label → mujik quality 映射
# BTC 输出 "Root:quality"（root 是 C/C#/D/.../A#/B 形式），quality 是 14 种之一
_BTC_QUALITY_MAP: dict[str, str] = {
    "maj": "",        # C → ("C", "")
    "min": "m",       # C:min → ("C", "m")
    "dim": "dim",
    "aug": "aug",
    "min6": "m6",
    "maj6": "maj6",
    "min7": "m7",
    "minmaj7": "mM7",  # BTC minmaj7 → mujik mM7
    "maj7": "maj7",
    "7": "7",
    "dim7": "dim7",
    "hdim7": "hdim7",
    "sus2": "sus2",
    "sus4": "sus4",
}


def check_btc_hcqt_available() -> bool:
    """检查 BTC-HCQT 是否可用：torch + librosa 导入 + model file 配置存在。

    Returns:
        True 如果依赖齐全；False 任何环节缺失。
    """
    try:
        import torch  # noqa: F401
        import librosa  # noqa: F401
    except ImportError:
        return False
    # model file 路径由用户配置；此处不强制检查（无 ENV/默认）
    return True


def _parse_btc_chord_label(label: str) -> ChordEvent | None:
    """BTC chord label 字符串 → ChordEvent 或 None（skip）。

    Examples:
        "C"        → ChordEvent(root="C", quality="")  (BTC 输出 bare root 表示 maj)
        "C:min"    → ChordEvent(root="C", quality="m")
        "C:maj7"   → ChordEvent(root="C", quality="maj7")
        "C:7"      → ChordEvent(root="C", quality="7")
        "C:dim7"   → ChordEvent(root="C", quality="dim7")
        "F#:sus4"  → ChordEvent(root="F#", quality="sus4")
        "Bb:hdim7" → ChordEvent(root="Bb", quality="hdim7")
        "N"        → None (no chord)
        "X"        → None (unknown)
    """
    label = label.strip()
    if not label or label in ("N", "X"):
        return None

    # BTC bare root 表示 maj (e.g. "C" → ("C", ""))
    if ":" not in label:
        if _is_valid_btc_root(label):
            return ChordEvent(
                start=0.0, end=0.0, root=label, quality="", vocab="btc-extended",
            )
        logger.warning("btc_hcqt: skipping malformed label: %r", label)
        return None

    root, _, quality_short = label.partition(":")
    root = root.strip()
    quality_short = quality_short.strip()
    if not root or not _is_valid_btc_root(root):
        logger.warning(
            "btc_hcqt: skipping invalid root: %r", label,
        )
        return None
    if not quality_short:
        return None

    quality = _BTC_QUALITY_MAP.get(quality_short, quality_short)
    return ChordEvent(
        start=0.0, end=0.0, root=root, quality=quality, vocab="btc-extended",
    )


def _is_valid_btc_root(root: str) -> bool:
    """检查 root 是否合法（BTC 用 # 不用 b，root 形如 C/C#/D/.../A#/B）。"""
    if not root:
        return False
    # BTC 用 # 不用 b；A-G + 可选 #
    return len(root) in (1, 2) and root[0] in "ABCDEFG" and (
        len(root) == 1 or root[1] == "#"
    )


_BTC_PREDICT_WRAPPER = '''"""
BTC-ISMIR19 推理 wrapper（v0.4.8）。

Usage: _btc_predict_wrapper.py <input_audio> <output_json> <model_path> <voca>
  - input_audio: 输入 wav 路径
  - output_json: 输出 JSON 路径
  - model_path: pretrained .pt 文件路径
  - voca: "large" (170 类) or "simple" (25 类)

Exit codes:
  2: usage
  3: missing dep (torch/librosa)
  4: BTC code not importable
  5: model file missing
  6: inference failure
"""
import json
import os
import sys


def main():
    if len(sys.argv) != 5:
        print("usage: _btc_predict_wrapper.py <input> <output_json> <model_path> <voca>",
              file=sys.stderr)
        sys.exit(2)

    audio_path = sys.argv[1]
    json_path = sys.argv[2]
    model_path = sys.argv[3]
    voca = sys.argv[4]

    try:
        import torch
        import librosa
        import numpy as np
    except ImportError as e:
        print(f"missing dep: {e}; install via `[chord-btc]` extra",
              file=sys.stderr)
        sys.exit(3)

    # 尝试 import BTC-ISMIR19 代码（用户需提供）
    # 优先 sys.path，然后 BTC_ISMIR19_PATH 环境变量，最后 vendor 目录
    btc_paths = [
        os.environ.get("BTC_ISMIR19_PATH", ""),
        os.path.join(os.path.dirname(__file__), "_btc"),
    ]
    btc_paths = [p for p in btc_paths if p and os.path.isdir(p)]
    for p in btc_paths:
        sys.path.insert(0, p)

    try:
        from btc_model import BTC_model
        from utils.mir_eval_modules import (
            audio_file_to_features, idx2chord, idx2voca_chord,
        )
    except ImportError as e:
        print(f"BTC-ISMIR19 code not importable: {e}; "
              f"set BTC_ISMIR19_PATH or vendor in _btc/", file=sys.stderr)
        sys.exit(4)

    if not os.path.isfile(model_path):
        print(f"model file not found: {model_path}", file=sys.stderr)
        sys.exit(5)

    # model config (硬编码与 BTC-ISMIR19/run_config.yaml 一致)
    from utils.hparams import HParams
    config = HParams.load("run_config.yaml") if os.path.isfile("run_config.yaml") \\
        else _default_config(voca == "large")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BTC_model(config=config.model).to(device)
    try:
        # torch 2.6+ 默认 weights_only=True 会拒载含 numpy 标量（mean/std）的
        # checkpoint；权重来自上游 MIT 仓库（jayg996/BTC-ISMIR19），显式关闭
        checkpoint = torch.load(model_path, map_location=device,
                                weights_only=False)
        mean = checkpoint["mean"]
        std = checkpoint["std"]
        model.load_state_dict(checkpoint["model"])
    except Exception as e:
        print(f"failed to load model: {e}", file=sys.stderr)
        sys.exit(6)

    # 特征提取
    try:
        feature, feature_per_second, _ = audio_file_to_features(
            audio_path, config,
        )
    except Exception as e:
        print(f"feature extraction failed: {e}", file=sys.stderr)
        sys.exit(6)

    if voca == "large":
        idx_to_chord = idx2voca_chord()
    else:
        idx_to_chord = idx2chord

    # 推理
    feature = feature.T
    feature = (feature - mean) / std
    time_unit = feature_per_second
    n_timestep = config.model["timestep"]
    num_pad = n_timestep - (feature.shape[0] % n_timestep)
    feature = np.pad(feature, ((0, num_pad), (0, 0)),
                     mode="constant", constant_values=0)
    num_instance = feature.shape[0] // n_timestep

    lines = []
    try:
        with torch.no_grad():
            model.eval()
            feature = torch.tensor(
                feature, dtype=torch.float32,
            ).unsqueeze(0).to(device)
            start_time = 0.0
            for t in range(num_instance):
                self_attn_output, _ = model.self_attn_layers(
                    feature[:, n_timestep * t:n_timestep * (t + 1), :],
                )
                prediction, _ = model.output_layer(self_attn_output)
                prediction = prediction.squeeze()
                for i in range(n_timestep):
                    if t == 0 and i == 0:
                        prev_chord = prediction[i].item()
                        continue
                    if prediction[i].item() != prev_chord:
                        lines.append({
                            "start": float(start_time),
                            "end": float(time_unit * (n_timestep * t + i)),
                            "label": idx_to_chord[prev_chord],
                        })
                        start_time = time_unit * (n_timestep * t + i)
                        prev_chord = prediction[i].item()
            # 收尾
            lines.append({
                "start": float(start_time),
                "end": float(time_unit * (n_timestep * num_instance)),
                "label": idx_to_chord[prev_chord],
            })
    except Exception as e:
        print(f"inference failed: {e}", file=sys.stderr)
        sys.exit(6)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(lines, f, ensure_ascii=False, indent=2)


def _default_config(large_voca: bool):
    """BTC-ISMIR19/run_config.yaml 默认值（避免文件查找）。"""
    from utils.hparams import HParams
    return HParams(
        mp3={"song_hz": 22050, "inst_len": 10.0, "skip_interval": 5.0},
        feature={
            "n_bins": 144, "bins_per_octave": 24,
            "hop_length": 2048, "large_voca": large_voca,
        },
        experiment={"learning_rate": 0.0001, "weight_decay": 0.0,
                    "max_epoch": 100, "batch_size": 128, "save_step": 40,
                    "data_ratio": 0.8},
        model={
            "feature_size": 144, "timestep": 108,
            "num_chords": 170 if large_voca else 25,
            "input_dropout": 0.2, "layer_dropout": 0.2,
            "attention_dropout": 0.2, "relu_dropout": 0.2,
            "num_layers": 8, "num_heads": 4, "hidden_size": 128,
            "total_key_depth": 128, "total_value_depth": 128,
            "filter_size": 128, "loss": "ce", "probs_out": False,
        },
    )


if __name__ == "__main__":
    main()
'''


def detect_chords_with_btc(
    audio_path: str | Path,
    config: "BtcHcqtConfig | None" = None,
    out_dir: str | Path | None = None,
) -> list[ChordEvent]:
    """用 BTC-HCQT (large_voca) 检测和弦 → ChordEvent 列表。

    Args:
        audio_path: 输入 wav 路径
        config: BtcHcqtConfig（含 model_path / voca / timeout）
        out_dir: 输出目录；None 时写到 tmp

    Returns:
        list[ChordEvent]：BTC 输出的和弦事件（N 和 X 已过滤）

    Raises:
        FileNotFoundError: 音频文件不存在
        BtcHcqtAdapterError: subprocess 失败 / 超时 / 输出解析失败
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"audio not found: {audio_path}")

    timeout = (
        getattr(config, "btc_timeout_sec", BTC_HCQT_TIMEOUT_DEFAULT)
        if config is not None
        else BTC_HCQT_TIMEOUT_DEFAULT
    )
    voca = (
        getattr(config, "btc_voca", None)
        or getattr(config, "voca", "large")
        if config is not None
        else "large"
    )

    # 权重路径解析顺序：config.btc_model_path → env MUJIK_BTC_MODEL → None
    # （镜像内默认装到 /app/models/btc_model_large_voca.pt 并设 env，实现零配置）
    model_path = (
        getattr(config, "btc_model_path", None)
        if config is not None
        else None
    )
    if not model_path:
        model_path = os.environ.get("MUJIK_BTC_MODEL") or None

    if out_dir is None:
        out_dir = Path(tempfile.mkdtemp(prefix="mujik_btc_chord_"))
    else:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / f"btc_chords_{audio_path.stem}.json"
    # v0.5.1 修 5：wrapper 脚本写系统临时目录，不再泄漏进产物目录
    wrapper_path = Path(tempfile.gettempdir()) / f"mujik_btc_predict_wrapper_{os.getpid()}.py"
    wrapper_path.write_text(_BTC_PREDICT_WRAPPER)

    cmd = [
        sys.executable, str(wrapper_path),
        str(audio_path), str(json_path),
        str(model_path) if model_path else "",
        voca,
    ]
    logger.info(
        "btc_hcqt: input={input}, model={model}, voca={voca}, timeout={sec}s",
        input=audio_path, model=model_path, voca=voca, sec=timeout,
    )

    # vendor 目录注入：wrapper 写在系统临时目录，其自身 __file__ 解析不到包内
    # _btc/，因此这里显式把包内 vendored BTC-ISMIR19 目录放进 BTC_ISMIR19_PATH
    # （用户显式设置的 env 优先，不覆盖）
    env = dict(os.environ)
    if not env.get("BTC_ISMIR19_PATH"):
        vendor_dir = Path(__file__).resolve().parent / "_btc"
        if vendor_dir.is_dir():
            env["BTC_ISMIR19_PATH"] = str(vendor_dir)

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, env=env,
        )
    except subprocess.TimeoutExpired as e:
        raise BtcHcqtAdapterError(
            f"btc-hcqt chord detection timeout after {timeout}s"
        ) from e

    try:
        wrapper_path.unlink()
    except OSError:
        pass

    if result.returncode != 0:
        raise BtcHcqtAdapterError(
            f"btc-hcqt chord failed (exit={result.returncode}): "
            f"{result.stderr[:500]}"
        )

    if not json_path.exists():
        raise BtcHcqtAdapterError(
            f"btc-hcqt chord output json not found: {json_path}"
        )

    try:
        raw = json.loads(json_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        raise BtcHcqtAdapterError(
            f"failed to parse btc-hcqt chord output: {e}"
        ) from e

    chord_track: list[ChordEvent] = []
    for entry in raw:
        try:
            label = entry["label"]
            start = float(entry["start"])
            end = float(entry["end"])
        except (KeyError, TypeError, ValueError) as e:
            logger.warning(
                "btc_hcqt: skipping malformed entry %r: %s", entry, e,
            )
            continue
        chord = _parse_btc_chord_label(label)
        if chord is None:
            continue
        chord.start = start
        chord.end = end
        chord_track.append(chord)

    logger.info(
        "btc_hcqt: {n} chords detected from {audio}",
        n=len(chord_track), audio=audio_path.name,
    )
    return chord_track


__all__ = [
    "BTC_HCQT_TIMEOUT_DEFAULT",
    "BtcHcqtAdapterError",
    "check_btc_hcqt_available",
    "detect_chords_with_btc",
    "_parse_btc_chord_label",
]
