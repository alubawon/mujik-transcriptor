"""Time signature JSON I/O.

读写 `list[TimeSignatureSegment]` 到 JSON 文件。

JSON 格式（与 `pipeline.py` 写出格式保持一致）：
    [
      {
        "start": float,
        "end": float,
        "sig": [int, int],
        "confidence": float,
        "source": str
      },
      ...
    ]

设计决策（v0.2.3）：
- key 命名为 "start"/"end"/"sig"，与 v0.2.2 pipeline 写出保持一致
- 不用 "start_time"/"end_time"/"time_signature" 的全名（避免破坏既有的 time_signatures.json）
- "source" 必须落在 TimeSigSource 的 Literal 集合内
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mujik.time_signature.model import TimeSignatureSegment

_VALID_SOURCES: set[str] = set()
# 静态初始化允许的 source 字符串（与 Literal 保持一致）
for _s in ("auto_resnet18", "auto_beatnet", "manual", "default_4_4"):
    _VALID_SOURCES.add(_s)


def _segment_to_dict(seg: TimeSignatureSegment) -> dict[str, Any]:
    """序列化为 dict。"""
    return {
        "start": float(seg.start_time),
        "end": float(seg.end_time),
        "sig": [int(seg.time_signature[0]), int(seg.time_signature[1])],
        "confidence": float(seg.confidence),
        "source": seg.source,
    }


def _dict_to_segment(d: dict[str, Any]) -> TimeSignatureSegment:
    """反序列化为 TimeSignatureSegment。"""
    # 兼容两种 key 命名：新格式 "start"/"end"/"sig"；旧格式 "start_time"/"end_time"/"time_signature"
    start = d.get("start", d.get("start_time"))
    end = d.get("end", d.get("end_time"))
    sig = d.get("sig", d.get("time_signature"))
    confidence = d.get("confidence", 1.0)
    source = d.get("source", "manual")

    if start is None or end is None or sig is None:
        raise ValueError(
            f"time-signature entry missing required keys: start/end/sig, got {d}"
        )

    sig_tuple: tuple[int, int]
    if isinstance(sig, (list, tuple)) and len(sig) == 2:
        sig_tuple = (int(sig[0]), int(sig[1]))
    else:
        raise ValueError(f"sig must be [num, den], got {sig!r}")

    if source not in _VALID_SOURCES:
        # 默认 manual（最保守，CLI 改拍号场景）
        source = "manual"

    return TimeSignatureSegment(
        start_time=float(start),
        end_time=float(end),
        time_signature=sig_tuple,
        confidence=float(confidence),
        source=source,  # type: ignore[arg-type]
    )


def read_time_signatures_json(path: str | Path) -> list[TimeSignatureSegment]:
    """读取 JSON 文件，返回 TimeSignatureSegment 列表。

    若文件不存在或为空，返回空列表。
    """
    p = Path(path)
    if not p.exists():
        return []
    raw = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(
            f"time_signatures.json must be a list, got {type(raw).__name__}"
        )
    return [_dict_to_segment(d) for d in raw]


def write_time_signatures_json(
    segments: list[TimeSignatureSegment],
    path: str | Path,
) -> None:
    """写入 JSON 文件（原子：先写 .tmp 再 rename）。

    父目录若不存在则创建。
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = [_segment_to_dict(s) for s in segments]
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(p)


__all__ = [
    "read_time_signatures_json",
    "write_time_signatures_json",
]
