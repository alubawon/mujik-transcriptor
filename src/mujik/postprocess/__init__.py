"""postprocess 模块（v0.4.0 新增）。

公开 API：
    pitch_bend:
        inject_pitch_bends_to_pretty_midi
        extract_pitch_bends_from_pretty_midi
        bend_to_pretty_pitch / pretty_pitch_to_bend
"""
from __future__ import annotations

from mujik.postprocess.pitch_bend import (
    DEFAULT_FRAME_RATE_HZ,
    PITCH_BEND_CENTER,
    PITCH_BEND_MAX,
    bend_to_pretty_pitch,
    extract_pitch_bends_from_pretty_midi,
    inject_pitch_bends_to_pretty_midi,
    pretty_pitch_to_bend,
)

__all__ = [
    "PITCH_BEND_CENTER",
    "PITCH_BEND_MAX",
    "DEFAULT_FRAME_RATE_HZ",
    "bend_to_pretty_pitch",
    "pretty_pitch_to_bend",
    "inject_pitch_bends_to_pretty_midi",
    "extract_pitch_bends_from_pretty_midi",
]
