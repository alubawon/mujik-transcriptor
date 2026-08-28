"""和声识别模块（v0.4.4）。"""
from __future__ import annotations

from mujik.chord.madmom_adapter import (
    MADMOM_CHORD_TIMEOUT_DEFAULT,
    MadmomChordAdapterError,
    check_madmom_chord_available,
    detect_chords_with_madmom,
)

__all__ = [
    "MADMOM_CHORD_TIMEOUT_DEFAULT",
    "MadmomChordAdapterError",
    "check_madmom_chord_available",
    "detect_chords_with_madmom",
]
