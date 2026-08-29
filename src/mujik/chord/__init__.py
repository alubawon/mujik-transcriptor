"""和声识别模块（v0.4.4 madmom + v0.4.5 quantize + v0.4.8 BTC-HCQT + v0.4.9 groove）。"""
from __future__ import annotations

from mujik.chord.btc_hcqt_adapter import (
    BTC_HCQT_TIMEOUT_DEFAULT,
    BtcHcqtAdapterError,
    check_btc_hcqt_available,
    detect_chords_with_btc,
)
from mujik.chord.groove import apply_groove_to_chord_track
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
    "BTC_HCQT_TIMEOUT_DEFAULT",
    "BtcHcqtAdapterError",
    "check_btc_hcqt_available",
    "detect_chords_with_btc",
    "apply_groove_to_chord_track",
]
