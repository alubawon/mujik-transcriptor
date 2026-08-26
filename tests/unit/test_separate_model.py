"""Tests for separate.model.Stem / Stems."""
from __future__ import annotations

from pathlib import Path

import pytest

from mujik.separate.model import Stem, Stems


class TestStem:
    def test_valid(self, tmp_path: Path):
        audio = tmp_path / "vocals.wav"
        audio.write_bytes(b"RIFF")
        s = Stem(
            name="vocals", audio_path=audio,
            sample_rate=44100, duration=180.0, source_model="htdemucs_ft",
        )
        assert s.name == "vocals"
        assert s.sample_rate == 44100

    def test_invalid_stem_name(self, tmp_path: Path):
        with pytest.raises(ValueError):
            Stem(
                name="nonexistent",  # type: ignore[arg-type]
                audio_path=tmp_path / "x.wav",
                sample_rate=44100, duration=10.0, source_model="x",
            )

    def test_invalid_sample_rate(self, tmp_path: Path):
        with pytest.raises(ValueError):
            Stem(
                name="vocals", audio_path=tmp_path / "x.wav",
                sample_rate=0, duration=10.0, source_model="x",
            )


class TestStems:
    def test_add_and_query(self, tmp_path: Path):
        st = Stems(separation_model="demucs", sample_rate=44100, total_duration=100.0)
        for name in ("vocals", "drums", "bass", "other"):
            audio = tmp_path / f"{name}.wav"
            audio.write_bytes(b"x")
            st.add(Stem(
                name=name, audio_path=audio,
                sample_rate=44100, duration=100.0, source_model="htdemucs_ft",
            ))
        assert st.stem_count == 4
        assert set(st.names) == {"vocals", "drums", "bass", "other"}

    def test_require_missing(self):
        st = Stems()
        with pytest.raises(KeyError):
            st.require("drums")

    def test_primary_stems(self, tmp_path: Path):
        st = Stems()
        for name in ("vocals", "drums", "bass", "other"):
            audio = tmp_path / f"{name}.wav"
            audio.write_bytes(b"x")
            st.add(Stem(
                name=name, audio_path=audio,
                sample_rate=44100, duration=100.0, source_model="x",
            ))
        primaries = st.primary_stems()
        assert [s.name for s in primaries] == ["vocals", "drums", "bass", "other"]
