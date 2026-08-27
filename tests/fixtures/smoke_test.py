"""E2E smoke test（不下载模型）。

跑：
1. 生成合成测试 wav
2. 跑 Pipeline.run() 但 mock 掉 demucs/basic-pitch/adtof
3. 验证 out/project.mid 存在 + pretty-midi 可解析

用法（在 dev-v0.2.1 容器内）：
    python tests/fixtures/smoke_test.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np


def main() -> int:
    """执行 smoke test。"""
    repo_root = Path(__file__).resolve().parents[2]
    fixture_path = repo_root / "tests" / "fixtures" / "synthetic_5s.wav"

    # 1. 生成 wav
    print(f"[1/5] generate synthetic wav → {fixture_path}")
    sample_rate = 44100
    duration = 5.0
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    audio = (
        0.2 * np.sin(2 * np.pi * 440 * t) +
        0.3 * np.sin(2 * np.pi * 200 * np.mod(t, 1.0))
    ).astype(np.float32)
    fixture_path.parent.mkdir(parents=True, exist_ok=True)
    import soundfile as sf
    sf.write(str(fixture_path), audio, sample_rate)
    print(f"      ok ({audio.shape[0]} samples)")

    # 2. 准备 output dir
    with tempfile.TemporaryDirectory(prefix="mujik_smoke_") as tmp:
        out_dir = Path(tmp)
        print(f"[2/5] output dir: {out_dir}")

        # 3. mock 所有重 adapter，写 fake stems
        print("[3/5] mock heavy adapters...")

        def fake_separate(input_path, stems_dir, config=None):
            stems_dir = Path(stems_dir)
            track_dir = stems_dir / "htdemucs_ft" / Path(input_path).stem
            track_dir.mkdir(parents=True, exist_ok=True)
            for name in ("vocals", "drums", "bass", "other"):
                p = track_dir / f"{name}.wav"
                sf.write(str(p), audio[:22050], sample_rate)  # 0.5s 占位
            from mujik.separate.model import Stem, Stems
            s = Stems(
                separation_model="demucs/htdemucs_ft",
                sample_rate=sample_rate,
                total_duration=duration,
            )
            for name in ("vocals", "drums", "bass", "other"):
                s.add(Stem(
                    name=name,  # type: ignore[arg-type]
                    audio_path=track_dir / f"{name}.wav",
                    sample_rate=sample_rate,
                    duration=duration,
                    source_model="demucs/htdemucs_ft",
                ))
            return s

        def fake_transcribe(stem, config=None, out_dir=None):
            from mujik.midi.model import Note
            if stem.name == "vocals":
                return [
                    Note(0.0, 0.5, 60, 100),
                    Note(0.5, 1.0, 62, 90),
                ]
            if stem.name == "drums":
                return [
                    Note(1.0, 1.05, 36, 100),
                    Note(2.0, 2.05, 38, 100),
                ]
            if stem.name == "bass":
                return [Note(1.0, 2.0, 40, 110)]
            return []

        # 4. 跑 pipeline
        print("[4/5] run pipeline (mocked)...")
        sys.path.insert(0, str(repo_root / "src"))

        from mujik.config.schema import (
            LoudnormConfig, PipelineConfig, TranscribeConfig,
        )
        from mujik.pipeline import Pipeline

        cfg = PipelineConfig(
            input_path=str(fixture_path),
            output_dir=str(out_dir),
            loudnorm=LoudnormConfig(enabled=False),  # 跳过 pyloudnorm
            transcribe=TranscribeConfig(),
        )
        with patch("mujik.pipeline.separate_with_demucs", side_effect=fake_separate), \
             patch("mujik.pipeline.transcribe_stem", side_effect=fake_transcribe), \
             patch("soundfile.info") as mock_info:
            mock_info.return_value = MagicMock(duration=duration, samplerate=sample_rate)
            project = Pipeline(cfg).run()

        # 5. 验证产物
        print("[5/5] verify outputs...")
        midi_path = out_dir / "project.mid"
        assert midi_path.exists(), f"missing {midi_path}"
        meta_path = out_dir / "project.json"
        assert meta_path.exists(), f"missing {meta_path}"

        import pretty_midi
        pm = pretty_midi.PrettyMIDI(str(midi_path))
        n_tracks = len(pm.instruments)
        n_notes = sum(len(i.notes) for i in pm.instruments)
        print(f"      project.mid: {n_tracks} tracks, {n_notes} notes")
        assert n_tracks >= 3, f"expected >= 3 tracks, got {n_tracks}"
        assert n_notes >= 4, f"expected >= 4 notes, got {n_notes}"

        meta = json.loads(meta_path.read_text())
        assert meta["mujik_version"] == "0.2.1"
        print(f"      project.json: {meta}")

        print("\n✅ E2E smoke test PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
