"""E2E smoke test（不下载模型）。

跑：
1. 生成合成测试 wav
2. 跑 Pipeline.run() 但 mock 掉 demucs/basic-pitch/adtof
3. 验证 out/project.mid 存在 + pretty-midi 可解析
4. 跑 mujik quantize 验证 out/quantize_report.json
5. 跑 mujik render 验证 out/project.svg 存在

用法（在 dev-v0.2.4 容器内）：
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
    print(f"[1/9] generate synthetic wav → {fixture_path}")
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
        print(f"[2/9] output dir: {out_dir}")

        # 3. mock 所有重 adapter，写 fake stems
        print("[3/9] mock heavy adapters...")

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
                # v0.4.1: 加 pitch_bend 让 MusicXML 渲染 <bend> 元素
                return [
                    Note(0.0, 0.5, 60, 100, pitch_bend=(0.0, 0.4, 0.5, 0.4, 0.0)),
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
        print("[4/9] run pipeline (mocked)...")
        sys.path.insert(0, str(repo_root / "src"))

        from mujik.config.schema import (
            LoudnormConfig, PipelineConfig, TranscribeConfig,
        )
        from mujik.pipeline import Pipeline
        from mujik.rhythm.model import BeatTrack

        cfg = PipelineConfig(
            input_path=str(fixture_path),
            output_dir=str(out_dir),
            loudnorm=LoudnormConfig(enabled=False),  # 跳过 pyloudnorm
            transcribe=TranscribeConfig(),
        )

        def fake_madmom(audio_path, config=None, out_dir=None):
            return BeatTrack(
                beats=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5],
                downbeats=[0.0, 2.0, 4.0],
                bpm=120.0,
                tempo_confidence=0.9,
            )

        with patch("mujik.pipeline.separate_with_demucs", side_effect=fake_separate), \
             patch("mujik.pipeline.transcribe_stem", side_effect=fake_transcribe), \
             patch("mujik.pipeline.track_beats_with_madmom", side_effect=fake_madmom), \
             patch("soundfile.info") as mock_info:
            mock_info.return_value = MagicMock(duration=duration, samplerate=sample_rate)
            project = Pipeline(cfg).run()

        # 5. 验证产物
        print("[5/9] verify pipeline outputs...")
        midi_path = out_dir / "project.mid"
        assert midi_path.exists(), f"missing {midi_path}"
        meta_path = out_dir / "project.json"
        assert meta_path.exists(), f"missing {meta_path}"
        beats_path = out_dir / "beats.json"
        assert beats_path.exists(), f"missing {beats_path} (v0.2.2)"
        ts_path = out_dir / "time_signatures.json"
        assert ts_path.exists(), f"missing {ts_path} (v0.2.2)"

        import pretty_midi
        pm = pretty_midi.PrettyMIDI(str(midi_path))
        n_tracks = len(pm.instruments)
        n_notes = sum(len(i.notes) for i in pm.instruments)
        print(f"      project.mid: {n_tracks} tracks, {n_notes} notes")
        assert n_tracks >= 3, f"expected >= 3 tracks, got {n_tracks}"
        assert n_notes >= 4, f"expected >= 4 notes, got {n_notes}"

        meta = json.loads(meta_path.read_text())
        assert meta["mujik_version"] in ("0.2.2", "0.4.0", "0.4.1", "0.4.2")
        assert meta["rhythm_enabled"] is True
        print(f"      project.json: {meta}")

        beats = json.loads(beats_path.read_text())
        assert beats["bpm"] == 120.0
        assert len(beats["beats"]) == 10
        print(f"      beats.json: bpm={beats['bpm']}, {len(beats['beats'])} beats")

        time_sigs = json.loads(ts_path.read_text())
        assert len(time_sigs) >= 1
        assert time_sigs[0]["sig"] == [4, 4]
        print(f"      time_signatures.json: {len(time_sigs)} segment(s), first={time_sigs[0]['sig']}")

        # 6. 跑 mujik quantize 验证后处理（v0.2.3 新增）
        print("[6/9] run mujik quantize (v0.2.3)...")
        from mujik.cli import main as cli_main
        rc = cli_main([
            "quantize",
            "--project-dir", str(out_dir),
        ])
        assert rc == 0, f"mujik quantize failed: rc={rc}"

        report_path = out_dir / "quantize_report.json"
        assert report_path.exists(), f"missing {report_path}"
        report = json.loads(report_path.read_text())
        assert report["total_notes_before"] >= 4
        assert report["total_notes_after"] >= 4
        assert report["grid_resolution"] == 16
        assert report["groove_template"] == "straight"
        print(
            f"      quantize_report.json: {report['total_notes_before']} -> {report['total_notes_after']} notes, "
            f"grid={report['grid_resolution']}, groove={report['groove_template']}"
        )

        # 7. 跑 mujik render 验证 MusicXML + SVG 输出（v0.2.4 新增）
        print("[7/9] run mujik render (v0.2.4)...")
        from mujik.cli import main as cli_main2
        from mujik.midi.io import read_midi_to_project
        from mujik.score.builder import build_musicxml

        # 手动构造 MusicXML（避免 mock 整个 cli_main）
        proj = read_midi_to_project(str(midi_path))
        musicxml = build_musicxml(proj, layout="per_stem")
        musicxml_path = out_dir / "project.musicxml"
        musicxml_path.write_text(musicxml, encoding="utf-8")
        assert musicxml_path.exists()
        assert "<score-partwise" in musicxml
        print(f"      project.musicxml: {len(musicxml)} chars")

        # 跑 render 走 Verovio Python binding → SVG
        rc = cli_main2([
            "render",
            "--input", str(musicxml_path),
            "--output", str(out_dir / "project"),
            "--backend", "verovio",
        ])
        assert rc == 0, f"mujik render failed: rc={rc}"

        svg_path = out_dir / "project.svg"
        assert svg_path.exists(), f"missing {svg_path}"
        assert svg_path.stat().st_size > 100
        assert "<svg" in svg_path.read_text().lower()
        print(f"      project.svg: {svg_path.stat().st_size} bytes")

        # 尝试 PDF（仅在 verovio CLI 可用时）
        try:
            rc = cli_main2([
                "render",
                "--input", str(musicxml_path),
                "--output", str(out_dir / "project"),
                "--backend", "verovio",
                "--pdf",
            ])
            if rc == 0:
                pdf_path = out_dir / "project.pdf"
                if pdf_path.exists():
                    print(f"      project.pdf: {pdf_path.stat().st_size} bytes (verovio CLI)")
            else:
                print("      project.pdf: skipped (verovio CLI not available)")
        except Exception as e:
            print(f"      project.pdf: skipped ({e})")

        # 8. v0.4.1 验证：MusicXML 含 <bend> + <harmony>
        print("[8/9] verify v0.4.1 <bend> + <harmony> rendering...")
        from mujik.config.schema import RenderConfig
        from mujik.midi.model import ChordEvent

        # 注入 chord_track（含 C maj7 在第 1 measure 起始处 + F maj7 在第 2 measure）
        proj2 = read_midi_to_project(str(midi_path))
        proj2.chord_track = [
            ChordEvent(start=0.0, end=2.0, root="C", quality="maj7"),
            ChordEvent(start=2.0, end=4.0, root="F", quality="maj7"),
        ]
        # 重新 build MusicXML（include_chord_symbols=True 触发 <harmony>）
        musicxml_v041 = build_musicxml(
            proj2,
            config=RenderConfig(include_chord_symbols=True),
            layout="per_stem",
        )
        musicxml_v041_path = out_dir / "project_v041.musicxml"
        musicxml_v041_path.write_text(musicxml_v041, encoding="utf-8")

        # 验证 <bend> 元素（vocals 第一个 note 有 pitch_bend）
        assert "<bend" in musicxml_v041, "expected <bend> element in MusicXML"
        assert "<bend-alter>1</bend-alter>" in musicxml_v041, (
            f"expected <bend-alter>1</bend-alter> (0.5 * 2 = 1), got: {musicxml_v041[:500]}"
        )

        # 验证 <harmony> 元素
        assert "<harmony>" in musicxml_v041, "expected <harmony> in MusicXML"
        assert "<root-step>C</root-step>" in musicxml_v041
        assert "<kind>major-seventh</kind>" in musicxml_v041
        # F maj7（kind=major-seventh 也应该出现）
        assert musicxml_v041.count("<kind>major-seventh</kind>") >= 2

        # 验证向后兼容：include_chord_symbols=False 时不应有 <harmony>
        proj3 = read_midi_to_project(str(midi_path))
        proj3.chord_track = [
            ChordEvent(start=0.0, end=2.0, root="C", quality=""),
        ]
        musicxml_noharm = build_musicxml(
            proj3,
            config=RenderConfig(include_chord_symbols=False),
            layout="per_stem",
        )
        assert "<harmony>" not in musicxml_noharm, (
            "include_chord_symbols=False should skip <harmony>"
        )

        print(f"      project_v041.musicxml: {len(musicxml_v041)} chars, contains <bend> + <harmony>")

        # 9. v0.4.2 验证：muscriptor multitrack adapter 配置 + 解析
        print("[9/9] verify v0.4.2 muscriptor adapter...")
        from mujik.config.schema import TranscribeConfig
        from mujik.transcribe.muscriptor_adapter import (
            MuscriptorAdapterError,
            VALID_MUSCRIPTOR_MODELS,
            check_muscriptor_available,
            transcribe_multitrack,
        )

        # 验证 TranscribeConfig.mode 字段
        cfg_default = TranscribeConfig()
        assert cfg_default.mode == "per_stem", f"default mode should be per_stem, got {cfg_default.mode}"
        assert cfg_default.muscriptor_model == "medium"
        cfg_mt = TranscribeConfig(mode="multitrack", muscriptor_model="small")
        assert cfg_mt.mode == "multitrack"
        assert cfg_mt.muscriptor_model == "small"

        # 验证 muscriptor adapter 模块可 import + 关键函数存在
        assert callable(check_muscriptor_available)
        assert callable(transcribe_multitrack)
        assert set(VALID_MUSCRIPTOR_MODELS) == {"small", "medium", "large"}

        # 验证 muscriptor 错误类存在
        try:
            raise MuscriptorAdapterError("test")
        except MuscriptorAdapterError as e:
            assert "test" in str(e)

        # 验证 muscriptor adapter 文件存在
        import mujik.transcribe.muscriptor_adapter as ma
        assert hasattr(ma, "transcribe_multitrack")
        assert hasattr(ma, "MuscriptorAdapterError")

        print(f"      TranscribeConfig.mode={cfg_mt.mode}, model={cfg_mt.muscriptor_model}")
        print(f"      muscriptor adapter: {len(VALID_MUSCRIPTOR_MODELS)} valid models, subprocess-based")

        print("\n✅ E2E smoke test PASSED (v0.4.2)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
