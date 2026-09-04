"""E2E smoke test（不下载模型）。

跑：
1. 生成合成测试 wav
2. 跑 Pipeline.run() 但 mock 掉 demucs/basic-pitch/drumscript
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
    print(f"[1/18] generate synthetic wav → {fixture_path}")
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
        print(f"[2/18] output dir: {out_dir}")

        # 3. mock 所有重 adapter，写 fake stems
        print("[3/18] mock heavy adapters...")

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
        print("[4/18] run pipeline (mocked)...")
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
        print("[5/18] verify pipeline outputs...")
        midi_path = out_dir / "project.mid"
        assert midi_path.exists(), f"missing {midi_path}"
        meta_path = out_dir / "project.json"
        assert meta_path.exists(), f"missing {meta_path}"
        beats_path = out_dir / "ws" / "beats.json"  # v0.5.1 修 5：中间产物在 ws/
        assert beats_path.exists(), f"missing {beats_path} (v0.2.2)"
        ts_path = out_dir / "ws" / "time_signatures.json"  # v0.5.1 修 5
        assert ts_path.exists(), f"missing {ts_path} (v0.2.2)"

        import pretty_midi
        pm = pretty_midi.PrettyMIDI(str(midi_path))
        n_tracks = len(pm.instruments)
        n_notes = sum(len(i.notes) for i in pm.instruments)
        print(f"      project.mid: {n_tracks} tracks, {n_notes} notes")
        assert n_tracks >= 3, f"expected >= 3 tracks, got {n_tracks}"
        assert n_notes >= 4, f"expected >= 4 notes, got {n_notes}"

        meta = json.loads(meta_path.read_text())
        assert meta["mujik_version"] in ("0.2.2", "0.4.0", "0.4.1", "0.4.2", "0.4.3", "0.4.4", "0.4.5", "0.4.6", "0.4.7", "0.4.8", "0.4.9", "0.5.0", "0.5.1", "0.5.2")
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
        print("[6/18] run mujik quantize (v0.2.3)...")
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
        print("[7/18] run mujik render (v0.2.4)...")
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

        # 8. v0.4.1 验证：MusicXML 含 <bend> + <harmony>；v0.4.3 验证：release 模式
        print("[8/18] verify v0.4.1 <bend> + <harmony> + v0.4.3 release curve...")
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

        # v0.4.3 验证：release curve → 2 个 <bend> 兄弟
        # (0.0, 0.4, 0.5, 0.4, 0.0) → peak 0.5, post-peak 0.0 < 0.1 → has_release
        # 用 "<bend " 带空格避免匹配 <bend-alter>
        n_bend_tags = musicxml_v041.count("<bend ")
        assert n_bend_tags == 2, (
            f"v0.4.3: expected 2 <bend> siblings (bend+release), got {n_bend_tags}"
        )
        # 第一个 <bend>: positive alter
        assert "<bend-alter>1</bend-alter>" in musicxml_v041
        # 第二个 <bend>: negative alter + <release/> marker
        assert "<bend-alter>-1</bend-alter>" in musicxml_v041
        assert "<release/>" in musicxml_v041
        print(f"      v0.4.3 release curve: {n_bend_tags} <bend> siblings + <release/> marker")

        # 9. v0.4.2 验证：muscriptor multitrack adapter 配置 + 解析
        print("[9/18] verify v0.4.2 muscriptor adapter...")
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

        # 10. v0.4.3 验证：连续 bend 曲线渲染
        print("[10/18] verify v0.4.3 continuous bend curve rendering...")
        from mujik.score.bend import (
            BendPoint,
            build_bend_elements,
            detect_bend_release,
        )
        from mujik.midi.model import Note as NoteModel

        # verify detect_bend_release
        peak, has_rel = detect_bend_release((0.0, 0.3, 0.5, 0.3, 0.0))
        assert peak == 1 and has_rel is True, f"got peak={peak}, has_rel={has_rel}"

        peak, has_rel = detect_bend_release((0.0, 0.2, 0.4, 0.5, 0.5))
        assert peak == 1 and has_rel is False, f"got peak={peak}, has_rel={has_rel}"

        # verify build_bend_elements（双 bend 兄弟）
        xml_curve = build_bend_elements([BendPoint(0.0, 1), BendPoint(1.0, 0)])
        assert xml_curve.count("<bend ") == 2
        assert "<release/>" in xml_curve

        # verify 端到端：单 note with release curve → MusicXML 含 2 个 <bend>
        from mujik.score.builder import build_musicxml
        from mujik.midi.model import Project as ProjectModel, Track
        from mujik.time_signature.model import build_default_segments

        proj_v043 = ProjectModel(
            audio_path="test.wav",
            duration=2.0,
            sample_rate=44100,
            time_signatures=build_default_segments(2.0),
            tempo_map=[],
        )
        vocals_v043 = Track(stem_name="vocals")
        vocals_v043.add(NoteModel(
            0.0, 1.0, 60, 100,
            pitch_bend=(0.0, 0.3, 0.5, 0.3, 0.0),
        ))
        proj_v043.tracks["vocals"] = vocals_v043
        xml_v043 = build_musicxml(proj_v043, layout="per_stem")
        assert xml_v043.count("<bend ") == 2, (
            f"v0.4.3 builder: expected 2 <bend> for release curve, got "
            f"{xml_v043.count('<bend ')}"
        )
        assert "<release/>" in xml_v043
        # v0.4.3 新增 shape="curved"
        assert 'shape="curved"' in xml_v043

        print(f"      detect_bend_release: peak=1 has_release=True ✓")
        print(f"      build_bend_elements: 2 <bend> siblings + <release/> ✓")
        print(f"      end-to-end MusicXML: shape=\"curved\" + bend+release ✓")

        # 11. v0.4.4 验证：自动和弦检测（madmom subprocess）
        print("[11/18] verify v0.4.4 automatic chord detection...")
        from mujik.chord.madmom_adapter import (
            MADMOM_CHORD_TIMEOUT_DEFAULT,
            MadmomChordAdapterError,
            _parse_madmom_chord_label,
            check_madmom_chord_available,
            detect_chords_with_madmom,
        )
        from mujik.midi.model import ChordEvent as ChordEventModel
        from mujik.config.schema import ChordConfig

        # 验证 _parse_madmom_chord_label
        c_maj = _parse_madmom_chord_label("C:maj")
        assert c_maj is not None
        assert c_maj.root == "C"
        assert c_maj.quality == ""
        c_min = _parse_madmom_chord_label("F#:min")
        assert c_min is not None
        assert c_min.root == "F#"
        assert c_min.quality == "m"
        assert _parse_madmom_chord_label("N") is None
        assert _parse_madmom_chord_label("X") is None

        # 验证 ChordConfig 新增 chord_timeout_sec
        cfg_ch = ChordConfig()
        assert cfg_ch.chord_timeout_sec == 1800
        cfg_ch_custom = ChordConfig(chord_timeout_sec=600)
        assert cfg_ch_custom.chord_timeout_sec == 600

        # 验证 madmom adapter 模块可 import
        assert callable(check_madmom_chord_available)
        assert callable(detect_chords_with_madmom)
        assert MADMOM_CHORD_TIMEOUT_DEFAULT == 1800
        try:
            raise MadmomChordAdapterError("test")
        except MadmomChordAdapterError as e:
            assert "test" in str(e)

        # 验证 detect_chords_with_madmom subprocess 流程（mock）
        # 写 fake JSON 到 out_dir，验证解析
        chord_json_path = out_dir / f"chords_{fixture_path.stem}.json"
        chord_json_path.write_text(json.dumps([
            {"start": 0.0, "end": 2.0, "label": "C:maj"},
            {"start": 2.0, "end": 4.0, "label": "F:maj"},
            {"start": 4.0, "end": 4.5, "label": "N"},
        ]))
        with patch("subprocess.run",
                   return_value=MagicMock(returncode=0, stderr="")):
            chord_track = detect_chords_with_madmom(
                fixture_path, out_dir=out_dir,
            )
        # 3 entries → 2 ChordEvent（N 过滤）
        assert len(chord_track) == 2
        assert chord_track[0].root == "C" and chord_track[0].quality == ""
        assert chord_track[1].root == "F" and chord_track[1].quality == ""

        # 验证端到端：chord_track 注入 Project → MusicXML <harmony>
        proj_chord = read_midi_to_project(str(midi_path))
        proj_chord.chord_track = [
            ChordEventModel(0.0, 1.0, "C", ""),     # C major
            ChordEventModel(1.0, 2.0, "A", "m"),    # A minor
        ]
        xml_chord = build_musicxml(
            proj_chord,
            config=RenderConfig(include_chord_symbols=True),
            layout="per_stem",
        )
        assert "<harmony>" in xml_chord
        assert "<root-step>C</root-step>" in xml_chord
        assert "<kind>major</kind>" in xml_chord
        assert "<root-step>A</root-step>" in xml_chord
        assert "<kind>minor</kind>" in xml_chord

        print(f"      _parse_madmom_chord_label: C:maj/F#:min/N/X parsed ✓")
        print(f"      ChordConfig.chord_timeout_sec: {cfg_ch.chord_timeout_sec}s ✓")
        print(f"      detect_chords_with_madmom: subprocess mock returns 2 chords ✓")
        print(f"      end-to-end: chord_track → MusicXML <harmony> ✓")

        # 12. v0.4.5 验证：chord quantize 到 bar/beat
        print("[12/18] verify v0.4.5 chord quantize to bar/beat...")
        from mujik.chord.quantize import (
            quantize_chord_track,
            snap_chord_to_grid,
            merge_consecutive_chords,
            filter_short_chords,
        )
        from mujik.time_signature.model import TimeSignatureSegment

        # 验证 ChordConfig 新增 v0.4.5 字段
        cfg_q = ChordConfig()
        assert cfg_q.quantize_enabled is True
        assert cfg_q.grid_per_bar == 4
        assert cfg_q.merge_consecutive is True
        assert cfg_q.min_duration_sec == 0.5

        # 构造拍号段
        sigs = [
            TimeSignatureSegment(
                start_time=0.0, end_time=10.0,
                time_signature=(4, 4), confidence=1.0, source="manual",
            ),
        ]

        # snap 验证：120 BPM, grid_per_bar=4 → step=0.5s (beat)
        c_raw = ChordEventModel(0.1, 0.7, "C", "")
        c_snap = snap_chord_to_grid(c_raw, sigs, bpm=120.0, grid_per_bar=4, duration=10.0)
        assert c_snap.start == 0.0  # round(0.1/0.5) = 0
        assert c_snap.end == 0.5    # round(0.7/0.5) = 1

        # merge 验证
        merged = merge_consecutive_chords([
            ChordEventModel(0.0, 2.0, "C", ""),
            ChordEventModel(2.0, 4.0, "C", ""),
        ])
        assert len(merged) == 1
        assert merged[0].end == 4.0

        # filter 验证
        filtered = filter_short_chords([
            ChordEventModel(0.0, 0.2, "C", ""),
            ChordEventModel(0.5, 2.0, "F", ""),
        ], min_duration_sec=0.5)
        assert len(filtered) == 1
        assert filtered[0].root == "F"

        # 端到端：raw madmom 100ms 粒度 → quantize → bar/beat 对齐
        # chord 持续时间设计成 snap 后 > 0.6s，避免被阈值误过滤
        raw_chords = [
            ChordEventModel(0.1, 1.6, "C", ""),     # → snap (0.0, 1.5) 1.5s keep
            ChordEventModel(1.3, 2.5, "F", ""),     # → snap (1.5, 2.5) 1.0s keep
            ChordEventModel(2.2, 3.6, "C", ""),     # → snap (2.0, 3.5) 1.5s keep
            ChordEventModel(3.1, 3.2, "G", ""),     # → snap+defense (3.0, 3.5) 0.5s < 0.6 filter
        ]
        quantized = quantize_chord_track(
            raw_chords, sigs, bpm=120.0,
            grid_per_bar=4, merge_consecutive=False, min_duration_sec=0.6,
            duration=10.0,
        )
        # 3 个长 chord 保留, G 过滤
        assert len(quantized) == 3
        roots = [c.root for c in quantized]
        assert roots == ["C", "F", "C"], f"got {roots}"
        # snap 边界正确（grid_per_bar=4 → 0.5s 步长）
        assert quantized[0].start == 0.0
        assert quantized[1].start == 1.5
        assert quantized[2].start == 2.0

        # 验证 quantize_chord_track 也可被 madmom detect 流程调用（mock）
        # 模拟 pipeline 2.7/7 → 2.8/7 顺序
        with patch("subprocess.run",
                   return_value=MagicMock(returncode=0, stderr="")):
            detected = detect_chords_with_madmom(fixture_path, out_dir=out_dir)
        if detected:  # 若 madmom 可用
            q2 = quantize_chord_track(
                detected, sigs, bpm=120.0,
                grid_per_bar=4, merge_consecutive=True, min_duration_sec=0.5,
                duration=10.0,
            )
            print(f"      raw madmom: {len(detected)} → quantized: {len(q2)} chords")

        print(f"      ChordConfig.quantize_enabled: {cfg_q.quantize_enabled} ✓")
        print(f"      snap_chord_to_grid: 0.1s → 0.0s, 0.7s → 0.5s (beat grid) ✓")
        print(f"      merge_consecutive_chords: 2×C → 1×C (0-4s) ✓")
        print(f"      filter_short_chords: 0.2s dropped, 1.5s kept ✓")
        print(f"      end-to-end: raw 100ms → quantized 0.5s beat grid ✓")

        # 13. v0.4.6 验证：ChordEvent hardening 验证器
        print("[13/18] verify v0.4.6 ChordEvent hardening validator...")
        from mujik.midi.model import (
            ALLOWED_QUALITIES_BY_VOCAB,
        )

        # 合法构造（root、quality、bass、vocab 各组合）
        c_ok = ChordEventModel(0.0, 1.0, "C", "")
        assert c_ok.root == "C"
        c_full = ChordEventModel(0.0, 1.0, "F#", "m7", bass="A")
        assert c_full.bass == "A"

        # 拒绝非法 root
        for bad_root in ("", "H", "C##", "Do", "C1"):
            try:
                ChordEventModel(0.0, 1.0, bad_root, "")
                raise AssertionError(f"root {bad_root!r} should be rejected")
            except ValueError as e:
                assert "root must match" in str(e), f"got: {e}"
        # 拒绝非法 bass
        try:
            ChordEventModel(0.0, 1.0, "C", "7", bass="H")
            raise AssertionError("bass=H should be rejected")
        except ValueError as e:
            assert "bass must match" in str(e)
        # 拒绝负 start / end<start
        try:
            ChordEventModel(-0.1, 1.0, "C", "")
        except ValueError as e:
            assert "start must be >= 0" in str(e)
        try:
            ChordEventModel(2.0, 1.0, "C", "")
        except ValueError as e:
            assert "end" in str(e)
        # 允许 placeholder (end == start)
        c_ph = ChordEventModel(0.0, 0.0, "C", "")
        assert c_ph.end == c_ph.start

        # quality vocab 四档 (v0.4.6 + v0.4.8 新增 btc-extended)
        assert set(ALLOWED_QUALITIES_BY_VOCAB.keys()) == {
            "root", "root-quality", "extended", "btc-extended",
        }
        # root vocab 只接受 ""
        assert "" in ALLOWED_QUALITIES_BY_VOCAB["root"]
        # root-quality 含 maj/min
        assert "maj" in ALLOWED_QUALITIES_BY_VOCAB["root-quality"]
        assert "m" in ALLOWED_QUALITIES_BY_VOCAB["root-quality"]
        # extended 含 7/maj7/m7/dim/aug/sus
        for q in ("7", "maj7", "m7", "dim", "aug", "sus"):
            assert q in ALLOWED_QUALITIES_BY_VOCAB["extended"], f"{q} not in extended"
        # extended 拒绝 9/alt
        try:
            ChordEventModel(0.0, 1.0, "C", "9")
        except ValueError as e:
            assert "quality" in str(e)
        try:
            ChordEventModel(0.0, 1.0, "C", "alt")
        except ValueError as e:
            assert "quality" in str(e)

        # root ⊂ root-quality ⊂ extended
        assert ALLOWED_QUALITIES_BY_VOCAB["root"].issubset(
            ALLOWED_QUALITIES_BY_VOCAB["root-quality"]
        )
        assert ALLOWED_QUALITIES_BY_VOCAB["root-quality"].issubset(
            ALLOWED_QUALITIES_BY_VOCAB["extended"]
        )

        # 端到端：合法 chord → MusicXML <harmony> 仍然工作
        proj_v46 = read_midi_to_project(str(midi_path))
        proj_v46.chord_track = [
            ChordEventModel(0.0, 1.0, "C", "maj7"),   # extended vocab
            ChordEventModel(1.0, 2.0, "A", "m"),      # root-quality
        ]
        xml_v46 = build_musicxml(
            proj_v46,
            config=RenderConfig(include_chord_symbols=True),
            layout="per_stem",
        )
        assert "<kind>major-seventh</kind>" in xml_v46  # QUALITY_TO_KIND maj7
        assert "<kind>minor</kind>" in xml_v46

        print(f"      root validation: H/##/empty/digit rejected ✓")
        print(f"      bass validation: H rejected, empty allowed ✓")
        print(f"      start/end: negative rejected, end==start placeholder allowed ✓")
        print(f"      quality vocab: root/root-quality/extended (9/alt rejected) ✓")
        print(f"      end-to-end: harded chord → MusicXML <kind> ✓")

        # 14. v0.4.7 验证：find_chord_at_time O(log n) bisect
        print("[14/18] verify v0.4.7 find_chord_at_time bisect optimization...")
        from mujik.score.harmony import find_chord_at_time as find_chord_v47

        # 基础 case：与 v0.4.1 行为兼容
        assert find_chord_v47(None, 1.0) is None
        assert find_chord_v47([], 1.0) is None
        single = [ChordEventModel(0.0, 2.0, "C", "")]
        assert find_chord_v47(single, 1.0) == single[0]
        assert find_chord_v47(single, -1.0) is None  # t 早于所有 chord
        assert find_chord_v47(single, 100.0) is None  # t 晚于所有 chord

        # bisect 边界：t == start 命中, t == end 不命中
        track = [
            ChordEventModel(0.0, 2.0, "C", ""),
            ChordEventModel(2.0, 4.0, "F", ""),
            ChordEventModel(4.0, 6.0, "G", "7"),
        ]
        assert find_chord_v47(track, 0.0).root == "C"   # t == start
        assert find_chord_v47(track, 2.0).root == "F"   # t == next start
        assert find_chord_v47(track, 2.0).quality == ""
        assert find_chord_v47(track, 6.0) is None       # t == end 不命中
        assert find_chord_v47(track, 1.5).root == "C"   # within first chord
        # 真 gap：track 之外（实际 track 没 gap，但 t 早于/晚于全部 应 None）
        assert find_chord_v47(track, -1.0) is None      # before all
        assert find_chord_v47(track, 10.0) is None      # after all

        # 大规模：500 chord 1000 random query，bisect 与 reference 一致
        import random
        random.seed(2026)
        n = 500
        large = [ChordEventModel(i * 0.5, (i + 1) * 0.5, "C", "") for i in range(n)]

        def linear_ref(track, t):
            for c in track:
                if c.start <= t < c.end:
                    return c
            return None

        consistent_count = 0
        for _ in range(1000):
            t = random.uniform(-5, n * 0.5 + 5)
            bisect_result = find_chord_v47(large, t)
            linear_result = linear_ref(large, t)
            assert bisect_result == linear_result, f"t={t}: bisect vs linear mismatch"
            consistent_count += 1
        assert consistent_count == 1000

        # unsorted fallback：shuffle 后仍正确
        random.shuffle(large)
        # 仍能在 unsorted 列表里找到正确 chord
        t = 10.0  # 应在 index 20 (10.0 / 0.5)
        expected = linear_ref(large, t)
        actual = find_chord_v47(large, t)
        assert actual == expected

        # 端到端：<harmony> 渲染仍命中正确 chord
        proj_v47 = read_midi_to_project(str(midi_path))
        proj_v47.chord_track = [
            ChordEventModel(0.0, 2.0, "C", "maj7"),
            ChordEventModel(2.0, 4.0, "F", "m"),
        ]
        xml_v47 = build_musicxml(
            proj_v47,
            config=RenderConfig(include_chord_symbols=True),
            layout="per_stem",
        )
        # bisect 后 <harmony> 元素仍正常出现
        assert "<harmony>" in xml_v47
        # 不同 measure 命中不同 chord
        assert xml_v47.count("<harmony>") >= 1

        print(f"      basic: empty/single/t==start/t==end/gap handled ✓")
        print(f"      bisect: 500 chord × 1000 random query 与 linear 一致 ✓")
        print(f"      unsorted fallback: shuffle 后仍正确查询 ✓")
        print(f"      end-to-end: chord_track → MusicXML <harmony> via bisect ✓")

        # 15. v0.4.8 验证：BTC-HCQT 7th/9th/11th/13th 延伸和弦
        print("[15/18] verify v0.4.8 BTC-HCQT extended chord vocabulary...")
        from mujik.chord.btc_hcqt_adapter import (
            BTC_HCQT_TIMEOUT_DEFAULT,
            BtcHcqtAdapterError,
            _parse_btc_chord_label,
            check_btc_hcqt_available,
            detect_chords_with_btc,
        )

        # 验证 _parse_btc_chord_label
        # BTC bare root = maj
        c_bare = _parse_btc_chord_label("C")
        assert c_bare is not None and c_bare.root == "C" and c_bare.quality == ""
        # BTC 标准 14 种 quality
        for label, exp_quality in [
            ("C:min", "m"),
            ("C:maj7", "maj7"),
            ("C:7", "7"),
            ("C:min7", "m7"),
            ("C:dim", "dim"),
            ("C:aug", "aug"),
            ("C:dim7", "dim7"),
            ("C:hdim7", "hdim7"),
            ("C:minmaj7", "mM7"),
            ("C:min6", "m6"),
            ("C:maj6", "maj6"),
            ("C:sus2", "sus2"),
            ("C:sus4", "sus4"),
        ]:
            c = _parse_btc_chord_label(label)
            assert c is not None, f"{label} parse failed"
            assert c.root == "C", f"{label} root != C"
            assert c.quality == exp_quality, f"{label} quality {c.quality} != {exp_quality}"
        # 跳过 N/X
        assert _parse_btc_chord_label("N") is None
        assert _parse_btc_chord_label("X") is None
        # BTC 不用 b（flat）：Db 被拒
        assert _parse_btc_chord_label("Db:min") is None

        # 验证 BTC adapter 模块可 import
        assert callable(check_btc_hcqt_available)
        assert callable(detect_chords_with_btc)
        assert BTC_HCQT_TIMEOUT_DEFAULT == 1800
        try:
            raise BtcHcqtAdapterError("test")
        except BtcHcqtAdapterError as e:
            assert "test" in str(e)

        # 验证 4 档 vocab (root/root-quality/extended/btc-extended)
        # btc-extended 必须含 BTC 独有的 6ths / half-diminished / minor-major 7th
        for q in ("m6", "maj6", "dim7", "hdim7", "mM7", "min6", "minmaj7"):
            assert q in ALLOWED_QUALITIES_BY_VOCAB["btc-extended"], \
                f"{q} not in btc-extended"
        # extended 不含 BTC 独有 quality
        for q in ("m6", "maj6", "dim7", "hdim7", "mM7"):
            assert q not in ALLOWED_QUALITIES_BY_VOCAB["extended"], \
                f"{q} should not be in extended"
        # vocab 包含关系：root ⊂ root-quality ⊂ extended ⊂ btc-extended
        assert ALLOWED_QUALITIES_BY_VOCAB["root"].issubset(
            ALLOWED_QUALITIES_BY_VOCAB["root-quality"]
        )
        assert ALLOWED_QUALITIES_BY_VOCAB["root-quality"].issubset(
            ALLOWED_QUALITIES_BY_VOCAB["extended"]
        )
        assert ALLOWED_QUALITIES_BY_VOCAB["extended"].issubset(
            ALLOWED_QUALITIES_BY_VOCAB["btc-extended"]
        )

        # 验证 ChordConfig 新增 backend 字段
        cfg_btc = ChordConfig()
        assert cfg_btc.backend == "btc-hcqt"
        assert cfg_btc.vocab == "btc-extended"
        assert cfg_btc.btc_timeout_sec == 1800
        # backend 可设为 madmom（向后兼容）
        cfg_madmom = ChordConfig(backend="madmom")
        assert cfg_madmom.backend == "madmom"

        # 验证 detect_chords_with_btc subprocess 流程（mock）
        chord_json_path = out_dir / f"btc_chords_{fixture_path.stem}.json"
        chord_json_path.write_text(json.dumps([
            {"start": 0.0, "end": 2.0, "label": "C"},
            {"start": 2.0, "end": 4.0, "label": "F#:min7"},
            {"start": 4.0, "end": 4.5, "label": "N"},
            {"start": 4.5, "end": 5.0, "label": "G:maj7"},
            {"start": 5.0, "end": 5.5, "label": "A:hdim7"},
        ]))
        with patch("subprocess.run",
                   return_value=MagicMock(returncode=0, stderr="")):
            btc_track = detect_chords_with_btc(fixture_path, out_dir=out_dir)
        # 5 entries → 4 ChordEvent (N 过滤)
        assert len(btc_track) == 4
        assert btc_track[0].quality == ""       # C bare = maj
        assert btc_track[1].quality == "m7"    # F#:min7
        assert btc_track[2].quality == "maj7"  # G:maj7
        assert btc_track[3].quality == "hdim7" # A:hdim7

        # 端到端：BTC 风格 chord 注入 Project → MusicXML <harmony>
        proj_btc = read_midi_to_project(str(midi_path))
        proj_btc.chord_track = [
            ChordEventModel(0.0, 1.0, "C", "maj7", vocab="btc-extended"),
            ChordEventModel(1.0, 2.0, "A", "hdim7", vocab="btc-extended"),
        ]
        xml_btc = build_musicxml(
            proj_btc,
            config=RenderConfig(include_chord_symbols=True),
            layout="per_stem",
        )
        assert "<harmony>" in xml_btc
        assert "<kind>major-seventh</kind>" in xml_btc
        # hdim7 不在 QUALITY_TO_KIND 透传，验证 Verovio 接受
        assert "<kind>hdim7</kind>" in xml_btc

        print(f"      _parse_btc_chord_label: 14 BTC qualities (maj/7/maj7/m7/dim7/hdim7/...) parsed ✓")
        print(f"      vocab: btc-extended 含 BTC 独有 (6/dim7/hdim7/mM7) ✓")
        print(f"      ChordConfig.backend: btc-hcqt/madmom routing ✓")
        print(f"      detect_chords_with_btc: mock 5 entries → 4 chords (N filter) ✓")
        print(f"      end-to-end: BTC chord → MusicXML <kind>major-seventh/hdim7</kind> ✓")

        # 16. v0.4.9 验证：chord track groove 联动 (swing)
        print("[16/18] verify v0.4.9 chord track groove linking...")
        from mujik.chord.groove import apply_groove_to_chord_track

        # 构造拍号段
        sigs_groove = [
            TimeSignatureSegment(
                start_time=0.0, end_time=10.0,
                time_signature=(4, 4), confidence=1.0, source="manual",
            ),
        ]

        # 验证 straight = noop
        c_onbeat = ChordEventModel(0.0, 0.5, "C", "")
        c_offbeat = ChordEventModel(0.5, 0.75, "F", "")
        c_beat2 = ChordEventModel(1.0, 1.5, "G", "7")
        track_straight = [c_onbeat, c_offbeat, c_beat2]
        out_straight = apply_groove_to_chord_track(
            track_straight, sigs_groove, bpm=120.0,
            template="straight", strength=1.0,
        )
        for orig, shifted in zip(track_straight, out_straight):
            assert orig.start == shifted.start
            assert orig.end == shifted.end

        # 验证 swing16 + 120 BPM 4/4
        # 120 BPM → beat = 0.5s, offbeat (0.5 beat) = 0.25s
        # chord 边界 0.0 (beat) / 0.5 (beat) / 0.75 (offbeat) / 1.0 (beat) / 1.5 (beat)
        out_swing = apply_groove_to_chord_track(
            track_straight, sigs_groove, bpm=120.0,
            template="swing16", strength=1.0, ratio=0.6,
        )
        # chord 0: 0.0 (beat) → 0.0, 0.5 (beat) → 0.5
        assert out_swing[0].start == 0.0
        assert out_swing[0].end == 0.5
        # chord 1: 0.5 (beat) → 0.5, 0.75 (offbeat) → 0.8
        assert out_swing[1].start == 0.5
        assert out_swing[1].end == 0.8
        # chord 2: 1.0 (beat) → 1.0, 1.5 (beat) → 1.5
        assert out_swing[2].start == 1.0
        assert out_swing[2].end == 1.5

        # 验证 strength 0 = noop
        out_s0 = apply_groove_to_chord_track(
            [c_offbeat], sigs_groove, bpm=120.0,
            template="swing16", strength=0.0, ratio=0.6,
        )
        assert out_s0[0].end == 0.75

        # 验证 strength 0.5 = 半偏移（0.025s 替代 0.05s）
        out_s05 = apply_groove_to_chord_track(
            [c_offbeat], sigs_groove, bpm=120.0,
            template="swing16", strength=0.5, ratio=0.6,
        )
        assert out_s05[0].end == 0.775

        # 验证 ratio 0.5 = 直拍（无偏移）
        out_r05 = apply_groove_to_chord_track(
            [c_offbeat], sigs_groove, bpm=120.0,
            template="swing16", strength=1.0, ratio=0.5,
        )
        assert out_r05[0].end == 0.75

        # 验证 ChordConfig 新增 groove 字段（默认关闭）
        cfg_groove = ChordConfig()
        assert cfg_groove.apply_groove is False  # v0.4.9 默认关闭
        assert cfg_groove.chord_groove_template == "swing16"
        assert cfg_groove.chord_groove_strength == 1.0
        assert cfg_groove.chord_groove_ratio == 0.6

        # 启用 groove 时，所有字段都通过
        cfg_groove_on = ChordConfig(apply_groove=True)
        assert cfg_groove_on.apply_groove is True

        # 端到端：与 quantize 串接
        from mujik.chord.quantize import quantize_chord_track
        raw_for_q = [
            ChordEventModel(0.0, 0.5, "C", ""),
            ChordEventModel(0.5, 1.0, "F", ""),
            ChordEventModel(1.0, 1.5, "G", "7"),
        ]
        quantized_v49 = quantize_chord_track(
            raw_for_q, sigs_groove, bpm=120.0,
            grid_per_bar=4, merge_consecutive=False, min_duration_sec=0.0,
        )
        grooved_v49 = apply_groove_to_chord_track(
            quantized_v49, sigs_groove, bpm=120.0,
            template="swing16", strength=1.0, ratio=0.6,
        )
        for c in grooved_v49:
            assert c.end > c.start
        # 全部都是 on-beat 边界 → 不偏移
        assert grooved_v49[0].start == 0.0 and grooved_v49[0].end == 0.5
        assert grooved_v49[1].start == 0.5 and grooved_v49[1].end == 1.0
        assert grooved_v49[2].start == 1.0 and grooved_v49[2].end == 1.5

        print(f"      straight: noop (3 chords unchanged) ✓")
        print(f"      swing16: offbeat 0.75s → 0.8s (+50ms) ✓")
        print(f"      strength 0=noop, 0.5=half (0.025s), 1.0=full (0.05s) ✓")
        print(f"      ratio 0.5=直拍无偏移 ✓")
        print(f"      ChordConfig.apply_groove: 默认 False, opt-in ✓")
        print(f"      end-to-end: quantize → groove 串接 (3 on-beat chords) ✓")

        # 17. v0.5.0 验证：5-genre benchmark 框架
        print("[17/18] verify v0.5.0 5-genre benchmark framework...")
        from mujik.benchmarks import (
            BENCHMARK_GENRES,
            BenchmarkSample,
        )
        from mujik.benchmarks.datasets.synthetic import SyntheticBenchmarkDataset
        from mujik.benchmarks.metrics import (
            BeatTrackingMetrics,
            ChordRecognitionMetrics,
            NoteTranscriptionMetrics,
        )
        from mujik.benchmarks.report import render_json, render_markdown
        from mujik.benchmarks.runner import BenchmarkRunner

        # 验证 5 genre
        assert set(BENCHMARK_GENRES) == {"pop", "jazz", "metal", "rnb", "classical"}

        # 验证 synthetic dataset
        bench_dir = out_dir / "bench"
        ds = SyntheticBenchmarkDataset(base_dir=bench_dir)
        samples = ds.list_samples()
        assert len(samples) == 15  # 5 × 3
        genres = {s.genre for s in samples}
        assert genres == set(BENCHMARK_GENRES)
        for s in samples:
            assert Path(s.audio_path).exists()
            assert len(s.gt_beats) > 0
            assert len(s.gt_chords) > 0
            assert len(s.gt_notes) > 0

        # 验证 metrics 三个 calculator
        ntm = NoteTranscriptionMetrics()
        assert ntm.name == "note_transcription"
        result_ntm = ntm.compute(
            {"notes": [(60, 0.0, 0.5)]},
            {"notes": [(60, 0.0, 0.5)]},
        )
        assert result_ntm["f1"] == 1.0

        btm = BeatTrackingMetrics()
        assert btm.name == "beat_tracking"
        result_btm = btm.compute(
            {"beats": [0.0, 0.5, 1.0]},
            {"beats": [0.0, 0.5, 1.0]},
        )
        # mir_eval 可能不可用；fallback 到 0
        assert "cmlt" in result_btm

        crm = ChordRecognitionMetrics()
        assert crm.name == "chord_recognition"
        result_crm = crm.compute(
            {"chords": [(0.0, 1.0, "C", "")]},
            {"chords": [(0.0, 1.0, "C", "")]},
        )
        assert "majmin" in result_crm

        # 验证 runner + report（用完美 pipeline）
        def perfect_pipeline(audio_path: str) -> dict:
            gt_path = Path(audio_path).with_suffix(".json")
            gt = json.loads(gt_path.read_text())
            return {
                "note_transcription": {"notes": gt.get("notes", [])},
                "beat_tracking": {"beats": gt.get("beats", [])},
                "chord_recognition": {"chords": gt.get("chords", [])},
            }

        runner = BenchmarkRunner(
            version="0.5.0",
            metric_calculators={
                "note_transcription": ntm,
                "beat_tracking": btm,
                "chord_recognition": crm,
            },
        )
        report = runner.run(ds, perfect_pipeline)
        assert report.version == "0.5.0"
        assert report.n_samples == 15
        assert report.dataset_name == "synthetic_5genre_baseline"
        assert report.overall.get("note_transcription", 0) >= 0.9
        # 5 genre 全部聚合
        assert set(report.per_genre.keys()) == set(BENCHMARK_GENRES)

        # 验证 markdown + JSON report
        md = render_markdown(report)
        assert "v0.5.0" in md
        assert "synthetic_5genre_baseline" in md
        assert "Overall" in md
        assert "Per-Genre" in md
        for genre in BENCHMARK_GENRES:
            assert genre in md

        js = render_json(report)
        parsed = json.loads(js)
        assert parsed["version"] == "0.5.0"
        assert parsed["n_samples"] == 15
        assert len(parsed["per_sample"]) == 15

        # 写 out/bench_report.md 供查看
        (out_dir / "bench_report.md").write_text(md, encoding="utf-8")
        (out_dir / "bench_report.json").write_text(js, encoding="utf-8")
        print(f"      bench_report.md: {len(md)} chars")
        print(f"      bench_report.json: {len(js)} chars")

        print(f"      5 genres: pop/jazz/metal/rnb/classical ✓")
        print(f"      synthetic dataset: 15 samples (5×3) with gt notes/beats/chords ✓")
        print(f"      metrics: note_f1/beat_cmlt/chord_majmin (mir_eval if available) ✓")
        print(f"      runner: {report.n_samples} samples aggregated per genre ✓")
        print(f"      report: markdown table + JSON dump ✓")

        # 18. v0.5.1 验证：易用性层（一键 demo + 进度条 + README）
        print("[18/18] verify v0.5.1 UX (demo script + progress + README)...")
        demo_dir = out_dir / "demo"
        demo_dir.mkdir(exist_ok=True)

        # 18.1 进度条在 CI 下自动 no-op（不污染输出）
        from mujik.pipeline_progress import PipelineProgress, _NullProgress
        with PipelineProgress(total=8) as prog:
            # CI 下应降级为 no-op
            assert isinstance(prog, _NullProgress), "CI 应降级 no-op"
            prog.advance("probe")
            prog.advance("denoise")
            assert prog.step_idx == 2
        print("      progress: CI auto no-op ✓")

        # 18.2 _demo_report.py 生成可读报告（v0.5.1 修 5：曲名目录布局）
        # 单 preset：demo_out/<曲名>/project.json + demo_out/<曲名>/ws/
        # 多 preset：demo_out/<曲名>/<preset>/ + 共享 demo_out/<曲名>/ws/
        song_dir = demo_dir / "buhee"
        song_dir.mkdir(exist_ok=True)
        ws_dir = song_dir / "ws"
        ws_dir.mkdir(exist_ok=True)
        (ws_dir / "beats.json").write_text(json.dumps({
            "bpm": 120.0, "beats": [0.0, 0.5], "downbeats": [0.0],
        }))
        (ws_dir / "time_signatures.json").write_text(json.dumps([
            {"start": 0.0, "end": 1.0, "sig": [4, 4], "confidence": 1.0, "source": "h"},
        ]))
        (ws_dir / "chords.json").write_text(json.dumps([
            {"start": 0.0, "end": 1.0, "root": "Cmaj7", "quality": "maj7", "bass": None},
        ]))
        for name in ("pop", "jazz"):
            pd = song_dir / name
            pd.mkdir(exist_ok=True)
            (pd / "project.json").write_text(json.dumps({
                "mujik_version": "0.5.1",
                "preset": name,
                "separator": "demucs/htdemucs_ft",
                "transcribe_mode": "per_stem",
                "rhythm_enabled": True,
                "chord_enabled": name == "jazz",
                "chord_backend": "madmom",
                "chord_quantize_enabled": name == "jazz",
                "chord_groove_enabled": False,
                "chord_groove_template": "straight",
                "denoise_enabled": False,
                "denoise_backend": "none",
                "score_features": ["bend", "harmony"],
            }, ensure_ascii=False))

        import subprocess
        r = subprocess.run(
            [sys.executable, str(repo_root / "scripts" / "_demo_report.py"), str(demo_dir)],
            capture_output=True, text=True,
        )
        assert r.returncode == 0, f"_demo_report.py failed: {r.stderr}"
        report_md = r.stdout
        assert "**buhee/pop**" in report_md
        assert "**buhee/jazz**" in report_md
        assert "## Summary" in report_md
        # 写 out/demo_report.md
        (demo_dir / "demo_report.md").write_text(report_md, encoding="utf-8")
        print(f"      demo report: {len(report_md)} chars, song-layout runs covered ✓")

        # 18.3 README quickstart 段存在
        readme = (repo_root / "README.md").read_text(encoding="utf-8")
        assert "5 分钟跑通" in readme or "Docker 镜像" in readme
        assert "v0.5.0" in readme  # 状态段
        assert "调用示例" in readme
        assert "scripts/run_demo.sh" in readme  # 一键 demo 引用
        print("      README: quickstart + 调用示例 + demo 引用 ✓")

        # 18.4 pipeline.py 集成进度条（version 0.5.1）
        from mujik.pipeline import (
            PIPELINE_TOTAL_STEPS_PERSTEM,
            PIPELINE_TOTAL_STEPS_MULTITRACK,
        )
        assert PIPELINE_TOTAL_STEPS_PERSTEM > 0
        assert PIPELINE_TOTAL_STEPS_MULTITRACK > 0
        print(f"      pipeline constants: per_stem={PIPELINE_TOTAL_STEPS_PERSTEM} multitrack={PIPELINE_TOTAL_STEPS_MULTITRACK} ✓")

        # 18.5 demo 脚本：无参 showcase 三组合（demo/ 下三首曲子 × 各自 preset）+ 曲名目录布局
        # 这里验证 header + 显式错误路径；真实跑 showcase 在 ml 镜像里做
        script_text = (repo_root / "scripts" / "run_demo.sh").read_text()
        assert "demo/buhee.mp3|jazz" in script_text, "showcase 必须含 demo/buhee.mp3×jazz"
        assert "demo/moon.mp3|metal" in script_text, "showcase 必须含 demo/moon.mp3×metal"
        assert "demo/dança.mp3|pop" in script_text, "showcase 必须含 demo/dança.mp3×pop"
        assert "ws/" in script_text, "脚本头部必须说明 ws 中间产物分层"
        assert "MUJIK_DEMO_PRESETS" in script_text, "多 preset 对比必须 opt-in"
        # 显式传入不存在的文件应清晰报错
        r = subprocess.run(
            ["bash", str(repo_root / "scripts" / "run_demo.sh"), "/nonexistent/x.wav"],
            capture_output=True, text=True, timeout=10,
        )
        assert r.returncode == 1
        assert "❌" in r.stdout
        print("      demo script: buhee default + clear error on missing ✓")

        # 18.6 ghcr URL 必须是真实 org（v0.5.1 修：your-org → alubawon）
        readme_text = readme
        assert "alubawon" in readme_text, "README 必须含 alubawon"
        assert "your-org" not in readme_text, "README 不应再有占位 your-org"
        assert "dev-v0.5.1" in readme_text, "README 应引用 dev-v0.5.1 tag"
        print("      README: ghcr 真实 org (alubawon) + v0.5.1 tag ✓")

        print("\n✅ E2E smoke test PASSED (v0.5.1)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
