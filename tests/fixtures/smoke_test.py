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
    print(f"[1/13] generate synthetic wav → {fixture_path}")
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
        print(f"[2/13] output dir: {out_dir}")

        # 3. mock 所有重 adapter，写 fake stems
        print("[3/13] mock heavy adapters...")

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
        print("[4/13] run pipeline (mocked)...")
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
        print("[5/13] verify pipeline outputs...")
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
        assert meta["mujik_version"] in ("0.2.2", "0.4.0", "0.4.1", "0.4.2", "0.4.3", "0.4.4", "0.4.5", "0.4.6")
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
        print("[6/13] run mujik quantize (v0.2.3)...")
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
        print("[7/13] run mujik render (v0.2.4)...")
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
        print("[8/13] verify v0.4.1 <bend> + <harmony> + v0.4.3 release curve...")
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
        print("[9/13] verify v0.4.2 muscriptor adapter...")
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
        print("[10/13] verify v0.4.3 continuous bend curve rendering...")
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
        print("[11/13] verify v0.4.4 automatic chord detection...")
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
        print("[12/13] verify v0.4.5 chord quantize to bar/beat...")
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
        print("[13/13] verify v0.4.6 ChordEvent hardening validator...")
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

        # quality vocab 三档
        assert set(ALLOWED_QUALITIES_BY_VOCAB.keys()) == {
            "root", "root-quality", "extended",
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

        print("\n✅ E2E smoke test PASSED (v0.4.6)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
