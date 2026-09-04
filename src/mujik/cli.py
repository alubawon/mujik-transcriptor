"""mujik-transcriptor CLI 入口。

v0.2.3 子命令：run / render / separate / quantize / time-signature change
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from loguru import logger

from mujik import __version__


def cmd_run(args: argparse.Namespace) -> int:
    """运行管线。"""
    from mujik.config.schema import PipelineConfig
    from mujik.pipeline import Pipeline

    logger.info("mujik run: input={}, output={}, config={}", args.input, args.output, args.config)

    # 加载配置
    if args.config:
        cfg = PipelineConfig.from_yaml(args.config)
    else:
        cfg = PipelineConfig(input_path=args.input, output_dir=args.output)

    if args.preset:
        cfg = cfg.apply_preset(args.preset)

    # 覆盖 input/output（CLI 优先）
    cfg.input_path = args.input
    cfg.output_dir = args.output
    if args.workspace:
        cfg.workspace_dir = args.workspace

    # 跑管线
    pipeline = Pipeline(cfg)
    project = pipeline.run()

    # v0.5.1: 输出 score.musicxml（乐谱产物；`mujik render` 可将其转 PDF/SVG）。
    # MIDI 是核心产物，MusicXML 导出失败只降级不中断（fail-soft，日志明示）
    try:
        from mujik.score.builder import build_musicxml
        score_xml = build_musicxml(project)
        score_path = Path(args.output) / "score.musicxml"
        score_path.write_text(score_xml, encoding="utf-8")
        logger.info("wrote MusicXML → {}", score_path)
    except Exception as e:  # noqa: BLE001
        logger.warning("MusicXML export failed: {}", e)

    logger.info(
        "Pipeline done: {n} tracks, {m} notes total",
        n=len(project.tracks), m=project.total_notes(),
    )
    return 0


def cmd_render(args: argparse.Namespace) -> int:
    """单独跑渲染（v0.2.4：SVG + PDF）。

    支持 backend: verovio (默认) / lilypond / musescore
    --pdf 标志：verovio backend 走 CLI subprocess 出 PDF
    """
    from mujik.config.schema import RenderConfig
    from mujik.render import render_musicxml_to_file

    musicxml = Path(args.input).read_text()

    # 构造 RenderConfig
    cfg = RenderConfig(
        pdf_backend=args.backend,
        page_size=args.page_size,
        include_chord_symbols=args.include_chord_symbols,
        verovio_cli_path=args.verovio_cli_path,
    )

    out_path = render_musicxml_to_file(
        musicxml, args.output, config=cfg, prefer_pdf=args.pdf,
    )
    logger.info("Wrote: {} (backend={}, pdf={})", out_path, args.backend, args.pdf)
    return 0


def cmd_separate(args: argparse.Namespace) -> int:
    """仅跑源分离（调试用）。"""
    from mujik.separate.router import separate_audio

    stems = separate_audio(args.input, args.output)
    for name, stem in stems.stems.items():
        logger.info("  - {}: {}", name, stem.audio_path)
    return 0


def _parse_time_arg(s: str) -> float:
    """解析时间参数：支持浮点秒或 mm:ss.SSS 格式。"""
    s = s.strip()
    if ":" in s:
        parts = s.split(":")
        if len(parts) != 2:
            raise ValueError(f"time arg must be float or mm:ss.SSS, got {s!r}")
        minutes = int(parts[0])
        seconds = float(parts[1])
        return float(minutes) * 60.0 + seconds
    return float(s)


def _parse_sig_arg(s: str) -> tuple[int, int]:
    """解析拍号字符串 '4/4' → (4, 4)。"""
    s = s.strip()
    if "/" not in s:
        raise ValueError(f"signature must be like '4/4', got {s!r}")
    num_s, den_s = s.split("/", 1)
    num = int(num_s)
    den = int(den_s)
    if den not in (1, 2, 4, 8, 16, 32):
        raise ValueError(f"denominator must be power of 2 up to 32, got {den}")
    if num < 1 or num > 32:
        raise ValueError(f"numerator out of range: {num}")
    return (num, den)


def _argparse_sig(s: str) -> tuple[int, int]:
    """argparse type= 包装：失败抛 ArgumentTypeError 让 argparse 友好报错。"""
    try:
        return _parse_sig_arg(s)
    except ValueError as e:
        import argparse as _ap
        raise _ap.ArgumentTypeError(str(e))


def cmd_quantize(args: argparse.Namespace) -> int:
    """CLI: mujik quantize --project-dir DIR [--config-yaml CFG] [--out-dir DIR]"""
    from mujik.config.schema import QuantizeConfig
    from mujik.quantize.core import (
        load_beat_track_from_json,
        quantize_project,
        write_quantize_report,
    )
    from mujik.time_signature.io import read_time_signatures_json

    project_dir = Path(args.project_dir)
    midi_in = project_dir / "project.mid"
    # v0.5.1 修 5：beats/time-signatures 中间产物在 ws/（兼容旧 flat 布局）
    def _find_artifact(name: str) -> Path:
        ws_path = project_dir / "ws" / name
        if ws_path.exists():
            return ws_path
        return project_dir / name
    beats_json = _find_artifact("beats.json")
    ts_json = _find_artifact("time_signatures.json")

    if not midi_in.exists():
        logger.error("missing {}", midi_in)
        return 1
    if not beats_json.exists():
        logger.error("missing {} (required for BPM)", beats_json)
        return 2
    if not ts_json.exists():
        logger.warning("missing {} → use default 4/4", ts_json)
        time_signatures: list = []
    else:
        time_signatures = read_time_signatures_json(ts_json)

    beat_track = load_beat_track_from_json(beats_json)

    # 加载配置
    if args.config_yaml:
        cfg_path = Path(args.config_yaml)
        cfg_data = json.loads(cfg_path.read_text(encoding="utf-8"))
        cfg = QuantizeConfig(**cfg_data)
    else:
        cfg = QuantizeConfig()

    # 写盘路径
    out_dir = Path(args.out_dir) if args.out_dir else project_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    midi_out = out_dir / "project.mid"

    logger.info(
        "mujik quantize: dir={}, grid={}, strength={}, groove={}",
        project_dir, cfg.grid_resolution, cfg.strength, cfg.groove_template,
    )

    _, report = quantize_project(
        midi_in, beat_track, time_signatures, cfg, output_midi_path=midi_out,
    )

    if not args.no_write_report:
        report_path = out_dir / "quantize_report.json"
        write_quantize_report(report, report_path)
        logger.info("wrote {}", report_path)

    logger.info(
        "quantize done: tracks={n} notes_before={b} notes_after={a}",
        n=len(report.per_track),
        b=report.total_notes_before,
        a=report.total_notes_after,
    )
    return 0


def cmd_multitrack(args: argparse.Namespace) -> int:
    """v0.4.2: 用 muscriptor 一次性转写多乐器音频。

    跳过 4/5/6-stem 源分离，直接 muscriptor 多乐器转录。
    """
    audio_path = Path(args.input)
    if not audio_path.exists():
        logger.error(f"input not found: {audio_path}")
        return 1

    from mujik.transcribe.muscriptor_adapter import (
        MuscriptorAdapterError,
        check_muscriptor_available,
        transcribe_multitrack,
    )

    if not check_muscriptor_available():
        logger.error(
            "`uvx` not found. Install uv: https://docs.astral.sh/uv/getting-started/installation/"
        )
        return 2

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        project = transcribe_multitrack(
            audio_path,
            out_dir=out_dir,
            model=args.model,
            timeout_sec=args.timeout,
        )
    except MuscriptorAdapterError as e:
        logger.error(f"muscriptor failed: {e}")
        return 3
    except subprocess.TimeoutExpired:
        logger.error(f"muscriptor timeout after {args.timeout}s")
        return 4
    except Exception as e:
        logger.error(f"unexpected error: {e}")
        return 5

    # 写 project.mid + project.json
    from mujik.midi.io import write_project_to_midi
    from mujik.midi.model import TempoSegment
    from mujik.time_signature.model import build_default_segments
    # 兜底：muscriptor 输出可能没 time_signatures
    if not project.time_signatures:
        project.time_signatures = build_default_segments(project.duration or 1.0)
    if not project.tempo_map:
        project.tempo_map = [TempoSegment(0.0, project.duration or 1.0, 120.0)]
    project.metadata.update({
        "mujik_version": __version__,
        "transcribe_mode": "multitrack",
        "muscriptor_model": args.model,
    })
    write_project_to_midi(project, out_dir / "project.mid")
    (out_dir / "project.json").write_text(
        json.dumps({
            "mujik_version": __version__,
            "transcribe_mode": "multitrack",
            "muscriptor_model": args.model,
            "tracks": list(project.tracks.keys()),
            "total_notes": project.total_notes(),
            "duration": project.duration,
        }, ensure_ascii=False, indent=2)
    )
    logger.info(
        f"multitrack done: {len(project.tracks)} tracks, "
        f"{project.total_notes()} notes → {out_dir / 'project.mid'}"
    )
    return 0


def cmd_time_signature_change(args: argparse.Namespace) -> int:
    """CLI: mujik time-signature change --project-dir DIR --at T --new SIG --mode {A,B}"""
    from mujik.time_signature.io import (
        read_time_signatures_json,
        write_time_signatures_json,
    )
    from mujik.time_signature.operations import (
        change_time_signature_at_boundary,
        redraw_bars_under_new_time_signature,
    )

    project_dir = Path(args.project_dir)
    ts_json = project_dir / "time_signatures.json"
    if not ts_json.exists():
        logger.error("missing {}", ts_json)
        return 1

    segments = read_time_signatures_json(ts_json)
    if not segments:
        logger.error("empty time_signatures.json")
        return 2

    at = _parse_time_arg(args.at)
    # args.new 已被 argparse type= 转换为 tuple[int, int]
    new_sig = tuple(args.new)
    mode = args.mode.upper()

    if mode == "A":
        new_segments = redraw_bars_under_new_time_signature(
            segments, (at, segments[-1].end_time), new_sig,
        )
    elif mode == "B":
        change_mode = "regrid" if args.regrid else "preserve_time"
        new_segments = change_time_signature_at_boundary(
            segments, at, new_sig, mode=change_mode,
        )
    else:
        logger.error("--mode must be A or B, got {}", args.mode)
        return 3

    out_dir = Path(args.out_dir) if args.out_dir else project_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "time_signatures.json"
    write_time_signatures_json(new_segments, out_path)

    logger.info(
        "time-signature change done: mode={} at={} new={} segments_before={} segments_after={}",
        mode, at, new_sig, len(segments), len(new_segments),
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    """构建 CLI 解析器。"""
    parser = argparse.ArgumentParser(
        prog="mujik",
        description="mujik-transcriptor: end-to-end music audio to MIDI + PDF",
    )
    sub = parser.add_subparsers(dest="command", required=False)

    # run
    p_run = sub.add_parser("run", help="run the full pipeline")
    p_run.add_argument("--input", "-i", required=True, help="input audio file")
    p_run.add_argument("--output", "-o", required=True, help="output directory")
    p_run.add_argument(
        "--workspace", "-w", default=None,
        help="intermediate artifacts dir (default: <output>/ws)",
    )
    p_run.add_argument("--config", "-c", help="config YAML file")
    p_run.add_argument("--preset", choices=["pop", "jazz", "metal", "custom"])
    p_run.set_defaults(func=cmd_run)

    # render
    p_render = sub.add_parser(
        "render",
        help="render MusicXML to SVG (v0.1) or PDF (v0.2.4: verovio CLI / lilypond / musescore)",
    )
    p_render.add_argument("--input", "-i", required=True, help="input MusicXML file")
    p_render.add_argument("--output", "-o", required=True, help="output file (auto .svg/.pdf)")
    p_render.add_argument(
        "--backend", choices=["verovio", "lilypond", "musescore"],
        default="verovio", help="rendering backend (default: verovio)",
    )
    p_render.add_argument(
        "--pdf", action="store_true",
        help="(verovio backend only) render to PDF via verovio CLI",
    )
    p_render.add_argument(
        "--page-size", choices=["A4", "Letter"], default="A4",
    )
    p_render.add_argument(
        "--include-chord-symbols", action="store_true", default=True,
    )
    p_render.add_argument(
        "--verovio-cli-path", default="verovio",
        help="path to verovio CLI (for --pdf)",
    )
    p_render.set_defaults(func=cmd_render)

    # separate
    p_sep = sub.add_parser("separate", help="source separation only (debug)")
    p_sep.add_argument("--input", "-i", required=True)
    p_sep.add_argument("--output", "-o", required=True)
    p_sep.set_defaults(func=cmd_separate)

    # quantize (v0.2.3)
    p_q = sub.add_parser("quantize", help="quantize MIDI notes to a grid (v0.2.3)")
    p_q.add_argument(
        "--project-dir", required=True,
        help="directory containing project.mid + beats.json + time_signatures.json",
    )
    p_q.add_argument(
        "--config-yaml", default=None,
        help="optional JSON file with QuantizeConfig overrides (default: QuantizeConfig())",
    )
    p_q.add_argument(
        "--out-dir", default=None,
        help="output directory (default: in-place overwrite of project-dir)",
    )
    p_q.add_argument(
        "--no-write-report", action="store_true",
        help="skip writing quantize_report.json",
    )
    p_q.set_defaults(func=cmd_quantize)

    # time-signature (v0.2.3)
    p_ts = sub.add_parser(
        "time-signature",
        help="time-signature operations (v0.2.3)",
    )
    ts_sub = p_ts.add_subparsers(dest="ts_command", required=False)

    p_tsc = ts_sub.add_parser(
        "change",
        help="change time signature at a given time (mode A or B)",
    )
    p_tsc.add_argument("--project-dir", required=True, help="dir containing time_signatures.json")
    p_tsc.add_argument(
        "--at", required=True,
        help="time of change: float seconds or mm:ss.SSS",
    )
    p_tsc.add_argument(
        "--new", required=True, type=_argparse_sig,
        help="new time signature, e.g. '4/4' '3/4' '6/8' '7/8'",
    )
    p_tsc.add_argument(
        "--mode", required=True, choices=["A", "B"],
        help="A=redraw bars in range, B=split at boundary",
    )
    p_tsc.add_argument(
        "--regrid", action="store_true",
        help="(only with --mode B) regrid notes to new time signature grid",
    )
    p_tsc.add_argument(
        "--out-dir", default=None,
        help="output directory (default: in-place overwrite)",
    )
    p_tsc.set_defaults(func=cmd_time_signature_change)

    # multitrack (v0.4.2)
    p_mt = sub.add_parser(
        "multitrack",
        help="transcribe multitrack audio via muscriptor (v0.4.2)",
    )
    p_mt.add_argument("--input", "-i", required=True, help="input audio file")
    p_mt.add_argument("--output", "-o", required=True, help="output directory")
    p_mt.add_argument(
        "--model", choices=["small", "medium", "large"], default="medium",
        help="muscriptor model size (small=CPU friendly, large=GPU recommended)",
    )
    p_mt.add_argument(
        "--timeout", type=int, default=1800,
        help="subprocess timeout in seconds (default 1800)",
    )
    p_mt.set_defaults(func=cmd_multitrack)

    # chords (v0.4.4)
    p_ch = sub.add_parser(
        "chords",
        help="detect chords from audio via madmom (v0.4.4, major/minor only)",
    )
    p_ch.add_argument("--input", "-i", required=True, help="input audio file")
    p_ch.add_argument(
        "--output", "-o", required=True,
        help="output chords.json path (or directory; default: <stem>.chords.json)",
    )
    p_ch.add_argument(
        "--timeout", type=int, default=1800,
        help="subprocess timeout in seconds (default 1800)",
    )
    p_ch.set_defaults(func=cmd_chords)

    return parser


def cmd_chords(args: argparse.Namespace) -> int:
    """v0.4.4: 用 madmom CRNN 检测和弦 → 写 chords.json。

    输出 major / minor 两种 quality（madmom CRNN 限制）；
    7th / 延伸和弦留 v0.4.5+ 用 BTC-HCQT。
    """
    audio_path = Path(args.input)
    if not audio_path.exists():
        logger.error(f"input not found: {audio_path}")
        return 1

    from mujik.chord.madmom_adapter import (
        MadmomChordAdapterError,
        check_madmom_chord_available,
        detect_chords_with_madmom,
    )

    if not check_madmom_chord_available():
        logger.error(
            "madmom not installed. Install via `uv pip install madmom` "
            "(or `pip install mujik-transcriptor[chord]`)"
        )
        return 2

    # 解析 output 路径：如果是目录 → <stem>.chords.json
    out_path = Path(args.output)
    if out_path.is_dir() or str(args.output).endswith("/"):
        out_path.mkdir(parents=True, exist_ok=True)
        out_path = out_path / f"{audio_path.stem}.chords.json"
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)

    # 用临时 out_dir 给 madmom adapter 写 wrapper + 中间 JSON
    out_dir = out_path.parent
    try:
        from mujik.config.schema import ChordConfig
        chord_cfg = ChordConfig(chord_timeout_sec=args.timeout)
        chord_track = detect_chords_with_madmom(
            audio_path, config=chord_cfg, out_dir=out_dir,
        )
    except MadmomChordAdapterError as e:
        logger.error(f"madmom chord failed: {e}")
        return 3
    except subprocess.TimeoutExpired:
        logger.error(f"madmom chord timeout after {args.timeout}s")
        return 4
    except Exception as e:
        logger.error(f"unexpected error: {e}")
        return 5

    out_path.write_text(json.dumps(
        [
            {
                "start": c.start,
                "end": c.end,
                "root": c.root,
                "quality": c.quality,
                "bass": c.bass,
            }
            for c in chord_track
        ],
        ensure_ascii=False, indent=2,
    ))
    logger.info(
        f"chords done: {len(chord_track)} chords → {out_path}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    """主入口。"""
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 1
    # time-signature 必须有子命令
    if args.command == "time-signature" and not getattr(args, "ts_command", None):
        # 打印 time-signature 子命令的 help
        for action in parser._actions:
            if isinstance(action, argparse._SubParsersAction):
                for choice, subparser in action.choices.items():
                    if choice == "time-signature":
                        subparser.print_help()
                        return 1
        return 1
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
