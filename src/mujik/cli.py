"""mujik-transcriptor CLI 入口。

v0.1 阶段仅做骨架：subcommand 框架 + run 最小路径占位。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from loguru import logger


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

    # 跑管线
    pipeline = Pipeline(cfg)
    project = pipeline.run()

    logger.info(
        "Pipeline done: {n} tracks, {m} notes total",
        n=len(project.tracks), m=project.total_notes(),
    )
    return 0


def cmd_render(args: argparse.Namespace) -> int:
    """单独跑渲染（用于快速迭代乐谱排版）。

    v0.1：仅 SVG（Verovio 6.x Python 工具包不支持 renderToPDF）。
    v0.3+：PDF 走 GPL 隔离的 LilyPond/MuseScore 服务。
    """
    from mujik.render.verovio_backend import render_musicxml_to_svg

    musicxml = Path(args.input).read_text()
    result = render_musicxml_to_svg(musicxml)
    Path(args.output).write_text(result)
    logger.info("Wrote SVG: {} ({} chars)", args.output, len(result))
    return 0


def cmd_separate(args: argparse.Namespace) -> int:
    """仅跑源分离（调试用）。"""
    from mujik.separate.demucs_adapter import separate_with_demucs

    stems = separate_with_demucs(args.input, args.output)
    for name, stem in stems.stems.items():
        logger.info("  - {}: {}", name, stem.audio_path)
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
    p_run.add_argument("--config", "-c", help="config YAML file")
    p_run.add_argument("--preset", choices=["pop", "jazz", "metal", "custom"])
    p_run.set_defaults(func=cmd_run)

    # render
    p_render = sub.add_parser("render", help="render MusicXML to SVG (v0.1: BSD inline)")
    p_render.add_argument("--input", "-i", required=True, help="input MusicXML file")
    p_render.add_argument("--output", "-o", required=True, help="output SVG file")
    p_render.set_defaults(func=cmd_render)

    # separate
    p_sep = sub.add_parser("separate", help="source separation only (debug)")
    p_sep.add_argument("--input", "-i", required=True)
    p_sep.add_argument("--output", "-o", required=True)
    p_sep.set_defaults(func=cmd_separate)

    return parser


def main(argv: list[str] | None = None) -> int:
    """主入口。"""
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 1
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
