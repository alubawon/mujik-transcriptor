"""渲染后端统一入口。

按 RenderConfig.pdf_backend 自动选择：
- "verovio"  → VerovioBackend（Python binding）→ SVG / VerovioCliBackend → PDF
- "lilypond" → LilyPondClient（GPL 隔离 HTTP 服务）→ PDF
- "musescore" → MuseScoreClient（GPL 隔离 HTTP 服务）→ PDF

设计文档：docs/design.md §8
"""
from __future__ import annotations

from loguru import logger

from mujik.config.schema import RenderConfig
from mujik.render.verovio_backend import (
    VerovioBackend,
    VerovioBackendError,
    render_musicxml_to_svg,
)
from mujik.render.verovio_cli import (
    VerovioCliBackend,
    VerovioCliBackendError,
    render_musicxml_to_pdf,
)
from mujik.render.lilypond_client import (
    LilyPondClient,
    LilyPondClientError,
    render_via_lilypond,
)
from mujik.render.musescore_client import (
    MuseScoreClient,
    MuseScoreClientError,
    render_via_musescore,
)


def render_musicxml(
    musicxml_str: str,
    config: RenderConfig | None = None,
) -> bytes:
    """统一入口：MusicXML → bytes。

    v0.2.4：
    - verovio backend 默认出 SVG（Python binding）；要 PDF 走 VerovioCliBackend
    - lilypond / musescore 走外部 HTTP 服务（GPL 隔离）
    """
    cfg = config or RenderConfig()
    backend = cfg.pdf_backend

    if backend == "verovio":
        # 默认走 Python binding → SVG（向后兼容 v0.1）
        svg = render_musicxml_to_svg(musicxml_str, config=cfg)
        return svg.encode("utf-8")
    elif backend == "lilypond":
        return render_via_lilypond(musicxml_str, config=cfg)
    elif backend == "musescore":
        return render_via_musescore(musicxml_str, config=cfg)
    else:
        raise ValueError(f"unknown pdf_backend: {backend}")


def render_musicxml_to_file(
    musicxml_str: str,
    out_path: str,
    config: RenderConfig | None = None,
    prefer_pdf: bool = False,
) -> str:
    """渲染并写到文件，根据后端决定后缀（.pdf / .svg）。

    Args:
        musicxml_str: MusicXML 内容
        out_path: 输出路径
        config: RenderConfig
        prefer_pdf: 若 True 且 backend=verovio 且 CLI 可用，走 CLI 出 PDF

    Returns:
        实际写入的路径
    """
    cfg = config or RenderConfig()
    out_path = str(out_path)

    # verovio + prefer_pdf → CLI
    if cfg.pdf_backend == "verovio" and prefer_pdf:
        cli = VerovioCliBackend(
            cli_path=cfg.verovio_cli_path,
            timeout_sec=cfg.cli_timeout_sec,
        )
        if not cli.is_available():
            raise VerovioCliBackendError(
                f"verovio CLI not found at {cfg.verovio_cli_path!r}; "
                f"install via `apt install verovio` or `brew install verovio`"
            )
        if not out_path.endswith(".pdf"):
            out_path = out_path + ".pdf"
        cli.render_to_pdf(musicxml_str, out_path, page_size=cfg.page_size)
        logger.info("Rendered: backend=verovio-cli, output={}", out_path)
        return out_path

    # 默认路径
    content = render_musicxml(musicxml_str, config=cfg)

    if cfg.pdf_backend == "verovio":
        if not out_path.endswith(".svg"):
            out_path = out_path + ".svg"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(content.decode("utf-8"))
    else:
        if not out_path.endswith(".pdf"):
            out_path = out_path + ".pdf"
        with open(out_path, "wb") as f:
            f.write(content)

    logger.info("Rendered: backend={}, output={}", cfg.pdf_backend, out_path)
    return out_path


__all__ = [
    "render_musicxml",
    "render_musicxml_to_file",
    "VerovioBackend",
    "VerovioBackendError",
    "VerovioCliBackend",
    "VerovioCliBackendError",
    "LilyPondClient",
    "LilyPondClientError",
    "MuseScoreClient",
    "MuseScoreClientError",
]
