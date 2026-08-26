"""渲染后端统一入口。

按 RenderConfig.pdf_backend 自动选择：
- "verovio"  → VerovioBackend（BSD，主线内嵌）→ SVG（v0.1）
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
    """统一入口：MusicXML → PDF（或 SVG，取决于后端）。

    v0.1：仅 verovio 后端返回 SVG；lilypond/musescore 后端依赖外部服务。
    """
    cfg = config or RenderConfig()
    backend = cfg.pdf_backend

    if backend == "verovio":
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
) -> str:
    """渲染并写到文件，根据后端决定后缀（.pdf / .svg）。"""
    cfg = config or RenderConfig()
    content = render_musicxml(musicxml_str, config=cfg)

    if cfg.pdf_backend == "verovio":
        # SVG output
        if not out_path.endswith(".svg"):
            out_path = out_path + ".svg"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(content.decode("utf-8"))
    else:
        # PDF output
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
    "LilyPondClient",
    "LilyPondClientError",
    "MuseScoreClient",
    "MuseScoreClientError",
]
