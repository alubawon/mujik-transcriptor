"""Verovio CLI subprocess 包装（v0.2.4 PDF 输出）。

Verovio Python binding（6.x）只支持 renderToSVG()。要出 PDF 必须用
`verovio` CLI 工具（apt install verovio / brew install verovio）。

调用形式：
    verovio -f pdf -o out.pdf input.musicxml
    verovio -f svg -o out.svg input.musicxml

本模块：
- 检测 CLI 是否可用
- 写 tmp .musicxml + 调 subprocess + 验产物
- 失败抛 VerovioCliBackendError

依赖：仅 stdlib subprocess + pathlib。Verovio CLI 本身 MIT/BSD。
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Literal

from loguru import logger

from mujik.config.schema import RenderConfig


class VerovioCliBackendError(RuntimeError):
    pass


class VerovioCliBackend:
    """Verovio CLI subprocess 后端。"""

    def __init__(self, cli_path: str = "verovio", timeout_sec: int = 60) -> None:
        self.cli_path = cli_path
        self.timeout_sec = timeout_sec

    def is_available(self) -> bool:
        """检查 `verovio` CLI 是否在 PATH 中。"""
        return shutil.which(self.cli_path) is not None

    def render_to_pdf(
        self,
        musicxml_str: str,
        out_path: str | Path,
        page_size: Literal["A4", "Letter"] = "A4",
    ) -> Path:
        """MusicXML → PDF（原子：写 tmp .musicxml → CLI → 验产物）。"""
        return self._render(musicxml_str, out_path, fmt="pdf", page_size=page_size)

    def render_to_svg(
        self,
        musicxml_str: str,
        out_path: str | Path,
        page_size: Literal["A4", "Letter"] = "A4",
    ) -> Path:
        """MusicXML → SVG。"""
        return self._render(musicxml_str, out_path, fmt="svg", page_size=page_size)

    def _render(
        self,
        musicxml_str: str,
        out_path: str | Path,
        fmt: Literal["pdf", "svg"],
        page_size: Literal["A4", "Letter"] = "A4",
    ) -> Path:
        if not musicxml_str or not musicxml_str.strip():
            raise VerovioCliBackendError("empty MusicXML input")
        if not self.is_available():
            raise VerovioCliBackendError(
                f"verovio CLI not found at {self.cli_path!r}; "
                f"install via `apt install verovio` or `brew install verovio`"
            )

        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".musicxml", delete=False, encoding="utf-8",
        ) as tmp:
            tmp.write(musicxml_str)
            tmp_path = Path(tmp.name)

        try:
            # verovio CLI 接受 -o 输出文件 + 输入文件
            cmd = [
                self.cli_path,
                "-f", fmt,
                "-o", str(out),
                str(tmp_path),
            ]
            if page_size:
                cmd.extend(["-p", str(_PAGE_SIZES[page_size])])

            logger.info("verovio cli: {cmd}", cmd=cmd)
            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_sec,
                    check=False,
                )
            except subprocess.TimeoutExpired as e:
                raise VerovioCliBackendError(
                    f"verovio CLI timed out after {self.timeout_sec}s"
                ) from e
            except FileNotFoundError as e:
                raise VerovioCliBackendError(
                    f"verovio CLI not executable: {self.cli_path!r}"
                ) from e

            if proc.returncode != 0:
                raise VerovioCliBackendError(
                    f"verovio CLI failed (rc={proc.returncode}): "
                    f"stderr={proc.stderr.strip()[:500]}"
                )

            if not out.exists():
                raise VerovioCliBackendError(
                    f"verovio CLI did not produce output: {out}"
                )
            if out.stat().st_size == 0:
                raise VerovioCliBackendError(
                    f"verovio CLI produced empty output: {out}"
                )
        finally:
            # 清理 tmp
            try:
                tmp_path.unlink()
            except FileNotFoundError:
                pass

        logger.info(
            "verovio cli: rendered {fmt} → {out} ({n} bytes)",
            fmt=fmt, out=out, n=out.stat().st_size,
        )
        return out


_PAGE_SIZES: dict[str, int] = {
    "A4": 210,   # 210mm 宽
    "Letter": 216,  # 216mm 宽
}


def render_musicxml_to_pdf(
    musicxml_str: str,
    out_path: str | Path,
    config: RenderConfig | None = None,
    page_size: Literal["A4", "Letter"] = "A4",
) -> Path:
    """便捷函数：MusicXML → PDF 文件。"""
    cfg = config or RenderConfig()
    backend = VerovioCliBackend(
        cli_path=cfg.verovio_cli_path,
        timeout_sec=cfg.cli_timeout_sec,
    )
    return backend.render_to_pdf(musicxml_str, out_path, page_size=page_size)


def render_musicxml_to_svg_via_cli(
    musicxml_str: str,
    out_path: str | Path,
    config: RenderConfig | None = None,
    page_size: Literal["A4", "Letter"] = "A4",
) -> Path:
    """便捷函数：MusicXML → SVG 文件（走 CLI；Python binding 见 verovio_backend）。"""
    cfg = config or RenderConfig()
    backend = VerovioCliBackend(
        cli_path=cfg.verovio_cli_path,
        timeout_sec=cfg.cli_timeout_sec,
    )
    return backend.render_to_svg(musicxml_str, out_path, page_size=page_size)


__all__ = [
    "VerovioCliBackend",
    "VerovioCliBackendError",
    "render_musicxml_to_pdf",
    "render_musicxml_to_svg_via_cli",
]
