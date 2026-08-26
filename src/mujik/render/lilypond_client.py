"""LilyPond 渲染服务 HTTP 客户端（主线调用 GPL 隔离服务）。

主进程不引入任何 GPL 代码；通过 HTTP 调用 `render-lilypond` 子服务。
"""
from __future__ import annotations

import base64
from typing import Any

import httpx

from mujik.config.schema import RenderConfig


class LilyPondClientError(RuntimeError):
    pass


class LilyPondClient:
    """LilyPond 渲染服务客户端。"""

    def __init__(
        self,
        base_url: str = "http://localhost:5001",
        timeout_sec: int = 120,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout_sec

    def health(self) -> dict[str, Any]:
        """健康检查。"""
        try:
            r = httpx.get(f"{self.base_url}/health", timeout=5.0)
            r.raise_for_status()
            return r.json()
        except httpx.HTTPError as e:
            raise LilyPondClientError(f"health check failed: {e}") from e

    def render(
        self,
        musicxml_or_midi: str | bytes,
        input_type: str = "musicxml",
        options: dict | None = None,
    ) -> bytes:
        """MusicXML/MIDI → PDF。

        Args:
            musicxml_or_midi: MusicXML 字符串或 MIDI 字节
            input_type: "musicxml" | "midi"
            options: 渲染选项

        Returns:
            PDF 字节
        """
        if isinstance(musicxml_or_midi, str):
            payload = musicxml_or_midi.encode()
        else:
            payload = musicxml_or_midi
        b64 = base64.b64encode(payload).decode()

        try:
            r = httpx.post(
                f"{self.base_url}/render",
                json={
                    "input_type": input_type,
                    "input_b64": b64,
                    "options": options or {},
                },
                timeout=self.timeout,
            )
        except httpx.HTTPError as e:
            raise LilyPondClientError(f"request failed: {e}") from e

        if r.status_code != 200:
            raise LilyPondClientError(
                f"render failed (status={r.status_code}): {r.text[:500]}"
            )

        try:
            data = r.json()
            return base64.b64decode(data["pdf_b64"])
        except (KeyError, ValueError) as e:
            raise LilyPondClientError(f"invalid response: {e}") from e


def render_via_lilypond(
    musicxml_str: str,
    config: RenderConfig | None = None,
) -> bytes:
    """便捷函数：MusicXML → PDF via LilyPond 服务。"""
    cfg = config or RenderConfig()
    client = LilyPondClient(
        base_url=cfg.lilypond_url,
        timeout_sec=cfg.timeout_sec,
    )
    return client.render(musicxml_str)


__all__ = [
    "LilyPondClient",
    "LilyPondClientError",
    "render_via_lilypond",
]
