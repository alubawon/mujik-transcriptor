"""Tests for scripts/run_demo.sh (v0.5.1 + 修 2 + 修 5 曲名目录布局).

验证 demo 脚本：
1. 默认用 buhee/buhee.mp3（仓库自带），产物按曲名目录隔离
2. $1 覆盖默认；$3 显式 preset；MUJIK_DEMO_PRESETS opt-in 多 preset 对比
3. 显式传入不存在的文件 → 清晰报错
4. 头部注释 + executable
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT = REPO_ROOT / "scripts" / "run_demo.sh"
DEFAULT_WAV = REPO_ROOT / "buhee" / "buhee.mp3"


def _bash(*args: str, timeout: int = 30, env_extra: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    if not SCRIPT.exists():
        pytest.skip(f"script not found: {SCRIPT}")
    if not shutil.which("bash"):
        pytest.skip("bash not available")
    import os
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        ["bash", str(SCRIPT), *args],
        capture_output=True, text=True, timeout=timeout, env=env,
    )


class TestDefaults:
    def test_default_uses_buhee(self):
        """无参 → 走 buhee/buhee.mp3 默认（如果存在），产物在 demo_out/buhee/。

        timeout 放宽到 600s：在装齐 ML 栈的环境（如 dev-*-ml 镜像）里，
        run_demo.sh 会真实跑完整 pipeline（CPU demucs，且无 HF 缓存挂载时
        还要现场下载 ~336MB 权重）；在 CI/无 ML 环境里 pipeline 快速失败，
        不受影响。
        """
        if not DEFAULT_WAV.exists():
            pytest.skip(f"default wav missing: {DEFAULT_WAV}")
        r = _bash(timeout=600)
        # 接受 exit 0（成功）或 1（run 失败但脚本继续）
        assert r.returncode in (0, 1)
        # 必须在 stdout 看见 buhee.mp3 路径 + 曲名目录
        assert "buhee/buhee.mp3" in r.stdout or str(DEFAULT_WAV) in r.stdout
        assert "demo_out/buhee/" in r.stdout

    def test_first_arg_overrides_default(self, tmp_path: Path):
        """$1 覆盖默认，曲名目录跟随文件名。"""
        custom = tmp_path / "my_custom_song.wav"
        # 写真实 wav 头（4 字节 RIFF 即可）
        custom.write_bytes(b"RIFF" + b"\x00" * 100)
        r = _bash(str(custom))
        assert "my_custom_song.wav" in r.stdout or str(custom) in r.stdout

    def test_preset_arg_via_env_comparison(self, tmp_path: Path):
        """MUJIK_DEMO_PRESETS opt-in 多 preset（开发/评测用）。"""
        custom = tmp_path / "env_song.wav"
        custom.write_bytes(b"RIFF" + b"\x00" * 100)
        r = _bash(str(custom), timeout=120,
                  env_extra={"MUJIK_DEMO_PRESETS": "pop,jazz"})
        assert r.returncode in (0, 1)
        assert "[preset=pop]" in r.stdout
        assert "[preset=jazz]" in r.stdout


class TestErrorPaths:
    def test_missing_custom_file(self):
        """传入不存在的文件 → exit 1。"""
        r = _bash("/nonexistent/path/missing.wav")
        assert r.returncode == 1
        assert "❌" in r.stdout
        assert "not found" in r.stdout or "不存在" in r.stdout

    def test_default_missing_warns_with_hint(self):
        """默认文件不存在时给出明确提示。"""
        r = _bash("/definitely/not/a/real/file.wav")
        assert r.returncode == 1
        assert "❌" in r.stdout


class TestHeader:
    def test_script_header_documents_buhee_default(self):
        content = SCRIPT.read_text(encoding="utf-8")
        assert "buhee/buhee.mp3" in content
        assert "默认" in content or "default" in content.lower()

    def test_script_header_documents_song_layout(self):
        """修 5：头部注释说明曲名目录 + ws 分层。"""
        content = SCRIPT.read_text(encoding="utf-8")
        assert "ws/" in content
        assert "曲名" in content

    def test_script_chmod_executable(self):
        mode = SCRIPT.stat().st_mode
        assert mode & 0o111, f"script not executable: {SCRIPT}"
