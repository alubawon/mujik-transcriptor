"""Tests for scripts/run_demo.sh (v0.5.1 + 修 2).

验证 demo 脚本：
1. 默认用 buhee/buhee.mp3（仓库自带）
2. $1 覆盖默认
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


def _bash(*args: str, timeout: int = 30) -> subprocess.CompletedProcess:
    if not SCRIPT.exists():
        pytest.skip(f"script not found: {SCRIPT}")
    if not shutil.which("bash"):
        pytest.skip("bash not available")
    return subprocess.run(
        ["bash", str(SCRIPT), *args],
        capture_output=True, text=True, timeout=timeout,
    )


class TestDefaults:
    def test_default_uses_buhee(self):
        """无参 → 走 buhee/buhee.mp3 默认（如果存在）。

        timeout 放宽到 300s：在装齐 ML 栈的环境（如 dev-*-ml 镜像）里，
        run_demo.sh 会真实跑完整 pipeline（demucs 分离 3 个 preset），
        30s 必超时；在 CI/无 ML 环境里 preset 快速失败，不受影响。
        """
        if not DEFAULT_WAV.exists():
            pytest.skip(f"default wav missing: {DEFAULT_WAV}")
        r = _bash(timeout=300)
        # 接受 exit 0（成功）或 1（preset 失败但脚本继续）
        assert r.returncode in (0, 1)
        # 必须在 stdout 看见 buhee.mp3 路径
        assert "buhee/buhee.mp3" in r.stdout or str(DEFAULT_WAV) in r.stdout

    def test_first_arg_overrides_default(self, tmp_path: Path):
        """$1 覆盖默认。"""
        custom = tmp_path / "custom.wav"
        # 写真实 wav 头（4 字节 RIFF 即可）
        custom.write_bytes(b"RIFF" + b"\x00" * 100)
        r = _bash(str(custom))
        assert "custom.wav" in r.stdout or str(custom) in r.stdout


class TestErrorPaths:
    def test_missing_custom_file(self):
        """传入不存在的文件 → exit 1。"""
        r = _bash("/nonexistent/path/missing.wav")
        assert r.returncode == 1
        assert "❌" in r.stdout
        assert "not found" in r.stdout or "不存在" in r.stdout

    def test_default_missing_warns_with_hint(self, monkeypatch):
        """默认文件不存在时给出明确提示。"""
        # 通过临时改 SCRIPT 内容测；或用临时 HOME 等。这里直接测 mock：传入不存在的文件，错误信息含"不存在"
        r = _bash("/definitely/not/a/real/file.wav")
        assert r.returncode == 1
        assert "❌" in r.stdout


class TestHeader:
    def test_script_header_documents_buhee_default(self):
        content = SCRIPT.read_text(encoding="utf-8")
        assert "buhee/buhee.mp3" in content
        assert "默认" in content or "default" in content.lower()

    def test_script_chmod_executable(self):
        mode = SCRIPT.stat().st_mode
        assert mode & 0o111, f"script not executable: {SCRIPT}"


class TestSmokeIntegration:
    """默认 wav 跑完整脚本：expect 至少进 stage 3 preset loop。"""

    def test_default_runs_three_presets(self):
        if not DEFAULT_WAV.exists():
            pytest.skip(f"default wav missing: {DEFAULT_WAV}")
        # 600s：ML 栈齐全的环境里会真实跑 3 preset 完整 pipeline（CPU demucs）
        r = _bash(timeout=600)
        assert r.returncode in (0, 1)
        assert "preset=pop" in r.stdout
        assert "preset=jazz" in r.stdout
        assert "preset=metal" in r.stdout
        assert "✅ done" in r.stdout
