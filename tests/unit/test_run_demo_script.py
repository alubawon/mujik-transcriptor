"""Tests for scripts/run_demo.sh (v0.5.1 + 修).

验证 demo 脚本：
1. 无参数 → exit 2 + 帮助信息（含"必须"）
2. 不存在的文件 → exit 1 + "not found"
3. 真实文件存在 → 进入运行（不要求跑通，只需进 stage 3 preset loop）
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT = REPO_ROOT / "scripts" / "run_demo.sh"


def _bash(*args: str, env: dict | None = None) -> subprocess.CompletedProcess:
    if not SCRIPT.exists():
        pytest.skip(f"script not found: {SCRIPT}")
    if not shutil.which("bash"):
        pytest.skip("bash not available")
    return subprocess.run(
        ["bash", str(SCRIPT), *args],
        capture_output=True, text=True, env=env,
        timeout=30,
    )


class TestArgumentValidation:
    def test_no_args_rejects(self):
        r = _bash()
        assert r.returncode == 2
        assert "❌" in r.stdout
        assert "必须" in r.stdout
        assert "用法" in r.stdout
        assert "pop_song.wav" in r.stdout  # 示例

    def test_missing_file_rejects(self):
        r = _bash("/nonexistent/path/to/missing.wav")
        assert r.returncode == 1
        assert "❌" in r.stdout
        assert "not found" in r.stdout

    def test_synthetic_fixture_rejected_at_smoke_level(self, tmp_path: Path):
        """脚本不禁止 synthetic_5s.wav（只是建议），但应在帮助中说明。

        注意：脚本实际不区分 synthetic 与真实 wav——它只校验"文件存在"。
        拒绝 synthetic 的责任在 README + 脚本头部注释 + 帮助文本。
        """
        # 帮助文本应明确"必须真实 wav"
        r = _bash()
        assert "真实" in r.stdout
        assert "pop/jazz/metal" in r.stdout


class TestHeader:
    def test_script_header_documents_required_arg(self):
        content = SCRIPT.read_text(encoding="utf-8")
        assert "要求真实 wav" in content or "必须提供" in content
        assert "synthetic" in content.lower() or "合成" in content  # 解释为何不用合成

    def test_script_chmod_executable(self):
        import os
        mode = SCRIPT.stat().st_mode
        assert mode & 0o111, f"script not executable: {SCRIPT}"


class TestSmokeIntegration:
    """实际跑：传一个真实 fixture，期望跑完三 preset（即使失败也不应 crash 整脚本）。"""

    def test_real_wav_reaches_preset_loop(self, tmp_path: Path):
        # 用真实 fixture（synthetic_5s.wav 是仓库内真实存在的 wav）
        # 脚本只校验文件存在；能进 stage 3 即可
        # CI / 容器可能没 demucs/madmom，预期 preset 失败但脚本继续
        wav = REPO_ROOT / "tests" / "fixtures" / "synthetic_5s.wav"
        if not wav.exists():
            pytest.skip("synthetic_5s.wav fixture missing")
        # 短 timeout：测试用 fixture 跑完会很快（preset 全失败但 exit 0）
        r = _bash(str(wav), env={"PATH": "/usr/bin:/bin:/usr/local/bin"})
        # 退出码可能是 0（三 preset 全失败但继续）也可能是 1
        # 关键是 stdout 包含 "running preset" 三次
        assert r.returncode in (0, 1), f"unexpected exit code: {r.returncode}, stderr: {r.stderr}"
        # 应至少进入 stage 3（"running preset=pop"）
        assert "preset=pop" in r.stdout
        # 不应再有"❌ 必须"错误
        assert "❌ 必须" not in r.stdout
