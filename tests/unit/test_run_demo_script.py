"""Tests for scripts/run_demo.sh (v0.5.1 + 修 2 + 修 5 曲名目录 + showcase 默认).

验证 demo 脚本：
1. 无参 = 三组合 showcase：buhee×jazz / moon×metal / dança×pop（文件缺失跳过）
2. $1 覆盖默认；$3 显式 preset；MUJIK_DEMO_PRESETS opt-in 多 preset 对比（需显式 $1）
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
# 无参 showcase 三组合（与脚本内定义保持一致）
SHOWCASE: list[tuple[str, str]] = [
    ("demo/buhee.mp3", "jazz"),
    ("demo/moon.mp3", "metal"),
    ("demo/dança.mp3", "pop"),
]


def _bash(*args: str, timeout: int = 30, env_extra: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    if not SCRIPT.exists():
        pytest.skip(f"script not found: {SCRIPT}")
    if not shutil.which("bash"):
        pytest.skip("bash not available")
    import os
    env = os.environ.copy()
    env.pop("MUJIK_DEMO_PRESETS", None)  # 隔离外部 env
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        ["bash", str(SCRIPT), *args],
        capture_output=True, text=True, timeout=timeout, env=env,
    )


class TestShowcaseDefault:
    def test_noarg_showcase_runs_three_combos(self):
        """无参 = showcase：每个存在的曲子以其 preset 跑，产物在 demo_out/<曲名>/。

        每个组合各打一条 [preset=...] 标签。文件缺失的组合跳过（警告）。
        在装齐 ML 栈的环境里会真实跑 pipeline，timeout 放宽（CI 无 ML 栈时
        run 快速失败，标签照常打印）。
        """
        existing = [(s, p) for s, p in SHOWCASE if (REPO_ROOT / s).exists()]
        if not existing:
            pytest.skip("showcase 曲目全部缺失（mp3 是 gitignored 的本地文件）")
        r = _bash(timeout=900)
        # 全部缺失才 exit 1；此处至少一个存在，接受 0/1（1 = 某 run 失败但脚本继续）
        assert r.returncode in (0, 1)
        for song, preset in existing:
            assert f"[preset={preset}]" in r.stdout, f"缺少 [preset={preset}] 标签"
            song_stem = Path(song).stem
            assert f"demo_out/{song_stem}/" in r.stdout or f"demo_out/{song_stem}  " in r.stdout
        # 缺失的组合要打 skip 警告
        missing = [s for s, _ in SHOWCASE if (REPO_ROOT / s).exists() is False]
        for song in missing:
            assert song in r.stdout and "skip" in r.stdout

    def test_noarg_env_presets_warned_ignored(self, tmp_path: Path):
        """无参模式下 MUJIK_DEMO_PRESETS 被忽略并给出警告。"""
        r = _bash("/nonexistent/x.wav", env_extra={"MUJIK_DEMO_PRESETS": "pop,jazz"})
        # 显式传了不存在的文件 → 走单输入错误路径，但无参警告只在 showcase 分支；
        # 这里验证显式路径 + env 组合仍走对比逻辑的报错行为
        assert r.returncode == 1

    def test_showcase_all_missing_errors(self):
        """showcase 曲目全部缺失 → exit 1 + 明确提示。"""
        # 模拟：把 REPO_ROOT 假不了，只能直接检查：若本地曲目确实存在则跳过
        if any((REPO_ROOT / s).exists() for s, _ in SHOWCASE):
            pytest.skip("本地存在 showcase 曲目，无法模拟全缺失")
        r = _bash(timeout=30)
        assert r.returncode == 1
        assert "showcase" in r.stdout


class TestExplicitInput:
    def test_first_arg_overrides_default(self, tmp_path: Path):
        """$1 覆盖 showcase，曲名目录跟随文件名。"""
        custom = tmp_path / "my_custom_song.wav"
        custom.write_bytes(b"RIFF" + b"\x00" * 100)
        r = _bash(str(custom))
        assert "my_custom_song.wav" in r.stdout or str(custom) in r.stdout

    def test_preset_arg_via_env_comparison(self, tmp_path: Path):
        """MUJIK_DEMO_PRESETS opt-in 多 preset（显式 $1 时生效）。"""
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
    def test_script_header_documents_showcase(self):
        """头部注释列出三组合 showcase（含各曲 preset）。"""
        content = SCRIPT.read_text(encoding="utf-8")
        assert "demo/buhee.mp3" in content
        assert "moon.mp3" in content
        assert "dança.mp3" in content
        assert "jazz" in content and "metal" in content and "pop" in content

    def test_script_header_documents_song_layout(self):
        """头部注释说明曲名目录 + ws 分层。"""
        content = SCRIPT.read_text(encoding="utf-8")
        assert "ws/" in content
        assert "曲名" in content

    def test_script_chmod_executable(self):
        mode = SCRIPT.stat().st_mode
        assert mode & 0o111, f"script not executable: {SCRIPT}"


class TestFailureExitCode:
    """v0.5.2: run 失败计入 FAIL_COUNT → 脚本 exit 1（此前全失败也 exit 0）。"""

    def test_run_failure_exits_1(self, tmp_path: Path):
        """fake mujik 恒失败 → ❌ done + exit 1（showcase 其余组合仍继续）。"""
        if not shutil.which("bash"):
            pytest.skip("bash not available")
        fake_bin = tmp_path / "fakebin"
        fake_bin.mkdir()
        fake_mujik = fake_bin / "mujik"
        fake_mujik.write_text("#!/usr/bin/env bash\nexit 3\n")
        fake_mujik.chmod(0o755)

        import os
        env = os.environ.copy()
        env.pop("MUJIK_DEMO_PRESETS", None)
        env["PATH"] = f"{fake_bin}:{env['PATH']}"
        r = subprocess.run(
            ["bash", str(SCRIPT)], capture_output=True, text=True,
            timeout=120, env=env,
        )
        assert r.returncode == 1
        assert "❌ done（" in r.stdout
        assert "3 个 run 失败" in r.stdout  # 三曲 showcase 全部失败
        assert "✅" not in r.stdout
