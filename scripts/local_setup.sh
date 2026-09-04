#!/usr/bin/env bash
# 本地开发环境一键 setup（macOS arm64 / Linux 通用，v0.5.2）
#
# 两层开发流程：功能先在本 repo 的 .venv 里验证闭环，
# 容器（Dockerfile.ml）只留给构建 / 发布 / 重 ML 栈验证。
#
# 覆盖（macOS arm64 全部实测，v0.5.2）：单测 + CLI + demucs 分离(CPU/MPS)
#       + basic-pitch 转录 + drumscript 鼓 + madmom 节拍/和弦 + BTC 和弦
#       + MusicXML/PDF 渲染（需 brew cairo）
# 不覆盖（容器专属）：bytedance piano(TF git 包) / muscriptor(CC-BY-NC,
#       uvx 隔离) 的真实推理，以及镜像构建本身。
#
# 用法：./scripts/local_setup.sh
set -euo pipefail

cd "$(dirname "$0")/.."

PY=3.11

# 1) venv
if [ ! -x ".venv/bin/python" ]; then
  uv venv --python "$PY"
fi

# 2) 依赖：核心 + 分离(torch/demucs) + 转录(basic-pitch/drumscript) +
#    和弦(madmom) + 渲染(PDF) + dev
#    madmom 需要 Cython/setuptools 且不能 build-isolation
uv pip install -e ".[core-io,loudnorm,midi,render,benchmark,dev,separate,transcribe,transcribe-tf,transcribe-drumscript]"
uv pip install Cython "setuptools<81"
uv pip install "madmom>=0.16" --no-build-isolation

# 3) madmom 0.16 兼容补丁（与 Dockerfile.ml 步骤 4 相同，安装后执行）：
#    a) `from collections import MutableSequence`（py3.10+ 移除）→ collections.abc
#    b) np.float / np.int（numpy 1.24+ 移除）→ 内建
#    注意：macOS BSD sed 不支持 \b，用 perl
MADMON_DIR=".venv/lib/python${PY}/site-packages/madmom"
find "$MADMON_DIR" -name '*.py' \
  -exec perl -pi -e 's/from collections import MutableSequence/from collections.abc import MutableSequence/g; s/np\.float\b/float/g; s/np\.int\b/int/g' {} +

# 4) 硬校验（失败即中止）
.venv/bin/python -c "import madmom; print('madmom OK')"
.venv/bin/python -c "import torch; print('torch', torch.__version__, 'mps:', torch.backends.mps.is_available())"
.venv/bin/python -c "import demucs; print('demucs OK')"
.venv/bin/python -c "import basic_pitch; print('basic-pitch OK')"
.venv/bin/python -c "import drumscript; print('drumscript OK')"
.venv/bin/python -c "import verovio; print('verovio OK')"

# 5) cairosvg 需要系统 libcairo2：
#    - macOS: brew install cairo；且 ctypes 默认搜不到 /opt/homebrew/lib，
#      需要设 DYLD_FALLBACK_LIBRARY_PATH（本脚本已对当前 shell 生效，
#      长期使用请加进 ~/.zshrc）
#    - Linux: apt install libcairo2
if [ "$(uname)" = "Darwin" ]; then
  export DYLD_FALLBACK_LIBRARY_PATH="/opt/homebrew/lib${DYLD_FALLBACK_LIBRARY_PATH:+:$DYLD_FALLBACK_LIBRARY_PATH}"
fi
.venv/bin/python -c "import cairosvg, pypdf; print('svg-pdf OK')"

echo ""
echo "本地环境就绪。常用命令："
echo "  .venv/bin/python -m pytest tests/unit -q     # 全量单测（含 PDF e2e）"
echo "  .venv/bin/mujik run --input <audio> --output out/ --preset pop"
echo "  .venv/bin/mujik chords --input <audio> --output chords.json"
echo ""
echo "提示：bytedance piano / muscriptor 仍在容器验证；"
echo "      如需本地尝试：uv pip install -e '.[transcribe-bytedance,transcribe-muscriptor]'"
