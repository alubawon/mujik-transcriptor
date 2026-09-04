# Dockerfile.ml —— 完整 ML 栈镜像（基于 dev-v0.5.1 加装 separate/chord/transcribe extras）
#
# 注意：
# - demucs 4.x 依赖 sphn (Rust/maturin)，aarch64 无预编译 wheel → 必须装 Rust toolchain
# - 所有 RUN 都 set -o pipefail：任何一步失败立刻中止 build（不静默吞错）
#
# 用法：
#   docker build -f Dockerfile.ml -t mujik-transcriptor:dev-v0.5.1-ml --build-arg BASE_TAG=dev-v0.5.1 .
#
ARG BASE_TAG=dev-v0.5.1
FROM mujik-transcriptor:${BASE_TAG}

USER root

# 把当前 src + pyproject.toml 烧进镜像（覆盖 base 里旧版 mujik wheel；no-deps 只重装自身）。
# pyproject.toml 必须一起覆盖：否则后面 `uv pip install ".[extras]"` 用的是 base 镜像里
# 烘焙的旧 pyproject，新增 extra 会被 uv 以 "unknown extra" 警告静默跳过（exit 0 不装包）。
COPY --chown=mujik:mujik src ./src
COPY --chown=mujik:mujik pyproject.toml README.md ./
RUN set -euxo pipefail; \
    . /app/.venv/bin/activate; \
    UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
    uv pip install --no-cache --no-deps .

# 0) 系统库：pkg-config + libopus（sphn→audiopus_sys 优先链接系统 opus，
#    跳过 CMake 源码编译；CMake 4.x 已移除 <3.5 兼容，加 policy 开关兜底）
#    + libcairo2（PDF 渲染链 verovio toolkit SVG→cairosvg→pypdf 需要；
#      Debian/Ubuntu apt 源没有 verovio CLI 包，走纯 Python 路径）
RUN set -euxo pipefail; \
    apt-get update; \
    apt-get install -y --no-install-recommends pkg-config libopus-dev libcairo2; \
    rm -rf /var/lib/apt/lists/*

ENV CMAKE_POLICY_VERSION_MINIMUM=3.5

# 1) Rust toolchain（sphn 编译需要；用 tuna 镜像加速）
ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    RUSTUP_DIST_SERVER=https://mirrors.tuna.tsinghua.edu.cn/rustup \
    RUSTUP_UPDATE_ROOT=https://mirrors.tuna.tsinghua.edu.cn/rustup/rustup \
    PATH=/usr/local/cargo/bin:${PATH}

RUN set -euxo pipefail; \
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs -o /tmp/rustup-init.sh; \
    sh /tmp/rustup-init.sh -y --default-toolchain stable --profile minimal --no-modify-path; \
    rustc --version; cargo --version; \
    rm -f /tmp/rustup-init.sh

# 2) Cython / numpy / hatchling（madmom 编译需要 Cython；
#    后续 --no-build-isolation 装 .[chord] 时 mujik 自身需要 hatchling）
RUN set -euxo pipefail; \
    . /app/.venv/bin/activate; \
    UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
    uv pip install --no-cache "Cython" "numpy==1.26.4" "hatchling"

# 3) demucs + transcribe + drumscript + benchmark + render（PDF 链）
RUN set -euxo pipefail; \
    . /app/.venv/bin/activate; \
    UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
    uv pip install --no-cache ".[separate,transcribe,transcribe-tf,transcribe-drumscript,benchmark,render]"

# 4) madmom（单独装；失败即中止——主线 chord 功能必需）
#    madmom 0.16 兼容补丁（装后执行）：
#    a) `from collections import MutableSequence`（py3.10+ 移除）→ collections.abc
#    b) `np.float` / `np.int`（numpy 1.24+ 移除）→ 内建 float / int
#       （\b 保证不误伤 np.float32 / np.float64）
#       注：Cython 编译产物（ml/hmm .so）里的 np.int 由 wrapper 运行时 monkey-patch 兜底
RUN set -euxo pipefail; \
    . /app/.venv/bin/activate; \
    UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
    uv pip install --no-cache --no-build-isolation ".[chord]"; \
    find /app/.venv/lib/python3.11/site-packages/madmom -name '*.py' \
      -exec sed -i \
        -e 's/from collections import MutableSequence/from collections.abc import MutableSequence/g' \
        -e 's/np\.float\b/float/g' \
        -e 's/np\.int\b/int/g' \
        {} +

# 4.5) v0.5.2 起 drums 转录用 DrumScript（transcribe-drumscript extra，见步骤 3），
#      替代 adtof（原仓库 git URL 死链 + LICENSE 实为 CC-BY-NC-SA + 移植版无 LICENSE）。

# 4.7) BTC-ISMIR19 large_voca 预训练权重（MIT，随上游仓库分发，12MB）。
#      权重不进 git（模型权重分离策略），镜像构建时下载到固定路径；
#      adapter 按 config.btc_model_path → env MUJIK_BTC_MODEL 顺序解析。
ARG BTC_MODEL_URL=https://raw.githubusercontent.com/jayg996/BTC-ISMIR19/master/test/btc_model_large_voca.pt
RUN set -euxo pipefail; \
    mkdir -p /app/models; \
    curl -sfL "${BTC_MODEL_URL}" -o /app/models/btc_model_large_voca.pt; \
    test "$(stat -c%s /app/models/btc_model_large_voca.pt)" -gt 1000000
ENV MUJIK_BTC_MODEL=/app/models/btc_model_large_voca.pt

# 5) 硬校验：任何一个 import 失败 → build 失败
RUN set -euxo pipefail; \
    . /app/.venv/bin/activate; \
    python -c "import demucs; print('demucs OK')"; \
    python -c "import basic_pitch; print('basic_pitch OK')"; \
    python -c "import drumscript; print('drumscript OK')"; \
    python -c "import mir_eval; print('mir_eval OK')"; \
    python -c "import madmom; print('madmom OK')"; \
    python -c "import torch; print('torch', torch.__version__)"; \
    python -c "import cairosvg, pypdf; print('svg-pdf OK')"; \
    test -s "${MUJIK_BTC_MODEL}"

USER mujik

ENTRYPOINT ["/app/.venv/bin/mujik"]
CMD ["--help"]
