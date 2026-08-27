# mujik-transcriptor 镜像（多 stage）
# 见 docs/design.md §7

ARG PYTHON_VERSION=3.11
ARG CUDA_VERSION=12.1.1
ARG APT_MIRROR=mirrors.tuna.tsinghua.edu.cn

# ============================================================
# Stage 1: base —— Python 3.11 + 系统依赖 + uv（CPU）
# ============================================================
FROM python:${PYTHON_VERSION}-slim AS base

ARG APT_MIRROR

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LANG=C.UTF-8

RUN set -eux; \
    if [ -f /etc/apt/sources.list.d/debian.sources ]; then \
        sed -i "s#deb.debian.org#${APT_MIRROR}#g" /etc/apt/sources.list.d/debian.sources; \
    fi; \
    if [ -f /etc/apt/sources.list ]; then \
        sed -i "s#deb.debian.org#${APT_MIRROR}#g" /etc/apt/sources.list; \
        sed -i "s#security.debian.org#${APT_MIRROR}#g" /etc/apt/sources.list; \
    fi; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
        git \
        curl \
        ca-certificates \
        build-essential; \
    rm -rf /var/lib/apt/lists/*; \
    useradd -m -u 1000 mujik

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app
USER mujik
ENV PATH="/home/mujik/.local/bin:/app/.venv/bin:${PATH}"

# ============================================================
# Stage 2: deps —— 装项目依赖（缓存友好层）
# ============================================================
FROM base AS deps

ARG PYTHON_VERSION

USER root
COPY --chown=mujik:mujik pyproject.toml ./
COPY --chown=mujik:mujik README.md ./
COPY --chown=mujik:mujik src ./src
RUN chown -R mujik:mujik /app
USER mujik

RUN UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
    /usr/local/bin/python${PYTHON_VERSION} -m venv .venv \
    && . .venv/bin/activate \
    && UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
       uv pip install --no-cache ".[dev,all]"

# ============================================================
# Stage 3: gpu —— 加 CUDA runtime（生产用）
# ============================================================
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu22.04 AS gpu

ARG PYTHON_VERSION
ARG APT_MIRROR

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    LANG=C.UTF-8

RUN set -eux; \
    sed -i "s#deb.debian.org#${APT_MIRROR}#g; s#archive.ubuntu.com#${APT_MIRROR}#g; s#security.ubuntu.com#${APT_MIRROR}#g" \
        /etc/apt/sources.list; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        software-properties-common; \
    add-apt-repository -y ppa:deadsnakes/ppa; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        python${PYTHON_VERSION} \
        python3-pip \
        python3-dev \
        python3-venv \
        ffmpeg \
        libsndfile1 \
        git \
        curl \
        ca-certificates \
        build-essential; \
    rm -rf /var/lib/apt/lists/*; \
    useradd -m -u 1000 mujik

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

USER root
COPY --chown=mujik:mujik pyproject.toml ./
COPY --chown=mujik:mujik README.md ./
COPY --chown=mujik:mujik src ./src
RUN chown -R mujik:mujik /app
USER mujik

RUN UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
    /usr/local/bin/python${PYTHON_VERSION} -m venv .venv \
    && . .venv/bin/activate \
    && UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
       uv pip install --no-cache ".[all]"

ENV PATH="/home/mujik/.local/bin:/app/.venv/bin:${PATH}"

# ============================================================
# Stage 4: dev —— 源码以 volume 挂载（开发用）
# ============================================================
FROM deps AS dev

USER root
COPY --chown=mujik:mujik . /app
RUN chown -R mujik:mujik /app
USER mujik

CMD ["sleep", "infinity"]
