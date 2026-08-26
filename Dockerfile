# mujik-transcriptor 镜像（多 stage）
# 基础层 + GPU 变体 + dev 变体
# 见 docs/design.md §7

ARG PYTHON_VERSION=3.11
ARG CUDA_VERSION=12.1.1
ARG UBUNTU_VERSION=22.04

# ============================================================
# Stage 1: base —— Python 3.11 + 系统依赖 + uv
# ============================================================
FROM ubuntu:${UBUNTU_VERSION} AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LANG=C.UTF-8

RUN apt-get update && apt-get install -y --no-install-recommends \
        python${PYTHON_VERSION} \
        python3-pip \
        python3-dev \
        python3-venv \
        ffmpeg \
        libsndfile1 \
        git \
        curl \
        ca-certificates \
        build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -u 1000 mujik

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app
USER mujik
ENV PATH="/home/mujik/.local/bin:/app/.venv/bin:${PATH}"

# ============================================================
# Stage 2: deps —— 装项目依赖（缓存友好层）
# ============================================================
FROM base AS deps

USER root
COPY --chown=mujik:mujik pyproject.toml ./
COPY --chown=mujik:mujik src ./src
USER mujik

RUN uv venv --python ${PYTHON_VERSION} .venv \
    && . .venv/bin/activate \
    && uv pip install --no-cache ".[dev,core-io,render]"

# ============================================================
# Stage 3: gpu —— 加 CUDA runtime（生产用）
# ============================================================
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION} AS gpu

ARG PYTHON_VERSION=3.11

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    LANG=C.UTF-8

RUN apt-get update && apt-get install -y --no-install-recommends \
        python${PYTHON_VERSION} \
        python3-pip \
        python3-dev \
        python3-venv \
        ffmpeg \
        libsndfile1 \
        git \
        curl \
        ca-certificates \
        build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -u 1000 mujik

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

COPY --chown=mujik:mujik pyproject.toml ./
COPY --chown=mujik:mujik src ./src

USER mujik
RUN uv venv --python ${PYTHON_VERSION} .venv \
    && . .venv/bin/activate \
    && uv pip install --no-cache ".[all]"

ENV PATH="/home/mujik/.local/bin:/app/.venv/bin:${PATH}"

# ============================================================
# Stage 4: dev —— 源码以 volume 挂载（开发用）
# ============================================================
FROM deps AS dev

USER mujik
# 源码在 docker-compose 中以 -v 形式挂载；COPY 仅作 fallback
COPY --chown=mujik:mujik . /app

CMD ["sleep", "infinity"]
