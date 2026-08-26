# mujik-transcriptor GPU 镜像（主线）
# 见 docs/design.md §7

ARG PYTHON_VERSION=3.11
ARG CUDA_VERSION=12.1.1
ARG UBUNTU_VERSION=22.04

FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION} AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LANG=C.UTF-8

# 系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
        python${PYTHON_VERSION} \
        python3-pip \
        python3-dev \
        ffmpeg \
        libsndfile1 \
        git \
        curl \
        ca-certificates \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# uv 安装
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

# 依赖（独立 layer，缓存友好）
COPY pyproject.toml ./
COPY src ./src

# 安装主线 + 全部 extras
RUN uv pip install --system --no-cache ".[all]"

# 代码
COPY . /app

# 非 root 运行
RUN useradd -m -u 1000 mujik && chown -R mujik:mujik /app
USER mujik

ENV PATH="/home/mujik/.local/bin:$PATH"

ENTRYPOINT ["mujik"]
CMD ["--help"]
