# mujik-transcriptor MuseScore 渲染服务（GPL 隔离）
# 主线 mujik 通过 HTTP 调用本服务
# 自身许可证：GPL-2.0+（因 MuseScore 传染）

FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

# MuseScore 4 + xvfb（headless）
RUN apt-get update && apt-get install -y --no-install-recommends \
        musescore3 \
        xvfb \
        libqt5widgets5 \
        libqt5gui5 \
        libqt5network5 \
        libqt5xml5 \
        ffmpeg \
        ca-certificates \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY server.py requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5002

# 健康检查
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s \
    CMD curl -f http://localhost:5002/health || exit 1

# 用 xvfb-run 包裹 mscore 调用
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "5002"]
