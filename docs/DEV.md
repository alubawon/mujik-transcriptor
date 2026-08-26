# Development Guide

本项目的所有开发/测试/运行**都应在隔离容器中**进行，不在裸机跑。

## 1. 前置条件

### 容器运行时
推荐 OrbStack（macOS 原生，资源占用低）。

```bash
brew install --cask orbstack
```

启动 OrbStack.app（首次需要授权系统扩展，可能要重启）。

### 验证安装
```bash
docker --version
docker compose version
docker run --rm hello-world
```

## 2. 开发工作流

### 方式 A：devcontainer（推荐，VSCode/Cursor）

直接在 IDE 中打开项目：
- VSCode：安装 `Dev Containers` 扩展，命令面板 `Dev Containers: Reopen in Container`
- Cursor：内置支持

`.devcontainer/devcontainer.json` 会：
1. 基于 `Dockerfile` 的 `dev` 阶段构建镜像
2. 挂载源码到 `/app`
3. 启动后自动跑 `python -m pytest tests/unit/`

### 方式 B：docker compose（CLI）

```bash
# 启动 dev 容器（CPU，源码挂载）
docker compose --profile dev up -d pipeline-dev

# 进入容器
docker compose --profile dev exec pipeline-dev bash

# 容器内：跑测试
cd /app
python -m pytest tests/unit/ -v

# 跑 CLI
mujik --help
mujik run -i data/song.wav -o out/ --preset pop
mujik render -i score.musicxml -o score.svg

# 退出容器（容器仍在跑）
exit

# 停止容器
docker compose --profile dev down
```

### 方式 C：纯 docker（不用 compose）

```bash
# 构建 dev 镜像
docker build --target dev -t mujik-transcriptor:dev .

# 跑容器（源码挂载）
docker run -it --rm \
  -v $(pwd)/src:/app/src \
  -v $(pwd)/tests:/app/tests \
  -v $(pwd)/config:/app/config \
  -v mujik-venvs:/app/.venv \
  -v mujik-weights:/weights \
  mujik-transcriptor:dev \
  bash

# 容器内
cd /app
python -m pytest tests/unit/ -v
```

## 3. 端到端 Pipeline（GPU）

```bash
# 启动完整栈
docker compose --profile all up -d

# 跑管线
docker compose --profile pipeline exec pipeline-gpu \
  mujik run -i /data/song.wav -o /out/ --preset pop

# 产物在 ./out/
ls out/
```

## 4. 仅 GPL 渲染服务

```bash
# 单独启动 LilyPond 服务
docker compose --profile lilypond up -d render-lilypond

# 健康检查
curl http://localhost:5001/health

# 关闭
docker compose --profile lilypond down
```

## 5. 重建镜像

代码改了 `pyproject.toml` 或 `Dockerfile` 时：

```bash
docker compose --profile dev build pipeline-dev
docker compose --profile dev up -d pipeline-dev
```

仅源码改动（`src/`、`tests/`）不需要重建，因为是 volume 挂载。

## 6. 重置 venv

```bash
# 容器内
rm -rf /app/.venv
uv venv --python 3.11 /app/.venv
source /app/.venv/bin/activate
uv pip install -e ".[dev,core-io,render]"
```

或用 volume 重置：
```bash
docker volume rm mujik-dev-venv
docker compose --profile dev up -d pipeline-dev  # 自动重建
```

## 7. 一次性跑命令

```bash
# 在新容器里跑测试（不进入）
docker compose --profile dev run --rm pipeline-dev \
  python -m pytest tests/unit/

# 在新容器里跑 CLI
docker compose --profile dev run --rm pipeline-dev \
  mujik --help
```

## 8. 排错

```bash
# 看容器日志
docker compose --profile dev logs pipeline-dev

# 进容器排查
docker compose --profile dev exec pipeline-dev bash
which python
python --version
ls -la /app
```

## 9. 切换容器运行时

如果从 Docker Desktop 切到 OrbStack（反之亦然），无需改项目配置——Docker CLI 在两者上都一致。

## 10. 不用容器（不推荐）

仅在调试容器构建本身时才在裸机跑：

```bash
# 裸机
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -e ".[dev,core-io,render]"
python -m pytest tests/unit/
```

⚠️ **生产开发必须在容器内**，避免污染系统 Python。
