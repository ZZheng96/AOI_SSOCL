# AOI 后端服务镜像（桌面 UI 请原生运行 PySide6，不在容器内）
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# OpenCV 等依赖的系统库
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# torch 轮子源按平台参数化：默认 CPU；GPU 构建传
#   --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121
# （或直接使用 docker-compose.gpu.yml 覆盖文件，无需手工改本文件）
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
RUN pip install torch torchvision --index-url ${TORCH_INDEX_URL}

COPY requirements.txt .
RUN pip install -r requirements.txt

# 构建上下文为 AOI_tree，代码和 DINOv2 权重一起打入镜像，支持离线启动。
COPY AOI_Core/backend ./backend
COPY AOI_Core/algo ./algo
COPY AOI_Core/assets ./assets
COPY assets/dinov2 ./assets/dinov2
COPY AOI_Core/configs ./configs
COPY AOI_Core/server.py .

# 运行数据通过卷挂载：/data/storage /data/database
ENV AOI_CONFIG=/app/configs/docker.yaml
EXPOSE 8017

# A16（2026-10-03）：非 root 运行——此前容器以 root 跑服务，被 RCE/反序列化
# 命中即为容器内 root。建专用用户并把 /data 卷与 /app 配置目录交给它。
RUN useradd -r -u 10001 -m aoi && \
    mkdir -p /data/storage /data/database && \
    chown -R aoi:aoi /data /app
USER aoi

# 生产启动自检：/api/health 为探活豁免路径（无需 API Key）。
# start-period 放宽到 120s：torch 导入与引擎快照扫描在冷机上较慢。
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8017/api/health', timeout=4)"]

CMD ["python", "server.py"]
