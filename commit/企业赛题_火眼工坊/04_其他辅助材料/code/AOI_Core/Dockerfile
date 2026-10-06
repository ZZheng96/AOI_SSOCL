# AOI 后端服务镜像（桌面 UI 请原生运行 PySide6，不在容器内）
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# OpenCV 等依赖的系统库
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 先装 CPU 版 torch（GPU 部署请改用 cu121 源或 nvidia 基础镜像）
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install -r requirements.txt

# 后端代码 + 自学习算法引擎及本地模型资源
# 2026-09-01：补 algo/ 与 assets/。缺这两层时镜像内 import algo 失败、
# DINOv2 vendor 权重缺失会触发联网下载，容器交付实际不可自包含运行。
COPY backend ./backend
COPY algo ./algo
COPY assets ./assets
COPY configs ./configs
COPY server.py .

# 运行数据通过卷挂载：/data/storage /data/database
ENV AOI_CONFIG=/app/configs/docker.yaml
EXPOSE 8017

CMD ["python", "server.py"]
