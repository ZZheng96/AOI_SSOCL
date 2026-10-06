# AOI 实时在线 AI 质检系统

> 面向工业质检的可自学习异常检测系统：图片/视频实时检测 + 操作员反馈驱动在线学习 + 可解释输出。
> 本仓库为参赛与工程评审交付包，主系统为 `AOI_Core`。

---

## 1. 系统组成

| 目录 | 定位 | 说明 |
|------|------|------|
| `AOI_Core/` | **主系统** | 桌面端（PySide6）+ FastAPI 后端 + 检测引擎（DINOv2 多槽位融合 + SSOCL 持续学习）+ Docker 交付 |
| `AOI_feature/` | 特征分析子系统（内置） | 传统特征贡献评估 / 特征集管理 GUI（主干同源，parity 校验通过） |
| `PCB_Dual/` | 双引擎分支 | 传统 CV 算法仓库 + 特征引擎双检融合（PCB 场景专用） |
| `research/` | 复现包 | 赛题要求、研究报告、算法设计文档 |
| `assets/` | 模型权重 | DINOv2 主干（`assets/dinov2`）、判别头预训练权重、SHA-256 校验清单 |

---

## 2. 快速启动

### 2.1 桌面端（推荐，内嵌后端）

```powershell
# 环境：Python 3.10+，依赖见 AOI_Core/requirements.txt
cd AOI_tree/AOI_Core
pip install -r requirements.txt          # 或 pip install -r requirements-lock.txt（锁定版本）
python main.py                           # 自动内嵌启动后端 + 打开桌面 UI
```

桌面端默认连接 `http://127.0.0.1:8017`，首次启动自动完成模型加载与自检。

### 2.2 独立后端服务

```powershell
cd AOI_tree/AOI_Core
python server.py                         # 前台启动 uvicorn，端口 8017
```

### 2.3 Docker 部署（产线推荐）

```powershell
# CPU 版（默认）
cd AOI_tree/AOI_Core
set AOI_API_KEY=<强随机密钥>             # 生产必填，容器内强制鉴权
docker compose up -d --build

# GPU 版（CUDA 12.1，需 nvidia-container-toolkit）
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
```

- 构建上下文为 `AOI_tree`，DINOv2 权重已打入镜像，支持离线启动。
- 数据持久化卷：`aoi-storage`（图像/快照/学习产物）+ `aoi-database`（SQLite）。
- 健康检查：`GET /api/health`（探活豁免，无需 Key）。
- 生产模式：环境变量 `AOI_PRODUCTION=1` 已默认开启；未设置 `AOI_API_KEY` 时拒绝启动。

---

## 3. CPU / GPU 方案

| 场景 | 入口 | 说明 |
|------|------|------|
| 开发/演示 | `python main.py` | 自动检测 CUDA，有 GPU 用 GPU，无 GPU 回落 CPU |
| CPU 容器 | `docker-compose.yml` | 默认 CPU torch，适合无显卡产线服务器 |
| GPU 容器 | `docker-compose.yml + docker-compose.gpu.yml` | cu121 轮子，需 NVIDIA 驱动 ≥525 |
| 自定义 CUDA | `docker build --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cuXXX` | 按实际驱动选择轮子源 |

权重与依赖校验：

```bash
# 权重完整性（SHA-256）
sha256sum -c assets/checksums.sha256      # Windows 可用 Git Bash

# 依赖锁定复现（同平台同 Python 版本）
pip install -r AOI_Core/requirements-lock.txt
```

---

## 4. 测试与验收

```powershell
cd AOI_tree/AOI_Core

# 全部冒烟测试（功能正确性，独立临时环境不污染生产库）
python -m tests.run_all

# 关键模块
python -m tests.smoke.m01_engine          # 引擎：训练→预测→快照→反馈即学
python -m tests.smoke.m02_e2e             # 后端端到端全链路
python -m tests.smoke.m10_line_safety     # 产线安全：鉴权/白名单/PLC
python -m tests.smoke.m11_roles           # 权限角色/对位预警/日志轮转
python -m tests.smoke.m11_production_mode # 生产模式启动自检（2026-10-02 新增）

# 性能基准（需后端已启动且品类已 prepare）
python -m tests.bench.fullchain --category <品类名>                    # 工程红线 <1s
python -m tests.bench.fullchain --category <品类名> --competition \
    --paths <2500x2500图目录> --warmup 5 --rounds 3                     # 赛题口径 <200ms

# 视频异常检测协议（2026-10-02 新增）
python -m tests.bench.video_protocol --quick

# 自学习回归门禁协议（2026-10-02 新增）
python -m tests.bench.learning_protocol --quick
python -m tests.bench.learning_protocol --seeds 42,1,2 --n-rounds 3    # 标准档
```

测试报告统一落盘 `AOI_Core/storage/logs/`（JSON 格式，含环境 manifest）。

---

## 5. 配置说明

主配置文件：`AOI_Core/configs/default.yaml`（开发）/ `AOI_Core/configs/docker.yaml`（容器）。

关键配置项：

| 配置 | 默认 | 说明 |
|------|------|------|
| `pipeline.latency_budget_ms` | 200 | 延迟考核预算（对应赛题 2060 GPU 2500² <200ms） |
| `security.api_key` | 空 | 开发模式关闭鉴权；生产通过环境变量 `AOI_API_KEY` 注入 |
| `security.keys` | `{}` | 多角色密钥表（operator/engineer/admin） |
| `system.engine_storage` | `storage/engine` | 模型快照根目录（Docker 下为 `/data/storage/engine`，已入卷） |

完整配置说明见 `AOI_Core/docs/用户手册.md` §配置中心。

---

## 6. 文档索引

| 文档 | 位置 |
|------|------|
| 用户手册 | `AOI_Core/docs/用户手册.md` |
| 全流程操作 | `AOI_Core/docs/全流程操作.md` |
| 数据生命周期与清理策略 | `AOI_Core/docs/数据生命周期与清理策略.md` |
| 赛题要求 | `research/赛题/可自学习的AOI实时在线AI质检.md` |
| 系统评审报告 | `review/AOI系统问题发现与完善.md` |

---

## 7. 已知限制与诚实声明

- **性能指标**：赛题硬指标（2500×2500 / 2060 GPU / <200ms）需在指定硬件上复测确认；当前报告为开发机实测，未在 2060 实机验证前不作为正式承诺。
- **视频检测**：支持视频文件逐帧检测与异常注入协议验证；实时流（RTSP）需产线环境实测。
- **自学习**：反馈驱动学习闭环已验证（正确反馈不退化、错误反馈有门禁拦截）；跨品类统计收益需多品类多种子复跑。
- **数据隔离**：`algo/common/isolation.py` 强制 fit/test 零交集，违反即拒绝运行。

---

## 8. 许可证

- DINOv2 主干代码与权重：Apache 2.0（见 `assets/dinov2/LICENSE`）。
- 本项目代码：参赛交付用途，内部评审。
