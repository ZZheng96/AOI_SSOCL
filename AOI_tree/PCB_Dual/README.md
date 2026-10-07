# PCB_Ins v2

融合传统算法 PCB 检测（PCB_Ins）+ 特征学习通用检测（AOI_sys/demo5）的工业 PCB 检测系统。
用户入口统一为 PCB_Ins，无感 AOI_sys 存在，同一张图双检并行：传统引擎检具体缺陷类型（缺件/多锡/虚焊…），特征引擎检通用外观异常并可在线学习。

## 快速开始

```powershell
cd d:\CGAIC\PCB_Ins_v2
python main.py             # 桌面端（内嵌服务 + UI，推荐入口）
python server.py           # 仅 REST 服务 http://127.0.0.1:8022
python tests\_verify_traditional.py   # 传统引擎验证（模板+检测）
python tests\_verify_feature.py       # 特征引擎验证（品类模型）
python tests\_verify_dual.py          # 双检闭环验证
python tests\_verify_pipeline.py      # 产线集成验证
python tests\_smoke_full.py           # API 冒烟（先启动 server）
```

## 目录结构

```text
PCB_Ins_v2/
├── main.py                # 桌面入口（内嵌服务 + UI）
├── server.py              # 独立服务启动
├── configs/               # yaml 配置（default.yaml / demo5_fast.yaml）
├── app/
│   ├── api/               # FastAPI：templates/detect/models/feedback/workorder/datasources
│   ├── core/              # 配置/DB/数据导入/产线服务
│   ├── db/                # SQLite ORM（AOI_sys 迁入 + 双检字段）
│   ├── detect/            # 传统引擎：catalog/registry/scheduler/gate/adapters
│   ├── inspect/           # 检测服务编排 + 双检融合（fusion.py / detect_service.py）
│   ├── engines/           # 特征引擎客户端（FeatureClient → HTTP 调用树干 AOI_Core）
│   ├── template/          # 模板库（InspectionTemplate 配方 + engine_mode/model_category）
│   ├── recipe/            # 参数/预处理配方存储
│   ├── calibration/       # ROI 标定存储
│   ├── ui/                # PySide6 前端（PCB_Ins 复用 + 嵌入 AOI_Core 页面）
│   └── utils/             # 图像 IO 等
├── alg_repo/              # 算法仓库位置（smt 已入；through_hole/gold_finger/wrinkle/body 预留）
├── templates/             # 模板数据（templates/{id}/template.json）
├── storage/               # 参数/标定/DB/输出/日志
└── docs/架构与设计.md      # 架构、接口、关键发现、路线
```

## 核心概念

- **模板**：金样板 + 检测项 + ROI + 参数（传统差分参考）；`engine_mode` 配置双检组合，`model_category` 绑定品类模型
- **品类模型**：由树干 AOI_Core 统一管理（PCB_Dual 启动时以子进程拉起 Core :8017），支持反馈即学/版本管理
- **算法仓库**：`alg_repo/<name>` 放外部算法源码，adapter 契约接入；未就绪自动降级
- **工单/数据源**：产线任务实例（绑定模板 + 数据源，暂停/恢复/回队），数据源目录导入
- **学习闭环**：检测 → 复判反馈 → AOI 即学（双库/重校准/孵化），学习状态可查

## 验证结果（P0-P7）

- 传统引擎：SMT 四类缺陷，NG 图检出多锡框、OK 图通过
- 特征引擎：品类模型 prepare 5s，predict 45-144ms（fast）
- 双检闭环：NG 图双判 NG（~300ms）、OK 图双过
- 学习闭环：反馈即学 18ms，defect_bank 累积
- 产线集成：工单自动消费（NG 308ms / OK 90ms），回队插队/暂停恢复/PLC 触发
- 桌面 UI：6 页签（含品类模型页），检测页双检分栏展示
- **单图双检 <400ms，满足 1s 红线**
