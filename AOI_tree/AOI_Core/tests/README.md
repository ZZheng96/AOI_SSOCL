tests/                        # 唯一测试包
├── __init__.py               # 包说明：各模块职责一览
├── run_all.py                # 统一调度：python -m tests.run_all [关键字过滤]
├── framework.py              # 断言/HTTP/结果落盘框架
├── testdata.py               # 数据集自动探测 + 合成兜底
├── smoke/                    # 功能冒烟（独立临时环境，不污染生产库）
│   ├── m01_engine.py         #   引擎层：训练→预测→快照→反馈即学→版本固化
│   ├── m02_e2e.py            #   后端端到端全链路
│   ├── m10_line_safety.py    #   产线安全：鉴权/白名单/PLC/归档/导出
│   ├── m11_roles.py          #   权限角色/对位预警/日志轮转
│   ├── m11_production_mode.py #  生产模式安全：拒绝启动/文档关闭/CORS/鉴权（12 断言）
│   ├── m14_template.py       #   L3 模板/在线学习/分层徽章
│   └── m15_feedback.py       #   反馈/主动选样/导出端点
├── ui/offscreen.py           # UI 离屏冒烟（10 页面实例化+切换）
├── bench/                    # 性能实测（需已启动的后端，不进 run_all 编排）
│   ├── fullchain.py          #   全链路延迟三层口径（200ms 竞赛/1s 红线/2s CPU 挑战）
│   ├── stability.py          #   并发长稳+内存泄漏监控
│   ├── video_protocol.py     #   视频异常检测协议复现（FFV1/原生分辨率/可配置截断）
│   └── learning_protocol.py  #   自学习双臂对照回归门禁（隔离红线+固定 holdout）
└── results/legacy/           # 历史验收快照（归档保留，不污染新结果）