"""AOI_sys 测试包。

目录结构：
- run_all.py     全量回归调度入口：python -m tests.run_all [关键字过滤]
- framework.py   测试公共框架（断言/HTTP/结果落盘）
- testdata.py    测试数据供给（自动探测可用数据集，无则合成兜底）
- smoke/         功能冒烟（独立临时环境，不依赖已启动的服务）
- ui/            UI 离屏冒烟
- bench/         性能/长稳实测（需已启动的后端服务）
"""
