# dev_async 进度（render/persist 移出关键路径）

- [x] 1 读主流程 + 分段计时
- [x] 2 选改法并实现
- [x] 3 tests/smoke/test_async_persist.py
- [x] 4 A/B ≤10 张
- [x] 5 全量 smoke（后台）
- [x] 6 文档 §/SYNC/清单

## 日志
- 2026-10-13 建立进度文件
- 2026-10-13 步骤 1 完成（bench_breakdown.py，n=24，6 图 × 4 轮）：
  - e2e p50/p95 ≈ 150/221。
  - 只有异常图有 render 段，32~36ms。其中 overlay 22~25、heatmap_u8 14/26、mask_boxes 6.6/9.1、resize 5.2、cvt 1.7、colormap 1.9、addWeighted 3.5。
  - persist 4.6/20.3，不计入 e2e。io_wait 恒为 0，所以改法 (a) 对 e2e 没有收益。
  - 选改法 (b)：把 resize、cvt、着色、叠加、画框移入线程池，预计省 12~19ms。改前副本存为 service_before.py。
- 2026-10-13 步骤 2/3 完成：render 已移入线程池，补了 _track/wait_artifact/_safe_job，routes_system.api_file 会等待在途产物。另做两项逐位等价的微优化。test_async_persist 5 项 OK。
- 2026-10-13 步骤 4 完成：三次 A/B 的 render P95 为 34→19。e2e P95 分别为 225.3/225.3、203.4/214.5、207.8/190.4，受机器噪声影响大。breakdown 空闲时为 180.6。三次精度检查都是 identical。
- 2026-10-13 步骤 5 完成：smoke 383 tests OK（98.3s）。
- 2026-10-13 步骤 6 完成：验证记录 §73、协同 SYNC-23、任务清单 G1（HARD-01）。
