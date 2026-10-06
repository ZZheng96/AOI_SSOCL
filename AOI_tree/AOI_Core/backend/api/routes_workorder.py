"""工单制核心接口（W-workorder，2026-08-27 前端反馈 v2 重构）。

契约（与用户澄清对齐）：
- 工单 = 产线任务实例（如"3号线 SMT 质检"），下挂数据源，可含多品类，
  可保存为任务模板 / 从模板导入配置；
- 数据源 = 独立登记的数据实体（名称 + 模态图片/视频 + 采样配置），
  可被多个工单复用；数据按数据源管理（数据源 -> 品类 -> 批次）；
- 数据条件在工单级声明，结构为"三档 + 两开关"（非平铺五选一）：
  * 标注档位（信息量递进，选高档自动兼容低档）：
    L0 仅正常图 / L1a 图像级标注 / L1b 缺陷位置标注
  * 两个可叠加开关：品类独立（数据已分品类）/ 模板比对（提供模板图）
- 导入数据后体检：声明条件 vs 数据实际支撑能力，不符给警告并
  自动调整为合适条件（可修复数据后再改工单、再次体检）；
- 统计以工单为基础：工单行汇总 + 选中切换 + 时间范围（today/7d/all）。

A18（2026-10-03）：本模块为聚合入口，路由实现按域拆分——
数据源/分组方案见 routes_wo_datasource，工单 CRUD/体检/模板见
routes_wo_order，统计/产线控制/队列/学习/报告见 routes_wo_stats，
共享常量与辅助函数见 _wo_shared。对外契约不变：app.py 仍以
routes_workorder.router 注册全部 /api 路由。
"""
from fastapi import APIRouter

from . import routes_wo_datasource, routes_wo_order, routes_wo_stats

router = APIRouter()
router.include_router(routes_wo_datasource.router)
router.include_router(routes_wo_order.router)
router.include_router(routes_wo_stats.router)
