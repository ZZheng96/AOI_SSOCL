"""F1 产线集成测试：工单(绑定模板) + 数据源 → 产线消费未检测图 → 双检落库。

不依赖 server，直接驱动 PipelineService。
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.DEBUG,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
SMT = ROOT / "alg_repo" / "smt"
PAIR = SMT / "test_images" / "excess_pair" / "pair4"


def main() -> None:
    from app.core.pipeline_service import PipelineService
    from app.core.ingest import import_folder
    from app.db.database import session_scope
    from app.db.models import DataSource, Detection, WorkOrder, WorkOrderSource
    from app.template.store import TemplateStore

    tpl = TemplateStore().find_by_standard_path(PAIR / "PixPin_2026-07-01_11-21-15_OK.png")
    assert tpl is not None, "模板不存在，先跑 _verify_traditional.py"
    if tpl.model_category != "smt_solder":
        tpl.model_category = "smt_solder"
        tpl.engine_mode = "dual"
        TemplateStore().save(tpl)

    # 清理旧测试数据
    with session_scope() as s:
        s.query(WorkOrderSource).filter(
            WorkOrderSource.workorder_id.in_(
                s.query(WorkOrder.id).filter(WorkOrder.name == "pipeline_test")))
        s.query(Detection).filter(Detection.workorder_id.is_(None)).filter(
            Detection.template_id == tpl.id).delete()
        old = s.query(WorkOrder).filter(WorkOrder.name == "pipeline_test").all()
        for wo in old:
            s.query(WorkOrderSource).filter(WorkOrderSource.workorder_id == wo.id).delete()
            s.delete(wo)
        ds = s.query(DataSource).filter(DataSource.name == "pipeline_test_src").first()
        if ds is not None:
            s.delete(ds)

    # 数据源 + 导入（pair4 目录：2 张图）
    with session_scope() as s:
        ds = DataSource(name="pipeline_test_src", modality="image",
                        label_tier="L1a", pretrain_normal=100,
                        pretrain_anomaly=30, batch_size=30)
        s.add(ds)
        s.flush()
        ds_id = ds.id
    import_folder(str(PAIR), category="smt_solder", source="datasource",
                  copy_to_storage=False, datasource_id=ds_id)

    # 工单绑定模板 + 挂源
    with session_scope() as s:
        wo = WorkOrder(name="pipeline_test", pipeline_status="running",
                       template_ref=tpl.id, label_tier="L1a")
        s.add(wo)
        s.flush()
        s.add(WorkOrderSource(workorder_id=wo.id, datasource_id=ds_id))
        wo_id = wo.id
    print(f"工单 {wo_id} 已创建（模板 {tpl.id}，源 {ds_id}）", flush=True)

    # 启动产线服务，消费未检测图
    svc = PipelineService.get()
    svc.start()
    deadline = time.time() + 60
    n_det = 0
    while time.time() < deadline:
        with session_scope() as s:
            n_det = (s.query(Detection)
                     .filter(Detection.workorder_id == wo_id).count())
        if n_det >= 1:
            break
        time.sleep(1)
    svc.stop()

    with session_scope() as s:
        rows = (s.query(Detection).filter(Detection.workorder_id == wo_id)
                .order_by(Detection.id).all())
        print("产线落库检测数:", len(rows))
        for r in rows:
            print(f"  det{r.id}: {Path(r.image_path).name} "
                  f"engine={r.engine_mode} trad={r.traditional_overall} "
                  f"feat={r.feature_decision} anomaly={r.is_anomaly} "
                  f"({r.latency_ms:.0f}ms)")
    assert n_det >= 1, "产线未消费任何图"


if __name__ == "__main__":
    main()
