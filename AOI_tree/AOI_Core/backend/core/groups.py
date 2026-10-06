"""数据源分组引擎（前端反馈 v9）。

数据源内分两组：
- 预训练组：按品类取前 N 正常 + M 异常，严格只取 train/valid 域；
  test 域只进入检测组，避免训练/测试泄漏。
- 检测组：该品类其余全部图片，按 batch_size 分成检测批次。

分组是动态计算（按当前 N/M/批次量与库内图片），不落表；
「另存方案」把预训练组图片清单固化到 DataSourcePlan，导入时可按方案复用。
"""
from __future__ import annotations

from ..db.models import Dataset, Image as ImageRow

# 预训练组正常图优先来源（split）；不足时用其余 split 补充
_SPLIT_PRIORITY = ("train", "valid")


def _pick(rows, label: str, n: int, allow_test: bool = True) -> list:
    """按 id 升序取该 label 的前 n 张。

    优先 _SPLIT_PRIORITY（train/valid）；不足时：
    - anomaly：只取 train/valid 域缺陷；
    - normal：只取 train/valid 域正常图；test 域不进入预训练。
    """
    pri = [i for i, lbl, sp in rows if lbl == label and sp in _SPLIT_PRIORITY]
    if n <= 0:
        return []
    if not allow_test:
        return pri[:n]
    rest = [i for i, lbl, sp in rows if lbl == label and sp not in _SPLIT_PRIORITY]
    return (pri + rest)[:n]


def compute_groups(s, datasource_id: int, pretrain_normal: int,
                   pretrain_anomaly: int, batch_size: int,
                   plan_pretrain_ids: list | None = None) -> dict:
    """按品类分组：{cat: {pretrain:[id], detect_batches:[[id]...], counts}}。

    plan_pretrain_ids：方案提供的预训练组图片 id 清单（复用方案时固定分配）。
    """
    ds_ids = [r[0] for r in s.query(Dataset.id).filter(
        Dataset.datasource_id == datasource_id).all()]
    if not ds_ids:
        return {}
    rows = (s.query(ImageRow.id, ImageRow.category, ImageRow.label,
                    ImageRow.split)
            .filter(ImageRow.dataset_id.in_(ds_ids))
            .order_by(ImageRow.id.asc()).all())
    # 模板图（split=template）是 L3 比对参考，不参与预训练组/检测组
    # （UI 实测发现：不过滤会混进检测批次被产线当检测样本消费）
    rows = [(i, cat, lbl, sp) for i, cat, lbl, sp in rows if sp != "template"]
    cats = sorted({c for _, c, _, _ in rows})
    out: dict = {}
    plan_set = set(plan_pretrain_ids or [])
    for cat in cats:
        cat_rows = [(i, lbl, sp) for i, c, lbl, sp in rows if c == cat]
        if plan_pretrain_ids is not None:
            # 方案清单也必须遵守训练/测试隔离；旧方案可能包含历史 test id。
            pretrain = [i for i, _, sp in cat_rows
                        if i in plan_set and sp in _SPLIT_PRIORITY]
        else:
            pretrain = (_pick(cat_rows, "normal", max(pretrain_normal, 0),
                              allow_test=False)
                        + _pick(cat_rows, "anomaly", max(pretrain_anomaly, 0),
                                allow_test=False))
        ps = set(pretrain)
        detect = [i for i, _, _ in cat_rows if i not in ps]
        bs = max(batch_size, 1)
        batches = [detect[i:i + bs] for i in range(0, len(detect), bs)]
        n_pre_n = sum(1 for i, lbl, _ in cat_rows
                      if lbl == "normal" and i in ps)
        n_pre_a = sum(1 for i, lbl, _ in cat_rows
                      if lbl == "anomaly" and i in ps)
        out[cat] = {
            "pretrain": pretrain,
            "pretrain_counts": {"normal": n_pre_n, "anomaly": n_pre_a},
            "detect_batches": batches,
            "detect_count": len(detect),
            "total": len(cat_rows),
        }
    return out
