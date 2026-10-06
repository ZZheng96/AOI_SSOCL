"""组级特征贡献评估（诚实口径）

三层证据回答"每组特征实际起了什么作用"：
  1. 独立 AUROC：只用该组维度打分——"它单独行不行"
  2. 留一消融：全向量去掉该组后的 AUROC 损失——"它对整体是否不可替代"
  3. 排列重要性：测试集内打乱该组维度后的 AUROC 损失（n_perm 次均值±std）——
     "打乱它整体掉多少"，对冗余组更敏感

红线（继承主干评测规范）：
  - fit 只用 init_normal（train 域），过 isolation.guard_bundle 校验
  - AUROC 不做方向翻转（max(auc,1-auc) 是用测试标签的乐观偏差，禁止）
  - 负贡献如实登记，不删不报
"""
import numpy as np

from . import ensure_core_importable

ensure_core_importable()
from algo.common.isolation import guard_bundle      # noqa: E402
from algo.eval.metrics import image_metrics          # noqa: E402


def evaluate_category(extractor, bundle, load_image, groups_meta: list,
                      k: int = 3, n_perm: int = 5, seed: int = 42,
                      protocol: dict | None = None, name: str | None = None) -> dict:
    """对单个品类跑三层贡献评估，返回贡献档案 dict。

    extractor: 任何带 extract(img)->vec 的对象（主干 extractor 或 FeatureSet）
    groups_meta: FeatureSet.probe() 的输出（[{key,name,dim,slice,...}]），
                 组集合与切片完全由当前特征配置决定
    bundle 需满足 algo.data 适配器约定（init_normal/init_defect/val/test；
    适配器 as_dict 不含 name，品类名由调用方经 name 传入）。
    """
    from .scoring import GroupKNN, extract_features

    guard_bundle(bundle, protocol or {})
    rng = np.random.default_rng(seed)

    ref_paths = list(bundle["init_normal"])
    test_paths = [p for p, _ in bundle["test"]]
    test_labels = np.array([lb for _, lb in bundle["test"]])

    ref_vecs, _ = extract_features(extractor, ref_paths, load_image)
    test_vecs, ok_paths = extract_features(extractor, test_paths, load_image)
    # 解码失败剔除后按 ok_paths 对齐标签（不静默错位）
    keep = [i for i, p in enumerate(test_paths) if p in set(ok_paths)]
    test_labels = test_labels[keep]

    scorer = GroupKNN(k=k).fit(ref_vecs)
    m_full = image_metrics(test_labels, scorer.score(test_vecs))
    full = m_full["auroc"]

    groups = {}
    for m in groups_meta:
        sl = m["slice"]
        standalone = image_metrics(test_labels, scorer.score(test_vecs, sl))["auroc"]
        rest = np.array([i for i in range(test_vecs.shape[1])
                         if not (sl.start <= i < sl.stop)])
        loo = image_metrics(test_labels, scorer.score(test_vecs, rest))["auroc"]
        perm_drops = [full - image_metrics(
            test_labels, scorer.score_permuted(test_vecs, sl, rng))["auroc"]
            for _ in range(n_perm)]
        groups[m["key"]] = {
            "name": m["name"], "dim": m["dim"],
            "standalone_auroc": standalone,
            "loo_auroc": loo,
            "loo_drop": full - loo,                  # >0 该组不可替代；<0 该组在拖累
            "perm_drop_mean": float(np.mean(perm_drops)),
            "perm_drop_std": float(np.std(perm_drops)),
        }

    return {
        "category": name or bundle.get("name") or "unknown",
        "n_ref": len(ref_vecs), "n_test": int(len(test_labels)),
        "n_defect": int(test_labels.sum()),
        "full_auroc": full,
        "full_ap": m_full["ap"],
        "groups": groups,
        "meta": {
            "k": k, "n_perm": n_perm, "seed": seed,
            "feature_set": (extractor.config.get("name")
                            if hasattr(extractor, "config") else "trunk_raw"),
            "caliber": "U27 播种 / U36 z-clip ±10 / 独立测试查询保留最近邻（训练自身查询才跳过）",
            "honesty": "AUROC 不翻转；fit 仅用 train 域 init_normal；负贡献如实登记",
        },
    }
