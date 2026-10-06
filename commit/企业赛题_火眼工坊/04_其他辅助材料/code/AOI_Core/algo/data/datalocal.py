"""data_local 适配器（诚实协议，无 pairs——文件名配对是红线 2 禁止的隐式信息）

目录：{category}/{train/good, val/{good,...defect dirs}, test/...}
协议：
  init_normal = train/good 抽 100；init_defect = test 域缺陷抽 ≤30（赛题协议）；
  val 域做验证（阈值/选型）；test 域剔除 init_defect 后做最终评测。
  注意：test/good 只出现在评测侧，绝不进 fit（与 demo4 泄漏版的关键区别）。
"""
import os
import numpy as np
from .base import Bundle

IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp")

# 预置本地测试品类（M14 通用化：load_all 改为按 root 目录自动发现——
# 客户数据集品类任意，不再依赖硬编码清单；CATEGORIES 仅作预置示例保留）
CATEGORIES = ["component", "extra_part", "gold_finger", "solder_smt"]


def _list_images(d):
    if not os.path.isdir(d):
        return []
    out = []
    for dp, _, fn in os.walk(d):
        for f in fn:
            if f.lower().endswith(IMG_EXT):
                out.append(os.path.join(dp, f))
    return sorted(out)


def _split_good_defect(cat_dir, split):
    sdir = os.path.join(cat_dir, split)
    good = _list_images(os.path.join(sdir, "good"))
    defect = []
    if os.path.isdir(sdir):
        for sub in sorted(os.listdir(sdir)):
            if sub != "good" and os.path.isdir(os.path.join(sdir, sub)):
                defect += _list_images(os.path.join(sdir, sub))
    return good, defect


def load_category(root, category, n_init_normal=100, n_init_defect=30, seed=42,
                  max_test=150):
    """max_test（2026-08-16 用户规则）：test 总样本上限——超过则随机抽 max_test 张，
    调优迭代不跑全量；最终验证用 --max-test 0 关闭抽样。"""
    rng = np.random.default_rng(seed)
    cat_dir = os.path.join(root, category)
    tr_good, _ = _split_good_defect(cat_dir, "train")
    te_good, te_defect = _split_good_defect(cat_dir, "test")

    # 实际数据无 val 划分：阈值按红线 3 由 train/good 分位数定，test 域全量评测
    init_normal = sorted(rng.choice(tr_good, min(n_init_normal, len(tr_good)), replace=False).tolist())
    # 缺陷池对半分：init 拿一半（≤30），另一半留给评测（component 仅 14 张缺陷）
    n_init = min(n_init_defect, max(1, len(te_defect) // 2))
    sel = rng.choice(len(te_defect), n_init, replace=False) if n_init else []
    init_defect = sorted(te_defect[i] for i in sel)
    eval_defect = sorted(p for i, p in enumerate(te_defect) if i not in set(sel))

    test = [(p, 0) for p in te_good] + [(p, 1) for p in eval_defect]
    rng.shuffle(test)
    if max_test and len(test) > max_test:
        test = test[:max_test]
        rng.shuffle(test)
    return Bundle(name=category, init_normal=init_normal,
                  init_defect=init_defect, val=[], test=test).as_dict()


def load_all(root, cfg, max_test=150):
    p = cfg["protocol"]
    # M14 通用化：按 root 目录自动发现品类（客户数据集品类任意；跳过非目录/
    # 隐藏目录/无 train 结构的目录）
    cats = [c for c in sorted(os.listdir(root))
            if os.path.isdir(os.path.join(root, c)) and not c.startswith(".")
            and os.path.isdir(os.path.join(root, c, "train"))]
    if not cats:
        cats = CATEGORIES
    return {c: load_category(root, c, p["n_init_normal"], p["n_init_defect"],
                             cfg["seed"], max_test=max_test)
            for c in cats}
