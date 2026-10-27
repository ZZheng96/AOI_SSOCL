"""MVTec 格式数据集统一适配器（mvtec / MPDD / BTAD，2026-08-16 目标模式）

目录约定：
  {root}/{category}/train/good/...            正常（BTAD 为 train/ok）
  {root}/{category}/test/{good|缺陷类型}/...   评测（BTAD 为 test/{ok,ko}）

协议（与 datalocal 相同的诚实口径）：
  init_normal = train/good 抽 n（大数据集"只用少量"：n_init_normal 上限）
  init_defect = test 缺陷抽一半（≤30，赛题协议）
  test = te_good 全量（可抽样上限）+ 剩余缺陷（shuffle）
  test/good 绝不进 fit。
"""
import os
import numpy as np
from .base import Bundle

IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp")

# BTAD 子目录映射
_BTAD_MAP = {"ok": "good", "ko": "defect"}


def _list_images(d):
    if not os.path.isdir(d):
        return []
    out = []
    for dp, _, fn in os.walk(d):
        for f in fn:
            if f.lower().endswith(IMG_EXT):
                out.append(os.path.join(dp, f))
    return sorted(out)


def _split_good_defect(cat_dir, btad=False):
    """test 侧正常/缺陷切分（BTAD: ok/ko 映射）。"""
    sdir = os.path.join(cat_dir, "test")
    good = []
    defect = []
    if not os.path.isdir(sdir):
        return good, defect
    for sub in sorted(os.listdir(sdir)):
        full = os.path.join(sdir, sub)
        if not os.path.isdir(full):
            continue
        name = _BTAD_MAP.get(sub, sub) if btad else sub
        if name == "good":
            good += _list_images(full)
        else:
            defect += _list_images(full)
    return good, defect


def _train_good_dir(cat_dir, btad=False):
    sub = "ok" if btad else "good"
    return os.path.join(cat_dir, "train", sub)


def load_category(root, category, n_init_normal=100, n_init_defect=30,
                  seed=42, n_eval_good=None, btad=False, max_test=150):
    """n_eval_good: test/good 评测抽样上限（大数据集"只用少量"，None=全量）。
    max_test（2026-08-16 用户规则）：test 总样本上限——超过则随机抽 max_test 张，
    调优迭代不跑全量（如 gyudet 1113 张）；最终验证用 --max-test 0 关闭抽样。"""
    rng = np.random.default_rng(seed)
    cat_dir = os.path.join(root, category)
    tr_good = _list_images(_train_good_dir(cat_dir, btad))
    te_good, te_defect = _split_good_defect(cat_dir, btad)

    init_normal = sorted(rng.choice(
        tr_good, min(n_init_normal, len(tr_good)), replace=False).tolist())
    n_init = min(n_init_defect, max(1, len(te_defect) // 2))
    sel = rng.choice(len(te_defect), n_init, replace=False) if n_init else []
    init_defect = sorted(te_defect[i] for i in sel)
    eval_defect = sorted(p for i, p in enumerate(te_defect) if i not in set(sel))
    if n_eval_good and len(te_good) > n_eval_good:
        te_good = sorted(rng.choice(te_good, n_eval_good, replace=False).tolist())

    test = [(p, 0) for p in te_good] + [(p, 1) for p in eval_defect]
    rng.shuffle(test)
    if max_test and len(test) > max_test:
        test = test[:max_test]
        rng.shuffle(test)
    return Bundle(name=category, init_normal=init_normal,
                  init_defect=init_defect, val=[], test=test).as_dict()


def load_btad(root, category, **kw):
    return load_category(root, category, btad=True, **kw)


MVTEC_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "transistor", "toothbrush", "wood", "zipper",
]
MPDD_CATEGORIES = [
    "bracket_black", "bracket_brown", "bracket_white",
    "connector", "metal_plate", "tubes",
]
BTAD_CATEGORIES = ["01", "02", "03"]
