"""GYU-DET 适配器（诚实协议，YOLO 标注格式）

目录：{train,valid,test}/{images,labels}，YOLO txt 标注。
标签判定：label 文件存在且含 ≥1 非空行 → 缺陷；缺失或空 → 正常
（demo4 教训：标签检验必须看内容，不能只看文件有无）。
协议（红线 1/3）：
  init_normal = train 域正常图抽 100；init_defect = train 域缺陷图抽 ≤30；
  valid 域做验证/调阈值；test 域做最终评测。test/good 一律不进 fit。
"""
import os
import numpy as np
from .base import Bundle

IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp")


def _load_split(root, split):
    img_dir = os.path.join(root, split, "images")
    lbl_dir = os.path.join(root, split, "labels")
    normal, defect = [], []
    for f in sorted(os.listdir(img_dir)):
        if not f.lower().endswith(IMG_EXT):
            continue
        lbl = os.path.join(lbl_dir, os.path.splitext(f)[0] + ".txt")
        is_defect = False
        if os.path.exists(lbl):
            with open(lbl, encoding="utf-8", errors="ignore") as fh:
                is_defect = any(line.strip() for line in fh)
        (defect if is_defect else normal).append(os.path.join(img_dir, f))
    return normal, defect


def load_gyudet(root, n_init_normal=100, n_init_defect=30, seed=42, max_test=150):
    """max_test（2026-08-16 用户规则）：val/test 各随机抽 max_test 张（全量 1113+ 太耗时），
    调优迭代用；最终验证用 --max-test 0 关闭抽样。
    U43：gyudet 缺陷率 95%，纯随机抽样会让 eval 正常样本太少（AUROC 噪声 0.26 级）——
    抽样改为分层：正常样本全保留，缺陷随机补足到 max_test。"""
    rng = np.random.default_rng(seed)
    tr_n, tr_d = _load_split(root, "train")
    va_n, va_d = _load_split(root, "valid")
    te_n, te_d = _load_split(root, "test")

    init_normal = sorted(rng.choice(tr_n, min(n_init_normal, len(tr_n)), replace=False).tolist())
    init_defect = sorted(rng.choice(tr_d, min(n_init_defect, len(tr_d)), replace=False).tolist()) if tr_d else []

    def _cap(good, bad, cap):
        """分层抽样：good 全保留（正常样本稀有，是 AUROC 排序的锚），bad 补足到 cap。"""
        if cap and len(good) + len(bad) > cap:
            n_good = min(len(good), max(cap // 4, 1))
            n_bad = cap - n_good
            good = sorted(rng.choice(good, n_good, replace=False).tolist())
            bad = sorted(rng.choice(bad, min(n_bad, len(bad)), replace=False).tolist())
        return good, bad

    va_n, va_d = _cap(va_n, va_d, max_test)
    te_n, te_d = _cap(te_n, te_d, max_test)
    val = [(p, 0) for p in va_n] + [(p, 1) for p in va_d]
    test = [(p, 0) for p in te_n] + [(p, 1) for p in te_d]
    rng.shuffle(val)
    rng.shuffle(test)
    return Bundle(name="gyudet", init_normal=init_normal,
                  init_defect=init_defect, val=val, test=test).as_dict()
