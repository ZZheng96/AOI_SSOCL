"""CLI 入口：特征贡献评估（基于当前特征集配置 feature_set.yaml）

用法:
  # 合成数据冒烟（无真实数据时验证管线）
  python run_eval.py --dataset synthetic

  # 真实数据集
  python run_eval.py --dataset mvtec --root e:/CPIPC/CGAIC/data_origin/mvtec --category bottle
  python run_eval.py --dataset datalocal --root <data_local根> --category solder_smt
  python run_eval.py --dataset mvtec --root ... --category all   # 全品类

特征集合管理见 manage.py（启停/改参/增删）；改完配置重跑本脚本即得新贡献报告。
"""
import argparse
import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

# 以包方式导入（模块内使用相对导入），同时把 AOI_Core 加入 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from AOI_feature import ensure_core_importable                     # noqa: E402
from AOI_feature.feature_set import FeatureSet                      # noqa: E402
from AOI_feature.contribution import evaluate_category              # noqa: E402
from AOI_feature.report import save_category_report, save_summary   # noqa: E402

ensure_core_importable()
from algo.data import mvtec_like, datalocal           # noqa: E402


def load_image_unicode(path):
    """中文路径兼容读图（cv2.imread 不支持非 ASCII 路径），失败返回 None"""
    try:
        data = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return None if img is None else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception:
        return None


def load_bundle(dataset, root, category, args):
    kw = dict(n_init_normal=args.n_init_normal, n_init_defect=args.n_init_defect,
              seed=args.seed, max_test=args.max_test)
    if dataset in ("mvtec", "mpdd"):
        return mvtec_like.load_category(root, category, **kw)
    if dataset == "btad":
        return mvtec_like.load_btad(root, category, **kw)
    if dataset == "datalocal":
        return datalocal.load_category(root, category, **kw)
    raise ValueError(dataset)


def run_evaluation(fs: FeatureSet, dataset: str, root: str | None,
                   category: str | None, n_init_normal=60, n_init_defect=30,
                   max_test=150, k=3, n_perm=5, seed=42, out=None,
                   progress=None) -> tuple[list, list]:
    """执行贡献评估并落盘报告，返回 (results, groups_meta)。供 CLI 与 GUI 复用。

    progress: 可选回调 fn(msg: str)。
    """
    groups_meta = fs.probe()

    def log(msg):
        if progress:
            progress(msg)
        else:
            print(msg)

    log(f"[ok] 特征集 {fs.config.get('name')}：{len(groups_meta)} 组启用，"
        f"合计 {sum(m['dim'] for m in groups_meta)} 维")

    def run(bundle, name):
        return evaluate_category(fs, bundle, load_image_unicode, groups_meta,
                                 k=k, n_perm=n_perm, seed=seed, name=name)

    results = []
    if dataset == "synthetic":
        from AOI_feature.synthetic import make_synthetic
        sroot = tempfile.mkdtemp(prefix="aoi_feat_synth_")
        make_synthetic(sroot, "synthetic", seed=seed)
        bundle = datalocal.load_category(sroot, "synthetic", n_init_normal,
                                         n_init_defect, seed, max_test=max_test)
        res = run(bundle, "synthetic")
        results.append(res)
        if out:
            save_category_report(res, out)
        log(f"[ok] synthetic 全向量 AUROC={res['full_auroc']:.4f} "
            f"(ref={res['n_ref']}, test={res['n_test']})")
    else:
        if not root:
            raise ValueError("dataset 非 synthetic 时必须给 root")
        if dataset == "mvtec":
            cats = mvtec_like.MVTEC_CATEGORIES if category in (None, "all") \
                else [category]
        elif dataset == "mpdd":
            cats = mvtec_like.MPDD_CATEGORIES if category in (None, "all") \
                else [category]
        elif dataset == "btad":
            cats = mvtec_like.BTAD_CATEGORIES if category in (None, "all") \
                else [category]
        else:
            cats = [category] if category else datalocal.CATEGORIES
        for cat in cats:
            bundle = load_bundle(dataset, root, cat,
                                 argparse.Namespace(
                                     n_init_normal=n_init_normal,
                                     n_init_defect=n_init_defect,
                                     seed=seed, max_test=max_test))
            if not bundle["init_normal"] or not bundle["test"]:
                log(f"[skip] {cat}: 数据为空（root={root}）")
                continue
            res = run(bundle, cat)
            results.append(res)
            if out:
                save_category_report(res, out)
            log(f"[ok] {cat}: 全向量 AUROC={res['full_auroc']:.4f}")

    if results and out:
        save_summary(results, out, groups_meta)
        log(f"[done] 报告已落盘: {out}")
    return results, groups_meta


def main():
    ap = argparse.ArgumentParser(description="AOI 特征组贡献评估（特征集配置驱动）")
    ap.add_argument("--dataset", default="synthetic",
                    choices=["synthetic", "mvtec", "mpdd", "btad", "datalocal"])
    ap.add_argument("--root", default=None, help="数据集根目录（synthetic 时忽略）")
    ap.add_argument("--category", default=None, help="品类；缺省 synthetic/all 逻辑")
    ap.add_argument("--config", default=None,
                    help="特征集配置 YAML（缺省用 AOI_feature/feature_set.yaml）")
    ap.add_argument("--n-init-normal", type=int, default=60)
    ap.add_argument("--n-init-defect", type=int, default=30)
    ap.add_argument("--max-test", type=int, default=150)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--n-perm", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "outputs"))
    args = ap.parse_args()

    fs = FeatureSet.load(args.config) if args.config else FeatureSet.load()
    run_evaluation(fs, args.dataset, args.root, args.category,
                   n_init_normal=args.n_init_normal,
                   n_init_defect=args.n_init_defect, max_test=args.max_test,
                   k=args.k, n_perm=args.n_perm, seed=args.seed, out=args.out)


if __name__ == "__main__":
    main()
