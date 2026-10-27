"""数据隔离红线（红线 1/2 的代码级强制）

任何数据集适配器必须经 guard_bundle 校验后才能进入 fit/eval 管线：
- fit 侧只允许 train/good + 协议内的缺陷样本；
- test/good、配对参考、文件名元信息一律不得作为训练输入。
"""
import os

FORBIDDEN_FIT_KEYS = {"pairs", "test_good", "ref_pool"}


def _domain(path):
    """从路径推断数据域：train / val / test / unknown"""
    parts = os.path.normpath(str(path)).lower().split(os.sep)
    for d in ("train", "val", "valid", "test"):
        if d in parts:
            return "val" if d == "valid" else d
    return "unknown"


def guard_bundle(bundle, protocol):
    """对适配器返回的数据包做红线断言，违规直接抛错。

    bundle 约定字段：
      init_normal: list[path]   训练正常（必须全部来自 train 域）
      init_defect: list[path]   训练缺陷（数量 ≤ protocol.n_init_defect）
      val:  list[(path,label)]  验证（调阈值/选型，不得进 fit）
      test: list[(path,label)]  最终评测（不得进 fit）
    """
    for key in FORBIDDEN_FIT_KEYS:
        assert key not in bundle, f"红线违规：bundle 含有禁止字段 {key}"

    bad = [p for p in bundle["init_normal"] if _domain(p) != "train"]
    assert not bad, f"红线违规：init_normal 含非 train 域样本，如 {bad[:3]}"

    n_max = protocol.get("n_init_defect", 30)
    assert len(bundle["init_defect"]) <= n_max, \
        f"红线违规：init_defect {len(bundle['init_defect'])} > 协议上限 {n_max}"

    fit_set = set(bundle["init_normal"]) | set(bundle["init_defect"])
    for split in ("val", "test"):
        leak = fit_set & {p for p, _ in bundle[split]}
        assert not leak, f"红线违规：fit 集与 {split} 集有交集，如 {list(leak)[:3]}"
    return True
