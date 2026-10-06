"""数据隔离红线（红线 1/2 的代码级强制）

任何数据集适配器必须经 guard_bundle 校验后才能进入 fit/eval 管线：
- fit 侧只允许 train/good + 协议内的缺陷样本；
- test/good、配对参考、文件名元信息一律不得作为训练输入。
"""
import hashlib
import os

FORBIDDEN_FIT_KEYS = {"pairs", "test_good", "ref_pool"}


def _domain(path):
    """从路径推断数据域：train / val / test / unknown"""
    parts = os.path.normpath(str(path)).lower().split(os.sep)
    for d in ("train", "val", "valid", "test"):
        if d in parts:
            return "val" if d == "valid" else d
    return "unknown"


class IsolationViolation(RuntimeError):
    """数据隔离红线违规（显式异常，不受 python -O 影响——原 assert 在
    优化模式下被整体剥离，红线形同虚设，评审问题 A12）。"""


def guard_bundle(bundle, protocol):
    """对适配器返回的数据包做红线校验，违规直接抛 IsolationViolation。

    bundle 约定字段：
      init_normal: list[path]   训练正常（必须全部来自 train 域）
      init_defect: list[path]   训练缺陷（数量 ≤ protocol.n_init_defect）
      val:  list[(path,label)]  验证（调阈值/选型，不得进 fit）
      test: list[(path,label)]  最终评测（不得进 fit）
    """
    for key in FORBIDDEN_FIT_KEYS:
        if key in bundle:
            raise IsolationViolation(f"红线违规：bundle 含有禁止字段 {key}")

    bad = [p for p in bundle["init_normal"] if _domain(p) != "train"]
    if bad:
        raise IsolationViolation(
            f"红线违规：init_normal 含非 train 域样本，如 {bad[:3]}")

    n_max = protocol.get("n_init_defect", 30)
    if len(bundle["init_defect"]) > n_max:
        raise IsolationViolation(
            f"红线违规：init_defect {len(bundle['init_defect'])} "
            f"> 协议上限 {n_max}")

    groups = {"init_normal": bundle["init_normal"],
              "init_defect": bundle["init_defect"],
              "val": [p for p, _ in bundle["val"]],
              "test": [p for p, _ in bundle["test"]]}
    seen_paths = {}
    seen_contents = {}
    for split, paths in groups.items():
        for path in paths:
            key = os.path.normcase(os.path.realpath(os.path.abspath(path)))
            previous = seen_paths.get(key)
            if previous is not None and previous != split:
                raise IsolationViolation(
                    f"红线违规：{previous} 与 {split} 共用图片 {path}")
            seen_paths[key] = split
            digest = hashlib.sha256()
            with open(path, "rb") as image:
                for chunk in iter(lambda: image.read(1 << 20), b""):
                    digest.update(chunk)
            previous = seen_contents.get(digest.digest())
            if previous is not None and previous != split:
                raise IsolationViolation(
                    f"红线违规：{previous} 与 {split} 存在重复内容，如 {path}")
            seen_contents[digest.digest()] = split
    return True
