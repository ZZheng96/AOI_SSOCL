"""依赖锁一致性检查（第三轮决策项：锁不一致时告警或阻断）。

对比 requirements-lock.txt 的 pkg==ver 与当前环境已安装版本；
比较时忽略本地版本后缀（+cu126 等，CPU/GPU 轮子差异不算不一致）。
"""
from __future__ import annotations

from importlib import metadata
from pathlib import Path
from typing import Any, Dict

from .config import AOI_ROOT

LOCK_PATH = AOI_ROOT / "requirements-lock.txt"


def _base(ver: str) -> str:
    return ver.split("+", 1)[0].strip()


def parse_lock(path: Path) -> Dict[str, str]:
    pins: Dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if "==" in line:
            name, ver = line.split("==", 1)
            pins[name.strip()] = ver.strip()
    return pins


def lock_status(path: Path | None = None) -> Dict[str, Any]:
    """{ok, lock_path, mismatches:[{package,locked,installed}], missing:[pkg]}。
    锁文件不存在时 ok=True 且 lock_found=False（无锁可比）。"""
    p = Path(path or LOCK_PATH)
    if not p.is_file():
        return {"ok": True, "lock_found": False, "lock_path": str(p),
                "mismatches": [], "missing": []}
    mismatches, missing = [], []
    for name, ver in parse_lock(p).items():
        try:
            inst = metadata.version(name)
        except metadata.PackageNotFoundError:
            missing.append(name)
            continue
        if _base(inst) != _base(ver):
            mismatches.append({"package": name, "locked": ver, "installed": inst})
    return {"ok": not mismatches and not missing, "lock_found": True,
            "lock_path": str(p), "mismatches": mismatches, "missing": missing}


def dependency_guard(policy: str, path: Path | None = None) -> Dict[str, Any]:
    """policy=block 且不一致 → 抛 CandidateReleaseError(DEPENDENCY_LOCK_MISMATCH)；
    warn 仅返回状态（调用方附带告警）。"""
    st = lock_status(path)
    if policy == "block" and not st["ok"]:
        from ..self_learning.candidate_release import CandidateReleaseError
        viol = [f"{m['package']} 锁定 {m['locked']} 实装 {m['installed']}"
                for m in st["mismatches"]] + [f"{n} 未安装" for n in st["missing"]]
        raise CandidateReleaseError(
            "依赖与 requirements-lock.txt 不一致，按治理决策（block）阻断发布",
            "DEPENDENCY_LOCK_MISMATCH", violations=viol)
    return st
