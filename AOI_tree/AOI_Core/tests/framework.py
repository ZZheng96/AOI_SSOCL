"""测试公共框架：结果记录、HTTP 辅助、断言。

每个测试模块实现 run(ctx) -> None，通过 ctx.check() 记录断言结果，
测试结果落盘 tests/results/{test_id}.json，汇总于 results/summary.json。
"""
from __future__ import annotations

import json
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

TEST_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = TEST_ROOT / "results"
FIGURES_DIR = TEST_ROOT / "report" / "figures"
REPORT_DIR = TEST_ROOT / "report"
BASE_URL = "http://127.0.0.1:8018/api"

for d in (RESULTS_DIR, FIGURES_DIR, REPORT_DIR):
    d.mkdir(parents=True, exist_ok=True)


class TestFail(AssertionError):
    """测试断言失败。"""


class Ctx:
    """测试上下文：HTTP 客户端 + 断言与结果记录。"""

    def __init__(self, test_id: str, name: str):
        self.test_id = test_id
        self.name = name
        self.checks: List[Dict[str, Any]] = []
        self.artifacts: List[str] = []      # 供报告引用的产物（图片/路径）
        self.metrics: Dict[str, Any] = {}   # 供报告图表引用的指标
        self.t0 = time.perf_counter()

    # ── 断言 ────────────────────────────────────────────────
    def check(self, desc: str, ok: bool, detail: str = "") -> bool:
        self.checks.append({"desc": desc, "ok": bool(ok), "detail": str(detail)[:500]})
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}] {desc}" + (f" | {detail}" if detail and not ok else ""))
        return ok

    def require(self, desc: str, ok: bool, detail: str = "") -> None:
        if not self.check(desc, ok, detail):
            raise TestFail(f"{desc}: {detail}")

    # ── HTTP ────────────────────────────────────────────────
    def get(self, path: str, **params) -> Any:
        r = requests.get(f"{BASE_URL}{path}", params=params, timeout=60)
        r.raise_for_status()
        return r.json()

    def post(self, path: str, body: Optional[dict] = None, timeout: int = 300,
             files=None, data=None) -> Any:
        r = requests.post(f"{BASE_URL}{path}", json=body, files=files,
                          data=data, timeout=timeout)
        r.raise_for_status()
        return r.json()

    def delete(self, path: str) -> Any:
        r = requests.delete(f"{BASE_URL}{path}", timeout=60)
        r.raise_for_status()
        return r.json() if r.content else {}

    def wait_task(self, task_id: int, timeout: int = 600,
                  interval: float = 2.0) -> Dict:
        """轮询后台任务直至 done/failed。"""
        deadline = time.time() + timeout
        while time.time() < deadline:
            t = self.get(f"/tasks/{task_id}")
            if t["status"] in ("done", "failed"):
                return t
            time.sleep(interval)
        raise TestFail(f"任务 {task_id} 超时（{timeout}s）")

    # ── 结果落盘 ────────────────────────────────────────────
    def finish(self, error: Optional[Exception] = None) -> Dict:
        n_fail = sum(1 for c in self.checks if not c["ok"])
        result = {
            "test_id": self.test_id,
            "name": self.name,
            "status": "error" if error and not self.checks else (
                "failed" if (error or n_fail) else "passed"),
            "n_checks": len(self.checks),
            "n_failed": n_fail,
            "duration_s": round(time.perf_counter() - self.t0, 2),
            "checks": self.checks,
            "metrics": self.metrics,
            "artifacts": self.artifacts,
            "error": (f"{error}\n{traceback.format_exc()}"[:2000]
                      if error else None),
        }
        with open(RESULTS_DIR / f"{self.test_id}.json", "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        return result


def run_test(test_id: str, name: str, fn) -> Dict:
    """执行单个测试模块并记录结果。"""
    print(f"\n{'='*60}\n[{test_id}] {name}\n{'='*60}")
    ctx = Ctx(test_id, name)
    err: Optional[Exception] = None
    try:
        fn(ctx)
    except Exception as e:  # noqa: BLE001
        err = e
        print(f"  [ERROR] {e}")
    return ctx.finish(err)


def collect_summary() -> Dict:
    """汇总 results/*.json → summary.json。"""
    results = []
    for p in sorted(RESULTS_DIR.glob("t*.json")):
        with open(p, encoding="utf-8") as f:
            results.append(json.load(f))
    summary = {
        "total": len(results),
        "passed": sum(1 for r in results if r["status"] == "passed"),
        "failed": sum(1 for r in results if r["status"] != "passed"),
        "total_checks": sum(r["n_checks"] for r in results),
        "failed_checks": sum(r["n_failed"] for r in results),
        "duration_s": round(sum(r["duration_s"] for r in results), 1),
        "results": [{k: r[k] for k in
                     ("test_id", "name", "status", "n_checks", "n_failed",
                      "duration_s")} for r in results],
    }
    with open(RESULTS_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary
