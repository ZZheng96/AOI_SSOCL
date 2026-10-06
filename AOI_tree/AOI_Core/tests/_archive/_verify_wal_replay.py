# -*- coding: utf-8 -*-
"""高#1 WAL 联跑验证：反馈即学 → 模拟崩溃（不 consolidate）→ 重启重放。

验证点：
A. submit_feedback 学习成功后写 learning_journal.jsonl（base=激活版本）；
B. 不 consolidate 直接丢弃引擎（模拟崩溃），新引擎装载快照时 _replay_journal
   重放 base==当前版本 的条目，内存学习态恢复（defect_bank.samples 一致）；
C. 即学超时保护（评审#7）：timeout_ms 很小 → 返回 async=True，学习转后台完成。

不落盘新快照（不 consolidate），结束后删除 journal 还原现场。
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

STORAGE = r"D:\CGAIC\AOI_sys\storage\demo5"
BASE_CFG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "configs", "demo5_fast.yaml")
CAT = "bracket_white"
IMG = r"D:\CGAIC\AOI_sys\storage\archive\bracket_white\20260830\det669_062.png"
JOURNAL = os.path.join(STORAGE, "snapshots", CAT, "learning_journal.jsonl")


def journal_lines() -> int:
    if not os.path.exists(JOURNAL):
        return 0
    with open(JOURNAL, "r", encoding="utf-8") as f:
        return sum(1 for ln in f if ln.strip())


def main() -> None:
    assert os.path.exists(IMG), f"缺图: {IMG}"
    # 幂等：清掉历史中断残留的 journal，保证基线干净
    if os.path.exists(JOURNAL):
        os.remove(JOURNAL)
    from backend.engine import Demo5Engine

    # ---- A. 加载 v3 + 基线 ----
    e1 = Demo5Engine(STORAGE, BASE_CFG)
    e1._ensure_loaded(CAT)
    ins0 = e1.learning_insight(CAT)
    n0 = ins0["defect_bank"]["samples"]
    print(f"[A] 基线 journal={journal_lines()} 行, defect_bank={n0}", flush=True)

    # ---- A. 反馈即学（verdict=wrong, label=1 缺陷反馈）----
    r = e1.submit_feedback(CAT, IMG, verdict="wrong", label=1)
    print(f"[A] submit_feedback -> {r}", flush=True)
    assert "async" not in r, f"同步学习意外转后台: {r}"
    assert journal_lines() == 1, f"WAL 未写/多写: {journal_lines()}"
    with open(JOURNAL, "r", encoding="utf-8") as f:
        rec = __import__("json").loads(f.readline())
    assert rec["base"] == 3 and rec["path"] == IMG and rec["verdict"] == "wrong"
    print(f"[A] WAL 已写 1 条: base={rec['base']} verdict={rec['verdict']} ✅", flush=True)

    ins1 = e1.learning_insight(CAT)
    n1 = ins1["defect_bank"]["samples"]
    print(f"[A] 学习后 defect_bank={n1}（基线 {n0}）", flush=True)

    # ---- B. 模拟崩溃：不 consolidate，丢弃引擎，重启重放 ----
    del e1
    e2 = Demo5Engine(STORAGE, BASE_CFG)
    e2._ensure_loaded(CAT)   # 装载 v3 时 _replay_journal 应打印重放 1 条
    ins2 = e2.learning_insight(CAT)
    n2 = ins2["defect_bank"]["samples"]
    print(f"[B] 重放后 defect_bank={n2}", flush=True)
    assert n2 == n1, f"重放未恢复学习态: n1={n1} n2={n2}"
    print("[B] 崩溃→重启→重放：未巩固学习已恢复 ✅", flush=True)

    # ---- C. 即学超时保护：timeout_ms=1 强制转后台 ----
    r3 = e2.submit_feedback(CAT, IMG, verdict="wrong", label=1, timeout_ms=1)
    print(f"[C] timeout_ms=1 -> {r3}", flush=True)
    assert r3.get("async") is True, f"未转后台: {r3}"
    time.sleep(3)  # 等后台线程完成学习并写 WAL
    assert journal_lines() == 2, f"后台学习未写 WAL: {journal_lines()}"
    print(f"[C] 即学超时转后台完成，WAL 现 {journal_lines()} 条 ✅", flush=True)

    # ---- 还原现场：删除 journal（学习未 consolidate，重放语义已验毕）----
    if os.path.exists(JOURNAL):
        os.remove(JOURNAL)
    print(f"[cleanup] 已删除 {JOURNAL}", flush=True)
    print("✅ WAL 重放联跑验证全部通过")


if __name__ == "__main__":
    main()
