"""传统算法量化评测：对带 GT 标签的测试图跑各传统算法，输出检出率 + 耗时。

目标：把"单张 NG 检出"的轶事验证升级为可复现的量化基线（混淆矩阵/检出率）。
当前 PCB_Ins_v2 自包含的带标签样本有限（见 CASES 注释），本脚本是量化框架起点，
扩充数据后往 CASES 加条目即可。

GT 约定（不臆造）：
- 文件名 `*_OK.png` → 正常（GT=OK）
- 缺陷样本目录（如 excess_pair / 金手指 第一对）→ 缺陷（GT=NG）

用法：
    python tests/eval_traditional.py
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.utils.cv_io import imread_unicode  # noqa: E402


@dataclass
class Case:
    group: str
    defect: str
    gt_ng: bool
    std: str | None
    test: str


@dataclass
class Verdict:
    group: str
    defect: str
    gt_ng: bool
    pred_ng: bool
    detail: str
    ms: float


def pad_json_to_frames(pad_json: dict) -> list[list[int]]:
    frames: list[list[int]] = []
    for sh in pad_json.get("shapes", []):
        pts = sh.get("points") or []
        if sh.get("shape_type") == "rectangle" and len(pts) == 2:
            (x0, y0), (x1, y1) = pts[0], pts[1]
            x, y = min(x0, x1), min(y0, y1)
            w, h = abs(x1 - x0), abs(y1 - y0)
            frames.append([int(x), int(y), int(w), int(h)])
    return frames


def run_smt(c: Case) -> Verdict:
    """SMT 焊点：直接调算法（等价 _verify_smt_alg.py），避免走模板/DB 副作用。"""
    smt_src = ROOT / "alg_repo" / "smt" / "src"
    smt_contract = ROOT / "alg_repo" / "smt" / "contract_reference"
    for p in (smt_src, smt_contract):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    from algorithms.solder_smt.composite_smt_all.solder_smt_all_alg import SolderSmtAllAlg

    pair = Path(c.test).parent
    pad_json = json.loads((pair / "pad.json").read_text(encoding="utf-8"))
    params = json.loads((pair / "params.json").read_text(encoding="utf-8"))
    alg = SolderSmtAllAlg()
    test = imread_unicode(c.test)
    std = imread_unicode(c.std)
    cfg = alg.default_config()
    cfg.update(params)
    cfg["pad_frames"] = pad_json_to_frames(pad_json)
    cfg.update({"detect_excess": True, "detect_insufficient": True,
                "detect_bridge": True, "detect_cold_solder": True,
                "enable_visualization": False})
    t0 = time.perf_counter()
    r = alg.run(test, cfg, original_template_image=std)
    ms = (time.perf_counter() - t0) * 1000
    labels = [str(p.label) for p in r.parts]
    pred_ng = bool(labels)
    return Verdict("smt", c.defect, c.gt_ng, pred_ng,
                   f"code={r.code} labels={labels or '-'} msg={r.message[:60]}", ms)


def run_gold(c: Case) -> Verdict:
    from app.detect.adapters.gold_finger_adapter import GoldFingerAdapter
    from app.detect.types import DetectRequest

    gf = GoldFingerAdapter()
    std = imread_unicode(c.std)
    test = imread_unicode(c.test)
    req = DetectRequest(image_std=std, image_std_raw=std,
                        image_test=test, image_test_raw=test,
                        template_name="eval_gold")
    t0 = time.perf_counter()
    r = gf.run("board_板面金手指", req, {})
    ms = (time.perf_counter() - t0) * 1000
    return Verdict("gold", c.defect, c.gt_ng, not r.ok,
                   f"缺陷数={len(r.boxes)} {r.message[:70]}", ms)


def run_wrinkle(c: Case) -> Verdict:
    from app.detect.adapters.wrinkle_adapter import WrinkleAdapter
    from app.detect.types import DetectRequest

    wr = WrinkleAdapter()
    test = imread_unicode(c.test)
    req = DetectRequest(image_test=test, image_test_raw=test, template_name="eval_wrinkle")
    t0 = time.perf_counter()
    r = wr.run("wrinkle_起皱", req, {})
    ms = (time.perf_counter() - t0) * 1000
    return Verdict("wrinkle", c.defect, c.gt_ng, not r.ok,
                   f"缺陷数={len(r.boxes)} {r.message[:70]}", ms)


RUNNERS = {"smt": run_smt, "gold": run_gold, "wrinkle": run_wrinkle}


def build_cases() -> list[Case]:
    smt = ROOT / "alg_repo" / "smt" / "test_images"
    gold = ROOT / "alg_repo" / "gold_finger" / "test_images" / "pairs"
    cases: list[Case] = []

    # SMT 多锡：excess_pair 含 *_OK.png（正常）+ 无后缀（多锡缺陷）
    ep = smt / "excess_pair" / "pair4"
    if (ep / "PixPin_2026-07-01_11-21-15.png").exists():
        ok = ep / "PixPin_2026-07-01_11-21-15_OK.png"
        ng = ep / "PixPin_2026-07-01_11-21-15.png"
        cases.append(Case("smt", "多锡", True, str(ok), str(ng)))
        cases.append(Case("smt", "多锡(OK)", False, str(ok), str(ok)))

    # 金手指 第一对：已知 NG（test1 检出 15 缺陷）
    g1 = gold / "第一对"
    if (g1 / "test1.png").exists() and (g1 / "templ1.png").exists():
        cases.append(Case("gold", "金手指", True, str(g1 / "templ1.png"), str(g1 / "test1.png")))

    # 金手指 第二/三对：按"缺陷对"约定视作 NG（未逐张人工复核，仅登记为待核）
    for name in ("第二对", "第三对"):
        d = gold / name
        tpls = sorted(d.glob("templ*")) if d.is_dir() else []
        tests = sorted(d.glob("test*")) if d.is_dir() else []
        if tpls and tests:
            cases.append(Case("gold", f"金手指({name})", True, str(tpls[0]), str(tests[0])))

    return cases


def main() -> None:
    cases = build_cases()
    if not cases:
        print("未找到评测用例（检查 alg_repo/smt、alg_repo/gold_finger 数据）")
        return

    verdicts: list[Verdict] = []
    print(f"{'组':<8}{'缺陷/类别':<14}{'GT':<4}{'预测':<4}{'耗时(ms)':<10}{'详情'}")
    for c in cases:
        runner = RUNNERS.get(c.group)
        if runner is None:
            continue
        try:
            v = runner(c)
        except Exception as exc:  # noqa: BLE001
            v = Verdict(c.group, c.defect, c.gt_ng, False, f"[ERROR] {exc}", 0.0)
        verdicts.append(v)
        print(f"{v.group:<8}{v.defect:<14}{'NG' if v.gt_ng else 'OK':<4}"
              f"{'NG' if v.pred_ng else 'OK':<4}{v.ms:<10.1f}{v.detail}")

    # 汇总：检出率（召回）+ 误报
    ng_cases = [v for v in verdicts if v.gt_ng]
    ok_cases = [v for v in verdicts if not v.gt_ng]
    tp = sum(1 for v in ng_cases if v.pred_ng)
    fn = len(ng_cases) - tp
    fp = sum(1 for v in ok_cases if v.pred_ng)
    tn = len(ok_cases) - fp
    print("\n=== 汇总（当前样本量小，仅作基线，勿当统计结论）===")
    print(f"缺陷样本: {len(ng_cases)}  检出 {tp}  漏检 {fn}  召回率={tp / max(1, len(ng_cases)):.3f}")
    print(f"正常样本: {len(ok_cases)}  误报 {fp}  通过 {tn}  误报率={fp / max(1, len(ok_cases)):.3f}")
    lat = [v.ms for v in verdicts if v.ms > 0]
    if lat:
        print(f"平均耗时={sum(lat) / max(1, len(lat)):.1f}ms  最大={max(lat):.1f}ms")

    # 落盘原始记录（随包证据链：docs 引用此 JSON）
    import datetime
    import platform
    out = {
        "bench": "eval_traditional",
        "ts": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "env": {"python": platform.python_version(), "platform": platform.platform()},
        "caveat": "自包含带 GT 标签样本量小，仅作量化基线，勿当统计结论；GT 约定 *_OK.png=正常、缺陷样本目录=NG",
        "cases": [vars(v) for v in verdicts],
        "summary": {
            "ng_total": len(ng_cases), "tp": tp, "fn": fn,
            "recall": round(tp / max(1, len(ng_cases)), 4),
            "ok_total": len(ok_cases), "fp": fp, "tn": tn,
            "fpr": round(fp / max(1, len(ok_cases)), 4),
            "ms_mean": round(sum(lat) / max(1, len(lat)), 1) if lat else None,
            "ms_max": round(max(lat), 1) if lat else None,
        },
    }
    res_dir = ROOT / "tests" / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    out_path = res_dir / "eval_traditional_latest.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n原始记录已落盘: {out_path}")


if __name__ == "__main__":
    main()
