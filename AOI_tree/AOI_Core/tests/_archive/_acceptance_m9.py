"""M9 工业级可用性验收 + M9b 多数据集扩展：用 demo5 已测过的数据集走端到端工业实操流程。

对照基线（demo5 项目记忆.md，sanity 加权 + shead；data_local 为 equal 权重口径）：
  mvtec: bottle 0.9985 / cable 0.9619 / tile 0.9989 / screw 0.8715（学习后 0.9286）
  btad : btad_01 0.9524 / btad_02 0.8600 / btad_03 0.9980
  mpdd : mpdd_tubes 0.72（等权 fused 保守线）/ mpdd_bracket_white 0.94
  data_local: solder_smt 0.683（U16 裁剪弱槽后）

每品类验收流程（工业实操全链路）：
  1. 数据登记：train 正常 100 张 + test 各缺陷目录各 2 张（cap 5）入 Image 表
     （split=train / train_anomaly）；test 正常 15 张 + test 缺陷 15 张作评估集
     （不登记进库，仅本地留路径，防泄漏；样本不足时取实际可用数量）。
  2. 部署效率：POST /api/models/prepare（L1a/fast，force）→ 计时（红线 <120s）。
  3. 检测质量：评估集逐张 POST /api/detect/image → AUROC（对照基线-0.05）
     + 延迟均值/峰值（峰值 <1000ms 红线）。
  4. 反馈学习：挑系统判错的（good 但 score>0.5 / defect 但 score<0.5，cap 5）
     逐张 POST /api/feedback → 断言响应含 pre。
  5. 翻案曲线：GET /api/learning/flip_curve/{category} → total/flipped/flip_rate。
  6. 版本固化：POST /api/self_learning/update → v2 → GET /api/categories 确认。
  7. 数据治理：GET /api/datasets?category= 有批次；GET /api/images/tree 四维 datasets。

gyudet 跨域学习增益专项（--gyudet-learn，赛题"反馈驱动持续学习"杀手锏验证）：
  demo5 已知 gyudet 跨域静态失效（U38/U41：single 0.49 / tiles9 0.52），
  在线学习是唯一出路（U46/U51：53 反馈后 0.6599→0.7665，峰值 0.8180）。
  流程：train/normal 100 + train/defect 5 锚定 prepare（L1a）→ valid 初始 AUROC
  → 错检逐张反馈（cap 20，跨域需要更多反馈）→ 每 5 条在 30 张未反馈 holdout 上
  重估 AUROC → 学习曲线 k=0/5/10/15/20。验收：k=20 增益 ≥+0.05 且延迟峰值 <1s；
  静态 AUROC 不参与 PASS 判定（预期低）。

用法：
  python _acceptance_m9.py                          # 默认 mvtec 四品类（M9 基线）
  python _acceptance_m9.py bottle cable             # mvtec 子集
  python _acceptance_m9.py --datasets btad,mpdd,data_local   # 扩展数据集
  python _acceptance_m9.py --datasets btad btad_01           # 单品类调试
  python _acceptance_m9.py --datasets all                    # 全部静态数据集
  python _acceptance_m9.py --gyudet-learn                    # 仅 gyudet 专项
  python _acceptance_m9.py --datasets all --gyudet-learn     # 全量

按工作区规则随机取少量图片（seed 固定可复现），不跑全量数据集。
隔离环境：临时 sqlite + 临时 storage，不污染生产库（模式同 _smoke_m2.py）。
"""
import argparse
import os
import random
import sys
import tempfile
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # AOI_sys 根

# ---- 隔离环境（必须在 import backend 之前设置 AOI_CONFIG）----
TMP = tempfile.mkdtemp(prefix="aoi_m9_")
CFG_PATH = os.path.join(TMP, "cfg.yaml")
import yaml  # noqa: E402

with open(CFG_PATH, "w", encoding="utf-8") as f:
    yaml.safe_dump({"system": {
        "database_url": "sqlite:///" + os.path.join(TMP, "aoi.db").replace("\\", "/"),
        "storage_dir": os.path.join(TMP, "storage").replace("\\", "/"),
        "demo5_storage": os.path.join(TMP, "storage", "demo5").replace("\\", "/"),
        # 基线口径 = sanity 加权（demo5 项目记忆 U55/U59：bottle 0.9985 等数字均
        # 出自 fusion.weight_mode=sanity）。m0_fast.yaml 的双 fusion 键覆盖 bug
        # 已在 M9 中修复本体（合并为单个 fusion: {weight_mode: sanity, mode: fixed_equal}），
        # 此处直接引用修好的原文件。
        "demo5_base_cfg": "configs/demo5_fast.yaml",
    }}, f, allow_unicode=True)
os.environ["AOI_CONFIG"] = CFG_PATH

# ── 数据集配置层（M9b）──────────────────────────────────────────
# defect_dirs="auto" 表示 test 下除正常目录外的全部子目录均为缺陷目录
# （MPDD 各品类缺陷目录命名不统一：tubes=anomalous，bracket_white=defective_painting+scratches）。
# cats: {验收品类名: (数据集内子目录名, demo5 基线 AUROC)}
DATASETS = {
    "mvtec": {
        "root": r"D:\CGAIC\data_origin\mvtec",
        "normal_dir": "good", "defect_dirs": "auto",
        "exts": (".png", ".jpg", ".jpeg"), "source": "mvtec",
        "cats": {"bottle": ("bottle", 0.9985), "cable": ("cable", 0.9619),
                 "tile": ("tile", 0.9989), "screw": ("screw", 0.8715)},
    },
    "btad": {
        "root": r"D:\CGAIC\data_origin\BTAD\BTech_Dataset_transformed",
        "normal_dir": "ok", "defect_dirs": ["ko"],
        "exts": (".bmp", ".png", ".jpg", ".jpeg"), "source": "btad",
        "cats": {"btad_01": ("01", 0.9524), "btad_02": ("02", 0.8600),
                 "btad_03": ("03", 0.9980)},
    },
    "mpdd": {
        "root": r"D:\CGAIC\data_origin\MPDD",
        "normal_dir": "good", "defect_dirs": "auto",
        "exts": (".png", ".jpg", ".jpeg"), "source": "mpdd",
        "cats": {"mpdd_tubes": ("tubes", 0.72),
                 "mpdd_bracket_white": ("bracket_white", 0.94)},
    },
    "data_local": {
        "root": r"D:\CGAIC\data_local",
        "normal_dir": "good", "defect_dirs": "auto",
        "exts": (".png", ".jpg", ".jpeg", ".bmp"), "source": "data_local",
        "cats": {"solder_smt": ("solder_smt", 0.683)},
    },
}
DATASET_ORDER = ["mvtec", "btad", "mpdd", "data_local"]

GYUDET_ROOT = r"D:\CGAIC\demo3\outputs\crops\gyudet"

TOL = 0.05
PREPARE_REDLINE_S = 120
LAT_REDLINE_MS = 1000

# 对齐 demo5 基线协议 n_init_normal=100（见 M9 记录；20 张会低估 AUROC）
N_TRAIN_NORMAL = 100
N_ANCHOR_PER_DEFECT = 2
N_ANCHOR_CAP = 5
N_EVAL_GOOD = 15
N_EVAL_DEFECT = 15
N_FEEDBACK_CAP = 5
SEED = 7

# gyudet 跨域学习增益专项参数
GYU_N_TRAIN_NORMAL = 100
GYU_N_ANCHOR = 5            # train 域有 defect 目录，锚定从 train/defect 取
GYU_N_EVAL0 = 15            # 初始评估：valid normal 15 + defect 15
GYU_N_HOLDOUT = 15          # 学习曲线 holdout：valid 另取 normal 15 + defect 15
GYU_FB_CAP = 20             # 跨域需要更多反馈（常规 cap 5）
GYU_EVAL_EVERY = 5
GYU_GAIN_BAR = 0.05         # demo5 U46/U51 增益 +0.10~+0.14，取保守线


def _ls(d, exts=(".png", ".jpg", ".jpeg")):
    return sorted(os.path.join(d, x) for x in os.listdir(d)
                  if x.lower().endswith(exts))


def _defect_dirs(src, ds_cfg):
    test_dir = os.path.join(src, "test")
    if ds_cfg["defect_dirs"] == "auto":
        return [d for d in sorted(os.listdir(test_dir))
                if d != ds_cfg["normal_dir"]
                and os.path.isdir(os.path.join(test_dir, d))]
    return [d for d in ds_cfg["defect_dirs"]
            if os.path.isdir(os.path.join(test_dir, d))]


def collect(ds_key, cat):
    """返回 (train_normals, defect_anchors, eval_items[(path, label)])。"""
    ds = DATASETS[ds_key]
    sub, _ = ds["cats"][cat]
    src = os.path.join(ds["root"], sub)
    assert os.path.isdir(src), f"{ds_key} 品类目录不存在: {src}"
    exts = ds["exts"]
    rnd = random.Random(SEED)

    train_normals = _ls(os.path.join(src, "train", ds["normal_dir"]), exts)
    rnd.shuffle(train_normals)
    train_normals = train_normals[:N_TRAIN_NORMAL]

    defect_dirs = _defect_dirs(src, ds)
    assert defect_dirs, f"{src}/test 下无缺陷目录"
    anchors = []
    for d in defect_dirs:
        pool = _ls(os.path.join(src, "test", d), exts)
        rnd.shuffle(pool)
        anchors += pool[:N_ANCHOR_PER_DEFECT]
    anchors = anchors[:N_ANCHOR_CAP]
    anchor_set = set(anchors)

    eval_good = _ls(os.path.join(src, "test", ds["normal_dir"]), exts)
    rnd.shuffle(eval_good)
    eval_good = eval_good[:N_EVAL_GOOD]
    eval_defect_pool = [p for d in defect_dirs
                        for p in _ls(os.path.join(src, "test", d), exts)
                        if p not in anchor_set]
    rnd.shuffle(eval_defect_pool)
    eval_defects = eval_defect_pool[:N_EVAL_DEFECT]

    eval_items = ([(p, 0) for p in eval_good]
                  + [(p, 1) for p in eval_defects])
    rnd.shuffle(eval_items)
    return train_normals, anchors, eval_items


def collect_gyudet():
    """gyudet 专项取样：train 域建库，valid 域分初始评估集与 holdout（互斥防泄漏）。"""
    exts = (".jpg", ".jpeg", ".png")
    rnd = random.Random(SEED)
    train_normals = _ls(os.path.join(GYUDET_ROOT, "train", "normal"), exts)
    rnd.shuffle(train_normals)
    train_normals = train_normals[:GYU_N_TRAIN_NORMAL]
    anchors = _ls(os.path.join(GYUDET_ROOT, "train", "defect"), exts)
    rnd.shuffle(anchors)
    anchors = anchors[:GYU_N_ANCHOR]

    valid_n = _ls(os.path.join(GYUDET_ROOT, "valid", "normal"), exts)
    valid_d = _ls(os.path.join(GYUDET_ROOT, "valid", "defect"), exts)
    rnd.shuffle(valid_n)
    rnd.shuffle(valid_d)
    holdout = ([(p, 0) for p in valid_n[:GYU_N_HOLDOUT]]
               + [(p, 1) for p in valid_d[:GYU_N_HOLDOUT]])
    eval0 = ([(p, 0) for p in valid_n[GYU_N_HOLDOUT:
                                       GYU_N_HOLDOUT + GYU_N_EVAL0]]
             + [(p, 1) for p in valid_d[GYU_N_HOLDOUT:
                                        GYU_N_HOLDOUT + GYU_N_EVAL0]])
    rnd.shuffle(holdout)
    rnd.shuffle(eval0)
    assert eval0 and holdout, "gyudet valid 域样本不足"
    return train_normals, anchors, eval0, holdout


def register_images(cat, train_normals, anchors, source):
    from backend.db.database import session_scope
    from backend.db.models import Image as ImageRow
    with session_scope() as s:
        for p in train_normals:
            s.add(ImageRow(path=p, category=cat, split="train",
                           label="normal", source=source))
        for p in anchors:
            s.add(ImageRow(path=p, category=cat, split="train_anomaly",
                           label="anomaly", source=source))


def wait_task(task_manager, task_id, timeout=900):
    t0 = time.time()
    while time.time() - t0 < timeout:
        t = task_manager.get(task_id)
        if t["status"] in ("done", "failed"):
            return t
        print(f"  [task {task_id}] {t['progress']*100:5.1f}% {t['message']}",
              flush=True)
        time.sleep(3)
    raise TimeoutError(f"task {task_id} 超时")


def _prepare(client, task_manager, cat, ck):
    """POST prepare + 等待，返回 (ok, prep_s)。"""
    t0 = time.time()
    r = client.post("/api/models/prepare",
                    json={"category": cat, "scenario": "L1a",
                          "profile": "fast", "force": True})
    assert r.status_code == 200, f"prepare 提交失败: {r.text}"
    tid = r.json()["task_id"]
    print(f"[{cat}] prepare task_id={tid}，等待完成...", flush=True)
    t = wait_task(task_manager, tid)
    prep_s = time.time() - t0
    ok = t["status"] == "done"
    if ok:
        res = t["result"]
        detail = (f"{prep_s:.1f}s（红线 {PREPARE_REDLINE_S}s）"
                  f" version={res['version']} n_normal={res['n_normal']}"
                  f" n_defect={res['n_defect']}")
    else:
        detail = f"prepare 失败: {t['message']}"
    return ck("部署 prepare", ok and prep_s < PREPARE_REDLINE_S, detail), prep_s


def _detect_set(client, cat, eval_items, verbose=True):
    """逐张 detect，返回 (labels, scores, dets, lats)。"""
    labels, scores, dets, lats = [], [], [], []
    for i, (p, lab) in enumerate(eval_items):
        d = client.post("/api/detect/image",
                        json={"path": p, "category": cat,
                              "with_heatmap": False})
        assert d.status_code == 200, f"detect 失败 {p}: {d.text}"
        d = d.json()
        scores.append(d["final_score"])
        labels.append(lab)
        lats.append(d["latency_ms"])
        dets.append(d)
        if verbose:
            print(f"  [detect {i+1}/{len(eval_items)}] label={lab}"
                  f" score={d['final_score']:.4f} decision={d['decision']}"
                  f" latency={d['latency_ms']:.0f}ms", flush=True)
    return labels, scores, dets, lats


def _auroc(labels, scores):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(labels, scores))


def run_category(client, task_manager, ds_key, cat, data):
    """单品类全流程验收，返回 {category, checks{name:(ok,detail)}, metrics...}。"""
    baseline = DATASETS[ds_key]["cats"][cat][1]
    train_normals, anchors, eval_items = data
    row = {"category": cat, "dataset": ds_key, "baseline": baseline,
           "checks": {}}

    def ck(name, ok, detail):
        row["checks"][name] = (bool(ok), detail)
        mark = "✅" if ok else "❌"
        print(f"  {mark} {name}: {detail}", flush=True)
        return ok

    # ── 2) 部署效率 ─────────────────────────────────────────
    ok, prep_s = _prepare(client, task_manager, cat, ck)
    row["prep_s"] = prep_s
    if not ok:
        return row

    # ── 3) 检测质量：评估集 AUROC + 延迟 ─────────────────────
    labels, scores, dets, lats = _detect_set(client, cat, eval_items)
    auroc = _auroc(labels, scores)
    avg_lat = sum(lats) / len(lats)
    max_lat = max(lats)
    bar = baseline - TOL
    row["auroc"] = auroc
    row["avg_lat"] = avg_lat
    row["max_lat"] = max_lat
    ck("检测质量 AUROC", auroc >= bar,
       f"{auroc:.4f}（基线 {baseline}，验收线 {bar:.4f}）")
    if auroc < bar:
        print(f"  [{cat}] AUROC 低于验收线，逐图 score 分布：", flush=True)
        for (p, lab), sc in sorted(zip(eval_items, scores),
                                   key=lambda x: x[1]):
            print(f"    label={lab} score={sc:.4f} {os.path.basename(p)}",
                  flush=True)
    ck("延迟红线", max_lat < LAT_REDLINE_MS,
       f"均值 {avg_lat:.0f}ms / 峰值 {max_lat:.0f}ms（红线 {LAT_REDLINE_MS}ms）")

    # ── 4) 反馈学习：挑系统判错的（cap 5）反馈即学 ──────────
    mis = [(p, lab, sc, d) for (p, lab), sc, d in zip(eval_items, scores, dets)
           if (lab == 1 and sc < 0.5) or (lab == 0 and sc > 0.5)]
    mis = mis[:N_FEEDBACK_CAP]
    fb_n = 0
    for p, lab, sc, d in mis:
        ftype = "false_negative" if lab == 1 else "false_positive"
        r = client.post("/api/feedback", json={
            "detection_id": d["detection_id"], "feedback_type": ftype,
            "operator_label": lab, "comment": "M9 验收错检反馈"})
        assert r.status_code == 200, f"feedback 失败: {r.text}"
        fb = r.json()
        assert "pre" in fb and fb["pre"] is not None, f"响应缺 pre: {fb}"
        fb_n += 1
        print(f"  [feedback] {ftype} pre={fb['pre']['score']:.4f}"
              f" engine_update={fb['engine_update'] is not None}", flush=True)
    row["fb_n"] = fb_n
    ck("反馈即学", fb_n > 0 or not mis,
       f"{fb_n} 条错检反馈即学（评估集错检 {len(mis)} 张）"
       if mis else "评估集零错检，无反馈可提")

    # ── 5) 翻案曲线 ─────────────────────────────────────────
    if fb_n:
        r = client.get(f"/api/learning/flip_curve/{cat}")
        assert r.status_code == 200, f"flip_curve 失败: {r.text}"
        fc = r.json()["summary"]
        row["flip"] = fc
        ck("翻案曲线", True,
           f"total={fc['total']} flipped={fc['flipped_count']}"
           f" flip_rate={fc['flip_rate']:.0%}")

    # ── 6) 版本固化：巩固落盘 v2 ─────────────────────────────
    r = client.post("/api/self_learning/update",
                    json={"category": cat, "note": "M9 验收巩固"})
    assert r.status_code == 200, f"consolidate 提交失败: {r.text}"
    t = wait_task(task_manager, r.json()["task_id"])
    ok = t["status"] == "done" and t["result"].get("version") == "v2"
    if ok:
        cats = client.get("/api/categories").json()
        c = [x for x in cats if x["category"] == cat][0]
        ok = c["prepared"] and c["current_version"] == 2
        detail = f"当前版本 v{c['current_version']}"
    else:
        detail = f"巩固失败: {t.get('message')}"
    row["version"] = 2 if ok else None
    ck("版本固化", ok, detail)

    # ── 7) 数据治理：批次建档 + 四维树 ───────────────────────
    ds = client.get(f"/api/datasets?category={cat}").json()["items"]
    tree = client.get("/api/images/tree").json()
    has_4d = "datasets" in tree and cat in tree["datasets"]
    row["n_datasets"] = len(ds)
    ck("数据治理", len(ds) >= 1 and has_4d,
       f"{len(ds)} 个批次，四维树 datasets 维={'有' if has_4d else '无'}")
    return row


def run_gyudet_learn(client, task_manager, data):
    """gyudet 跨域学习增益专项：静态失效 → 反馈即学 → 学习曲线增益验收。"""
    cat = "gyudet"
    _, _, eval0, holdout = data
    row = {"category": cat, "dataset": "gyudet", "checks": {}}

    def ck(name, ok, detail):
        row["checks"][name] = (bool(ok), detail)
        mark = "✅" if ok else "❌"
        print(f"  {mark} {name}: {detail}", flush=True)
        return ok

    # ── 部署（L1a：train/normal 100 + train/defect 5 锚定）───
    ok, prep_s = _prepare(client, task_manager, cat, ck)
    row["prep_s"] = prep_s
    if not ok:
        return row

    all_lats = []

    # ── 初始静态 AUROC（信息项，不参与 PASS；跨域预期 ~0.5）──
    print(f"[{cat}] 初始静态评估（valid {len(eval0)} 图）...", flush=True)
    labels0, scores0, dets0, lats0 = _detect_set(client, cat, eval0,
                                                 verbose=False)
    all_lats += lats0
    auroc0 = _auroc(labels0, scores0)
    row["auroc_static"] = auroc0
    ck("初始静态 AUROC（信息项）", True,
       f"{auroc0:.4f}（demo5 U38/U41 跨域静态失效 ~0.5，不参与 PASS 判定）")

    # holdout 在 k=0 的基准分（与后续曲线同一评估集，保证可比性）
    hl, hs, _, hlat = _detect_set(client, cat, holdout, verbose=False)
    all_lats += hlat
    curve = [(0, _auroc(hl, hs))]
    print(f"[{cat}] 学习曲线 k=0  AUROC={curve[0][1]:.4f}"
          f"（holdout {len(holdout)} 图，全程不反馈）", flush=True)

    # ── 反馈学习：初始评估集错检逐张反馈（cap 20）────────────
    mis = [(p, lab, sc, d) for (p, lab), sc, d in zip(eval0, scores0, dets0)
           if (lab == 1 and sc < 0.5) or (lab == 0 and sc > 0.5)]
    mis = mis[:GYU_FB_CAP]
    if not mis:
        ck("反馈学习", False, "初始评估集零错检，无法构造反馈流")
        return row
    print(f"[{cat}] 初始错检 {len(mis)} 张，逐张反馈即学"
          f"（每 {GYU_EVAL_EVERY} 条重估 holdout AUROC）...", flush=True)
    fb_done = 0
    for p, lab, sc, d in mis:
        ftype = "false_negative" if lab == 1 else "false_positive"
        r = client.post("/api/feedback", json={
            "detection_id": d["detection_id"], "feedback_type": ftype,
            "operator_label": lab, "comment": "M9b gyudet 跨域学习反馈"})
        assert r.status_code == 200, f"feedback 失败: {r.text}"
        fb = r.json()
        fb_done += 1
        print(f"  [feedback {fb_done}/{len(mis)}] {ftype}"
              f" pre={fb['pre']['score'] if fb.get('pre') else sc:.4f}"
              f" engine_update={fb['engine_update'] is not None}", flush=True)
        if fb_done % GYU_EVAL_EVERY == 0:
            hl, hs, _, hlat = _detect_set(client, cat, holdout, verbose=False)
            all_lats += hlat
            curve.append((fb_done, _auroc(hl, hs)))
            print(f"[{cat}] 学习曲线 k={fb_done:<3} AUROC={curve[-1][1]:.4f}",
                  flush=True)
    if curve[-1][0] != fb_done:     # 不足整档时补终点
        hl, hs, _, hlat = _detect_set(client, cat, holdout, verbose=False)
        all_lats += hlat
        curve.append((fb_done, _auroc(hl, hs)))
        print(f"[{cat}] 学习曲线 k={fb_done:<3} AUROC={curve[-1][1]:.4f}",
              flush=True)
    row["fb_n"] = fb_done
    row["curve"] = curve

    gain = curve[-1][1] - curve[0][1]
    row["gain"] = gain
    print(f"[{cat}] 跨域学习增益：初始 {curve[0][1]:.4f} → "
          f"{curve[-1][0]} 反馈后 {curve[-1][1]:.4f}（{gain:+.4f}）", flush=True)
    ck("跨域学习增益", gain >= GYU_GAIN_BAR,
       f"初始 {curve[0][1]:.4f} → k={curve[-1][0]} {curve[-1][1]:.4f}"
       f"（{gain:+.4f}，验收线 +{GYU_GAIN_BAR}；"
       f"demo5 U46/U51 参考 +0.10~+0.14）")

    # ── 延迟红线（专项全程所有 detect）───────────────────────
    row["max_lat"] = max(all_lats)
    row["avg_lat"] = sum(all_lats) / len(all_lats)
    ck("延迟红线", row["max_lat"] < LAT_REDLINE_MS,
       f"均值 {row['avg_lat']:.0f}ms / 峰值 {row['max_lat']:.0f}ms"
       f"（红线 {LAT_REDLINE_MS}ms，共 {len(all_lats)} 次 detect）")

    # ── 翻案曲线 summary ────────────────────────────────────
    r = client.get(f"/api/learning/flip_curve/{cat}")
    assert r.status_code == 200, f"flip_curve 失败: {r.text}"
    fc = r.json()["summary"]
    row["flip"] = fc
    ck("翻案曲线", True,
       f"total={fc['total']} flipped={fc['flipped_count']}"
       f" flip_rate={fc['flip_rate']:.0%}")
    return row


def _print_group(rows):
    hdr = (f"{'品类':<20}{'AUROC':<9}{'验收线':<9}{'prepare':<9}"
           f"{'延迟均/峰ms':<14}{'反馈':<6}{'翻案率':<8}{'版本':<6}"
           f"{'批次':<5}{'结论'}")
    print(hdr, flush=True)
    for row in rows:
        cat = row["category"]
        auroc = f"{row['auroc']:.4f}" if "auroc" in row else "-"
        bar = f"{row['baseline'] - TOL:.4f}" if "baseline" in row else "-"
        prep = f"{row['prep_s']:.1f}s" if "prep_s" in row else "-"
        lat = (f"{row['avg_lat']:.0f}/{row['max_lat']:.0f}"
               if "avg_lat" in row else "-")
        fb = str(row.get("fb_n", "-"))
        flip = (f"{row['flip']['flip_rate']:.0%}" if "flip" in row else "-")
        ver = f"v{row['version']}" if row.get("version") else "-"
        nds = str(row.get("n_datasets", "-"))
        ok = all(ok for ok, _ in row["checks"].values())
        print(f"{cat:<20}{auroc:<9}{bar:<9}{prep:<9}{lat:<14}{fb:<6}"
              f"{flip:<8}{ver:<6}{nds:<5}{'PASS' if ok else 'FAIL'}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cats", nargs="*", default=[],
                    help="品类过滤（在选定数据集内；缺省=该数据集全部品类）")
    ap.add_argument("--datasets", default=None,
                    help="逗号分隔数据集（mvtec,btad,mpdd,data_local）或 all；"
                         "缺省 mvtec；与 --gyudet-learn 同给则缺省不跑静态集")
    ap.add_argument("--gyudet-learn", action="store_true",
                    help="跑 gyudet 跨域学习增益专项")
    args = ap.parse_args()

    if args.datasets is None:
        ds_keys = [] if args.gyudet_learn else ["mvtec"]
    elif args.datasets.strip().lower() == "all":
        ds_keys = list(DATASET_ORDER)
    else:
        ds_keys = [k.strip() for k in args.datasets.split(",") if k.strip()]
    for k in ds_keys:
        assert k in DATASETS, f"未知数据集: {k}（可选 {list(DATASETS)} / all）"

    # 解析 (ds_key, cat) 任务清单
    plan = []
    for k in ds_keys:
        cats = args.cats or list(DATASETS[k]["cats"])
        for c in cats:
            assert c in DATASETS[k]["cats"], \
                f"品类 {c} 不在数据集 {k}（可选 {list(DATASETS[k]['cats'])}）"
            plan.append((k, c))

    print("=" * 72, flush=True)
    print(f"M9 工业级可用性验收  datasets={ds_keys or '(无静态集)'}"
          f" gyudet_learn={args.gyudet_learn}  TMP={TMP}", flush=True)
    print("=" * 72, flush=True)

    # 1) 数据登记（全部品类，TestClient 启动 lifespan 时统一归入历史批次）
    data_map = {}
    for ds_key, cat in plan:
        data = collect(ds_key, cat)
        register_images(cat, data[0], data[1], DATASETS[ds_key]["source"])
        data_map[cat] = data
        print(f"[data] {cat}: train_normal={len(data[0])}"
              f" anchor_defect={len(data[1])} eval={len(data[2])}"
              f"（good={sum(1 for _, l in data[2] if l == 0)}"
              f" defect={sum(1 for _, l in data[2] if l == 1)}）", flush=True)
    gyu_data = None
    if args.gyudet_learn:
        gyu_data = collect_gyudet()
        register_images("gyudet", gyu_data[0], gyu_data[1], "gyudet")
        print(f"[data] gyudet: train_normal={len(gyu_data[0])}"
              f" anchor_defect={len(gyu_data[1])}"
              f" eval0={len(gyu_data[2])} holdout={len(gyu_data[3])}",
              flush=True)

    from fastapi.testclient import TestClient

    from backend.api.app import app
    from backend.core.tasks import task_manager

    rows = []
    gyu_row = None
    all_pass = True
    with TestClient(app) as client:
        for ds_key, cat in plan:
            print(f"\n── {cat}（{ds_key}）──", flush=True)
            try:
                row = run_category(client, task_manager, ds_key, cat,
                                   data_map[cat])
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                row = {"category": cat, "dataset": ds_key,
                       "checks": {"异常中断": (False, repr(e))}}
            rows.append(row)
            all_pass = all_pass and all(ok for ok, _ in row["checks"].values())
        if args.gyudet_learn:
            print("\n── gyudet 跨域学习增益专项 ──", flush=True)
            try:
                gyu_row = run_gyudet_learn(client, task_manager, gyu_data)
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                gyu_row = {"category": "gyudet", "dataset": "gyudet",
                           "checks": {"异常中断": (False, repr(e))}}
            all_pass = all_pass and all(
                ok for ok, _ in gyu_row["checks"].values())

    # ── 总表（分数据集分组）──────────────────────────────────
    print("\n" + "=" * 72, flush=True)
    print("总表", flush=True)
    for ds_key in DATASET_ORDER:
        grp = [r for r in rows if r.get("dataset") == ds_key]
        if not grp:
            continue
        print(f"\n── {ds_key} ──", flush=True)
        _print_group(grp)
    if gyu_row is not None:
        print("\n── gyudet 跨域学习增益专项 ──", flush=True)
        if "curve" in gyu_row:
            seg = " → ".join(f"k={k} {a:.4f}" for k, a in gyu_row["curve"])
            print(f"学习曲线（holdout 30 图，未参与反馈）: {seg}", flush=True)
            print(f"跨域学习增益：初始 {gyu_row['curve'][0][1]:.4f} → "
                  f"{gyu_row['curve'][-1][0]} 反馈后 "
                  f"{gyu_row['curve'][-1][1]:.4f}（{gyu_row['gain']:+.4f}）",
                  flush=True)
        for name, (ok, detail) in gyu_row["checks"].items():
            print(f"  {'✅' if ok else '❌'} {name}: {detail}", flush=True)
        ok = all(ok for ok, _ in gyu_row["checks"].values())
        print(f"gyudet 专项结论：{'PASS' if ok else 'FAIL'}", flush=True)

    print("\n" + "=" * 72, flush=True)
    print("工业级可用性验收：" + ("PASS ✅" if all_pass else "FAIL ❌"), flush=True)
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
