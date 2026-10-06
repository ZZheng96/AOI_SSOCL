# -*- coding: utf-8 -*-
"""v10 冒烟：数据增强配置页（§15.29 伪异常合成 / 预处理 / 训练增强）。

1. GET /api/augment/config 返回 pseudo/preprocess/enhance 三节；
2. PUT /api/augment/config 写回白名单字段（改回原值，幂等）；
3. 引擎合成单元级：defect_transplant（含无缺陷池回退）与 preprocess 各开关；
4. /api/pseudo/preview 返回 图+掩码；
5. UI：页面可构建、三类配置控件存在、无训练参数（lr/epochs）与 demo5 字样。
"""
import os
import sys
import time
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.CRITICAL)
from server import start_server_background
start_server_background()
import requests  # noqa: E402

BASE = "http://127.0.0.1:8017"
for _ in range(100):
    try:
        if requests.get(f"{BASE}/api/health", timeout=1).json().get("status") == "ok":
            break
    except Exception:
        pass
    time.sleep(0.3)

ISSUES = []
_OK = {"n": 0}


def ck(name, cond, detail=""):
    if cond:
        _OK["n"] += 1
        print(f"  ✅ {name}" + (f"（{detail}）" if detail else ""), flush=True)
    else:
        ISSUES.append(f"{name}: {detail}")
        print(f"  ❌ {name}（{detail}）", flush=True)


# 0. 引擎合成单元级：defect_transplant（AHL 真实缺陷移植）────────
import cv2  # noqa: E402
import numpy as np  # noqa: E402
from algo.slots.disc import synth_pseudo, load_defect_mask  # noqa: E402
from algo.common.preprocess import preprocess_image  # noqa: E402

rng = np.random.default_rng(7)
normal = np.random.randint(0, 255, (128, 128, 3), np.uint8)
defect = np.random.randint(0, 255, (64, 64, 3), np.uint8)
# 缺陷位置掩码：中央小圆区域为缺陷（0/255）
dm = np.zeros((64, 64), np.uint8)
cv2.circle(dm, (32, 32), 10, 255, -1)
pcfg = {"methods": ["defect_transplant"], "transplant_scale": [0.35, 0.9],
        "feather": True}
out, mask = synth_pseudo(normal, rng, defect_pool=[(defect, dm)], pcfg=pcfg)
ck("defect_transplant 输出同尺寸", out.shape == normal.shape)
ck("defect_transplant 掩码非零", float(mask.max()) > 0.5)
ck("移植改动局限于合成掩码内（按位置抠取）",
   int(((np.abs(out.astype(int) - normal.astype(int)).sum(2) > 5)
        & (mask <= 0.1)).sum()) == 0)
# 无位置掩码的源不参与移植（整图/整块贴入无意义）→ 回退 CutPaste
zm = np.zeros((64, 64), np.uint8)
outp, maskp = synth_pseudo(normal, rng, defect_pool=[(defect, zm)], pcfg=pcfg)
ck("无位置掩码源自动回退（CutPaste）", outp.shape == normal.shape
   and float(maskp.max()) > 0.5)
# 无缺陷池回退（应回退 cutpaste，仍返回图+掩码）
out2, mask2 = synth_pseudo(normal, rng, defect_pool=None, pcfg=pcfg)
ck("无缺陷池自动回退（CutPaste）", out2.shape == normal.shape
   and float(mask2.max()) > 0.5)
# 仅保留 color_blot 时按配置只合成色斑
pcfg2 = {"methods": ["color_blot"]}
out3, _ = synth_pseudo(normal, rng, pcfg=pcfg2)
ck("color_blot 单方式可合成", out3.shape == normal.shape)
# 预处理各开关
out4 = preprocess_image(normal, {"enabled": True, "gray": True})
ck("预处理灰度 3 通道保持", out4.shape == normal.shape and out4.ndim == 3)
out5 = preprocess_image(normal, {"enabled": True, "clahe": 2.0})
ck("预处理 CLAHE 生效", out5.shape == normal.shape)
out6 = preprocess_image(normal, {"enabled": True, "median": 3})
ck("预处理中值生效", out6.shape == normal.shape)
out7 = preprocess_image(normal, None)
ck("预处理未启用直通", out7 is normal)

# 1. 读取配置
r = requests.get(f"{BASE}/api/augment/config", timeout=8)
b = r.json()
ck("GET 增强配置 200", r.status_code == 200)
ck("三节结构完整", all(k in b for k in ("pseudo", "preprocess", "enhance")),
   str(list(b.keys())))
pseudo = b.get("pseudo") or {}
ck("伪异常含真实缺陷移植方式",
   "defect_transplant" in (pseudo.get("methods") or []),
   str(pseudo.get("methods")))
ck("含预处理/增强字段", "enabled" in (b.get("preprocess") or {})
   and "flip" in (b.get("enhance") or {}))
ck("无训练参数节", "lr" not in (b.get("enhance") or {}))
ck("页面说明为系统视角", "demo5" not in (b.get("note") or ""))

# 2. 写回（原值幂等）
r = requests.put(f"{BASE}/api/augment/config", timeout=8, json={
    "pseudo": {"methods": pseudo.get("methods"),
               "transplant_scale": pseudo.get("transplant_scale"),
               "feather": pseudo.get("feather", True),
               "pseudo_per_image": pseudo.get("pseudo_per_image", 8)},
    "preprocess": b.get("preprocess"),
    "enhance": b.get("enhance")})
ck("PUT 增强配置 200", r.status_code == 200, str(r.status_code))
b2 = r.json()
ck("写回成功标记", b2.get("ok") is True)
r3 = requests.get(f"{BASE}/api/augment/config", timeout=8).json()
ck("读回一致",
   (r3.get("pseudo") or {}).get("pseudo_per_image")
   == pseudo.get("pseudo_per_image"))

# 3. 预览：取一张 normal 图
imgs = requests.get(f"{BASE}/api/images", params={"label": "normal",
                                                  "page": 1, "page_size": 5},
                    timeout=8).json().get("items", [])
if imgs:
    iid = imgs[0]["id"]
    for m in ("defect_transplant", "cutpaste", "color_blot"):
        r = requests.post(f"{BASE}/api/pseudo/preview", timeout=8,
                          json={"image_id": iid, "method": m})
        ck(f"预览 {m} 200", r.status_code == 200, str(r.status_code))
        bb = r.json()
        ck(f"预览 {m} 返回图+掩码",
           bool(bb.get("image_path")) and bool(bb.get("mask_path")),
           os.path.basename(bb.get("image_path", "")))
else:
    print("  ⚠ 无 normal 图，跳过预览组", flush=True)

# 4. UI 构建
from PySide6.QtCore import QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication, QLabel  # noqa: E402
from ui.main_window import MainWindow  # noqa: E402

app = QApplication([])
win = MainWindow()
win.resize(1280, 800)
win.show()


def pump(ms=300):
    end = time.time() + ms / 1000
    while time.time() < end:
        app.processEvents()
        time.sleep(0.02)


def step():
    win.goto("数据增强")
    pump(2500)
    pg = win.page_augment
    ck("页面可构建", pg is not None)
    # 三类配置控件
    ck("伪异常控件存在", hasattr(pg, "spin_pseudo_per_image")
       and hasattr(pg, "_method_checks")
       and "defect_transplant" in pg._method_checks)
    ck("预处理控件存在", hasattr(pg, "chk_pre_enabled")
       and hasattr(pg, "combo_pre_median"))
    ck("训练增强控件存在", hasattr(pg, "chk_en_flip")
       and hasattr(pg, "chk_en_brightness"))
    # 无训练参数控件（学习率/轮数等）
    txt = " ".join(w.text() for w in pg.findChildren(QLabel))
    ck("无学习率/轮数等训练参数", "lr" not in txt and "epochs" not in txt)
    ck("无 demo5 术语", "demo5" not in txt.lower())
    ck("合成方式下拉存在", hasattr(pg, "combo_method")
       and pg.combo_method.count() >= 3)
    print(f"\n{'='*46}", flush=True)
    print(f"验证全部通过 ✅（{_OK['n']}）" if not ISSUES
          else f"验证发现 {len(ISSUES)} 个问题 ❌（通过 {_OK['n']}）", flush=True)
    for it in ISSUES:
        print(f"  ❌ {it}", flush=True)
    import os as _os
    _os._exit(1 if ISSUES else 0)


QTimer.singleShot(400, step)
app.exec()
