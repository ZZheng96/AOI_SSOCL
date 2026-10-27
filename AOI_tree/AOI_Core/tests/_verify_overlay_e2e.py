# -*- coding: utf-8 -*-
"""端到端验证：实时监控页叠加图是否真实显示。

流程：
1. 启动/复用后端（加载当前磁盘代码）
2. 按产线路径触发一条检测（target_type="pipeline", with_heatmap=True）
3. 调 /api/detections/live 确认返回 heatmap_path / overlay_path
4. 用 file_url 下载三层图字节，验证均为有效图片
5. 启动 ImageViewer，模拟 monitor_page._show_detection 的调用序列，
   切到 overlay 层，离屏截图保存，并校验画面非空、与原图有差异
"""
import os
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

BASE = "http://127.0.0.1:8017"

import requests  # noqa: E402


def ensure_backend():
    try:
        if requests.get(f"{BASE}/api/health", timeout=1).json().get("status") == "ok":
            print("[backend] 复用已运行实例（注意：可能是旧代码）", flush=True)
            return False
    except Exception:
        pass
    from server import start_server_background
    start_server_background()
    for _ in range(100):
        try:
            if requests.get(f"{BASE}/api/health", timeout=1).json().get("status") == "ok":
                break
        except Exception:
            pass
        time.sleep(0.3)
    print("[backend] 已启动（当前磁盘代码）", flush=True)
    return True


def step_detect():
    """按产线路径触发检测，返回 detection 结果。"""
    import sqlite3
    from backend.pipeline.service import get_detection_service

    db = os.path.join(ROOT, "database", "aoi.db")
    row = sqlite3.connect(db).execute(
        "select path from images where category='02' and split='test' limit 1"
    ).fetchone()
    svc = get_detection_service()
    out = svc.detect_image(
        row[0], "02", persist=True, with_heatmap=True,
        target_type="pipeline", workorder_id=1,
    )
    print(f"[detect] id={out.get('detection_id') or out.get('id')} "
          f"heatmap={out.get('heatmap_path')} overlay={out.get('overlay_path')}", flush=True)
    return out


def step_api(det_id):
    r = requests.get(f"{BASE}/api/detections/live",
                     params={"after_id": det_id - 1, "limit": 1}, timeout=10)
    items = (r.json().get("items") or [])
    assert items, "live 接口无返回"
    d = items[0]
    print(f"[api] live 返回: heatmap_path={'有' if d.get('heatmap_path') else '无'} "
          f"overlay_path={'有' if d.get('overlay_path') else '无'}", flush=True)
    assert d.get("overlay_path"), "live 接口未返回 overlay_path"
    return d


def step_download(d):
    from ui.api_client import file_url
    data = {}
    for layer in ("original", "heatmap", "overlay"):
        p = d.get({"original": "image_path"}.get(layer, f"{layer}_path"))
        url = file_url(p)
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        data[layer] = r.content
        print(f"[download] {layer}: {len(r.content)} bytes", flush=True)
    assert len(set(data.values())) == 3, "三层图字节完全相同，疑似同一张图"
    print("[download] 三层图字节互不相同 ✓", flush=True)
    return data


def step_ui(data):
    """离屏渲染 ImageViewer：模拟 monitor 默认 overlay 层显示。"""
    from PySide6.QtCore import QTimer
    from PySide6.QtGui import QImage
    from PySide6.QtWidgets import QApplication

    from ui.widgets.image_viewer import ImageViewer

    app = QApplication.instance() or QApplication([])
    v = ImageViewer()
    v.resize(960, 600)

    # 模拟 monitor_page._show_detection：先清帧，再逐层喂字节
    v.clear_images()
    for layer in ("original", "heatmap", "overlay"):
        v.set_image_bytes(data[layer], layer)

    assert v._layer == "overlay", f"默认层异常: {v._layer}"
    assert v._layer_ready.get("overlay"), "overlay 层未就绪"
    v.set_layer("overlay")
    v.show()
    app.processEvents()

    out_png = os.path.join(ROOT, "tests", "_verify_overlay_shot.png")
    v.grab().save(out_png)
    img = QImage(out_png)
    colors = set()
    for y in range(0, img.height(), 20):
        for x in range(0, img.width(), 20):
            colors.add(img.pixelColor(x, y).rgba())
    print(f"[ui] 截图 {out_png}，采样颜色数={len(colors)}", flush=True)
    assert len(colors) > 50, "画面近乎纯色，疑似空白"

    # 原图截图对比：切到 original 再抓一张，两者应不同
    v.set_layer("original")
    app.processEvents()
    orig_png = os.path.join(ROOT, "tests", "_verify_original_shot.png")
    v.grab().save(orig_png)
    same = open(out_png, "rb").read() == open(orig_png, "rb").read()
    print(f"[ui] overlay 截图与原图截图 {'相同 ✗' if same else '不同 ✓'}", flush=True)
    assert not same, "overlay 显示内容与原图一致，问题仍存在"

    QTimer.singleShot(0, app.quit)
    app.exec()
    print("[ui] ImageViewer overlay 显示验证通过 ✓", flush=True)


if __name__ == "__main__":
    fresh = ensure_backend()
    out = step_detect()
    det_id = out.get("detection_id") or out.get("id")
    d = step_api(det_id)
    data = step_download(d)
    step_ui(data)
    print("\n[结论] 实时监控页叠加图链路验证通过", flush=True)
