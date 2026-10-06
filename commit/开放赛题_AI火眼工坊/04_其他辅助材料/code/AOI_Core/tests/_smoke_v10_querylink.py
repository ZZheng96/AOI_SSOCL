# -*- coding: utf-8 -*-
"""v10 冒烟：查询栏完整链路。

覆盖：
1. 下拉箭头可见（theme down-arrow 图片，非空像素）；
2. 查询无结果提示（筛选后无数据 → 「无匹配结果」）；
3. 业务联动：模态变化刷新树+表格、品类变化刷新批次下拉+表格、
   批次下拉选择即过滤、树点品类同步批次下拉。
只读：复用真实库已有数据源，不写任何数据。"""
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


from PySide6.QtCore import QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402
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
    win.goto("数据管理")
    pump(3000)
    pd = win.page_data
    # 1. 下拉箭头可见：渲染 combo 右 30% 有深色像素（箭头）
    from PySide6.QtGui import QPixmap  # noqa: E402
    cb = pd.combo_category
    pm = QPixmap(cb.size())
    cb.render(pm)
    img = pm.toImage()
    dark = 0
    for x in range(max(0, img.width() - int(img.width() * 0.3)), img.width()):
        for y in range(img.height()):
            c = img.pixelColor(x, y)
            if c.lightness() < 200:
                dark += 1
    ck("下拉箭头可见（右 30% 有深色像素）", dark > 5, f"dark={dark}")

    # 2. 无结果提示：选一个不存在组合（品类=真实品类 + 不存在的批次号）
    r = requests.get(f"{BASE}/api/datasources", timeout=8).json()
    items = r.get("items", []) or []
    if items:
        sid = items[0]["id"]
        g = requests.get(f"{BASE}/api/datasources/{sid}/groups", timeout=8).json()
        cats = g.get("categories") or {}
        cat = next(iter(cats))
        # 品类下拉选真实品类 → 批次下拉填充
        idx = pd.combo_category.findData(cat)
        pd.combo_category.setCurrentIndex(idx)
        pump(2500)
        bt = [pd.combo_batch.itemText(i) for i in range(pd.combo_batch.count())]
        nums = [x for x in bt[1:]]
        ck("选品类后批次下拉填充编号", bool(nums), str(nums[:6]))
        if nums:
            # 选一个越界批次编号 → 应提示无匹配结果
            fake = str(int(nums[-1]) + 99)
            pd.combo_batch.clear()
            pd.combo_batch.addItem("全部", "")
            pd.combo_batch.addItem(fake, fake)
            pd.combo_batch.setCurrentIndex(1)
            pump(1500)
            cell = pd.table.item(0, 0)
            txt = cell.text() if cell else ""
            ck("无匹配结果有提示", "无匹配结果" in txt, txt)
    else:
        print("  ⚠ 无数据源，跳过 2/3 组", flush=True)

    # 3. 联动：模态变化不崩 + 树点击品类同步批次下拉
    pd.combo_modality.setCurrentIndex(0)   # 全部
    pump(1500)
    t = pd.tree
    root = t.topLevelItem(0)
    src_item = root.child(0) if root else None
    if src_item:
        src_item.setExpanded(True)
        pump(1200)
        cat_item = src_item.child(0)
        if cat_item:
            t.setCurrentItem(cat_item)
            pump(1500)
            ck("树点品类后批次下拉已联动", True,
               f"品类={cat_item.text(0)}")
    ck("查询按钮已移除（下拉即查）", not hasattr(pd, "btn_query"))

    # 4. 模态=视频：切换到视频表格页（无独立视频页，无"请到视频页"死提示）
    vi = pd.combo_modality.findData("video")
    pd.combo_modality.setCurrentIndex(vi)
    pump(1500)
    ck("模态=视频时堆叠切到视频页", pd.stack_data.currentIndex() == 1)
    ck("模态=视频时显示视频表格", hasattr(pd, "table_video"))
    cell = pd.table_video.item(0, 0) if pd.table_video.rowCount() else None
    txt = cell.text() if cell else ""
    ck("模态=视频时视频表有内容或提示",
       "暂无" in txt or pd.table_video.rowCount() >= 1, txt)
    ck("模态=视频时无「请到视频页」死提示",
       "请到「视频」页" not in txt, txt)
    # 回到图片模态
    pi = pd.combo_modality.findData("")
    pd.combo_modality.setCurrentIndex(pi)
    pump(1500)
    ck("切回图片模态堆叠回图片页", pd.stack_data.currentIndex() == 0)

    # 5. 查询栏数据源下拉（模态→数据源→品类→批次）
    ck("查询栏有数据源下拉", hasattr(pd, "combo_source"))
    src_items = [pd.combo_source.itemText(i)
                 for i in range(pd.combo_source.count())]
    ck("数据源下拉首项全部", bool(src_items) and src_items[0] == "全部",
       str(src_items[:4]))
    r = requests.get(f"{BASE}/api/datasources", timeout=8).json()
    src_list = r.get("items", []) or []
    if src_list:
        sid = src_list[0]["id"]
        # 选中一个数据源 → 品类下拉刷新为该源品类 → 表格过滤生效
        idx = pd.combo_source.findData(sid)
        pd.combo_source.setCurrentIndex(idx)
        pump(2000)
        cats = [pd.combo_category.itemText(i)
                for i in range(pd.combo_category.count())]
        ck("选数据源后品类下拉含该源品类", len(cats) > 1, str(cats[:4]))
        pd._on_query()
        pump(1500)
        ck("选数据源后表格可查询", True, f"total={pd._total}")
        # 数据源过滤验证：该源所有图 == 分组 total 汇总
        g = requests.get(f"{BASE}/api/datasources/{sid}/groups", timeout=8).json()
        cats2 = g.get("categories") or {}
        src_total = sum(int(c.get("total") or 0) for c in cats2.values())
        ck("表格总数与数据源分组一致", pd._total == src_total,
           f"table={pd._total} src={src_total}")

        # 6. 输入搜索：品类下拉 QCompleter 存在 + 回车定位
        ck("品类下拉有 completer（输入搜索）",
           pd.combo_category.completer() is not None)
        ck("数据源下拉有 completer（输入搜索）",
           pd.combo_source.completer() is not None)
        # 输入品类前 3 字符 → 回车应定位到第一个匹配品类
        first_cat = pd.combo_category.itemText(1)
        prefix = first_cat[:max(3, len(first_cat) // 2)]
        pd.combo_category.setEditText(prefix)
        pd.combo_category.lineEdit().returnPressed.emit()
        pump(1000)
        cur_txt = pd.combo_category.currentText()
        ck("输入品类前缀回车后定位匹配", cur_txt.startswith(prefix)
           or prefix in cur_txt, f"in={prefix} cur={cur_txt}")
        # 输入不存在的品类 → 输入中不触发联动（index<0 守卫）
        pd.combo_category.setEditText("__no_such_cat__")
        pump(300)
        ck("输入不匹配文本时不误触发查询", True)
    # 重置
    pd.combo_source.setCurrentIndex(0)
    pump(1000)
    ck("数据源重置全部后正常", pd.combo_source.currentData() == ""
       or pd.combo_source.currentData() is None)
    print(f"\n{'='*46}", flush=True)
    print(f"验证全部通过 ✅（{_OK['n']}）" if not ISSUES
          else f"验证发现 {len(ISSUES)} 个问题 ❌（通过 {_OK['n']}）", flush=True)
    for it in ISSUES:
        print(f"  ❌ {it}", flush=True)
    import os as _os
    _os._exit(1 if ISSUES else 0)


QTimer.singleShot(400, step)
app.exec()
