"""REMAIN-02：UI↔API 离屏对照。

用 8031 真实 HTTP 响应填充学习页"当前特征集"卡片（PySide6 offscreen），
核对 UI 呈现与 API 字段一致，并截图留证。不做真实桌面交互。
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import json
from pathlib import Path

import requests

ROOT = Path(r"d:\CGAIC\AOI_SSOCL")
STABLE = ROOT / "stable_AOI_tree" / "AOI_Core"
TASK = ROOT / "review" / "task" / "remain_02"
RAW = TASK / "raw"
BASE = "http://127.0.0.1:8031"
KEYS = {"operator": "remain02-operator-key", "engineer": "remain02-engineer-key"}
CAT = "gold_finger"

sys.path.insert(0, str(STABLE))
os.chdir(STABLE)

from PySide6.QtWidgets import QApplication, QWidget  # noqa: E402
from ui.pages.learning_page import LearningPage  # noqa: E402

app = QApplication.instance() or QApplication([])

results = []
for role in ("operator", "engineer"):
    r = requests.get(f"{BASE}/api/models/feature-set/{CAT}",
                     headers={"X-API-Key": KEYS[role]}, timeout=30)
    fs = r.json()
    page = LearningPage.__new__(LearningPage)
    QWidget.__init__(page)
    card = page._build_feature_set_card()
    page._fill_feature_set(fs)
    card.resize(900, card.sizeHint().height())
    png = RAW / f"ui_feature_set_card_{role}.png"
    card.grab().save(str(png))
    # 一致性断言
    checks = {
        "http_status": r.status_code,
        "can_edit_api": fs.get("can_edit"),
        "btn_draft_enabled": page.btn_fs_draft.isEnabled(),
        "btn_report_enabled": page.btn_fs_report.isEnabled(),
        "btn_rebuild_enabled": page.btn_fs_rebuild.isEnabled(),
        "scoring_combo_enabled": page.combo_fs_scoring.isEnabled(),
        "status_text": page.lbl_fs_status.text(),
        "n_checks": len(page._fs_checks),
        "n_groups_api": len(fs.get("groups") or []),
        "all_checks_enabled_match_can_edit": all(
            cb.isEnabled() == bool(fs.get("can_edit"))
            for cb in page._fs_checks.values()),
        "pending_rebuild_version_api": fs.get("pending_rebuild_version"),
        "pending_shown_in_ui": ("待激活门控" in page.lbl_fs_status.text())
        if fs.get("pending_rebuild_version") else None,
        "screenshot": png.name,
    }
    ok = (checks["btn_draft_enabled"] == bool(fs.get("can_edit"))
          and checks["all_checks_enabled_match_can_edit"]
          and checks["n_checks"] == checks["n_groups_api"])
    checks["pass"] = ok
    results.append({"role": role, **checks})
    print(f"[{role}] can_edit={fs.get('can_edit')} draft_btn={checks['btn_draft_enabled']} "
          f"groups={checks['n_checks']}/{checks['n_groups_api']} "
          f"pending_v={fs.get('pending_rebuild_version')} -> {'PASS' if ok else 'FAIL'}",
          flush=True)
    card.deleteLater()
    page.deleteLater()

(RAW / "ui_offscreen.json").write_text(
    json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
allpass = all(x["pass"] for x in results)
print("UI_OFFSCREEN_PASS" if allpass else "UI_OFFSCREEN_FAIL", flush=True)
sys.exit(0 if allpass else 1)
