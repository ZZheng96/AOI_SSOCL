"""L1 判定条：整板结论 + NG 项数 + 总耗时。各结果页共用。"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget

from app.detect.types import DetectSummary
from app.ui import theme
from app.ui.widgets import StatusPill


_STATUS_COLOR = {
    "OK": theme.OK,
    "NG": theme.NG,
    "GRAY": theme.REVIEW,  # 灰区待复判：黄，绝不显示绿色通过
    "ERROR": theme.ERROR,
    "REVIEW": theme.REVIEW,
    "SKIP": theme.SKIP,
    "RUNNING": theme.RUNNING,
}


class ResultHud(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(6)

        row = QHBoxLayout()
        row.setSpacing(12)
        self.lbl_verdict = QLabel("—")
        self.lbl_verdict.setObjectName("VerdictHero")
        self.lbl_verdict.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        row.addWidget(self.lbl_verdict, 0)
        self.pill = StatusPill("")
        self.pill.setVisible(False)
        row.addWidget(self.pill, 0, Qt.AlignVCenter)
        row.addStretch(1)
        root.addLayout(row)

        self.lbl_meta = QLabel("")
        self.lbl_meta.setProperty("class", "Hint")
        self.lbl_meta.setWordWrap(True)
        root.addWidget(self.lbl_meta)

        self.lbl_context = QLabel("")
        self.lbl_context.setProperty("class", "Hint")
        self.lbl_context.setWordWrap(True)
        self.lbl_context.setVisible(False)
        root.addWidget(self.lbl_context)

        self.clear()

    def clear(self) -> None:
        self.set_status("", "等待检测", "")

    def set_running(self, text: str = "检测中…") -> None:
        self.set_status("RUNNING", text, "")

    def set_status(self, status: str, headline: str, meta: str, *, context: str = "") -> None:
        color = _STATUS_COLOR.get(status, theme.TEXT)
        self.lbl_verdict.setText(headline or "—")
        self.lbl_verdict.setStyleSheet(f"color: {color}; font-size: 28px; font-weight: 800;")
        self.lbl_meta.setText(meta)
        self.lbl_context.setText(context)
        self.lbl_context.setVisible(bool(context))

    def bind_summary(
        self,
        summary: DetectSummary | None,
        *,
        error: str = "",
        test_path: str = "",
        template_name: str = "",
        template_version: int | None = None,
        extra: str = "",
    ) -> None:
        if summary is None and not error:
            self.clear()
            return
        if error or (summary and (summary.gate_blocked or summary.overall == "ERROR")):
            msg = error or (summary.gate_message if summary else "") or "失败"
            elapsed = f"{summary.elapsed_ms} ms" if summary and summary.elapsed_ms else ""
            self.set_status("ERROR", "ERROR", "  ·  ".join(p for p in (msg, elapsed) if p))
            return
        assert summary is not None
        overall = summary.overall or ("OK" if summary.overall_ok else "NG")
        ng = summary.ng_count if summary.ng_count else len(summary.ng_list)
        elapsed = f"{summary.elapsed_ms} ms" if summary.elapsed_ms else ""
        name = Path(test_path).name if test_path else ""
        if overall == "GRAY":
            defect_note = "灰区待复判"
        else:
            defect_note = f"NG {ng} 项" if overall != "OK" else "无缺陷"
        meta_parts = [p for p in (defect_note, elapsed, name, extra) if p]
        ctx_parts = []
        if template_name:
            ver = f" v{template_version}" if template_version is not None else ""
            ctx_parts.append(f"模板 {template_name}{ver}")
        self.set_status(overall, overall, "  ·  ".join(meta_parts), context="  ·  ".join(ctx_parts))
