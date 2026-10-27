"""L1 判定条：整板结论 + NG 项数 + 总耗时。各结果页共用。"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QGridLayout, QHBoxLayout, QLabel, QVBoxLayout, QWidget

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

        self.dual_panel = QWidget()
        self.dual_grid = QGridLayout(self.dual_panel)
        self.dual_grid.setContentsMargins(0, 0, 0, 0)
        self.dual_grid.setHorizontalSpacing(12)
        self.dual_grid.setVerticalSpacing(3)
        self.dual_labels: list[QLabel] = []
        for col, title in enumerate(("传统引擎", "特征引擎")):
            title_label = QLabel(title)
            title_label.setProperty("class", "SectionLabel")
            self.dual_grid.addWidget(title_label, 0, col)
            value_label = QLabel("—")
            value_label.setWordWrap(True)
            self.dual_grid.addWidget(value_label, 1, col)
            self.dual_labels.append(value_label)
        root.addWidget(self.dual_panel)
        self.dual_panel.setVisible(False)

        self.clear()

    def clear(self) -> None:
        self.set_status("", "等待检测", "")
        self.dual_panel.setVisible(False)
        for label in self.dual_labels:
            label.setText("—")

    @staticmethod
    def _dual_text(data: dict, title: str) -> str:
        status = str(data.get("overall") or "ERROR")
        score = data.get("score")
        score_text = f" · 分数 {float(score):.4f}" if score is not None else ""
        elapsed = data.get("elapsed_ms")
        elapsed_text = f" · {elapsed:g} ms" if isinstance(elapsed, (int, float)) and elapsed else ""
        error = str(data.get("error") or "")
        suffix = f" · {error}" if error else ""
        return f"{status}{score_text}{elapsed_text}{suffix}"

    def _bind_dual(self, metadata: dict) -> None:
        dual = metadata.get("dual")
        if not dual:
            self.dual_panel.setVisible(False)
            return
        traditional = dict(metadata.get("traditional") or {})
        feature = dict(metadata.get("feature") or {})
        self.dual_labels[0].setText(self._dual_text(traditional, "传统引擎"))
        self.dual_labels[1].setText(self._dual_text(feature, "特征引擎"))
        fusion = str(metadata.get("fusion_overall") or "ERROR")
        if traditional.get("overall") != feature.get("overall"):
            self.lbl_context.setText((self.lbl_context.text() + "  ·  双引擎分歧，请复核").strip(" ·"))
            self.lbl_context.setVisible(True)
        self.dual_panel.setVisible(True)

    def set_running(self, text: str = "检测中…") -> None:
        self.set_status("RUNNING", text, "")
        self.dual_panel.setVisible(False)
        for label in self.dual_labels:
            label.setText("—")

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
            self._bind_dual(summary.metadata if summary else {})
            return
        assert summary is not None
        overall = summary.overall or ("OK" if summary.overall_ok else "NG")
        ng = summary.ng_count if summary.ng_count else len(summary.ng_list)
        elapsed = f"{summary.elapsed_ms} ms" if summary.elapsed_ms else ""
        name = Path(test_path).name if test_path else ""
        if overall in {"GRAY", "REVIEW"}:
            defect_note = "待人工复检"
        elif overall == "SKIP":
            defect_note = "未执行 / 条件不满足"
        else:
            defect_note = f"NG {ng} 项" if overall != "OK" else "无缺陷"
        if summary.requires_review and overall not in {"GRAY", "REVIEW", "SKIP"}:
            defect_note += "；另有未执行 / 待复检项"
        meta_parts = [p for p in (defect_note, elapsed, name, extra) if p]
        ctx_parts = []
        if template_name:
            ver = f" v{template_version}" if template_version is not None else ""
            ctx_parts.append(f"模板 {template_name}{ver}")
        self.set_status(overall, overall, "  ·  ".join(meta_parts), context="  ·  ".join(ctx_parts))
        self._bind_dual(summary.metadata)
