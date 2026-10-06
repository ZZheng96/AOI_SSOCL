from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from app.config import get_outputs_dir
from app.detect.types import DetectSummary
from app.utils.cv_io import imwrite_unicode


def _box_dict(box) -> dict:
    data = asdict(box)
    # 旧字段必须保留；新增字段允许缺省
    return {k: v for k, v in data.items() if v not in ("", None, []) or k in {"x", "y", "w", "h", "label"}}


def _result_dict(r) -> dict:
    return {
        "algorithm": r.algorithm,
        "item_id": r.item_id or r.algorithm,
        "display_name": r.display_name or r.algorithm,
        "ok": r.ok,
        "status": r.status or ("SKIP" if r.skipped else ("OK" if r.ok else "NG")),
        "message": r.message,
        "defect_type": r.defect_type,
        "defect_count": r.defect_count if r.defect_count else len(r.boxes),
        "elapsed_ms": r.elapsed_ms,
        "algorithm_version": r.algorithm_version,
        "error_code": r.error_code,
        "error_message": r.error_message,
        "skipped": r.skipped,
        "skip_reason": r.skip_reason,
        "hit_layer": r.hit_layer,
        "param_schema_version": r.param_schema_version,
        "boxes": [_box_dict(b) for b in r.boxes],
        "manual_verdict": r.manual_verdict,
    }


class ResultStore:
    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root else get_outputs_dir()
        self.root.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        template_name: str | None,
        standard_image: str,
        test_image: str,
        category: str | None,
        selected: list[str],
        summary: DetectSummary,
        *,
        template_id: str | None = None,
        template_version: int | None = None,
        rule_snapshot: dict | None = None,
        source: str | None = None,
    ) -> Path:
        now = datetime.now()
        day = now.strftime("%Y-%m-%d")
        stamp = now.strftime("%H%M%S_%f")[:-3]  # 毫秒级：同一秒同图重检会撞目录互相覆盖
        stem = Path(test_image).stem if test_image else "test"
        folder = self.root / day / f"{stamp}_{stem}"
        folder.mkdir(parents=True, exist_ok=True)

        vis_name = "result_vis.png"
        if summary.vis_image is not None:
            imwrite_unicode(folder / vis_name, summary.vis_image)

        overall = summary.overall or ("ERROR" if summary.gate_blocked else ("OK" if summary.overall_ok else "NG"))
        payload = {
            "time": now.isoformat(timespec="seconds"),
            "template_name": template_name,
            "template_id": template_id or template_name,
            "template_version": template_version or summary.template_version,
            "standard_image": standard_image,
            "test_image": test_image,
            "category": category,
            "overall": overall,
            "elapsed_ms": summary.elapsed_ms,
            "ng_count": summary.ng_count,
            "source": source or summary.source or "inspect",
            "algorithms_selected": selected,
            "rule_snapshot": rule_snapshot,
            "ng_list": [
                {
                    "algorithm": r.algorithm,
                    "item_id": r.item_id or r.algorithm,
                    "display_name": r.display_name,
                    "defect_type": r.defect_type,
                    "defect_count": r.defect_count,
                    "message": r.message,
                    "boxes": [_box_dict(b) for b in r.boxes],
                }
                for r in summary.ng_list
            ],
            "results": [_result_dict(r) for r in summary.results],
            "vis_file": vis_name,
            "gate_blocked": summary.gate_blocked,
            "gate_message": summary.gate_message,
            "gate_note": summary.gate_note,
            "manual_overall_verdict": None,
        }
        (folder / "result.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return folder

    def list_records(self, *, include_debug: bool = False,
                     limit: int | None = None) -> list[dict]:
        """列出历史归档（按时间倒序）。limit 限制返回条数，避免全量扫描卡死历史页。"""
        records = []
        if not self.root.exists():
            return records
        for json_path in sorted(self.root.rglob("result.json"), reverse=True):
            try:
                data = json.loads(json_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            source = str(data.get("source") or "inspect")
            if not include_debug and source in {"debug", "trial"}:
                continue
            data["_dir"] = str(json_path.parent)
            data["_json"] = str(json_path)
            records.append(data)
            if limit is not None and len(records) >= limit:
                break
        return records

    def delete_record(self, folder: str | Path) -> None:
        import shutil

        path = Path(folder)
        if path.exists() and path.is_dir():
            shutil.rmtree(path)

    def set_manual_verdict(
        self,
        folder: str | Path,
        algorithm: str | None,
        verdict: str | None,
        note: str = "",
    ) -> bool:
        """人工复判：algorithm=None 时设置整体复判结论。"""
        json_path = Path(folder) / "result.json"
        if not json_path.exists():
            return False
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return False

        if algorithm is None:
            data["manual_overall_verdict"] = verdict
            data["manual_overall_note"] = note
        else:
            found = False
            for item in data.get("results", []):
                if item.get("algorithm") == algorithm or item.get("item_id") == algorithm:
                    item["manual_verdict"] = verdict
                    item["manual_verdict_note"] = note
                    found = True
            if not found:
                return False

        json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return True
