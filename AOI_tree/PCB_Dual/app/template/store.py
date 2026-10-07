"""模板库：以标准图为基本单位，templates/{id}/ 独立管理。"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
from glob import escape as glob_escape
from pathlib import Path
from typing import Iterable

from app.calibration.store import load_calib, save_calib
from app.config import TEMPLATES_DIR, get_gate_settings
from app.template.model import GateConfig, InspectionTemplate, StandardImageRef
from app.template.regions import count_regions, region_sets_from_calibration

_SAFE_RE = re.compile(r'[\\/:*?"<>|]')


def safe_id(name: str) -> str:
    text = (name or "").strip() or "template"
    return _SAFE_RE.sub("_", text)


class TemplateStore:
    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root else TEMPLATES_DIR
        self.root.mkdir(parents=True, exist_ok=True)

    def _dir(self, template_id: str) -> Path:
        return self.root / safe_id(template_id)

    def _json_path(self, template_id: str) -> Path:
        return self._dir(template_id) / "template.json"

    def list_ids(self) -> list[str]:
        if not self.root.exists():
            return []
        ids = []
        for path in sorted(self.root.iterdir()):
            if path.is_dir() and (path / "template.json").exists():
                ids.append(path.name)
        return ids

    def list_templates(
        self,
        *,
        status: str | None = None,
        category: str | None = None,
    ) -> list[InspectionTemplate]:
        items: list[InspectionTemplate] = []
        for tid in self.list_ids():
            tpl = self.load(tid)
            if tpl is None:
                continue
            if status and tpl.status != status:
                continue
            if category and tpl.category != category:
                continue
            items.append(tpl)
        items.sort(key=lambda t: (t.status != "published", t.display_name.lower(), t.id))
        return items

    def exists(self, template_id: str) -> bool:
        return self._json_path(template_id).exists()

    def find_by_standard_path(self, image_path: str | Path) -> InspectionTemplate | None:
        """按标准图绝对路径或内容哈希查找已有模板（一图一模板）。"""
        src = Path(image_path) if image_path else None
        if src is None:
            return None
        target = ""
        try:
            if src.exists():
                target = str(src.resolve())
        except OSError:
            target = str(src)
        sha1 = self._file_sha1(src) if src.exists() else ""
        for tid in self.list_ids():
            tpl = self.load(tid)
            if tpl is None or not tpl.standard_image.path:
                continue
            if sha1 and tpl.standard_image.sha1 and tpl.standard_image.sha1 == sha1:
                return tpl
            try:
                if target and str(Path(tpl.standard_image.path).resolve()) == target:
                    return tpl
            except OSError:
                if tpl.standard_image.path == str(image_path):
                    return tpl
        return None

    def load(self, template_id: str) -> InspectionTemplate | None:
        path = self._json_path(template_id)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        if not isinstance(data, dict):
            return None
        tpl = InspectionTemplate.from_dict(data)
        if not tpl.id:
            tpl.id = safe_id(template_id)
        # 迁移：旧模板可能 calib_key != id，统一到 id
        if tpl.calib_key != tpl.id:
            legacy = load_calib(tpl.calib_key)
            if legacy and not tpl.calibration:
                tpl.calibration = dict(legacy)
                tpl.sync_regions_from_calibration()
            tpl.calib_key = tpl.id
        if not tpl.region_sets and tpl.calibration:
            tpl.sync_regions_from_calibration()
        return tpl

    def save(self, template: InspectionTemplate, *, sync_legacy: bool = True) -> Path:
        """保存草稿/已发布模板。标定键固定为 template.id。"""
        if not template.id:
            raise ValueError("template.id 不能为空")
        template.id = safe_id(template.id)
        template.calib_key = template.id
        if template.region_sets:
            template.sync_calibration_from_regions()
        elif template.calibration and not template.region_sets:
            template.sync_regions_from_calibration()
        template.touch()

        folder = self._dir(template.id)
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / "template.json"
        path.write_text(json.dumps(template.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

        if sync_legacy:
            self.sync_calibration_to_legacy(template)
        return path

    def delete(self, template_id: str) -> bool:
        folder = self._dir(template_id)
        if not folder.exists():
            return False
        shutil.rmtree(folder)
        self._purge_side_files(safe_id(template_id))
        return True

    @staticmethod
    def _purge_side_files(tid: str) -> None:
        """级联清理以模板 id 为键的外部文件：模板专属检测参数 + 旧版标定。"""
        from app.calibration.store import _path_for as calib_path
        from app.recipe.param_store import ParamStore

        ta_dir = ParamStore().root / "template_algorithm"
        if ta_dir.is_dir():
            for p in ta_dir.glob(f"{glob_escape(tid)}__*.json"):
                try:
                    owner = json.loads(p.read_text(encoding="utf-8")).get("template_name")
                except (json.JSONDecodeError, OSError, AttributeError):
                    continue
                if owner == tid:  # 防止 "A" 误删 "A__x" 模板的参数
                    p.unlink(missing_ok=True)
        calib_path(tid).unlink(missing_ok=True)

    def create_from_standard_image(
        self,
        standard_image_path: str | Path,
        *,
        display_name: str = "",
        category: str = "",
        notes: str = "",
        algorithm_ids: Iterable[str] | None = None,
        copy_image: bool = True,
    ) -> InspectionTemplate:
        """以标准图为基本单位新建独立模板（一图一模板）。"""
        src = Path(standard_image_path)
        if not src.exists():
            raise FileNotFoundError(f"标准图不存在：{src}")

        existing = self.find_by_standard_path(src)
        if existing is not None:
            raise ValueError(
                f"标准图已绑定模板「{existing.display_name}」（{existing.id}），"
                "请直接编辑该模板，或选用其它标准图。"
            )

        stem = src.stem or "template"
        tid = safe_id(stem)
        base = tid
        n = 2
        while self.exists(tid):
            tid = f"{base}_{n}"
            n += 1

        gate_settings = get_gate_settings()
        tpl = InspectionTemplate(
            id=tid,
            display_name=display_name or stem,
            status="draft",
            category=category,
            notes=notes,
            calib_key=tid,
            gate=GateConfig(
                size_gate_enabled=bool(gate_settings.get("gate_enabled", True)),
                size_gate_max_diff_px=int(gate_settings.get("size_gate_max_diff_px", 10)),
            ),
        )
        if algorithm_ids:
            tpl.set_algorithm_ids(list(algorithm_ids))

        folder = self._dir(tid)
        folder.mkdir(parents=True, exist_ok=True)
        if copy_image:
            dest = folder / f"standard{src.suffix.lower() or '.png'}"
            shutil.copy2(src, dest)
            tpl.standard_image = self._make_image_ref(dest)
            tpl.standard_image.source_name = src.name
        else:
            tpl.standard_image = self._make_image_ref(src)

        self.save(tpl)
        return tpl

    def create_draft(
        self,
        *,
        template_id: str | None = None,
        display_name: str = "",
        standard_image_path: str = "",
        category: str = "",
        notes: str = "",
        algorithm_ids: Iterable[str] | None = None,
        calib_key: str | None = None,  # noqa: ARG002 — 忽略，强制等于 id
    ) -> InspectionTemplate:
        """兼容旧调用：有标准图时走 create_from_standard_image。"""
        if standard_image_path:
            return self.create_from_standard_image(
                standard_image_path,
                display_name=display_name or Path(standard_image_path).stem,
                category=category,
                notes=notes,
                algorithm_ids=algorithm_ids,
                copy_image=True,
            )

        tid = safe_id(template_id or display_name or "template")
        base = tid
        n = 2
        while self.exists(tid):
            tid = f"{base}_{n}"
            n += 1

        gate_settings = get_gate_settings()
        tpl = InspectionTemplate(
            id=tid,
            display_name=display_name or tid,
            status="draft",
            category=category,
            notes=notes,
            calib_key=tid,
            gate=GateConfig(
                size_gate_enabled=bool(gate_settings.get("gate_enabled", True)),
                size_gate_max_diff_px=int(gate_settings.get("size_gate_max_diff_px", 10)),
            ),
        )
        if algorithm_ids:
            tpl.set_algorithm_ids(list(algorithm_ids))
        self.save(tpl)
        return tpl

    def publish(self, template: InspectionTemplate) -> InspectionTemplate:
        """发布模板。已发布模板再次发布时版本号 +1。"""
        issues = self.publish_checklist(template)
        blockers = [i for i in issues if i.startswith("[阻断]")]
        if blockers:
            raise ValueError("\n".join(blockers))
        was_published = template.status == "published"
        if was_published:
            template.version = int(template.version) + 1
        else:
            template.version = max(1, int(template.version or 1))
        template.status = "published"
        template.touch()
        self.save(template)
        return template

    def publish_checklist(self, template: InspectionTemplate) -> list[str]:
        """返回检查项文案；以 [阻断] 开头的为发布硬门槛。"""
        issues: list[str] = []
        from app.detect.catalog import get_catalog

        catalog = get_catalog()
        path = Path(template.standard_image.path) if template.standard_image.path else None
        need_std = self._requires_standard(template.enabled_algorithm_ids())
        if need_std:
            if not path or not path.exists():
                issues.append("[阻断] 标准图路径无效或不存在，请重新选择标准图创建模板")
        enabled = [a for a in template.enabled_algorithm_ids() if catalog.is_job_item(a)]
        if not enabled:
            issues.append("[阻断] 未勾选任何检测项（缺陷类型）")
        majors = self._major_groups(enabled)
        if len(majors) > 1:
            issues.append("[阻断] 一个模板只能绑定同一算法大类：当前为 " + "、".join(majors))

        # 检测区域：启用项所需区域种类至少有一处框选（表面类可无区域）
        needed_targets: set[str] = set()
        for item in template.enabled_detection_items():
            if not catalog.is_job_item(item.algorithm_id):
                continue
            from app.template.regions import region_targets_for_kind

            kind = item.region_kind or catalog.region_kind(item.algorithm_id)
            for t in region_targets_for_kind(kind if kind != "none" else ""):
                needed_targets.add(t)
        # smt 只要 pads 有即可；toe/rim 可选
        soft_optional = {"smt_toe", "smt_rim"}
        hard_targets = {t for t in needed_targets if t not in soft_optional}
        missing = [t for t in sorted(hard_targets) if not template.get_region_shapes(t)]
        if hard_targets and missing:
            # 仍交给 adapter.is_ready 做最终判定；这里给可读提示
            if count_regions(template.region_sets) == 0:
                issues.append("[阻断] 未框选检测区域，请在标准图上标定")

        from app.detect.registry import get_adapter_for
        from app.detect.roi import rois_from_region_sets

        self.sync_calibration_to_legacy(template)
        labels = catalog.labels()
        for alg_id in enabled:
            adapter = get_adapter_for(alg_id)
            if adapter is None:
                issues.append(f"[警告] 算法未加载适配器：{labels.get(alg_id, alg_id)}（产线检测将失败）")
                continue
            kind = catalog.region_kind(alg_id)
            item_rois = rois_from_region_sets(template.region_sets, kind if kind != "none" else None)
            ready, reason = adapter.is_ready(alg_id, template.calib_key, rois=item_rois)
            if not ready:
                issues.append(f"[阻断] {labels.get(alg_id, alg_id)} 标定未就绪：{reason}")

        # 引擎模式与品类模型绑定：feature/dual 缺绑定或模型未准备，产线检测必判 ERROR
        mode = template.engine_mode if template.engine_mode in ("traditional", "feature", "dual") else "dual"
        if mode in ("feature", "dual"):
            bound = (template.model_category or "").strip() or (template.category or "").strip()
            if not bound:
                issues.append(
                    f"[阻断] 引擎模式为「{mode}」但未绑定品类模型，请在右侧表单选择/填写品类模型"
                )
            elif not self._category_model_ready(bound):
                issues.append(
                    f"[阻断] 品类模型「{bound}」尚未准备，请先在「特征学习（AOI_Core）」页完成准备"
                )
        if not template.display_name.strip():
            issues.append("[警告] 显示名称为空")
        return issues

    @staticmethod
    def _category_model_ready(category: str) -> bool:
        """品类模型是否已在树干 AOI_Core 准备并激活。"""
        try:
            from app.engines.feature import get_engine

            return get_engine().current_version(category) is not None
        except Exception:  # noqa: BLE001 Core 不可达/判定失败时按未准备处理，避免带病发布
            return False

    def sync_calibration_to_legacy(self, template: InspectionTemplate) -> None:
        template.calib_key = template.id
        if template.region_sets:
            template.sync_calibration_from_regions()
        save_calib(template.calib_key, dict(template.calibration or {}))

    def pull_calibration_from_legacy(self, template: InspectionTemplate) -> None:
        template.calib_key = template.id
        template.calibration = dict(load_calib(template.calib_key) or {})
        template.sync_regions_from_calibration()

    def snapshot_for_task(self, template: InspectionTemplate, task_id: int) -> dict:
        """将标准图与模板数据一并冻结到任务私有目录。"""
        snapshot = template.to_dict()
        ref = snapshot["standard_image"]
        src = Path(ref.get("path") or "")
        if not ref.get("path") or not src.is_file():
            raise ValueError(f"标准图不存在：{ref.get('path') or '(空)'}")
        folder = self.root / ".task_snapshots" / str(task_id)
        folder.mkdir(parents=True, exist_ok=False)
        try:
            expected = self._file_sha1(src)
            if ref.get("sha1") and ref["sha1"] != expected:
                raise ValueError("标准图与模板记录的哈希不一致")
            dest = folder / f"standard{src.suffix.lower()}"
            shutil.copyfile(src, dest)
            if self._file_sha1(src) != expected or self._file_sha1(dest) != expected:
                raise ValueError("标准图复制期间发生变化或快照校验失败")
            ref["path"] = str(dest.resolve())
            ref["sha1"] = expected
            return snapshot
        except Exception:
            self.cleanup_task_snapshot(task_id)
            raise

    def cleanup_task_snapshot(self, task_id: int) -> None:
        shutil.rmtree(self.root / ".task_snapshots" / str(task_id), ignore_errors=True)

    @classmethod
    def verify_task_snapshot(cls, snapshot: dict) -> None:
        ref = snapshot.get("standard_image") or {}
        path = ref.get("path")
        expected = ref.get("sha1")
        if not path or not expected or cls._file_sha1(Path(path)) != expected:
            raise RuntimeError("任务标准图快照缺失或哈希校验失败")

    def validate_standard_image(self, template: InspectionTemplate) -> tuple[bool, str]:
        path = template.standard_image.path
        if not path:
            return False, "未设置标准图路径"
        p = Path(path)
        if not p.exists():
            return False, f"标准图不存在：{path}"
        return True, ""

    def rebind_standard_image(self, template: InspectionTemplate, new_path: str | Path) -> InspectionTemplate:
        """已废弃：禁止在同一模板内更换标准图。请改用 create_from_standard_image。"""
        raise RuntimeError(
            "同一模板不可更换标准图。请使用「选择标准图」新建独立模板，"
            f"当前模板「{template.display_name}」保持绑定 {template.standard_image.path}"
        )

    @classmethod
    def _file_sha1(cls, path: Path) -> str:
        if not path.exists():
            return ""
        h = hashlib.sha1()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    @classmethod
    def _make_image_ref(cls, path: str | Path) -> StandardImageRef:
        from app.utils.cv_io import imread_unicode

        p = Path(path)
        width = height = 0
        img = imread_unicode(p) if p.exists() else None
        if img is not None:
            height, width = img.shape[:2]
        sha1 = cls._file_sha1(p)
        return StandardImageRef(path=str(p.resolve() if p.exists() else p), sha1=sha1, width=width, height=height)

    @staticmethod
    def _requires_standard(algorithm_ids: list[str]) -> bool:
        from app.detect.catalog import get_catalog

        return get_catalog().requires_standard(list(algorithm_ids or []))

    @staticmethod
    def _major_groups(algorithm_ids: list[str]) -> list[str]:
        from app.detect.catalog import get_catalog

        catalog = get_catalog()
        groups: list[str] = []
        for alg_id in algorithm_ids:
            if not catalog.is_job_item(alg_id):
                continue
            g = catalog.group_name(alg_id)
            if g not in groups:
                groups.append(g)
        return groups


def _algorithm_label_map() -> dict[str, str]:
    from app.detect.catalog import get_catalog

    return get_catalog().labels()
