"""Round2 发布运维（AC-011～016/028～035/060/061）：release_ops 只读审计/计划/门禁。

全部合成数据 / fake 对象 / TEMP 目录；不碰生产 DB、快照、生产指针。
"""
import inspect
import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from algo import persist
from algo import release_ops as ro
from algo.decision.hierarchical import box_evidence
from algo.feature_revision import FeatureSetRevision, PARITY_CHECKS, production_hashes
from algo.features import registry as freg
from algo.ssocl.banks import NormalBank, DefectBank
from algo.ssocl.incubate import HeadManager

N = 12


def _p(i, msg):
    print(f"[{i}/{N}] {msg}", flush=True)


def _manifest(keys=("glcm", "edge"), parent=None):
    pos, groups = 0, []
    for k in keys:
        d = int(freg.group(k)["dim"])
        groups.append({"key": k, "dim": d, "slice": [pos, pos + d],
                       "params": dict(freg.group(k).get("params") or {})})
        pos += d
    return {"schema_version": 1, "groups": groups, "dimension": pos,
            "preprocess": {"max_side": 256}, "implementation_hashes": {"t": "a" * 64},
            "seed": 42, "dataset_split": {"test": [{"sha256": "b" * 64, "label": 0}]},
            "model_snapshot": {"category": "tmp", "version": 1,
                               "artifact_hashes": {"bank": "c" * 64}},
            "parent": parent}


class ReleaseOpsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="aoi_round2_release_")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    # ------------------------------------------------ AC-011
    def test_group_selection_plan(self):
        _p(1, "AC-011 组级独立选组：局部视图不被整图否定、样本不足/反向不选、不自动生效")
        res = [{"category": "c", "defect": "scratch", "view": "global",
                "groups": {"glcm": {"auroc": 0.52, "n": 5}, "edge": {"auroc": 0.40, "n": 5}}},
               {"category": "c", "defect": "scratch", "view": "tile",
                "groups": {"glcm": {"auroc": 0.80, "n": 5}, "hog": {"auroc": 0.9, "n": 1}}}]
        out = ro.group_selection_plan(res)
        g, t = out["plans"]["c/scratch/global"], out["plans"]["c/scratch/tile"]
        self.assertEqual(g["status"], "keep_current")
        self.assertEqual(g["rejected"], {"glcm": "weak", "edge": "reversed"})
        self.assertEqual(t["selected"], ["glcm"])
        self.assertEqual(t["rejected"]["hog"], "insufficient")
        self.assertEqual(t["overlay"]["trad"]["scoring"], "grouped")
        self.assertFalse(out["auto_apply"])

    # ------------------------------------------------ AC-012
    def test_profile_lineage_completeness(self):
        _p(2, "AC-012 档案血缘 code/data/config hash + 完整性/小样本判定")
        grp = {f: 0.6 for f in ro.PROFILE_GROUP_FIELDS}
        prof = {"status": "available", "full_ci95": [0.5, 0.7], "cluster_loo": {},
                "groups": {"glcm": dict(grp)},
                "meta": {"registry_version": 1, "registry_sha256": freg.registry_hash(),
                         "category": "c", "data_source": "synthetic", "config": {"k": 1},
                         "n_normal": 2, "n_defect": 5}}
        self.assertEqual(ro.profile_completeness(prof)["status"], "incomplete")
        st = ro.stamp_profile_lineage(prof, [("x" * 64, 0), ("y" * 64, 1)], code_hash="d" * 64)
        self.assertNotIn("code_sha256", prof["meta"])           # 不改原档案
        self.assertEqual(st["meta"]["data_hash"],
                         ro.stamp_profile_lineage(prof, [("y" * 64, 1), ("x" * 64, 0)],
                                                  code_hash="d" * 64)["meta"]["data_hash"])
        c = ro.profile_completeness(st)
        self.assertEqual((c["status"], c["small_sample"]), ("insufficient", True))
        st["meta"]["n_normal"] = 5
        self.assertEqual(ro.profile_completeness(st)["status"], "ok")
        del st["groups"]["glcm"]["permutation_auroc"]
        self.assertIn("glcm.permutation_auroc", ro.profile_completeness(st)["gaps"])
        self.assertEqual(ro.report_status(st), "ok")
        self.assertEqual(ro.report_status(st, current_registry_sha="e" * 64), "stale")
        self.assertEqual(ro.report_status(None), "missing")

    # ------------------------------------------------ AC-013
    def test_trad_protocol_audit(self):
        _p(3, "AC-013 trad 打分协议如实描述：legacy 全向量 L2 / grouped 组距离加权")
        legacy = SimpleNamespace(scoring="legacy", group_weight={}, groups=["glcm"],
                                 group_spec="all", selection_report=None)
        a = ro.trad_protocol_audit(legacy)
        self.assertIn("单一 L2", a["distance"])
        self.assertFalse(a["test_used_for_selection"])
        grouped = SimpleNamespace(
            scoring="grouped", group_weight={"glcm": 1.0, "edge": 0.0}, groups=["glcm", "edge"],
            group_spec="auto", selection_report={"near_const_dims": {"edge": 3},
                                                 "constant_groups": ["edge"], "data": "fit_dev",
                                                 "auto": {"status": "selected",
                                                          "reversed": ["hog"]}})
        b = ro.trad_protocol_audit(grouped)
        self.assertIn("组距离加权和", b["distance"])
        self.assertEqual(b["zero_weight_groups"], ["edge"])
        self.assertEqual(b["reversed_groups"], ["hog"])
        self.assertEqual(b["selection_data"], "fit_dev")

    # ------------------------------------------------ AC-014
    def test_box_attribution_audit(self):
        _p(4, "AC-014 逐框归因：区域证据 vs 整图回退，trad explain 参考")
        a = np.zeros((32, 32), np.float32)
        a[4:10, 4:10] = 1.0
        boxes = [{"bbox": [4, 4, 10, 10]}, {"bbox": [20, 20, 28, 28]}]
        box_evidence(boxes, {"a": a}, weights={"a": 1.0})
        out = ro.box_attribution_audit(boxes, {"group_breakdown": {"glcm": 1.0},
                                               "top_dims": [], "x": 1})
        self.assertEqual((out["n_region"], out["n_image_fallback"]), (1, 1))
        self.assertTrue(out["boxes"][1]["copied_from_image"])
        self.assertEqual(set(out["trad_reference"]), {"group_breakdown", "top_dims"})

    # ------------------------------------------------ AC-015 / AC-035
    def test_snapshot_completeness_matches_persist(self):
        _p(5, "AC-035 快照盘点字段 = persist 实际写入字段；未绑定 revision 不可复现")
        src = inspect.getsource(persist.write_snapshot_state) + \
            inspect.getsource(persist.finalize_snapshot)
        for k in ro.SNAPSHOT_REQUIRED:
            self.assertIn(f'"{k}"', src, f"persist 未写入 {k}")
        meta = {k: "x" for k in ro.SNAPSHOT_REQUIRED}
        meta.update(components={"bank": {"present": True}, "router": {"present": False}},
                    cache_inventory=dict(persist.CACHE_INVENTORY),
                    feature_revision_binding={"status": "unknown"})
        r = ro.snapshot_completeness(meta)
        self.assertEqual((r["status"], r["absent_components"], r["reproducible"]),
                         ("ok", ["router"], False))
        meta["feature_revision_binding"] = {"status": "bound"}
        self.assertTrue(ro.snapshot_completeness(meta)["reproducible"])
        del meta["rng"]
        self.assertEqual(ro.snapshot_completeness(meta)["missing"], ["rng"])

    def test_representation_change_plan(self):
        _p(6, "AC-015 表示变化 → 隔离旧库/重提/重建/重校准/G2；仅配置变 → 重校准")
        old = {"backbone_sha256": "a", "code_sha256": "c", "cfg_hash": "f",
               "production_feature_metadata": {"groups": ["glcm"]}}
        new = dict(old, backbone_sha256="b")
        r = ro.representation_change_plan(old, new)
        self.assertTrue(r["representation_changed"])
        self.assertEqual(r["steps"][0], "quarantine_old_banks")
        self.assertIs(r["mix_old_vectors"], False)
        r2 = ro.representation_change_plan(old, dict(old, cfg_hash="g"))
        self.assertEqual(r2["steps"], ["recalibrate", "g2_regression", "approve_then_bind"])
        r3 = ro.representation_change_plan(
            old, dict(old, production_feature_metadata={"groups": ["edge"]}))
        self.assertIn("feature_metadata", r3["changed"])
        self.assertEqual(ro.representation_change_plan(old, old)["steps"], [])

    # ------------------------------------------------ AC-016 / AC-061
    def _activate(self, store, rid, manifest):
        ev = {"revision_id": rid, "eval_run_id": "r", "status": "completed",
              "independent": True, "acceptance": "passed", "auroc": 0.9,
              **{k: deepcopy(manifest[k]) for k in ("seed", "dataset_split", "model_snapshot")}}
        par = {**ev, "status": "passed", "checks": dict.fromkeys(PARITY_CHECKS, True),
               "baseline_implementation_hashes": production_hashes(), "protocol_hash": "d" * 64,
               "numeric": {"n_samples": 3, "max_abs_diff": 0.0, "atol": 1e-6}}
        ep, pp = self.root / f"e_{rid[:8]}.json", self.root / f"p_{rid[:8]}.json"
        ep.write_text(json.dumps(ev), encoding="utf-8")
        pp.write_text(json.dumps(par), encoding="utf-8")
        store.transition(rid, "evaluated", actor="t", evidence_path=ep)
        store.transition(rid, "approved", actor="t")
        store.transition(rid, "active", actor="t", parity_path=pp)

    def test_feature_ops_toggle_bind_rollback(self):
        _p(7, "AC-016/061 角色只读、启停→draft(parent)、绑定/血缘/回滚/Detection 关联")
        store = FeatureSetRevision(self.root / "registry.json")
        base = _manifest()
        rid0 = store.register(base)
        with self.assertRaises(PermissionError):
            ro.submit_toggle(store, rid0, ["hog"], ["edge"], role="operator")
        with self.assertRaises(ValueError):
            ro.submit_toggle(store, rid0, [], ["glcm", "edge"], role="engineer")
        t = ro.submit_toggle(store, rid0, ["hog"], ["edge"], role="engineer")
        rid1 = t["revision_id"]
        self.assertEqual((t["state"], t["parent"], t["production_changed"]),
                         ("draft", rid0, False))
        m1 = store.list()[rid1]["manifest"]
        self.assertEqual([g["key"] for g in m1["groups"]], ["glcm", "hog"])
        self.assertEqual(store.lineage(rid1), [rid1, rid0])
        v = ro.feature_ops_view(store, role="operator")
        self.assertFalse(v["editable"])
        self.assertIsNone(v["production"])
        snap = base["model_snapshot"]
        self.assertEqual(ro.revision_binding(store, snap)["status"], "unbound")
        for rid in (rid0, rid1):
            self._activate(store, rid, store.list()[rid]["manifest"])
        store.bind_production(rid0, actor="t", model_snapshot=snap)
        store.bind_production(rid1, actor="t", model_snapshot=snap)
        b = ro.revision_binding(store, snap)
        self.assertEqual((b["status"], b["revision_id"], b["lineage"]),
                         ("bound", rid1, [rid1, rid0]))
        self.assertEqual(ro.revision_binding(store, {"other": 1})["reason"],
                         "model_snapshot_mismatch")
        self.assertTrue(ro.feature_ops_view(store, role="engineer")["revisions"])
        store.rollback_production(actor="t", model_snapshot=snap)
        self.assertEqual(ro.revision_binding(store, snap)["revision_id"], rid0)

    # ------------------------------------------------ AC-028 / AC-032
    def test_memory_asset_inventory(self):
        _p(8, "AC-028/032 记忆资产清单：库=memory、头/路由=weights、逐条血缘")
        rng = np.random.default_rng(0)
        nb = NormalBank()
        nb.seed_core(rng.normal(size=(2, 16, 6, 6)).astype(np.float32))
        db = DefectBank()
        db.add_patch_feats(rng.normal(size=(8, 16)), "init_defect_box", image_id="a.png")
        sem = SimpleNamespace(bank=rng.normal(size=(10, 16)).astype(np.float32))
        pipe = SimpleNamespace(handler=SimpleNamespace(normal_bank=nb, defect_bank=db),
                               slots={"sem": sem}, head_mgr=HeadManager(), router=None,
                               calibrators={"sem": object()},
                               _cal_normal_hist={"sem": [0.1, 0.2]})
        inv = ro.memory_asset_inventory(pipe)
        for k in ("sem_bank", "normal_bank", "defect_bank"):
            self.assertTrue(inv[k]["memory_not_weights"], k)
        self.assertEqual(len(inv["sem_bank"]["sha256"]), 64)
        self.assertEqual(inv["sem_bank"]["n"], 10)
        self.assertFalse(inv["incubate_head"]["memory_not_weights"])
        self.assertFalse(inv["incubate_head"]["trained"])
        self.assertFalse(inv["router"]["present"])
        self.assertEqual(inv["calibrators"]["n_normal"], {"sem": 2})
        self.assertIn("core", inv["normal_bank"])

    # ------------------------------------------------ AC-033
    def test_capability_card(self):
        _p(9, "AC-033 统一能力卡：槽 + 融合后通道，缺省显式 unknown")
        card = ro.capability_card()
        for n in ("sem", "disc", "trad", "open", "defect_knn", "incubate"):
            self.assertIn(n, card)
        self.assertFalse(card["open"]["in_fusion"])
        self.assertFalse(card["defect_knn"]["in_fusion"])
        self.assertTrue(card["sem"]["base_seven"])
        self.assertEqual(card["blob"]["train"], "unknown")
        self.assertTrue(all(len(r) >= 9 for r in card.values()))

    # ------------------------------------------------ AC-034
    def test_admission_registration_gate(self):
        _p(10, "AC-034 E_open 难例→准入登记门禁：缺证据阻断、齐全只登记 draft、永不自动上线")
        cand = {"kind": "new_group"}
        with self.assertRaises(PermissionError):
            ro.admission_registration_gate(cand, {}, 3, role="operator")
        r = ro.admission_registration_gate(cand, {"gain": {"status": "passed"}}, 3)
        self.assertEqual(r["status"], "blocked")
        self.assertIn("version", r["missing"])
        adm = {k: {"status": "passed"} for k in ro.ADMISSION_GATE if k != "version"}
        adm["snapshot_version"] = 3
        self.assertEqual(ro.admission_registration_gate(cand, adm, 3)["status"], "ready")
        self.assertEqual(ro.admission_registration_gate(cand, adm, 4)["missing"], ["version"])
        store = FeatureSetRevision(self.root / "reg.json")
        g = ro.admission_registration_gate(cand, adm, 3, store=store, manifest=_manifest())
        self.assertEqual(g["status"], "registered_draft")
        self.assertFalse(g["auto_online"])
        row = store.list()[g["revision_id"]]
        self.assertEqual(row["state"], "draft")
        self.assertTrue(row["manifest"]["admission_required"])
        self.assertIsNone(store.production_pointer())

    # ------------------------------------------------ AC-060
    def _pipe(self, names):
        slots = {n: SimpleNamespace(cfg={"n": n}) for n in names}
        return SimpleNamespace(slots=slots, open_det=None, calibrators={n: 1 for n in names},
                               slot_sanity={n: 0.9 for n in names},
                               _cal_normal_hist={n: [0.1] * 5 for n in names},
                               weights={n: 1.0 / len(names) for n in names},
                               cde_report=None, handler=None, head_mgr=None)

    def test_orchestration_fast_three(self):
        _p(11, "AC-060 编排总表：fast 三槽如实 is_seven=False、缺基础槽列出")
        t = ro.orchestration_table(self._pipe(["sem", "disc", "inp"]))
        self.assertFalse(t["is_seven"])
        self.assertEqual(set(t["missing_base"]), {"blob", "trad", "layout", "tpl"})
        self.assertEqual(t["candidates"], [])
        self.assertTrue(t["rows"]["sem"]["effective"])
        self.assertEqual(len(t["rows"]["sem"]["config_hash"]), 64)
        self.assertIsNone(t["diff"])

    def test_orchestration_seven_and_diff(self):
        _p(12, "AC-060 七槽齐 is_seven=True；候选槽分列；diff 列出变化与移除")
        prev = ro.orchestration_table(self._pipe(list(ro.SEVEN) + ["color"]))
        self.assertTrue(prev["is_seven"])
        self.assertEqual(prev["candidates"], ["color"])
        cur = ro.orchestration_table(self._pipe(list(ro.SEVEN)), prev=prev)
        self.assertEqual(cur["diff"]["color"], "removed")
        self.assertIn("weight", cur["diff"]["sem"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
