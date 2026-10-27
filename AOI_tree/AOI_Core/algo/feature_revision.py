"""不可变特征 manifest 的前门禁；不切换生产 extractor 或模型指针。"""
import hashlib
import json
import math
import os
import tempfile
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path


PARITY_CHECKS = ("extraction", "preprocess", "dimension_order_slice", "rng",
                 "knn_neighbors_self_match", "scoring", "report", "bank_compatibility")
STATES = ("draft", "evaluated", "approved", "active", "retired")
ADMISSION_CHECKS = ("gain", "redundancy", "regression", "g1", "g2")


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _hashes_valid(hashes):
    return isinstance(hashes, dict) and bool(hashes) and all(
        isinstance(k, str) and bool(k) and isinstance(v, str) and len(v) == 64
        and all(c in "0123456789abcdef" for c in v) for k, v in hashes.items())


def production_hashes():
    root = Path(__file__).parent
    return {name: file_hash(root / name) for name in ("vendor/traditional.py", "slots/trad.py")}


def _validate(manifest):
    if type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1:
        raise ValueError("未知 manifest schema")
    for key in ("preprocess", "implementation_hashes", "dataset_split", "model_snapshot"):
        if not isinstance(manifest.get(key), dict) or not manifest[key]:
            raise ValueError(f"缺少血缘: {key}")
    if type(manifest.get("seed")) is not int:
        raise ValueError("seed 必须明确")
    if not _hashes_valid(manifest["implementation_hashes"]):
        raise ValueError("实现 hash 无效")
    if manifest.get("parent") is not None and not _hashes_valid({"parent": manifest["parent"]}):
        raise ValueError("parent revision 无效")
    pos, keys = 0, set()
    for g in manifest.get("groups", []):
        if (not isinstance(g["key"], str) or not g["key"] or g["key"] in keys
                or type(g["dim"]) is not int or g["dim"] <= 0):
            raise ValueError("特征组重复或维度无效")
        keys.add(g["key"])
        if (g["slice"] != [pos, pos + g["dim"]]
                or any(type(v) is not int for v in g["slice"])
                or not isinstance(g.get("params"), dict)):
            raise ValueError("特征顺序/切片/参数无效")
        pos += g["dim"]
    if not pos or type(manifest.get("dimension")) is not int or manifest["dimension"] != pos:
        raise ValueError("总维度无效")


class FeatureSetRevision:
    def __init__(self, path):
        self.path = Path(path)

    @contextmanager
    def _lock(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock = str(self.path) + ".lock"
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            raise ValueError("revision 注册表正在写入；请重试（崩溃遗留锁需人工核查）")
        try:
            yield
        finally:
            os.close(fd)
            os.unlink(lock)

    def _read(self):
        if not self.path.exists():
            return {}
        with self.path.open(encoding="utf-8") as f:
            rows = json.load(f)
        for rid, row in rows.items():
            _validate(row["manifest"])
            if digest(row["manifest"]) != rid:
                raise ValueError("冻结 revision manifest 被修改")
            state = "draft"
            for event in row["history"]:
                if (event["from"] != state or state == "retired"
                        or event["to"] != STATES[STATES.index(state) + 1]
                        or not event.get("actor") or not event.get("at")):
                    raise ValueError("revision 状态历史无效")
                state = event["to"]
            if row["state"] != state:
                raise ValueError("revision 状态与历史不一致")
            for key in (("evaluation",) if state in ("evaluated", "approved") else
                        ("evaluation", "parity") if state in ("active", "retired") else ()):
                ref = row[key]
                if not _hashes_valid({ref["path"]: ref["sha256"]}):
                    raise ValueError("revision 证据引用无效")
        return rows

    def _write(self, rows, path=None):
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=".revision_")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(rows, f, ensure_ascii=False, indent=2, allow_nan=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path or self.path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def list(self):
        return deepcopy(self._read())

    def register(self, manifest):
        _validate(manifest)
        manifest = deepcopy(manifest)
        rid = digest(manifest)
        with self._lock():
            rows = self._read()
            if manifest.get("parent") is not None and manifest["parent"] not in rows:
                raise ValueError("parent revision 不存在")
            if rid not in rows:
                rows[rid] = {"manifest": manifest, "state": "draft", "history": [],
                             "created_at": datetime.now(timezone.utc).isoformat()}
                self._write(rows)
        return rid

    # —— E_open 自动注册（候选化）：new_defect 处置 → draft；同 key 归并计数，永不激活 ——
    @property
    def autoreg_path(self):
        return self.path.with_name(self.path.stem + ".open_autoreg.json")

    def open_candidates(self):
        if not self.autoreg_path.exists():
            return {}
        with self.autoreg_path.open(encoding="utf-8") as f:
            return json.load(f)

    def register_open_candidate(self, key, manifest, sample):
        """key=品类/缺陷类型。首例按 manifest 登记 draft；后续同 key 归并到同一 revision
        并计数（按 detection_id 去重）。只写 draft + 台账，不迁移状态、不动生产指针。"""
        _validate(manifest)
        manifest = deepcopy(manifest)
        now = datetime.now(timezone.utc).isoformat()
        with self._lock():
            rows, ledger = self._read(), self.open_candidates()
            entry = ledger.get(key)
            if entry is None or entry["revision_id"] not in rows:
                rid = digest(manifest)
                if rid not in rows:
                    rows[rid] = {"manifest": manifest, "state": "draft", "history": [],
                                 "created_at": now}
                    self._write(rows)
                entry = {"revision_id": rid, "key": key, "created_at": now, "count": 0,
                         "samples": [], "auto_online": False, "admission_required": True}
                action = "created"
            else:
                action = "merged"
            if any(s["detection_id"] == sample["detection_id"] for s in entry["samples"]):
                action = "duplicate"
            else:
                entry["samples"].append({**sample, "at": now})
                entry["count"] += 1
            entry["updated_at"] = now
            ledger[key] = entry
            self._write(ledger, self.autoreg_path)
            return {"action": action, "revision_id": entry["revision_id"], "key": key,
                    "count": entry["count"], "state": rows[entry["revision_id"]]["state"],
                    "auto_online": False}

    def _evidence(self, path, rid, manifest):
        if not path:
            raise ValueError("证据路径未知")
        path = str(Path(path).resolve())
        with open(path, "rb") as f:
            raw = f.read()
        data = json.loads(raw)
        if data.get("revision_id") != rid:
            raise ValueError("评价/等价证据不属于该 revision")
        for key in ("dataset_split", "model_snapshot", "seed"):
            if data.get(key) != manifest[key]:
                raise ValueError(f"证据血缘不匹配: {key}")
        return {"path": path, "sha256": hashlib.sha256(raw).hexdigest()}, data

    def _evaluation(self, row, rid):
        evidence = row.get("independent_evaluation", row["evaluation"])
        if file_hash(row["evaluation"]["path"]) != row["evaluation"]["sha256"]:
            raise ValueError("原评价工件已修改")
        ref, data = self._evidence(evidence["path"], rid, row["manifest"])
        if ref != evidence:
            raise ValueError("评价工件已修改")
        auc = data.get("auroc")
        if (not data.get("eval_run_id") or data.get("status") != "completed"
                or data.get("independent") is not True or data.get("acceptance") != "passed"
                or type(auc) not in (int, float) or not math.isfinite(auc) or not 0 <= auc <= 1):
            raise ValueError("独立评估不可用/未通过，拒绝批准或激活")
        if row["manifest"].get("admission_required"):
            adm = data.get("admission") or {}
            bad = [k for k in ADMISSION_CHECKS
                   if (adm.get(k) or {}).get("status") != "passed"]
            if (bad or adm.get("acceptance") != "passed"
                    or not _hashes_valid({"pool": adm.get("hard_pool_sha256")})):
                raise ValueError("难例准入试算缺失/未通过"
                                 + (f"（{','.join(bad)}）" if bad else "") + "，拒绝批准或激活")

    def _parity(self, path, rid, manifest):
        ref, parity = self._evidence(path, rid, manifest)
        if (parity.get("status") != "passed" or
                any(parity.get("checks", {}).get(k) is not True for k in PARITY_CHECKS)):
            raise ValueError("生产数值等价未证明，拒绝激活")
        if (parity.get("baseline_implementation_hashes") != production_hashes()
                or not _hashes_valid({"protocol": parity.get("protocol_hash")})):
            raise ValueError("旧生产实现/等价协议未知或生产实现已变化")
        numeric = parity.get("numeric") or {}
        diff, tolerance = numeric.get("max_abs_diff"), numeric.get("atol")
        if (type(numeric.get("n_samples")) is not int or numeric["n_samples"] <= 0
                or type(diff) not in (int, float) or not math.isfinite(diff)
                or type(tolerance) not in (int, float) or not math.isfinite(tolerance)
                or diff < 0 or tolerance < 0 or diff > tolerance):
            raise ValueError("数值等价样本/差异/容差缺失或失败")
        return ref

    def check_activation(self, rid, model_snapshot):
        row = self._read().get(rid)
        if not row or row["state"] != "active":
            raise ValueError("绑定特征 revision 尚未通过激活前门禁")
        if row["manifest"]["model_snapshot"] != model_snapshot:
            raise ValueError("revision 未绑定当前模型快照或模型资产已变化")
        self._evaluation(row, rid)
        if self._parity(row["parity"]["path"], rid, row["manifest"]) != row["parity"]:
            raise ValueError("等价工件已修改")
        return deepcopy(row)

    def transition(self, rid, target, *, actor, evidence_path=None, parity_path=None):
        if not isinstance(actor, str) or not actor.strip():
            raise ValueError("必须记录责任人")
        with self._lock():
            rows = self._read()
            if rid not in rows:
                raise ValueError("revision 不存在")
            row = rows[rid]
            allowed = dict(zip(STATES, STATES[1:]))
            if allowed.get(row["state"]) != target:
                raise ValueError("禁止越级、重复激活或复活 retired revision")
            manifest = row["manifest"]
            if target == "evaluated":
                ref, data = self._evidence(evidence_path, rid, manifest)
                if not data.get("eval_run_id") or data.get("status") != "completed":
                    raise ValueError("评价未完成/运行编号未知")
                row["evaluation"] = ref
            if target == "approved" and evidence_path:
                row["independent_evaluation"], _ = self._evidence(evidence_path, rid, manifest)
            if target in ("approved", "active"):
                self._evaluation(row, rid)
            if target == "active":
                row["parity"] = self._parity(parity_path, rid, manifest)
            row["history"].append({"from": row["state"], "to": target, "actor": actor,
                                   "at": datetime.now(timezone.utc).isoformat()})
            row["state"] = target
            self._write(rows)
            return deepcopy(row)

    # —— 生产指针（AC-061）：active 只是“可绑定”，绑定须显式 bind_production ——
    @property
    def pointer_path(self):
        return self.path.with_name(self.path.stem + ".production.json")

    def production_pointer(self):
        if not self.pointer_path.exists():
            return None
        with self.pointer_path.open(encoding="utf-8") as f:
            return json.load(f)

    def _write_pointer(self, data):
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=".pointer_")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.pointer_path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def bind_production(self, rid, *, actor, model_snapshot):
        if not isinstance(actor, str) or not actor.strip():
            raise ValueError("必须记录责任人")
        with self._lock():
            self.check_activation(rid, model_snapshot)
            ptr = self.production_pointer() or {"current": None, "history": []}
            if ptr["current"] == rid:
                return deepcopy(ptr)
            ptr["history"].append({"op": "bind", "from": ptr["current"], "to": rid, "actor": actor,
                                   "at": datetime.now(timezone.utc).isoformat()})
            ptr["current"] = rid
            ptr["model_snapshot"] = deepcopy(model_snapshot)
            self._write_pointer(ptr)
            return deepcopy(ptr)

    def rollback_production(self, *, actor, model_snapshot):
        if not isinstance(actor, str) or not actor.strip():
            raise ValueError("必须记录责任人")
        with self._lock():
            ptr = self.production_pointer()
            prev = next((e["from"] for e in reversed((ptr or {}).get("history", []))
                         if e["to"] == ptr["current"] and e["from"]), None)
            if not prev:
                raise ValueError("无可回滚的上一生产 revision")
            self.check_activation(prev, model_snapshot)
            ptr["history"].append({"op": "rollback", "from": ptr["current"], "to": prev,
                                   "actor": actor, "at": datetime.now(timezone.utc).isoformat()})
            ptr["current"] = prev
            ptr["model_snapshot"] = deepcopy(model_snapshot)
            self._write_pointer(ptr)
            return deepcopy(ptr)

    def lineage(self, rid):
        rows, chain = self._read(), []
        while rid:
            if rid not in rows or rid in chain:
                raise ValueError("revision 血缘断裂")
            chain.append(rid)
            rid = rows[rid]["manifest"].get("parent")
        return chain

    def detection_stamp(self, model_snapshot=None):
        """检测记录的生产 revision 血缘；未绑定时明确返回 unbound。"""
        ptr = self.production_pointer()
        if not ptr or not ptr.get("current"):
            return {"feature_revision": None, "status": "unbound"}
        stamp = {"feature_revision": ptr["current"], "status": "bound",
                 "model_snapshot": ptr.get("model_snapshot"),
                 "lineage": self.lineage(ptr["current"])}
        if model_snapshot is not None and model_snapshot != ptr.get("model_snapshot"):
            stamp["status"] = "snapshot_mismatch"
        return stamp
