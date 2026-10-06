"""后端 REST / WebSocket 客户端封装。

- ApiClient：requests 封装，统一超时与友好中文错误回调（error_occurred 信号）。
- ApiWorker：在后台线程执行任意调用，避免阻塞 UI。
- TaskWSClient：全局任务进度推送（/ws/tasks）。
"""
from __future__ import annotations

import atexit
import json
import os
import time
import weakref
from urllib.parse import quote

import requests
import websocket  # websocket-client
from PySide6.QtCore import QObject, QThread, Signal

DEFAULT_BASE = "http://127.0.0.1:8017"


def _load_api_key() -> str:
    """M10d：API Key 加载（优先级：环境变量 AOI_API_KEY > 仓库
    configs/default.yaml 的 security.api_key）。空串 = 后端未开鉴权。"""
    key = (os.environ.get("AOI_API_KEY") or "").strip()
    if key:
        return key
    try:
        import yaml
        cfg_path = os.environ.get("AOI_CONFIG")
        if not cfg_path:
            from pathlib import Path
            cfg_path = str(Path(__file__).resolve().parents[1]
                           / "configs" / "default.yaml")
        with open(cfg_path, "r", encoding="utf-8") as f:
            return str((yaml.safe_load(f).get("security") or {})
                       .get("api_key") or "").strip()
    except Exception:  # noqa: BLE001 配置缺失视为未开鉴权
        return ""

# ── 线程注册表：进程退出前统一停止并等待，避免 QThread 运行中被销毁导致 abort ──
_LIVE_THREADS: "weakref.WeakSet[QThread]" = weakref.WeakSet()


def register_thread(t: QThread) -> None:
    """登记后台线程，退出时由 atexit 统一清理。"""
    _LIVE_THREADS.add(t)


def _cleanup_threads() -> None:
    for t in list(_LIVE_THREADS):
        stop = getattr(t, "stop", None)
        if callable(stop):
            try:
                stop()
            except Exception:  # noqa: BLE001
                pass
    deadline = time.monotonic() + 10
    for t in list(_LIVE_THREADS):
        try:
            if t.isRunning():
                remaining = max(200, int((deadline - time.monotonic()) * 1000))
                t.wait(remaining)
        except Exception:  # noqa: BLE001
            pass


atexit.register(_cleanup_threads)


def file_url(path: str, base_url: str = DEFAULT_BASE) -> str:
    """本机绝对路径 → /api/file?path= 编码后的完整 URL。"""
    return f"{base_url.rstrip('/')}/api/file?path={quote(path)}"


class ApiClient(QObject):
    """REST 客户端：所有失败转为中文错误信号，绝不抛异常导致 UI 崩溃。"""

    error_occurred = Signal(str)
    connection_changed = Signal(bool)

    def __init__(self, base_url: str = DEFAULT_BASE, parent: QObject | None = None):
        super().__init__(parent)
        self.base_url = base_url.rstrip("/")
        self.ws_base = "ws://" + self.base_url.split("://", 1)[-1]
        self._session = requests.Session()
        # U-opsflow(2026-08-26)：UI 多线程并发（ApiWorker + 缩略图下载）默认
        # HTTPAdapter pool_maxsize=10，慢请求未归还时连接池打满 → urllib3
        # "Connection pool is full" 刷屏。抬升池上限，图片字节下载另用独立
        # Session（大而慢，避免占满 REST 请求池）。
        from requests.adapters import HTTPAdapter
        adapter = HTTPAdapter(pool_connections=10, pool_maxsize=64)
        self._session.mount("http://", adapter)
        self._session.mount("https://", adapter)
        self._bytes_session = requests.Session()
        self._bytes_session.mount("http://", HTTPAdapter(pool_connections=4,
                                                         pool_maxsize=16))
        api_key = _load_api_key()
        if api_key:  # M10d：后端开启鉴权时自动带 X-API-Key
            self._session.headers["X-API-Key"] = api_key
            self._bytes_session.headers["X-API-Key"] = api_key
        self.timeout = 15
        self._connected = False

    # ── 连接状态 ──────────────────────────────────────────
    @property
    def connected(self) -> bool:
        return self._connected

    def _set_connected(self, ok: bool) -> None:
        if ok != self._connected:
            self._connected = ok
            self.connection_changed.emit(ok)

    def ping(self) -> dict | None:
        """健康检查（静默，不触发错误弹窗）。"""
        try:
            r = self._session.get(self.base_url + "/api/health", timeout=2)
            if r.ok:
                self._set_connected(True)
                return r.json()
        except Exception:
            pass
        self._set_connected(False)
        return None

    def file_url(self, path: str) -> str:
        """本地绝对路径 → /api/file?path= 完整 URL（本地图片统一走此接口显示）。"""
        if not path:
            return ""
        return file_url(path, self.base_url)

    def ws_url(self, path: str) -> str:
        """REST 路径 → WebSocket 完整 URL（http 换 ws）。"""
        return self.ws_base + path

    # ── 通用请求 ──────────────────────────────────────────
    def _request(self, method: str, path: str, silent: bool = False,
                 timeout: float | None = None, conflict_ok: bool = False,
                 **kw):
        try:
            r = self._session.request(
                method, self.base_url + path,
                timeout=timeout or self.timeout, **kw)
        except requests.exceptions.ConnectionError:
            self._set_connected(False)
            if not silent:
                self.error_occurred.emit("无法连接后端服务，请确认后端已启动")
            return None
        except requests.exceptions.Timeout:
            if not silent:
                self.error_occurred.emit("请求超时，请稍后重试")
            return None
        except Exception as e:  # noqa: BLE001
            if not silent:
                self.error_occurred.emit(f"请求异常：{e}")
            return None
        self._set_connected(True)
        if r.status_code >= 400:
            try:
                detail = r.json().get("detail", r.text[:200])
            except Exception:  # noqa: BLE001
                detail = r.text[:200]
            # 409 冲突可由调用方显式接管（如激活门控拒绝需弹窗展示 gate_report）
            if conflict_ok and r.status_code == 409:
                return {"_status": 409, "detail": detail}
            if not silent:
                self.error_occurred.emit(f"请求失败（HTTP {r.status_code}）：{detail}")
            return None
        if not r.content:
            return {}
        try:
            return r.json()
        except Exception:  # noqa: BLE001
            return {"raw": r.text}

    def get(self, path: str, **kw):
        return self._request("GET", path, **kw)

    def post(self, path: str, **kw):
        return self._request("POST", path, **kw)

    def put(self, path: str, **kw):
        return self._request("PUT", path, **kw)

    def delete(self, path: str, **kw):
        return self._request("DELETE", path, **kw)

    def upload(self, path: str, file_path: str, form: dict | None = None,
               timeout: float = 120):
        """multipart 上传文件 + 表单字段。"""
        with open(file_path, "rb") as f:
            files = {"file": (os.path.basename(file_path), f)}
            return self._request("POST", path, files=files,
                                 data=form or {}, timeout=timeout)

    def fetch_bytes(self, url: str, timeout: float = 20) -> bytes | None:
        """下载二进制内容（用于图片字节 → QPixmap），静默失败。
        用独立 Session（图片下载量大且慢，不占 REST 请求池）。"""
        try:
            r = self._bytes_session.get(url, timeout=timeout)
            if r.ok:
                return r.content
        except Exception:  # noqa: BLE001
            pass
        return None

    # ══════════════════ 端点方法（每个端点一个方法） ══════════════════

    # ── 系统 ─────────────────────────────────────────────
    def health(self) -> dict | None:
        """GET /api/health → {status, device, gpu_name}"""
        return self.ping()

    def system_info(self) -> dict | None:
        """GET /api/system/info"""
        return self.get("/api/system/info")

    # ── M16 工作台 / 设置中心 ─────────────────────────────
    def models_readiness(self) -> dict | None:
        """GET /api/models/readiness → {items:[{category,stage,stage_cn,
        next_action,next_page,counts,n_versions,active_version,best_auroc}]}
        （M16a 品类上线状态机；静默失败）"""
        return self.get("/api/models/readiness", silent=True)

    def incomplete_report(self) -> dict | None:
        """GET /api/review/incomplete_report → {reports:[{category,warn,...}]}
        （M14c 不完备预警；静默失败）"""
        return self.get("/api/review/incomplete_report", silent=True)

    def get_config(self) -> dict | None:
        """GET /api/system/config → {items:[{section,key,type,name,desc,value}],
        config_path}（M16d 设置中心）"""
        return self.get("/api/system/config")

    def save_config(self, updates: dict) -> dict | None:
        """POST /api/system/config {"sec.key": value} → {applied, rejected}
        （M16d 白名单写回；admin 角色）"""
        return self.post("/api/system/config", json=dict(updates))

    def list_categories(self) -> list:
        """GET /api/categories -> [品类名...]（失败时回退 system_info）"""
        return [c["category"] for c in self.categories_full()]

    def categories_full(self) -> list:
        """GET /api/categories -> [{category, data_level, prepared, ...}]。

        U-workorder：data_level=工单指定的数据条件（None=未指定）。
        """
        data = self.get("/api/categories", silent=True)
        if isinstance(data, list):
            return [c for c in data if isinstance(c, dict) and c.get("category")]
        if isinstance(data, dict) and isinstance(data.get("categories"), list):
            return [c for c in data["categories"] if isinstance(c, dict)]
        info = self.system_info()
        if info and isinstance(info.get("categories"), list):
            return [c for c in info["categories"] if isinstance(c, dict)]
        return []

    # ── 数据源 / 工单 / 任务模板（U-workorder v2）─────────
    def list_datasources(self) -> list:
        """GET /api/datasources -> [{id,name,modality,capability,...}]"""
        data = self.get("/api/datasources", silent=True)
        if isinstance(data, dict) and isinstance(data.get("items"), list):
            return data["items"]
        return data if isinstance(data, list) else []

    def create_datasource(self, name: str, modality: str = "image",
                          label_tier: str = "L0",
                          sampling: dict | None = None,
                          per_category: bool = False,
                          has_template: bool = False,
                          note: str = "",
                          pretrain_normal: int | None = None,
                          pretrain_anomaly: int | None = None,
                          batch_size: int | None = None,
                          plan_id: int | None = None,
                          source_type: str = "local",
                          stream_config: dict | None = None) -> dict | None:
        """POST /api/datasources（v4 标注档位 / v6 分品类·模板声明 /
        v9 预训练组配置与方案复用 / v10 数据流类型 local|rtsp）"""
        payload: dict = {
            "name": name, "modality": modality, "label_tier": label_tier,
            "sampling": sampling or {}, "per_category": bool(per_category),
            "has_template": bool(has_template), "note": note,
            "source_type": source_type}
        if stream_config:
            payload["stream_config"] = stream_config
        if pretrain_normal is not None:
            payload["pretrain_normal"] = pretrain_normal
        if pretrain_anomaly is not None:
            payload["pretrain_anomaly"] = pretrain_anomaly
        if batch_size is not None:
            payload["batch_size"] = batch_size
        if plan_id is not None:
            payload["plan_id"] = plan_id
        return self.post("/api/datasources", json=payload)

    def delete_datasource(self, datasource_id: int) -> dict | None:
        """DELETE /api/datasources/{id}：删除数据源登记（解挂工单）。"""
        return self.delete(f"/api/datasources/{int(datasource_id)}")

    def update_datasource(self, datasource_id: int,
                          name: str | None = None,
                          label_tier: str | None = None,
                          sampling: dict | None = None,
                          per_category: bool | None = None,
                          has_template: bool | None = None,
                          note: str | None = None,
                          pretrain_normal: int | None = None,
                          pretrain_anomaly: int | None = None,
                          batch_size: int | None = None,
                          plan_id: int | None = None) -> dict | None:
        """PUT /api/datasources/{id}：编辑数据源（模态锁定；v9 分组配置）。"""
        payload: dict = {}
        if name is not None:
            payload["name"] = name
        if label_tier is not None:
            payload["label_tier"] = label_tier
        if sampling is not None:
            payload["sampling"] = sampling
        if per_category is not None:
            payload["per_category"] = per_category
        if has_template is not None:
            payload["has_template"] = has_template
        if note is not None:
            payload["note"] = note
        if pretrain_normal is not None:
            payload["pretrain_normal"] = pretrain_normal
        if pretrain_anomaly is not None:
            payload["pretrain_anomaly"] = pretrain_anomaly
        if batch_size is not None:
            payload["batch_size"] = batch_size
        if plan_id is not None:
            payload["plan_id"] = plan_id
        return self.put(f"/api/datasources/{int(datasource_id)}", json=payload)

    # ── 前端反馈 v9：预训练组/检测组 分组与方案 ──
    def datasource_groups(self, datasource_id: int) -> dict | None:
        """GET /api/datasources/{id}/groups → {categories:{cat:{pretrain,detect_batches}}}"""
        return self.get(f"/api/datasources/{int(datasource_id)}/groups",
                        silent=True)

    def save_plan(self, datasource_id: int, name: str,
                  note: str = "") -> dict | None:
        """POST /api/datasources/{id}/plan：把当前分组另存为可复用方案。"""
        return self.post(f"/api/datasources/{int(datasource_id)}/plan",
                         json={"name": name, "note": note})

    def list_plans(self) -> dict | None:
        """GET /api/datasource-plans：分组方案列表。"""
        return self.get("/api/datasource-plans", silent=True)

    def delete_plan(self, plan_id: int) -> dict | None:
        """DELETE /api/datasource-plans/{id}"""
        return self.delete(f"/api/datasource-plans/{int(plan_id)}")

    def attach_legacy(self, datasource_id: int) -> dict | None:
        """POST /api/datasources/{id}/attach-legacy：存量未挂源批次一键归入。"""
        return self.post(f"/api/datasources/{int(datasource_id)}/attach-legacy")

    def check_datasource(self, datasource_id: int, apply: bool = True) -> dict | None:
        """POST /api/datasources/{id}/check：数据源体检（apply 时自动校正档位）。"""
        return self.post(f"/api/datasources/{int(datasource_id)}/check",
                         params={"apply": str(bool(apply)).lower()})

    def clear_datasource_data(self, datasource_id: int) -> dict | None:
        """DELETE /api/datasources/{id}/data：清空源下批次与图片（保留源登记）。"""
        return self.delete(f"/api/datasources/{int(datasource_id)}/data")

    def list_workorders(self, range_name: str = "all") -> list:
        """GET /api/workorders?range=today|7d|all -> [{id,name,...,stats,check}]"""
        data = self.get("/api/workorders",
                        params={"range": range_name}, silent=True)
        if isinstance(data, dict) and isinstance(data.get("items"), list):
            return data["items"]
        return data if isinstance(data, list) else []

    def workorder_queue(self, workorder_id: int) -> dict | None:
        """GET /api/workorders/{id}/queue：数据流队列（批次+错检统计+回队）。"""
        return self.get(f"/api/workorders/{int(workorder_id)}/queue", silent=True)

    def queue_requeue(self, workorder_id: int, batch_keys: list,
                      requeue: bool = True) -> dict | None:
        """回队/取消回队：POST|DELETE /workorders/{id}/queue/requeue（检测批次粒度）。"""
        meth = self.post if requeue else self.delete
        return meth(f"/api/workorders/{int(workorder_id)}/queue/requeue",
                    json={"batch_keys": [str(k) for k in batch_keys]})

    def pipeline_control(self, workorder_id: int, action: str) -> dict | None:
        """产线控制：pause / resume / stop（stop 清空回队队列残留）。"""
        return self.post(f"/api/workorders/{int(workorder_id)}/pipeline/{action}")

    def download_workorder_report(self, workorder_id: int) -> bytes | None:
        """GET /api/workorders/{id}/report.csv → CSV 字节（工单检测报告导出；
        失败静默返回 None，由调用方提示）。"""
        return self.fetch_bytes(
            f"{self.base_url}/api/workorders/{int(workorder_id)}/report.csv")

    def workorder_learn(self, workorder_id: int) -> dict | None:
        """学习提升：批量应用反馈固化，按 auto_resume 恢复产线。"""
        return self.post(f"/api/workorders/{int(workorder_id)}/learn")

    def create_workorder(self, name: str, review_enabled: bool = False,
                         note: str = "",
                         datasource_ids: list | None = None,
                         from_template: int | None = None) -> dict | None:
        """POST /api/workorders：建工单（v4：数据源必选 + 复判开关；
        条件由数据源属性自动聚合，不再手填）。"""
        payload: dict = {
            "name": name, "review_enabled": bool(review_enabled),
            "note": note, "datasource_ids": datasource_ids or []}
        if from_template is not None:
            payload["from_template"] = from_template
        return self.post("/api/workorders", json=payload)

    def get_workorder(self, workorder_id: int) -> dict | None:
        """GET /api/workorders/{id}"""
        return self.get(f"/api/workorders/{int(workorder_id)}", silent=True)

    def update_workorder(self, workorder_id: int, **fields) -> dict | None:
        """PUT /api/workorders/{id}：改名/条件/状态/备注（只改传入字段）。"""
        return self.put(f"/api/workorders/{int(workorder_id)}", json=fields)

    def delete_workorder(self, workorder_id: int) -> dict | None:
        """DELETE /api/workorders/{id}"""
        return self.delete(f"/api/workorders/{int(workorder_id)}")

    def set_workorder_datasources(self, workorder_id: int,
                                  datasource_ids: list) -> dict | None:
        """POST /api/workorders/{id}/datasources：整体替换挂接（挂后体检）。"""
        return self.post(f"/api/workorders/{int(workorder_id)}/datasources",
                         json={"datasource_ids": datasource_ids})

    def check_workorder(self, workorder_id: int,
                        apply: bool = False) -> dict | None:
        """POST /api/workorders/{id}/check?apply=：体检（apply=true 自动调整）。"""
        return self.post(f"/api/workorders/{int(workorder_id)}/check",
                         params={"apply": str(bool(apply)).lower()})

    def workorder_stats(self, workorder_id: int,
                        range_name: str = "all") -> dict | None:
        """GET /api/workorders/{id}/stats?range=today|7d|all"""
        return self.get(f"/api/workorders/{int(workorder_id)}/stats",
                        params={"range": range_name}, silent=True)

    def list_workorder_templates(self) -> list:
        """GET /api/workorder-templates"""
        data = self.get("/api/workorder-templates", silent=True)
        if isinstance(data, dict) and isinstance(data.get("items"), list):
            return data["items"]
        return data if isinstance(data, list) else []

    def create_workorder_template(self, name: str,
                                  datasource_ids: list | None = None,
                                  review_enabled: bool = False,
                                  note: str = "") -> dict | None:
        """POST /api/workorder-templates（v4：存数据源挂接 + 复判开关）。"""
        return self.post("/api/workorder-templates", json={
            "name": name, "datasource_ids": datasource_ids or [],
            "review_enabled": bool(review_enabled), "note": note})

    def delete_workorder_template(self, template_id: int) -> dict | None:
        """DELETE /api/workorder-templates/{id}"""
        return self.delete(f"/api/workorder-templates/{int(template_id)}")

    # ── 图像 / 视频数据 ──────────────────────────────────
    def list_images(self, category: str = "", split: str = "", label: str = "",
                    dataset_id: int | str = "", page: int = 1,
                    page_size: int = 50, ids: str = "",
                    datasource_id: int | str = "") -> dict | None:
        """GET /api/images → {total, items}（M8b：dataset_id 过滤；
        v9：ids=逗号分隔 image id，用于预训练组/检测组批次查看；
        v10：datasource_id 按数据源过滤（查询栏数据源维度））"""
        params: dict = {"page": page, "page_size": page_size}
        if category:
            params["category"] = category
        if split:
            params["split"] = split
        if label:
            params["label"] = label
        if dataset_id != "":
            params["dataset_id"] = dataset_id
        if datasource_id != "":
            params["datasource_id"] = datasource_id
        if ids:
            params["ids"] = ids
        return self.get("/api/images", params=params)

    def images_tree(self) -> dict | None:
        """GET /api/images/tree → {"tree": 三维, "datasets": 四维批次维}（静默失败）"""
        return self.get("/api/images/tree", silent=True)

    def import_folder(self, folder: str, category: str, split: str = "train",
                      label: str = "normal", source: str = "folder",
                      copy_to_storage: bool = False, name: str = "",
                      note: str = "", dataset_name: str = "",
                      datasource_id: int | None = None) -> dict | None:
        """POST /api/images/import → {task_id, dataset_id}（M8a：name/note 批次命名）"""
        payload: dict = {
            "folder": folder, "category": category, "split": split,
            "label": label, "source": source, "copy_to_storage": copy_to_storage,
        }
        if name:
            payload["name"] = name
        if note:
            payload["note"] = note
        if dataset_name:
            payload["dataset_name"] = dataset_name
        if datasource_id:
            payload["datasource_id"] = datasource_id
        return self.post("/api/images/import", json=payload)

    def import_mvtec(self, root: str, name: str = "",
                     note: str = "", dataset_name: str = "",
                     datasource_id: int | None = None) -> dict | None:
        """POST /api/images/import_mvtec → {task_id, dataset_ids}"""
        payload: dict = {"root": root}
        if name:
            payload["name"] = name
        if note:
            payload["note"] = note
        if dataset_name:
            payload["dataset_name"] = dataset_name
        if datasource_id:
            payload["datasource_id"] = datasource_id
        return self.post("/api/images/import_mvtec", json=payload)

    def probe_dataset(self, root: str) -> dict | None:
        """POST /api/datasets/probe -> 识别报告（格式/品类/分组/计数/告警）。"""
        return self.post("/api/datasets/probe", json={"root": root})

    def adapt_import(self, root: str, fmt: str, category: str = "",
                     ok_role: str = "train", group_overrides: dict | None = None,
                     categories: list | None = None, dataset_name: str = "",
                     name: str = "", note: str = "",
                     datasource_id: int | None = None) -> dict | None:
        """POST /api/datasets/adapt_import -> {task_id}（确认后的映射导入）。"""
        payload: dict = {"root": root, "format": fmt}
        if category:
            payload["category"] = category
        if ok_role:
            payload["ok_role"] = ok_role
        if group_overrides:
            payload["group_overrides"] = group_overrides
        if categories is not None:
            payload["categories"] = categories
        if dataset_name:
            payload["dataset_name"] = dataset_name
        if name:
            payload["name"] = name
        if note:
            payload["note"] = note
        if datasource_id:
            payload["datasource_id"] = datasource_id
        return self.post("/api/datasets/adapt_import", json=payload)

    def upload_image(self, file_path: str, category: str, split: str = "train",
                     label: str = "normal", name: str = "",
                     note: str = "", datasource_id: int | None = None) -> dict | None:
        """POST /api/images/upload（multipart；v8：datasource_id 挂源）"""
        form: dict = {"category": category, "split": split, "label": label}
        if name:
            form["name"] = name
        if note:
            form["note"] = note
        if datasource_id is not None:
            form["datasource_id"] = str(int(datasource_id))
        return self.upload("/api/images/upload", file_path, form=form)

    def delete_image(self, image_id: int, delete_file: bool = False) -> dict | None:
        """DELETE /api/images/{id}?delete_file=（M8a：可选同时删磁盘文件）"""
        return self.delete(f"/api/images/{image_id}",
                           params={"delete_file": str(bool(delete_file)).lower()})

    def annotate_image(self, image_id: int, label: str | None = None,
                       split: str | None = None,
                       defect_type: str | None = None) -> dict | None:
        """POST /api/images/{id}/annotate（M8a 标注流转，只改传入字段）"""
        payload: dict = {}
        if label is not None:
            payload["label"] = label
        if split is not None:
            payload["split"] = split
        if defect_type is not None:
            payload["defect_type"] = defect_type
        return self.post(f"/api/images/{image_id}/annotate", json=payload)

    # ── 数据集批次（M8a/M8b）─────────────────────────────
    def list_datasets(self, category: str = "") -> dict | None:
        """GET /api/datasets?category= → {items:[{id,name,category,source_type,...}]}"""
        params = {"category": category} if category else {}
        return self.get("/api/datasets", params=params, silent=True)

    def dataset_detail(self, dataset_id: int, page: int = 1,
                       page_size: int = 50) -> dict | None:
        """GET /api/datasets/{id} → {dataset, total, page, page_size, items}"""
        return self.get(f"/api/datasets/{int(dataset_id)}",
                        params={"page": page, "page_size": page_size})

    def delete_dataset(self, dataset_id: int, delete_files: bool = False,
                       keep_detections: bool = True) -> dict | None:
        """DELETE /api/datasets/{id} → {deleted_images, deleted_files, affected_detections}"""
        return self.delete(
            f"/api/datasets/{int(dataset_id)}",
            params={"delete_files": str(bool(delete_files)).lower(),
                    "keep_detections": str(bool(keep_detections)).lower()})

    def attach_dataset(self, dataset_id: int, datasource_id: int) -> dict | None:
        """POST /api/datasets/{id}/attach?datasource_id=：批次挂到数据源。"""
        return self.post(
            f"/api/datasets/{int(dataset_id)}/attach",
            params={"datasource_id": int(datasource_id)})

    # ── 孤儿文件治理（M8a/M8b）─────────────────────────────
    def orphan_scan(self) -> dict | None:
        """POST /api/maintenance/orphan_scan → {task_id}（后台任务，result 含孤儿清单）"""
        return self.post("/api/maintenance/orphan_scan", json={})

    def orphan_clean(self, paths: list) -> dict | None:
        """POST /api/maintenance/orphan_clean {paths} → {deleted, missing, skipped}"""
        return self.post("/api/maintenance/orphan_clean",
                         json={"paths": list(paths)})

    def list_videos(self, category: str = "", page: int = 1,
                    page_size: int = 50) -> dict | None:
        """GET /api/videos → {total, items}"""
        params: dict = {"page": page, "page_size": page_size}
        if category:
            params["category"] = category
        return self.get("/api/videos", params=params)

    def import_video(self, path: str, category: str,
                     label: str = "",
                     datasource_id: int | None = None) -> dict | None:
        """POST /api/videos/import"""
        payload: dict = {"path": path, "category": category, "label": label}
        if datasource_id:
            payload["datasource_id"] = datasource_id
        return self.post("/api/videos/import", json=payload)

    # ── 检测 / 监控 ──────────────────────────────────────
    def detect_image(self, category: str, image_id: int | None = None,
                     path: str = "") -> dict | None:
        """POST /api/detect/image → 检测 dict"""
        payload: dict = {"category": category}
        if image_id is not None:
            payload["image_id"] = image_id
        elif path:
            payload["path"] = path
        return self.post("/api/detect/image", json=payload, timeout=120)

    def detect_upload(self, file_path: str, category: str) -> dict | None:
        """POST /api/detect/upload（multipart）→ 检测 dict"""
        return self.upload("/api/detect/upload", file_path,
                           form={"category": category}, timeout=120)

    def detect_video(self, video_id: int, category: str,
                     interval: int = 10) -> dict | None:
        """POST /api/detect/video → {task_id}"""
        return self.post("/api/detect/video", json={
            "video_id": video_id, "category": category, "interval": interval})

    def list_detections(self, category: str = "", is_anomaly: bool | None = None,
                        target_type: str = "", page: int = 1,
                        page_size: int = 50) -> dict | None:
        """GET /api/detections → {total, items}"""
        params: dict = {"page": page, "page_size": page_size}
        if category:
            params["category"] = category
        if is_anomaly is not None:
            params["is_anomaly"] = str(is_anomaly).lower()
        if target_type:
            params["target_type"] = target_type
        return self.get("/api/detections", params=params)

    def detections_live(self, after_id: int = 0, limit: int = 50,
                        category: str = "",
                        workorder_id: int | None = None) -> dict | None:
        """GET /api/detections/live（silent，轮询用）→ {items, alarm, next_after_id,
        latest_id, first_id, gap, dropped_count, reset_required, has_more}"""
        params: dict = {"after_id": after_id, "limit": limit}
        if category:
            params["category"] = category
        if workorder_id:
            params["workorder_id"] = workorder_id
        return self.get("/api/detections/live", params=params, silent=True)

    # ── 反馈 ─────────────────────────────────────────────
    def submit_feedback(self, detection_id: int, feedback_type: str,
                        operator_label: int, defect_type: str = "",
                        comment: str = "", operator: str = "",
                        region: list | None = None) -> dict | None:
        """POST /api/feedback → {id}（region 为漏检区域 [x,y,w,h] 原图像素坐标）"""
        payload: dict = {
            "detection_id": detection_id, "feedback_type": feedback_type,
            "operator_label": operator_label, "defect_type": defect_type,
            "comment": comment, "operator": operator,
        }
        if region:
            payload["region"] = [int(v) for v in region]
        return self.post("/api/feedback", json=payload)

    def invalidate_feedback(self, feedback_id: int) -> dict | None:
        """POST /api/feedback/{id}/invalidate → 作废反馈（标错撤回）"""
        return self.post(f"/api/feedback/{feedback_id}/invalidate", json={})

    def list_feedback(self, consumed: bool | None = None,
                      feedback_type: str = "", category: str = "",
                      page: int = 1, page_size: int = 100) -> dict | None:
        """GET /api/feedback → {total, items}"""
        params: dict = {"page": page, "page_size": page_size}
        if consumed is not None:
            params["consumed"] = str(consumed).lower()
        if feedback_type:
            params["feedback_type"] = feedback_type
        if category:
            params["category"] = category
        return self.get("/api/feedback", params=params)

    def feedback_summary(self, workorder_id: int | None = None) -> dict | None:
        """GET /api/feedback/summary（可工单口径）"""
        params = {"workorder_id": workorder_id} if workorder_id else {}
        return self.get("/api/feedback/summary", params=params)

    # ── 灰区复核队列（M5）───────────────────────────────────
    def review_queue(self, category: str = "", page: int = 1,
                     page_size: int = 20) -> dict | None:
        """GET /api/review/queue → {total, page, page_size, items}
        （静默失败：后端离线时由页面空态接管）"""
        params: dict = {"page": page, "page_size": page_size}
        if category:
            params["category"] = category
        return self.get("/api/review/queue", params=params, silent=True)

    def review_submit(self, detection_id: int, label: int,
                      comment: str = "", operator: str = "",
                      region: list | None = None) -> dict | None:
        """POST /api/review/{id} -> {feedback_id, engine_update, note, category}
        （409=已复核 由调用方弹窗接管；region 为缺陷定位框 [x,y,w,h] 原图像素
        坐标，label=1 时后端换算归一化 box 喂拦截通道，label=0 忽略）"""
        payload: dict = {
            "label": int(label), "comment": comment,
            "operator": operator or "operator"}
        if region:
            payload["region"] = [int(v) for v in region]
        return self.post(f"/api/review/{int(detection_id)}", json=payload,
                         conflict_ok=True)

    def review_stats(self) -> dict | None:
        """GET /api/review/stats -> {pending_total, pending_by_category,
        reviewed_today, reviewed_total, categories}（静默失败；
        categories=待复核品类全集，供待复核下拉）"""
        return self.get("/api/review/stats", silent=True)

    # ── 数据增强配置（§15.29：伪异常 / 预处理 / 训练增强）────────
    def get_augment_config(self) -> dict | None:
        """GET /api/augment/config：数据增强配置（pseudo/preprocess/enhance）。"""
        return self.get("/api/augment/config", silent=True)

    def save_augment_config(self, pseudo: dict | None = None,
                            preprocess: dict | None = None,
                            enhance: dict | None = None) -> dict | None:
        """PUT /api/augment/config：写回数据增强配置（白名单字段）。"""
        payload: dict = {}
        if pseudo:
            payload["pseudo"] = pseudo
        if preprocess:
            payload["preprocess"] = preprocess
        if enhance:
            payload["enhance"] = enhance
        return self.put("/api/augment/config", json=payload)

    def pseudo_preview(self, image_id: int, method: str,
                       params: dict | None = None) -> dict | None:
        """POST /api/pseudo/preview → {image_path, mask_path}

        method 支持引擎同款合成：defect_transplant（真实缺陷移植，默认）/
        cutpaste / color_blot。
        """
        return self.post("/api/pseudo/preview", json={
            "image_id": image_id, "method": method, "params": params or {}})

    # ── 模型 / 自学习 ────────────────────────────────────
    def list_models(self, category: str = "") -> list | None:
        """GET /api/models → [Model...]"""
        params = {"category": category} if category else {}
        data = self.get("/api/models", params=params)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return data.get("items", [])
        return None

    def prepare_model(self, category: str, scenario: str = "L1a",
                      profile: str = "fast", note: str = "",
                      force: bool = True,
                      datasource_id: int | None = None,
                      allow_experimental: bool = False) -> dict | None:
        """POST /api/models/prepare → {task_id}（少样本冷启动模型准备）

        scenario: L0/L1a/L1b/L2/L3（数据条件分层）；profile: fast/accuracy/cpu。
        datasource_id 非空时样本取自该数据源的「预训练组/检测组」（前端反馈 v9）。
        """
        payload: dict = {
            "category": category, "force": bool(force),
            "scenario": scenario, "profile": profile, "note": note,
            "allow_experimental": bool(allow_experimental)}
        if datasource_id is not None:
            payload["datasource_id"] = int(datasource_id)
        return self.post("/api/models/prepare", json=payload)

    def precheck(self, category: str, datasource_id: int | None = None) -> dict | None:
        """GET /api/models/precheck/{category} -> {counts, resolutions,
        suggested_scenario, suggested_profile, warnings}（prepare 前数据体检，
        纯 DB 统计；静默失败：后端离线/品类空时由调用方空态接管）。
        datasource_id 非空时后端追加数据血缘核验（前端反馈 2026-09-13 #4）。"""
        if not category:
            return None
        url = f"/api/models/precheck/{quote(category)}"
        if datasource_id:
            url += f"?datasource_id={int(datasource_id)}"
        return self.get(url, silent=True)

    def scan_models(self) -> dict | None:
        """POST /api/models/scan"""
        return self.post("/api/models/scan", json={})

    def activate_model(self, model_id: int, gate: bool = True,
                       force: bool = False) -> dict | None:
        """POST /api/models/{id}/activate?gate=&force=（M5 激活质量门控）

        成功返回激活结果（gate=true 时附 gate_report）；门控拒绝(409)时
        静默返回 {"_status": 409, "detail": {message, gate_report}}，
        由调用方弹窗展示对比数据并决定是否 force=True 强制激活。
        """
        return self.post(
            f"/api/models/{model_id}/activate",
            params={"gate": str(bool(gate)).lower(),
                    "force": str(bool(force)).lower()},
            conflict_ok=True)

    def rollback_model(self, category: str) -> dict | None:
        """POST /api/models/rollback"""
        return self.post("/api/models/rollback", json={"category": category})

    # ── 模型快照导出 / 导入（跨机器/跨品类复用）───────────────
    def export_snapshot(self, category: str, save_path: str,
                        version: str = "") -> str | None:
        """GET /api/models/{category}/snapshot/export → zip 落盘到 save_path。

        version 空 = 当前激活版本。成功返回 save_path；失败返回 None
        （错误信号已弹出）。"""
        params = {"version": version} if version else {}
        try:
            r = self._session.get(
                f"{self.base_url}/api/models/{quote(category)}/snapshot/export",
                params=params, timeout=300, stream=True)
        except requests.exceptions.ConnectionError:
            self._set_connected(False)
            self.error_occurred.emit("无法连接后端服务，请确认后端已启动")
            return None
        except Exception as e:  # noqa: BLE001
            self.error_occurred.emit(f"导出请求异常：{e}")
            return None
        self._set_connected(True)
        if r.status_code >= 400:
            try:
                detail = r.json().get("detail", r.text[:200])
            except Exception:  # noqa: BLE001
                detail = r.text[:200]
            self.error_occurred.emit(
                f"导出失败（HTTP {r.status_code}）：{detail}")
            return None
        try:
            with open(save_path, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
        except Exception as e:  # noqa: BLE001
            self.error_occurred.emit(f"写入文件失败：{e}")
            return None
        return save_path

    def import_snapshot(self, file_path: str, category: str) -> dict | None:
        """POST /api/models/snapshot/import（multipart：zip + 目标品类名）
        → {category, version, version_num, activated=False, message}"""
        return self.upload("/api/models/snapshot/import", file_path,
                           form={"category": category}, timeout=600)

    def self_learning_update(self, category: str, note: str = "") -> dict | None:
        """POST /api/self_learning/update → {task_id}（巩固落盘：在线学习结果固化为新快照版本）"""
        return self.post("/api/self_learning/update", json={
            "category": category, "note": note})

    def self_learning_status(self) -> dict | None:
        """GET /api/self_learning/status → {categories:[{category, versions,
        current_version, prepared, feedback_log_len, unconsumed_db}]}"""
        return self.get("/api/self_learning/status")

    def learning_insight(self, category: str) -> dict | None:
        """GET /api/self_learning/insight?category= → M14b 引擎内部状态
        {normal_bank, defect_bank, incubate, active_queue, weight_gate, ...}"""
        return self.get("/api/self_learning/insight", params={"category": category})

    # 注：/api/calibration/* 已随 检测引擎下线（410 Gone，在线 CDF 校准），
    # 前端不再提供校准集管理入口。

    # ── 评估 ─────────────────────────────────────────────
    def eval_accuracy(self, category: str, split: str = "test",
                      name: str = "", limit: int = 0) -> dict | None:
        """POST /api/eval/accuracy → {task_id}"""
        payload: dict = {"category": category, "split": split}
        if name:
            payload["name"] = name
        if limit:
            payload["limit"] = limit
        return self.post("/api/eval/accuracy", json=payload)

    def eval_benchmark(self, category: str, n_images: int = 20,
                       warmup: int = 3) -> dict | None:
        """POST /api/eval/benchmark → {task_id}"""
        return self.post("/api/eval/benchmark", json={
            "category": category, "n_images": n_images, "warmup": warmup})

    def eval_robustness(self, category: str, split: str = "test",
                        limit: int = 12) -> dict | None:
        """POST /api/eval/robustness → {task_id}（六维评价·维度2：
        亮度/高斯/模糊扰动下 AUC 衰减 + A_rob，默认小样本——约 limit×16 次前向）"""
        return self.post("/api/eval/robustness", json={
            "category": category, "split": split, "limit": limit})

    def list_eval_runs(self) -> list | None:
        """GET /api/eval/runs → [EvalRun...]"""
        data = self.get("/api/eval/runs")
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return data.get("items", [])
        return None

    def delete_eval_run(self, run_id: int) -> dict | None:
        """DELETE /api/eval/runs/{run_id}（admin 权限，仅删记录行）。

        2026-08-29 用户要求补充：评估记录此前只读不可删。
        """
        return self.delete(f"/api/eval/runs/{run_id}")

    def eval_compare(self, run_ids: list) -> dict | None:
        """POST /api/eval/compare"""
        return self.post("/api/eval/compare", json={"run_ids": run_ids})

    def ab_compare(self, category: str, version_b: int,
                   version_a: int | None = None,
                   split: str = "test", limit: int = 100) -> dict | None:
        """POST /api/eval/ab_compare → {task_id}（快照版本 A/B 对比，
        version_a 为空时对比当前激活版本）"""
        payload: dict = {"category": category, "version_b": int(version_b),
                         "split": split, "limit": int(limit)}
        if version_a is not None:
            payload["version_a"] = int(version_a)
        return self.post("/api/eval/ab_compare", json=payload)

    # ── 学习曲线 / 槽位贡献（引擎）─────────────────────────
    def learning_curve(self, category: str, n_stream: int = 20,
                       n_eval: int = 60, feedback_ratio: float = 1.0,
                       eval_every: int = 5,
                       max_feedback: int = 20,
                       mode: str = "rounds", round_size: int = 25,
                       n_rounds: int = 4,
                       eval_per_sample: int = 5) -> dict | None:
        """POST /api/learning/curve → {task_id}（离线回放，独立副本与产线并行）。
        mode=rounds（默认）：多轮持续学习；mode=stream：旧版单轮流。"""
        return self.post("/api/learning/curve", json={
            "category": category, "n_stream": int(n_stream),
            "n_eval": int(n_eval), "feedback_ratio": float(feedback_ratio),
            "eval_every": int(eval_every), "max_feedback": int(max_feedback),
            "mode": mode, "round_size": int(round_size),
            "n_rounds": int(n_rounds),
            "eval_per_sample": int(eval_per_sample)})

    def learning_curve_latest(self, category: str) -> dict | None:
        """GET /api/learning/curve/{category}/latest（静默失败：无记录返回 None）"""
        return self.get(f"/api/learning/curve/{quote(category)}/latest",
                        silent=True)

    def learning_contribution(self, category: str) -> dict | None:
        """POST /api/learning/contribution → {task_id}（槽位贡献档案）"""
        return self.post("/api/learning/contribution", json={
            "category": category})

    def learning_contribution_latest(self, category: str) -> dict | None:
        """GET /api/learning/contribution/{category}/latest（静默失败）"""
        return self.get(f"/api/learning/contribution/{quote(category)}/latest",
                        silent=True)

    def online_curve(self, category: str, window: int = 10) -> dict | None:
        """GET /api/learning/online_curve/{category}（M5 真实反馈流曲线，
        纯 DB 计算；静默失败）→ {points, summary, weights}"""
        return self.get(f"/api/learning/online_curve/{quote(category)}",
                        params={"window": int(window)}, silent=True)

    def weight_history(self, category: str) -> dict | None:
        """GET /api/learning/weight_history/{category}（权重随反馈演化；
        静默失败）→ {points:[{n_fb,action,weights}], current, n_labeled_fb}"""
        return self.get(f"/api/learning/weight_history/{quote(category)}",
                        silent=True)

    # ── 追溯→学习→提升（M7）─────────────────────────────
    def flip_curve(self, category: str) -> dict | None:
        """GET /api/learning/flip_curve/{category}（M7a 翻案曲线）
        → {points[], summary{total,flipped_count,flip_rate,legacy_count}}；
        反馈数 >50 返回 {task_id}（轮询 get_task 取 result）；
        409=品类未准备，由调用方接管提示（静默失败，其余错误不弹窗）。"""
        return self.get(f"/api/learning/flip_curve/{quote(category)}",
                        silent=True, conflict_ok=True)

    def detection_trace(self, detection_id: int) -> dict | None:
        """GET /api/detections/{id}/trace（M7a 单帧追溯链）
        → {detection, model|None, feedbacks[], post|None}"""
        return self.get(f"/api/detections/{int(detection_id)}/trace")

    def misjudged(self, category: str = "", page: int = 1,
                  page_size: int = 20) -> dict | None:
        """GET /api/detections/misjudged（M7a 错检集；静默失败）
        → {total, page, page_size, items[]}"""
        params: dict = {"page": page, "page_size": page_size}
        if category:
            params["category"] = category
        return self.get("/api/detections/misjudged", params=params,
                        silent=True)

    # ── 统计 / 日志 / 任务 ───────────────────────────────
    def stats_overview(self, category: str = "",
                       workorder_id: int | None = None,
                       range_name: str = "") -> dict | None:
        """GET /api/stats/overview（工单口径优先；category 兼容旧调用）"""
        params: dict = {}
        if workorder_id is not None:
            params["workorder_id"] = workorder_id
        elif category:
            params["category"] = category
        if range_name:
            params["range"] = range_name
        return self.get("/api/stats/overview", params=params)

    def stats_timeseries(self, days: int = 14, category: str = "") -> dict | None:
        """GET /api/stats/timeseries?days=&category= → {items: [...]}"""
        params: dict = {"days": days}
        if category:
            params["category"] = category
        return self.get("/api/stats/timeseries", params=params)

    # ── 质检报表（复核后口径，W-report 2026-08-29）─────────────
    @staticmethod
    def _scope_params(workorder_id: int | None, category: str,
                      range_name: str) -> dict:
        params: dict = {"range_name": range_name}
        if workorder_id is not None:
            params["workorder_id"] = workorder_id
        elif category:
            params["category"] = category
        return params

    def stats_quality(self, workorder_id: int | None = None,
                      category: str = "", range_name: str = "all") -> dict | None:
        """GET /api/stats/quality_overview（复核后口径 KPI：覆盖率/真实不良率/
        误报率/漏检率）"""
        return self.get("/api/stats/quality_overview",
                        params=self._scope_params(workorder_id, category,
                                                  range_name))

    def stats_mistake_trend(self, workorder_id: int | None = None,
                            category: str = "", range_name: str = "30d"
                            ) -> dict | None:
        """GET /api/stats/mistake_trend（按日 FP 率/FN 率/复核数）"""
        return self.get("/api/stats/mistake_trend",
                        params=self._scope_params(workorder_id, category,
                                                  range_name))

    def stats_defect_types(self, workorder_id: int | None = None,
                           category: str = "", range_name: str = "all"
                           ) -> dict | None:
        """GET /api/stats/defect_types（缺陷类型帕累托）"""
        return self.get("/api/stats/defect_types",
                        params=self._scope_params(workorder_id, category,
                                                  range_name))

    def stats_batch_summary(self, workorder_id: int | None = None,
                            category: str = "", range_name: str = "all"
                            ) -> dict | None:
        """GET /api/stats/batch_summary（按导入批次聚合对比）"""
        return self.get("/api/stats/batch_summary",
                        params=self._scope_params(workorder_id, category,
                                                  range_name))

    def stats_health(self, workorder_id: int | None = None,
                     category: str = "", range_name: str = "7d") -> dict | None:
        """GET /api/stats/health（灰区率/open 预警/对位预警）"""
        return self.get("/api/stats/health",
                        params=self._scope_params(workorder_id, category,
                                                  range_name))

    def list_logs(self, limit: int = 100, action: str = "",
                  category: str = "") -> list | None:
        """GET /api/logs?limit=&action=&category= → [日志...]（M8a 过滤参数）"""
        params: dict = {"limit": limit}
        if action:
            params["action"] = action
        if category:
            params["category"] = category
        data = self.get("/api/logs", params=params)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return data.get("items", [])
        return None

    def recent_tasks(self) -> list | None:
        """GET /api/tasks/recent"""
        data = self.get("/api/tasks/recent", silent=True)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return data.get("items", [])
        return None

    def get_task(self, task_id: str) -> dict | None:
        """GET /api/tasks/{id}"""
        return self.get(f"/api/tasks/{task_id}", silent=True)

    def cancel_task(self, task_id) -> dict | None:
        """POST /api/tasks/{id}/cancel（A15 协作式取消）"""
        return self.post(f"/api/tasks/{task_id}/cancel", silent=True)


class ApiWorker(QThread):
    """在后台线程执行函数（通常是 API 调用），结果经信号返回。"""

    succeeded = Signal(object)
    failed = Signal(str)

    def __init__(self, fn, parent: QObject | None = None):
        super().__init__(parent)
        self._fn = fn
        register_thread(self)

    def run(self) -> None:
        try:
            self.succeeded.emit(self._fn())
        except Exception as e:  # noqa: BLE001
            self.failed.emit(str(e))


class _WSClientBase(QThread):
    """websocket-client 后台线程基类，支持停止与空闲超时。"""

    error = Signal(str)

    def __init__(self, url: str, parent: QObject | None = None):
        super().__init__(parent)
        self._url = url
        self._stop = False
        self._ws = None
        register_thread(self)

    def stop(self) -> None:
        self._stop = True
        try:
            if self._ws is not None:
                self._ws.close()
        except Exception:  # noqa: BLE001
            pass

    def _connect(self):
        ws = websocket.create_connection(self._url, timeout=5)
        ws.settimeout(1.0)
        return ws

    def _recv_loop(self, on_message) -> None:
        while not self._stop:
            try:
                msg = self._ws.recv()
            except websocket.WebSocketTimeoutException:
                continue
            if not msg:
                break
            try:
                data = json.loads(msg)
            except Exception:  # noqa: BLE001
                continue
            if on_message(data):
                break


class TaskWSClient(_WSClientBase):
    """全局任务进度推送客户端，断线自动重连。"""

    task_updated = Signal(dict)

    def __init__(self, ws_base: str, parent: QObject | None = None):
        super().__init__(f"{ws_base}/ws/tasks", parent)

    def run(self) -> None:
        reported = False
        while not self._stop:
            try:
                self._ws = self._connect()
                reported = False
                self._recv_loop(lambda d: (self.task_updated.emit(d), False)[1])
            except Exception as e:  # noqa: BLE001
                if self._stop:
                    break
                if not reported:
                    self.error.emit(f"任务进度通道未连接：{e}")
                    reported = True
                time.sleep(3)
            finally:
                try:
                    if self._ws is not None:
                        self._ws.close()
                except Exception:  # noqa: BLE001
                    pass
