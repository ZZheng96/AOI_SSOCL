"""数据库引擎与会话管理（AOI_sys db 迁入，适配 v2 config）。

连接串取自 configs/default.yaml 的 database.url（默认 sqlite:///storage/aoi.db）。
WAL + busy_timeout：后台任务与 API 请求并发读写不互锁。
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import get_settings
from app.db.models import Base

_engine = None
_SessionLocal: sessionmaker | None = None


def _migrate_sqlite(engine) -> None:
    """轻量迁移：为已存在的 SQLite 库补新增列（create_all 不会改旧表）。"""
    from sqlalchemy import inspect, text
    insp = inspect(engine)
    tables = insp.get_table_names()

    def _add_col(table: str, col: str, ddl: str) -> None:
        if table not in tables:
            return
        cols = {c["name"] for c in insp.get_columns(table)}
        if col not in cols:
            with engine.begin() as conn:
                conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {ddl}"))

    _add_col("feedback", "calibration_excluded", "calibration_excluded BOOLEAN DEFAULT 0")
    _add_col("detections", "workorder_id", "workorder_id INTEGER REFERENCES work_orders(id)")
    _add_col("feedback", "invalidated", "invalidated BOOLEAN DEFAULT 0")
    _add_col("images", "dataset_id", "dataset_id INTEGER REFERENCES datasets(id)")
    _add_col("operation_logs", "extra", "extra JSON")
    _add_col("videos", "dataset_id", "dataset_id INTEGER REFERENCES datasets(id)")
    _add_col("datasets", "dataset_name", "dataset_name VARCHAR(64) DEFAULT ''")
    _add_col("datasets", "datasource_id", "datasource_id INTEGER REFERENCES data_sources(id)")
    _add_col("data_sources", "label_tier", "label_tier VARCHAR(8) DEFAULT 'L0'")
    _add_col("work_orders", "review_enabled", "review_enabled BOOLEAN DEFAULT 0")
    _add_col("data_sources", "source_type", "source_type VARCHAR(16) DEFAULT 'local'")
    _add_col("data_sources", "stream_config", "stream_config JSON")
    _add_col("work_orders", "pipeline_status", "pipeline_status VARCHAR(16) DEFAULT 'running'")
    _add_col("work_orders", "auto_resume", "auto_resume BOOLEAN DEFAULT 0")
    _add_col("work_orders", "queue_json", "queue_json JSON")
    _add_col("data_sources", "per_category", "per_category BOOLEAN DEFAULT 0")
    _add_col("data_sources", "has_template", "has_template BOOLEAN DEFAULT 0")
    _add_col("data_sources", "pretrain_normal", "pretrain_normal INTEGER DEFAULT 100")
    _add_col("data_sources", "pretrain_anomaly", "pretrain_anomaly INTEGER DEFAULT 30")
    _add_col("data_sources", "batch_size", "batch_size INTEGER DEFAULT 30")
    _add_col("data_sources", "plan_json", "plan_json JSON")
    # v2 双检扩展列
    _add_col("detections", "engine_mode", "engine_mode VARCHAR(16) DEFAULT 'dual'")
    _add_col("detections", "template_id", "template_id VARCHAR(64)")
    _add_col("detections", "template_version", "template_version INTEGER")
    _add_col("detections", "traditional_overall", "traditional_overall VARCHAR(16)")
    _add_col("detections", "feature_decision", "feature_decision VARCHAR(16)")
    _add_col("detections", "dual_json", "dual_json JSON")
    _add_col("work_orders", "template_ref", "template_ref VARCHAR(64)")
    # P0 板级模型补列
    _add_col("work_orders", "order_code", "order_code VARCHAR(64)")
    _add_col("work_orders", "customer", "customer VARCHAR(128)")
    _add_col("work_orders", "track_index", "track_index INTEGER")
    _add_col("work_orders", "order_state", "order_state VARCHAR(16) DEFAULT 'open'")
    _add_col("work_orders", "fixture_width", "fixture_width FLOAT")
    _add_col("work_orders", "fixture_height", "fixture_height FLOAT")
    _add_col("work_orders", "fixture_offset_x", "fixture_offset_x FLOAT")
    _add_col("work_orders", "fixture_offset_y", "fixture_offset_y FLOAT")
    _add_col("detections", "inspect_id", "inspect_id INTEGER REFERENCES inspects(id)")
    _add_col("detections", "position_on", "position_on VARCHAR(64)")
    _add_col("detections", "fov_index", "fov_index INTEGER")
    _add_col("detections", "board_x", "board_x FLOAT")
    _add_col("detections", "board_y", "board_y FLOAT")
    _add_col("detections", "align_offset_x", "align_offset_x FLOAT")
    _add_col("detections", "align_offset_y", "align_offset_y FLOAT")
    _add_col("detections", "align_warn", "align_warn BOOLEAN DEFAULT 0")
    _add_col("detections", "synced", "synced BOOLEAN DEFAULT 0")
    _add_col("feedback", "final_state", "final_state VARCHAR(16)")
    _add_col("feedback", "rejudge_time", "rejudge_time DATETIME")
    _add_col("feedback", "device_key", "device_key VARCHAR(64)")
    _add_col("feedback", "device_name", "device_name VARCHAR(128)")
    _add_col("feedback", "device_url", "device_url VARCHAR(256)")
    with engine.begin() as conn:
        conn.execute(text(
            "CREATE INDEX IF NOT EXISTS ix_images_dataset_id ON images (dataset_id)"))
        conn.execute(text(
            "CREATE INDEX IF NOT EXISTS ix_detections_workorder_id ON detections (workorder_id)"))


def get_engine():
    global _engine, _SessionLocal
    if _engine is None:
        url = str(get_settings().get("database", "url", "sqlite:///storage/aoi.db"))
        if not url.startswith("sqlite"):
            raise ValueError("仅支持 SQLite（database.url）")
        if not url.startswith("sqlite:///"):
            raise ValueError(f"不支持的 SQLite URL: {url}")
        from sqlalchemy import event
        _engine = create_engine(url, connect_args={"check_same_thread": False, "timeout": 30})

        @event.listens_for(_engine, "connect")
        def _set_sqlite_pragma(dbapi_conn, _record):  # noqa: ANN001
            cur = dbapi_conn.cursor()
            try:
                cur.execute("PRAGMA journal_mode=WAL")
                cur.execute("PRAGMA busy_timeout=30000")
            finally:
                cur.close()
        Base.metadata.create_all(_engine)
        _migrate_sqlite(_engine)
        _SessionLocal = sessionmaker(bind=_engine, expire_on_commit=False)
    return _engine


def get_session() -> Session:
    get_engine()
    return _SessionLocal()


@contextmanager
def session_scope() -> Iterator[Session]:
    s = get_session()
    try:
        yield s
        s.commit()
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()


def log_action(action: str, detail: str = "", user: str = "operator",
               extra: dict | None = None) -> None:
    from app.db.models import OperationLog
    with session_scope() as s:
        s.add(OperationLog(user=user, action=action, detail=detail, extra=extra))
