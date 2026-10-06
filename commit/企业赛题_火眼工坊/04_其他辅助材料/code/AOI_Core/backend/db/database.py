"""数据库引擎与会话管理。"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from ..core.config import get_settings
from .models import Base

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

    _add_col("feedback", "calibration_excluded",
             "calibration_excluded BOOLEAN DEFAULT 0")
    # 2026-08-29 收尾：检测工单归属（复核/实时流精确口径）+ 反馈作废
    _add_col("detections", "workorder_id",
             "workorder_id INTEGER REFERENCES work_orders(id)")
    _add_col("feedback", "invalidated", "invalidated BOOLEAN DEFAULT 0")
    # M8a 数据资产治理
    _add_col("images", "dataset_id",
             "dataset_id INTEGER REFERENCES datasets(id)")
    _add_col("operation_logs", "extra", "extra JSON")
    _add_col("videos", "dataset_id",
             "dataset_id INTEGER REFERENCES datasets(id)")
    # U-opsflow：数据集层级（数据集 -> 品类 -> 批次）
    _add_col("datasets", "dataset_name", "dataset_name VARCHAR(64) DEFAULT ''")
    # W-source：批次归属数据源
    _add_col("datasets", "datasource_id",
             "datasource_id INTEGER REFERENCES data_sources(id)")
    # 前端反馈 v4：条件下沉到数据源（标注档位）+ 工单人工复判开关
    _add_col("data_sources", "label_tier",
             "label_tier VARCHAR(8) DEFAULT 'L0'")
    _add_col("work_orders", "review_enabled",
             "review_enabled BOOLEAN DEFAULT 0")
    # 前端反馈 v5：数据源数据流（本地/实时）+ 工单产线状态与恢复配置
    _add_col("data_sources", "source_type",
             "source_type VARCHAR(16) DEFAULT 'local'")
    _add_col("data_sources", "stream_config", "stream_config JSON")
    _add_col("work_orders", "pipeline_status",
             "pipeline_status VARCHAR(16) DEFAULT 'running'")
    _add_col("work_orders", "auto_resume",
             "auto_resume BOOLEAN DEFAULT 0")
    _add_col("work_orders", "queue_json", "queue_json JSON")
    # 前端反馈 v6：数据源声明（分品类 / 提供模板图）
    _add_col("data_sources", "per_category",
             "per_category BOOLEAN DEFAULT 0")
    _add_col("data_sources", "has_template",
             "has_template BOOLEAN DEFAULT 0")
    # 前端反馈 v9：数据源内「预训练组」+「检测组」配置与分组方案
    _add_col("data_sources", "pretrain_normal",
             "pretrain_normal INTEGER DEFAULT 100")
    _add_col("data_sources", "pretrain_anomaly",
             "pretrain_anomaly INTEGER DEFAULT 30")
    _add_col("data_sources", "batch_size",
             "batch_size INTEGER DEFAULT 30")
    _add_col("data_sources", "plan_json", "plan_json JSON")
    # 端到端延迟口径（2026-08-30）：检测行与每日累计各补一列
    _add_col("detections", "latency_e2e_ms", "latency_e2e_ms FLOAT")
    _add_col("stats_daily", "total_latency_e2e_ms",
             "total_latency_e2e_ms FLOAT DEFAULT 0")
    with engine.begin() as conn:
        conn.execute(text(
            "CREATE INDEX IF NOT EXISTS ix_images_dataset_id "
            "ON images (dataset_id)"))
        conn.execute(text(
            "CREATE INDEX IF NOT EXISTS ix_detections_workorder_id "
            "ON detections (workorder_id)"))


def get_engine():
    global _engine, _SessionLocal
    if _engine is None:
        url = get_settings().database_url
        if url.startswith("sqlite"):
            from sqlalchemy import event
            # 2026-08-31 走查改进：SQLite 库文件目录自动创建。
            # commit 打包缺 database/ 目录时首次启动报 "unable to open
            # database file"（shift2 走查复现），此处幂等 mkdir 根治。
            if "sqlite:///" in url:
                db_path = Path(url[len("sqlite:///"):])
                if not db_path.is_absolute():
                    db_path = Path(__file__).resolve().parent.parent.parent / db_path
                db_path.parent.mkdir(parents=True, exist_ok=True)
            _engine = create_engine(
                url, connect_args={"check_same_thread": False, "timeout": 30})
            # WAL + busy_timeout：后台任务（导入/训练/产线检测）与
            # API 请求并发读写时，读不阻塞写、写等待 30s 而非 5s 报
            # "database is locked"（走查复现：导入大事务期 UI 读 500）
            @event.listens_for(_engine, "connect")
            def _set_sqlite_pragma(dbapi_conn, _record):  # noqa: ANN001
                cur = dbapi_conn.cursor()
                try:
                    cur.execute("PRAGMA journal_mode=WAL")
                    cur.execute("PRAGMA busy_timeout=30000")
                finally:
                    cur.close()
        else:
            _engine = create_engine(url)
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
    """操作日志；extra 为结构化字段（M8a，如 {image_id, dataset_id, category}）。"""
    from .models import OperationLog
    with session_scope() as s:
        s.add(OperationLog(user=user, action=action, detail=detail, extra=extra))
