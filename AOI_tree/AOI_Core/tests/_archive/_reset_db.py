# -*- coding: utf-8 -*-
"""一键清空数据库全部数据（保留表结构）：删除所有表并按当前模型重建。

用途：用户明确要求"清空所有数据"，从零开始按 数据源 → 导入 重新组织。
注意：只清数据库记录，storage 下已拷贝的图片文件保留（重新导入可复用，
      也可用「孤儿文件清理」清掉残留）。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.core.config import get_settings
from backend.db.database import get_engine
from backend.db.models import Base

if __name__ == "__main__":
    engine = get_engine()
    print("数据库:", get_settings().database_url, flush=True)
    # 外键约束在会话级关闭；直接 drop_all + create_all 重建全部表（结构最新）
    with engine.begin() as conn:
        conn.exec_driver_sql("PRAGMA foreign_keys=OFF")
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    with engine.begin() as conn:
        tables = [
            r for r in conn.exec_driver_sql(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name NOT LIKE 'sqlite_%'").fetchall()]
    print(f"已清空并重建表：{len(tables)} 张", flush=True)
    for (t,) in tables:
        print("  -", t, flush=True)
