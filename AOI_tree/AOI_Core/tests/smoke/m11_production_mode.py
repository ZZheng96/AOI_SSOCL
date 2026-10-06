"""M11 冒烟：生产模式安全默认值（2026-10-02 评审整改）。

覆盖：
1. AOI_PRODUCTION=1 且未配置任何 Key → create_app() 拒绝启动
2. 非回环 host 且未配置任何 Key（即使非生产模式）→ 拒绝启动
3. 生产模式 + Key → /docs /redoc /openapi.json 关闭（带 Key 也是 404）；
   鉴权强制生效（无 Key 401 / 对 Key 200）；CORS 默认空白名单（跨域预检不放行）
4. 开发模式回归：默认配置文档开放、CORS 放行 localhost:8017、鉴权关闭透传

隔离：独立临时 sqlite/storage/cfg。同进程内通过重置
backend.core.config._settings 单例 + 直接调用 create_app() 切换场景，
避免子进程重复 import torch 的开销；不使用 lifespan，不触碰引擎与数据集。
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))   # AOI_Core 根

# ---- 隔离环境（必须在 import backend 之前设置 AOI_CONFIG）----
TMP = tempfile.mkdtemp(prefix="aoi_m11_")
CFG_PATH = os.path.join(TMP, "cfg.yaml")
import yaml  # noqa: E402

for var in ("AOI_PRODUCTION", "AOI_API_KEY", "AOI_CORS_ORIGINS"):
    os.environ.pop(var, None)


def _write_cfg(host=None):
    system = {
        "database_url": "sqlite:///" + os.path.join(TMP, "aoi.db").replace("\\", "/"),
        "storage_dir": os.path.join(TMP, "storage").replace("\\", "/"),
        "engine_storage": os.path.join(TMP, "storage", "engine").replace("\\", "/"),
        "engine_base_cfg": "configs/engine_fast.yaml",
    }
    if host is not None:
        system["host"] = host
    with open(CFG_PATH, "w", encoding="utf-8") as f:
        yaml.safe_dump({"system": system}, f, allow_unicode=True)


_write_cfg()
os.environ["AOI_CONFIG"] = CFG_PATH

KEY = {"X-API-Key": "m11-key"}
ok_count = 0


def ck(name, cond, detail=""):
    global ok_count
    assert cond, f"❌ {name}: {detail}"
    ok_count += 1
    print(f"  ✅ {name}" + (f"（{detail}）" if detail else ""), flush=True)


def _reset_settings():
    import backend.core.config as cfgmod
    cfgmod._settings = None


def main():
    from fastapi.testclient import TestClient
    import backend.api.app as appmod

    # ── 1) 生产模式无 Key 拒绝启动 ─────────────────────
    print("[M11-1] 生产模式无 Key 拒绝启动", flush=True)
    os.environ["AOI_PRODUCTION"] = "1"
    _write_cfg()                # 回环 host、无 Key
    _reset_settings()
    try:
        appmod.create_app()
        refused = False
    except RuntimeError as exc:
        refused = "生产模式" in str(exc)
    ck("生产无 Key 拒绝启动", refused)

    # ── 2) 非回环 host 无 Key 拒绝启动（开发模式也一样）──
    print("[M11-2] 非回环 host 无 Key 拒绝启动", flush=True)
    os.environ["AOI_PRODUCTION"] = "0"
    _write_cfg(host="0.0.0.0")
    _reset_settings()
    try:
        appmod.create_app()
        refused = False
    except RuntimeError as exc:
        refused = "非回环" in str(exc)
    ck("非回环无 Key 拒绝启动", refused)

    # ── 3) 生产模式 + Key：文档关闭 / 鉴权强制 / CORS 空 ──
    print("[M11-3] 生产模式 + Key 的行为", flush=True)
    os.environ["AOI_PRODUCTION"] = "1"
    os.environ["AOI_API_KEY"] = "m11-key"
    _write_cfg()
    _reset_settings()
    prod_client = TestClient(appmod.create_app())
    r = prod_client.get("/docs", headers=KEY)
    ck("生产 /docs 关闭", r.status_code == 404, f"status={r.status_code}")
    r = prod_client.get("/redoc", headers=KEY)
    ck("生产 /redoc 关闭", r.status_code == 404, f"status={r.status_code}")
    r = prod_client.get("/openapi.json", headers=KEY)
    ck("生产 openapi 关闭", r.status_code == 404, f"status={r.status_code}")
    r = prod_client.get("/api/categories")
    ck("生产无 Key 401", r.status_code == 401, f"status={r.status_code}")
    r = prod_client.get("/api/categories", headers=KEY)
    ck("生产对 Key 200", r.status_code == 200, f"status={r.status_code}")
    r = prod_client.get("/api/health")
    ck("生产 health 豁免", r.status_code == 200, f"status={r.status_code}")
    r = prod_client.options(
        "/api/health",
        headers={"Origin": "http://evil.example",
                 "Access-Control-Request-Method": "GET"})
    ck("生产 CORS 默认空白名单",
       "access-control-allow-origin" not in r.headers)

    # ── 4) 开发模式回归：文档开放 / CORS localhost / 鉴权关 ──
    print("[M11-4] 开发模式回归", flush=True)
    os.environ["AOI_PRODUCTION"] = "0"
    os.environ.pop("AOI_API_KEY", None)
    _reset_settings()
    dev_client = TestClient(appmod.create_app())
    r = dev_client.get("/docs")
    ck("开发 /docs 开放", r.status_code == 200, f"status={r.status_code}")
    r = dev_client.get("/api/categories")
    ck("开发无 Key 透传", r.status_code == 200, f"status={r.status_code}")
    r = dev_client.options(
        "/api/health",
        headers={"Origin": "http://localhost:8017",
                 "Access-Control-Request-Method": "GET"})
    ck("开发 CORS 放行 localhost:8017",
       r.headers.get("access-control-allow-origin") == "http://localhost:8017")

    # 清理环境变量，避免影响同进程后续测试
    for var in ("AOI_PRODUCTION", "AOI_API_KEY"):
        os.environ.pop(var, None)
    print(f"\n✅ M11 冒烟通过（{ok_count} 项断言）", flush=True)


if __name__ == "__main__":
    main()
