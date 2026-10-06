"""前端反馈 v8 后端冒烟：数据源清数据（重建）+ 上传挂源。

隔离环境：独立临时 sqlite/storage/cfg。
"""
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

TMP = tempfile.mkdtemp(prefix="aoi_v8_")
CFG_PATH = os.path.join(TMP, "cfg.yaml")
import yaml  # noqa: E402

with open(CFG_PATH, "w", encoding="utf-8") as f:
    yaml.safe_dump({
        "system": {
            "database_url": "sqlite:///" + os.path.join(TMP, "aoi.db").replace("\\", "/"),
            "storage_dir": os.path.join(TMP, "storage").replace("\\", "/"),
            "demo5_storage": os.path.join(TMP, "storage", "demo5").replace("\\", "/"),
            "demo5_base_cfg": "configs/demo5_fast.yaml",
        },
        "archive": {"enabled": False},
    }, f, allow_unicode=True)
os.environ["AOI_CONFIG"] = CFG_PATH

ok = 0


def ck(name, cond, detail=""):
    global ok
    assert cond, f"❌ {name}: {detail}"
    ok += 1
    print(f"  ✅ {name}" + (f"（{detail}）" if detail else ""), flush=True)


def main() -> None:
    from fastapi.testclient import TestClient
    from backend.api.app import app

    with TestClient(app) as client:
        # ── 1) 清空数据源数据（重建用） ────────────────
        print("[V1] 清空数据源数据", flush=True)
        from backend.db.database import session_scope
        from backend.db.models import DataSource, Dataset, Image as ImageRow
        with session_scope() as s:
            src = DataSource(name="v8_src", modality="image")
            s.add(src); s.flush()
            ds = Dataset(name="批次A", category="catA", datasource_id=src.id)
            s.add(ds); s.flush()
            p = os.path.join(TMP, "a.png")
            with open(p, "wb") as fh:
                fh.write(b"\x89PNG\r\n\x1a\n" + b"0" * 32)
            s.add(ImageRow(path=p, category="catA", dataset_id=ds.id,
                           split="train", label="normal"))
            s.flush()
            sid = src.id
        r = client.delete(f"/api/datasources/{sid}/data")
        ck("清数据成功", r.status_code == 200 and r.json().get("cleared")
           and r.json().get("images") == 1, str(r.json()))
        r = client.get("/api/datasources")
        d = [x for x in r.json()["items"] if x["id"] == sid][0]
        ck("源登记保留、数据清空", d.get("capability", {}).get("n_images") == 0,
           str(d.get("capability")))
        r = client.delete("/api/datasources/99999/data")
        ck("清不存在 404", r.status_code == 404, str(r.status_code))

        # ── 2) 上传挂源 ────────────────────────────────
        print("[V2] 上传图片挂数据源", flush=True)
        p = os.path.join(TMP, "up.png")
        with open(p, "wb") as fh:
            fh.write(b"\x89PNG\r\n\x1a\n" + b"1" * 32)
        with open(p, "rb") as fh:
            r = client.post("/api/images/upload",
                            data={"category": "catA", "datasource_id": str(sid)},
                            files={"file": ("up.png", fh, "image/png")})
        ck("上传成功且挂源", r.status_code == 200, str(r.status_code))
        with session_scope() as s:
            img = (s.query(ImageRow).order_by(ImageRow.id.desc()).first())
            ds2 = s.get(Dataset, img.dataset_id)
            ck("上传图片批次挂到数据源", ds2.datasource_id == sid,
               str(ds2.datasource_id))

    shutil.rmtree(TMP, ignore_errors=True)
    print(f"\n全部通过 ✅（{ok} 项）", flush=True)


if __name__ == "__main__":
    main()
