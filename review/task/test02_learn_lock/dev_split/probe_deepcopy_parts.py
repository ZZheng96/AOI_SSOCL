"""探针：影子 deepcopy 按属性分解 + 共享 item 大数组后的成本（少量本地样本）。"""
import copy, json, os, random, shutil, sys, tempfile, time
from pathlib import Path

import yaml

ROOT = Path(r"d:\CGAIC\AOI_SSOCL\AOI_tree\AOI_Core")
OUT = Path(r"d:\CGAIC\AOI_SSOCL\review\task\test02_learn_lock\dev_split")
ARCHIVE = Path(r"d:\CGAIC\AOI_SSOCL\stable_AOI_tree\AOI_Core\storage\archive\gold_finger\20261011")


def main():
    tmp = Path(tempfile.mkdtemp(prefix="aoi_probe_dc_"))
    cfg = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text(encoding="utf-8"))
    cfg["system"].update(storage_dir=str(tmp / "storage"), engine_storage=str(tmp / "engine"),
                         engine_base_cfg=str(ROOT / "configs" / "engine_fast.yaml"),
                         database_url="sqlite:///" + (tmp / "p.db").as_posix(), host="127.0.0.1")
    cfg.setdefault("notify", {})["webhook_url"] = ""
    (tmp / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    os.environ["AOI_CONFIG"] = str(tmp / "config.yaml")
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    import numpy as np
    import torch
    from PIL import Image
    pool = []
    for p in sorted(ARCHIVE.iterdir()):
        if p.suffix.lower() in (".jpg", ".png"):
            with Image.open(p) as im:
                if im.size == (2500, 2500):
                    pool.append(p)
    random.seed(777)
    random.shuffle(pool)
    from backend.engine import get_engine
    engine = get_engine()
    engine.learning_parallel = False
    td = tmp / "train"
    td.mkdir(parents=True)
    paths = []
    for i, s in enumerate(pool[:40]):
        d = td / f"n{i:03d}{s.suffix}"
        shutil.copy2(s, d)
        paths.append(str(d))
    t0 = time.perf_counter()
    engine.prepare("gf", {"init_normal": paths, "init_defect": [], "val": [], "test": []},
                   activate=True, progress_cb=lambda f, m: None)
    print(f"[prepare] {time.perf_counter()-t0:.1f}s", flush=True)
    for i, p in enumerate(pool[40:46]):
        engine.predict_image_path("gf", str(p))
        engine.submit_feedback("gf", str(p), "correct", label=0)
        print(f"  [fb {i+1}/6]", flush=True)
    pipe, _ = engine._pipe_entry("gf")

    def base_memo():
        m = {id(pipe.backbone): pipe.backbone, id(pipe._cpu_exec): pipe._cpu_exec,
             id(pipe._trad_lock): pipe._trad_lock, id(pipe._item_cache): {}}
        if pipe.trace_api is not None:
            m[id(pipe.trace_api)] = pipe.trace_api
        return m

    def share_items(m):
        items = list(getattr(pipe, "_anchor_items", None) or [])
        h = pipe.handler
        for k in ("labeled_items", "_labeled_items", "labeled"):
            v = getattr(h, k, None)
            if isinstance(v, list):
                items += [x for x in v if isinstance(x, dict)]
        n = 0
        for it in items:
            for v in it.values():
                if isinstance(v, (np.ndarray, torch.Tensor)) or (
                        isinstance(v, list) and v and isinstance(v[0], np.ndarray)):
                    m[id(v)] = v
                    n += 1
        return n

    res = {"parts": {}, "handler_parts": {}}
    for k, v in pipe.__dict__.items():
        t = time.perf_counter()
        try:
            copy.deepcopy(v, base_memo())
            res["parts"][k] = round((time.perf_counter() - t) * 1000, 1)
        except Exception as e:  # noqa: BLE001
            res["parts"][k] = repr(e)[:80]
    for k, v in pipe.handler.__dict__.items():
        t = time.perf_counter()
        try:
            copy.deepcopy(v, base_memo())
            res["handler_parts"][k] = round((time.perf_counter() - t) * 1000, 1)
        except Exception as e:  # noqa: BLE001
            res["handler_parts"][k] = repr(e)[:80]
    for tag, mk in (("full", lambda: base_memo()), ("share_items", None)):
        ts = []
        for _ in range(3):
            m = base_memo()
            if mk is None:
                res["shared_arrays"] = share_items(m)
            t = time.perf_counter()
            copy.deepcopy(pipe, m)
            ts.append(round((time.perf_counter() - t) * 1000, 1))
        res[tag] = ts
    res["parts"] = dict(sorted(((k, v) for k, v in res["parts"].items()),
                               key=lambda kv: -(kv[1] if isinstance(kv[1], float) else 0))[:25])
    res["handler_parts"] = dict(sorted(((k, v) for k, v in res["handler_parts"].items()),
                                       key=lambda kv: -(kv[1] if isinstance(kv[1], float) else 0))[:25])
    print(json.dumps(res, ensure_ascii=False, indent=1), flush=True)
    (OUT / "probe_deepcopy_parts.json").write_text(json.dumps(res, ensure_ascii=False, indent=2),
                                                   encoding="utf-8")
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
