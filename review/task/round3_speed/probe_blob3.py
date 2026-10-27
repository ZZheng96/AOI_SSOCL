"""探针3：blob 带 halo 条带并行（灰度→blur→absdiff/max→top-k 融合）与整图实现逐位对照 + 计时。"""
import sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import cv2, numpy as np

sys.path.insert(0, r"d:\CGAIC\AOI_SSOCL\AOI_tree\AOI_Core")
from algo.slots.blob import BlobSlot  # noqa: E402

imgs = []
for p in Path(r"d:\CGAIC\data_local\gold_finger\test").rglob("*"):
    if p.suffix.lower() in (".png", ".jpg"):
        im = cv2.imread(str(p))
        if im is not None and not any(i.shape == im.shape for i in imgs):
            imgs.append(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    if len(imgs) >= 4:
        break
imgs.append(np.random.default_rng(0).integers(0, 256, (37, 53, 3), dtype=np.uint8))
imgs.append(np.random.default_rng(1).integers(0, 256, (5120, 5120, 3), dtype=np.uint8))
ex = ThreadPoolExecutor(8)


def T(f, n=5):
    best = 1e9
    for _ in range(n):
        t0 = time.perf_counter(); f(); best = min(best, (time.perf_counter() - t0) * 1000)
    return round(best, 1)


def strip_score(t, k=10, n_strips=8, scale=1.0, want_map=False):
    h = t.shape[0]
    halo = int(np.ceil(4 * 4 * scale)) + 4
    step = max(1, -(-h // n_strips))

    def one(y0):
        y1 = min(h, y0 + step)
        a, b = max(0, y0 - halo), min(h, y1 + halo)
        s = t[a:b]
        g = (cv2.cvtColor(s, cv2.COLOR_RGB2GRAY) if s.ndim == 3 else s).astype(np.float32)
        np.divide(g, np.float32(255.0), out=g)
        b1 = cv2.GaussianBlur(g, (0, 0), 1 * scale)
        b2 = cv2.GaussianBlur(g, (0, 0), 2 * scale)
        b4 = cv2.GaussianBlur(g, (0, 0), 4 * scale)
        r = cv2.max(cv2.absdiff(b1, b2), cv2.absdiff(b2, b4))[y0 - a:y0 - a + (y1 - y0)]
        flat = r.ravel()
        kk = min(k, flat.size)
        top = np.partition(flat, -kk)[-kk:]
        return top, (r if want_map else None)

    outs = list(ex.map(one, range(0, h, step)))
    allt = np.concatenate([o[0] for o in outs])
    kk = min(k, t.shape[0] * t.shape[1])
    return float(np.partition(allt, -kk)[-kk:].mean()), (np.vstack([o[1] for o in outs]) if want_map else None)


bs = BlobSlot({})
for t in imgs:
    def ref():
        return bs.score_tiles(tile_imgs=[t])[0][0]
    bs.with_heatmap = True
    _, hm = bs.score_tiles(tile_imgs=[t]); hm = hm[0]
    bs.with_heatmap = False
    s_ref = ref()
    for ns in (4, 8, 16):
        s_new, m = strip_score(t, n_strips=ns, want_map=True)
        print(t.shape, "strips", ns, "score_eq", s_ref == s_new, s_ref, s_new, "map_eq", np.array_equal(hm, m), flush=True)
    print("   ms ref", T(ref), *[("strips", ns, T(lambda: strip_score(t, n_strips=ns))) for ns in (4, 8, 16)], flush=True)
