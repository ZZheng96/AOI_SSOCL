"""探针：GPU DoG（cv2 同核 getGaussianKernel + reflect101）与 CPU cv2 分数差/计时。"""
import sys, time
from pathlib import Path
import cv2, numpy as np, torch
import torch.nn.functional as F

sys.path.insert(0, r"d:\CGAIC\AOI_SSOCL\AOI_tree\AOI_Core")
from algo.slots.blob import BlobSlot  # noqa: E402

dev = torch.device("cuda")
KS = {}


def kern(s):
    if s not in KS:
        k = int(round(s * 4 * 2 + 1)) | 1
        g = cv2.getGaussianKernel(k, s, cv2.CV_32F).ravel()
        KS[s] = torch.from_numpy(g).to(dev)
    return KS[s]


def blur(x, s):
    k = kern(s); r = k.numel() // 2
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="reflect"), k.view(1, 1, 1, -1))
    return F.conv2d(F.pad(x, (0, 0, r, r), mode="reflect"), k.view(1, 1, -1, 1))


@torch.inference_mode()
def gpu_score(t, k=10):
    gu = cv2.cvtColor(t, cv2.COLOR_RGB2GRAY)
    g = torch.from_numpy(gu).to(dev, non_blocking=True).float().div_(255.0)[None, None]
    b1, b2, b4 = blur(g, 1.0), blur(g, 2.0), blur(g, 4.0)
    r = torch.maximum((b1 - b2).abs_(), (b2 - b4).abs_())
    top = torch.topk(r.flatten(), k).values.cpu().numpy()
    return float(np.partition(top, -k)[-k:].mean())


def T(f, n=5):
    b = 1e9
    for _ in range(n):
        t0 = time.perf_counter(); f(); torch.cuda.synchronize(); b = min(b, (time.perf_counter() - t0) * 1000)
    return round(b, 1)


bs = BlobSlot({})
seen = set()
paths = sorted(Path(r"d:\CGAIC\AOI_SSOCL\stable_AOI_tree\AOI_Core\storage\archive\gold_finger\20261011").glob("*.png"))[:40]
paths += sorted(p for p in Path(r"d:\CGAIC\data_local\gold_finger\test").rglob("*.png"))[::10]
rel = []
for p in paths:
    im = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
    c = bs.score_tiles(tile_imgs=[im])[0][0]
    g = gpu_score(im)
    rel.append(abs(c - g) / max(abs(c), 1e-12))
    if im.shape not in seen:
        seen.add(im.shape)
        print(im.shape, "cpu", c, "gpu", g, "rel", f"{rel[-1]:.2e}",
              "ms cpu", T(lambda: bs.score_tiles(tile_imgs=[im])), "gpu", T(lambda: gpu_score(im)), flush=True)
print("n", len(rel), "max rel", f"{max(rel):.2e}", "mean rel", f"{np.mean(rel):.2e}",
      "peak MB", torch.cuda.max_memory_allocated() / 2**20, flush=True)
