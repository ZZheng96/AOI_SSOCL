import hashlib, json, random, sys
from pathlib import Path
from PIL import Image

SRC = Path(r"d:\CGAIC\AOI_SSOCL\stable_AOI_tree\AOI_Core\storage\archive\gold_finger\20261011")
OUT = Path(__file__).resolve().parent / "images.json"
N = 40
SEED = 20261013

def digest(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

files = []
for p in sorted(SRC.iterdir()):
    if p.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
        continue
    with Image.open(p) as im:
        if im.size == (2500, 2500):
            files.append(p)
print(f"native 2500x2500 pool: {len(files)}")
random.seed(SEED)
sample = random.sample(files, N)
rows = [{"path": str(p), "sha256": digest(p), "source_size": [2500, 2500]} for p in sample]
OUT.write_text(json.dumps({"native": True, "seed": SEED, "n": N, "images": rows},
                          ensure_ascii=False, indent=2), encoding="utf-8")
print(f"manifest: {OUT} ({N} images)")
