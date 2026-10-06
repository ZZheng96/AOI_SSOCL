"""特征函数库：15 组特征的参数化实现 + 自定义特征注册机制

每组特征是从 algo/vendor/traditional.py 的单体 extract() 移植的独立函数，
**默认参数与主干逐位一致**（parity 校验保证），差异仅在于：
  1. 所有超参数暴露在 params 中，可经 feature_set 配置修改
  2. 每组自包含（⑪逻辑组自算 Otsu 阈值，不再依赖 ⑤ 的中间变量）

函数签名：f(ctx, params) -> list[float]
  ctx = {"img": RGB 图, "gray": 灰度, "hsv": HSV}（均已按 preprocess 配置缩放）
自定义特征：在 custom_features.py 中用 @feature 装饰器注册即可被 manage.py add 使用。
"""
import cv2
import numpy as np
import pywt
from skimage.feature import local_binary_pattern, hog, graycomatrix, graycoprops
from skimage.measure import shannon_entropy
from skimage import morphology

# ── 注册表 ───────────────────────────────────────────────────────────

FEATURE_FUNCS = {}    # key -> {"func", "name", "rationale", "targets", "default_params"}


def feature(key: str, name: str, rationale: str, targets: str,
            default_params: dict | None = None):
    """特征函数注册装饰器（内置 15 组与自定义特征共用）"""
    def deco(f):
        FEATURE_FUNCS[key] = {
            "func": f, "name": name, "rationale": rationale,
            "targets": targets, "default_params": default_params or {},
        }
        return f
    return deco


def preprocess(img, max_side=256, min_side=32):
    """与主干一致的缩放：最长边 ≤max_side 降采样；最短边 ≥min_side（HOG 需要）"""
    h, w = img.shape[:2]
    if max(h, w) > max_side:
        s = max_side / max(h, w)
        img = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))))
    h, w = img.shape[:2]
    if min(h, w) < min_side:
        s = min_side / min(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)))
    return img


def make_ctx(img):
    return {"img": img,
            "gray": cv2.cvtColor(img, cv2.COLOR_RGB2GRAY),
            "hsv": cv2.cvtColor(img, cv2.COLOR_RGB2HSV)}


def _cv2_canny(gray, sigma=1, low=0.1, high=0.3):
    blur = cv2.GaussianBlur(gray, (0, 0), sigma)
    lo = max(1, int(255 * low))
    hi = min(255, int(255 * high))
    return cv2.Canny(blur, lo, hi)


# ── ① GLCM 纹理 ─────────────────────────────────────────────────────

@feature("glcm", "① GLCM 纹理",
         "灰度共生矩阵的对比度/相关性/能量/同质性是纹理统计的经典量，"
         "对规则纹理被破坏敏感且计算确定",
         "纹理划伤、织物/皮革表面异常、污染",
         {"distances": [1], "angles_deg": [0, 45, 90, 135], "levels": 256,
          "props": ["contrast", "correlation", "energy", "homogeneity"]})
def glcm(ctx, p):
    gray = ctx["gray"]
    levels = p["levels"]
    if levels < 256:
        # 非默认量化级数时先量化（skimage 要求图像最大值 < levels）；
        # levels=256 时保持原图，与主干逐位一致
        gray = np.floor(gray.astype(np.float32) / 256 * levels).astype(np.uint8)
    m = graycomatrix(gray, distances=p["distances"],
                     angles=[np.radians(a) for a in p["angles_deg"]],
                     levels=levels, symmetric=True, normed=True)
    return [float(graycoprops(m, prop).mean()) for prop in p["props"]]


# ── ② 区域自相似 ────────────────────────────────────────────────────

@feature("selfsim", "② 区域自相似",
         "正常工业品纹理/布局具重复性，块间均距破坏即异常信号；"
         ">max_blocks 时子采样（调用方负责播种，U27）",
         "重复纹理破坏、局部结构错乱",
         {"scales": [4, 8, 16], "max_blocks": 100})
def selfsim(ctx, p):
    gray = ctx["gray"]
    h, w = gray.shape
    out = []
    for scale in p["scales"]:
        blocks = [gray[i:i + scale, j:j + scale].ravel()
                  for i in range(0, h - scale + 1, scale)
                  for j in range(0, w - scale + 1, scale)]
        if len(blocks) > 1:
            blocks = np.array(blocks, dtype=np.float32)
            if len(blocks) > p["max_blocks"]:
                idx = np.random.choice(len(blocks), p["max_blocks"], replace=False)
                blocks = blocks[idx]
            bnorm = (blocks ** 2).sum(axis=1)
            dist2 = bnorm[:, None] + bnorm[None, :] - 2 * (blocks @ blocks.T)
            np.fill_diagonal(dist2, 0.0)
            out.append(float(np.sqrt(dist2[dist2 > 0]).mean()))
        else:
            out.append(0.0)
    out.append(float(np.std(gray[:8, :8] - gray[8:16, 8:16])))
    out.append(float(np.std(gray[::8, ::8])))
    return out


# ── ③ 边缘梯度 ──────────────────────────────────────────────────────

@feature("edge", "③ 边缘梯度",
         "多尺度 Canny 密度 + Sobel 梯度 + 方向直方图 + LBP，"
         "覆盖边缘强度/密度/方向三类信息（cv2 实现，比 skimage 快 ~50 倍）",
         "缺损、断线、崩边、裂纹",
         {"canny_sigmas": [1, 2, 3], "grad_bins": 16,
          "lbp_P": 8, "lbp_R": 1, "lbp_bins": 6})
def edge(ctx, p):
    gray = ctx["gray"]
    out = [float(_cv2_canny(gray, sigma=s).mean()) for s in p["canny_sigmas"]]
    sh = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sv = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    out += [float(sh.mean()), float(sv.mean()), float(np.sqrt(sh ** 2 + sv ** 2).mean())]
    mag = np.sqrt(sh ** 2 + sv ** 2)
    ang = np.arctan2(sv, sh + 1e-8)
    hist, _ = np.histogram(ang, bins=p["grad_bins"], range=(-np.pi, np.pi), weights=mag)
    out += hist.tolist()
    lbp = local_binary_pattern(gray, p["lbp_P"], p["lbp_R"], method="uniform")
    lbp_hist, _ = np.histogram(lbp, bins=p["lbp_bins"], range=(0, p["lbp_bins"]),
                               density=True)
    out += lbp_hist.tolist()
    return out


# ── ④ HOG ────────────────────────────────────────────────────────────

@feature("hog", "④ HOG",
         "cell 级方向梯度直方图捕获局部形状结构，与整图统计互补",
         "结构变形、形状畸变",
         {"orientations": 9, "cell": 16, "block": 2, "out_dim": 11})
def hog_(ctx, p):
    fd = hog(ctx["gray"], orientations=p["orientations"],
             pixels_per_cell=(p["cell"], p["cell"]),
             cells_per_block=(p["block"], p["block"]),
             feature_vector=True, block_norm="L2-Hys")
    idx = np.linspace(0, len(fd) - 1, p["out_dim"], dtype=int)
    return fd[idx].tolist()


# ── ⑤ 形状/几何 ─────────────────────────────────────────────────────

@feature("shape", "⑤ 形状/几何",
         "Hu 矩旋转/尺度不变 + 轮廓统计，是缺件/错位类几何异常的代理量",
         "缺件、错位、轮廓异常", {})
def shape(ctx, p):
    gray = ctx["gray"]
    h, w = gray.shape
    hu = cv2.HuMoments(cv2.moments(gray)).flatten()
    out = (-np.sign(hu) * np.log10(np.abs(hu) + 1e-10)).tolist()
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        areas = [cv2.contourArea(c) for c in contours]
        out.append(float(np.log1p(np.mean(areas))))
        out.append(float(np.log1p(np.std(areas))))
        out.append(len(contours) / (h * w) * 1e4)
        perims = [cv2.arcLength(c, True) for c in contours]
        out.append(float(np.mean(perims) / (h + w)))
    else:
        out += [0, 0, 0, 0]
    return out


# ── ⑥ 灰度统计 ──────────────────────────────────────────────────────

@feature("gray", "⑥ 灰度统计",
         "亮度/对比度/熵/分位数等全局量，捕获光照与整体灰度漂移",
         "污渍、氧化变色、曝光异常", {})
def gray_stats(ctx, p):
    gray = ctx["gray"]
    return [float(gray.mean()), float(gray.std()), float(gray.min()), float(gray.max()),
            float(np.percentile(gray, 25)), float(np.percentile(gray, 50)),
            float(np.percentile(gray, 75)), float(shannon_entropy(gray)),
            float(np.mean(gray ** 2)),
            float(np.mean(np.abs(gray - gray.mean()) ** 3)),
            float(np.mean(gray ** 4)),
            float((gray.max() - gray.min()) / (gray.mean() + 1e-8))]


# ── ⑦ 材质反光 ──────────────────────────────────────────────────────

@feature("gloss", "⑦ 材质反光",
         "高光占比/强度/连通性刻画镜面反射区，对表面凹凸与划伤敏感",
         "划痕、凹陷、抛光不良",
         {"sigma_mult": 2.0, "min_size": 50})
def gloss(ctx, p):
    gray = ctx["gray"]
    h, w = gray.shape
    thr = gray.mean() + p["sigma_mult"] * gray.std()
    specular = gray > thr
    out = [float(specular.mean()),
           float(gray[specular].std()) if specular.any() else 0.0]
    _, bw = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY)
    bw = morphology.remove_small_objects(bw.astype(bool), min_size=p["min_size"])
    out.append(float(bw.mean()))
    out.append(len(np.unique(bw)) / (h * w) * 1e4)
    return out


# ── ⑧ 色彩 HSV+矩 ───────────────────────────────────────────────────

@feature("color", "⑧ 色彩 HSV+矩",
         "HSV 直方图 + 色彩矩 + RGB 通道相关性，是异色类缺陷的直接通道",
         "变色、异色、脏污、错料",
         {"hsv_bins": 8})
def color(ctx, p):
    hsv, img = ctx["hsv"], ctx["img"]
    out = []
    for ch in range(3):
        hist, _ = np.histogram(hsv[:, :, ch], bins=p["hsv_bins"], range=(0, 256),
                               density=True)
        out += hist.tolist()
        d = hsv[:, :, ch].astype(float)
        out += [float(d.mean()), float(d.std() / 255),
                float(np.mean(((d - d.mean()) / (d.std() + 1e-8)) ** 3))]
    for i in range(3):
        for j in range(i + 1, 3):
            cc = np.corrcoef(img[:, :, i].ravel(), img[:, :, j].ravel())[0, 1]
            out.append(float(cc) if not np.isnan(cc) else 0.0)
    return out


# ── ⑨ FFT 频域 ──────────────────────────────────────────────────────

@feature("fft", "⑨ FFT 频域",
         "高低频能量比/频谱熵/径向分布刻画周期结构，网格/纹理类异常的频域指纹",
         "周期结构异常、纹理频率变化",
         {"radius_frac": 4, "radial_bins": 10})
def fft_(ctx, p):
    gray = ctx["gray"]
    h, w = gray.shape
    mag = np.abs(np.fft.fftshift(np.fft.fft2(gray)))
    cy, cx = h // 2, w // 2
    mask = np.zeros_like(gray, dtype=bool)
    cv2.circle(mask, (cx, cy), min(h, w) // p["radius_frac"], True, -1)
    total = mag.sum() + 1e-8
    low_ratio = mag[mask].sum() / total
    out = [float(low_ratio), float(1 - low_ratio)]
    prob = mag / total
    out.append(float(-np.sum(prob * np.log(prob + 1e-10))))
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    bins = np.linspace(0, r.max(), p["radial_bins"] + 1)
    for i in range(p["radial_bins"]):
        out.append(float(mag[(r >= bins[i]) & (r < bins[i + 1])].mean() / total))
    return out


# ── ⑩ Haar 小波 ─────────────────────────────────────────────────────

@feature("wavelet", "⑩ Haar 小波",
         "多尺度子带能量定位「哪一级尺度」出现异常，比 FFT 多空间定位能力",
         "多尺度边缘/纹理异常",
         {"levels": 3})
def wavelet(ctx, p):
    coeffs = pywt.wavedec2(ctx["gray"], "haar", level=p["levels"])
    out = []
    for i, c in enumerate(coeffs):
        if i == 0:
            out += [float(c.mean()), float(c.std()), float(c.min()), float(c.max())]
        else:
            for detail in c:
                out += [float(detail.mean()), float(detail.std()),
                        float(np.mean(detail ** 2)), float(np.mean(np.abs(detail)))]
    return out


# ── ⑪ 逻辑关系 ──────────────────────────────────────────────────────

@feature("logic", "⑪ 逻辑关系",
         "连通域统计 + 左右/上下对称性是缺件/错位的单图可计算代理"
         "（真正的逻辑判定靠模板槽位）",
         "缺件、错位、多/少部件", {})
def logic(ctx, p):
    gray = ctx["gray"]
    h, w = gray.shape
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    num, _, stats, _ = cv2.connectedComponentsWithStats(thresh, connectivity=8)
    out = [num / (h * w) * 1e4]
    if num > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        out += [float(np.mean(areas) / (h * w)), float(np.std(areas) / (h * w)),
                len(areas[areas > areas.mean()]) / max(1, len(areas))]
    else:
        out += [0, 0, 0]
    out.append(float(np.mean(np.abs(gray[:, :w // 2] - gray[:, :w // 2][:, ::-1]))))
    out.append(float(np.mean(np.abs(gray[:h // 2, :] - gray[:h // 2, :][::-1, :]))))
    return out


# ── ⑫ Gabor ─────────────────────────────────────────────────────────

@feature("gabor", "⑫ Gabor",
         "多方向多频率 Gabor 能量对方向性纹理具选择性",
         "划痕、断线等方向性缺陷",
         {"thetas_deg": [0, 45, 90, 135], "freqs": [0.1, 0.3],
          "ksize": 21, "sigma": 4.0, "gamma": 0.5})
def gabor(ctx, p):
    gray = ctx["gray"]
    out = []
    for theta in p["thetas_deg"]:
        for freq in p["freqs"]:
            kern = cv2.getGaborKernel((p["ksize"], p["ksize"]), p["sigma"],
                                      np.radians(theta), 1 / freq, p["gamma"], 0)
            out.append(float(cv2.filter2D(gray, cv2.CV_32F, kern).mean()))
    return out


# ── ⑬ LoG 斑点 ──────────────────────────────────────────────────────

@feature("log", "⑬ LoG 斑点",
         "多尺度高斯拉普拉斯响应捕获孤立点状异常",
         "污染点、气泡、针孔",
         {"sigmas": [1, 2, 4, 8, 16]})
def log_blob(ctx, p):
    gray = ctx["gray"]
    out = []
    for sigma in p["sigmas"]:
        blur = cv2.GaussianBlur(gray, (0, 0), sigma)
        out.append(float(np.abs(cv2.Laplacian(blur, cv2.CV_32F)).mean()))
    return out


# ── ⑭ 相位一致性 ────────────────────────────────────────────────────

@feature("pc", "⑭ 相位一致性",
         "相位同步性对光照/对比度变化鲁棒（简化实现：多尺度 LoG 响应近似）",
         "光照变化下的边缘类缺陷",
         {"nscale": 4})
def pc(ctx, p):
    gray = ctx["gray"]
    pcs = []
    for s in range(1, p["nscale"] + 1):
        blur = cv2.GaussianBlur(gray, (0, 0), 2 ** s)
        pcs.append(np.abs(cv2.Laplacian(blur, cv2.CV_32F)))
    pcm = np.mean(pcs, axis=0)
    pcm = pcm / (pcm.max() + 1e-8)
    return [float(pcm.mean()), float(pcm.std()), float(pcm.max())]


# ── ⑮ SURF ──────────────────────────────────────────────────────────

@feature("surf", "⑮ SURF",
         "多尺度 Hessian 斑点统计提供尺度不变的结构/纹理描述"
         "（自研简化版，OpenCV5 已移除 xfeatures2d）",
         "尺度变化的结构/纹理异常",
         {"n_octaves": 3, "n_scales": 3, "base_sigma": 1.6,
          "sigma_step": 1.26, "rel_thresh": 0.01})
def surf(ctx, p):
    return _surf_features(ctx["gray"], p["n_octaves"], p["n_scales"],
                          p["base_sigma"], p["sigma_step"], p["rel_thresh"])[0]


def _surf_features(gray, n_octaves=3, n_scales=3, base_sigma=1.6,
                   sigma_step=1.26, rel_thresh=0.01):
    """自研简化 SURF（与主干 vendor/traditional.py 逐行同源）"""
    h, w = gray.shape
    img = gray.astype(np.float32) / 255.0
    kps = []
    oct_resp = [[] for _ in range(n_octaves)]
    sigma = base_sigma
    for o in range(n_octaves):
        img_o = cv2.resize(img, (max(w >> o, 16), max(h >> o, 16)),
                           interpolation=cv2.INTER_LINEAR) if o else img
        oh, ow = img_o.shape[:2]
        if min(oh, ow) < 24:
            break
        s = sigma
        for _ in range(n_scales):
            blur = cv2.GaussianBlur(img_o, (0, 0), s)
            hxx = cv2.Sobel(blur, cv2.CV_32F, 2, 0, ksize=3)
            hyy = cv2.Sobel(blur, cv2.CV_32F, 0, 2, ksize=3)
            hxy = cv2.Sobel(blur, cv2.CV_32F, 1, 1, ksize=3)
            det = np.maximum((hxx * hyy - hxy * hxy) * (s ** 4), 0)
            gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
            ang = np.mod(np.degrees(np.arctan2(gy, gx)), 180.0)
            scale_abs = s * (2 ** o)
            peak = float(det.max()) if det.size else 0.0
            thr = peak * rel_thresh if peak > 1e-9 else 0.0
            if thr > 0:
                from scipy.ndimage import maximum_filter
                mask = (det >= thr) & (det == maximum_filter(det, size=3))
                yx = np.argwhere(mask)
                if yx.size:
                    keep = ((yx[:, 0] >= 2) & (yx[:, 0] < oh - 2)
                            & (yx[:, 1] >= 2) & (yx[:, 1] < ow - 2))
                    yx = yx[keep][:300]
                    if yx.size:
                        xs = yx[:, 1].astype(np.float64) / ow * w
                        ys = yx[:, 0].astype(np.float64) / oh * h
                        rv = det[yx[:, 0], yx[:, 1]].astype(np.float64)
                        ov = ang[yx[:, 0], yx[:, 1]].astype(np.float64)
                        for i in range(len(yx)):
                            kps.append((xs[i], ys[i], scale_abs, rv[i], ov[i]))
                            oct_resp[o].append(rv[i])
            s *= sigma_step
        sigma *= 2.0

    n_kp = len(kps)
    area = float(h * w)
    oct_means = [float(np.mean(v)) if v else 0.0 for v in oct_resp]
    if n_kp == 0:
        return [0.0] * 14, cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    arr = np.array(kps, dtype=np.float64)
    xs, ys, scales, resp = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
    orients = arr[:, 4]
    cx, cy = float(xs.mean()), float(ys.mean())
    dists = np.hypot(xs - w / 2, ys - h / 2)
    max_scale = max(scales.max(), 1e-9)
    ohist, _ = np.histogram(orients, bins=8, range=(0.0, 180.0))
    ohist = ohist / (ohist.sum() + 1e-9)
    o_nz = ohist[ohist > 0]
    orient_entropy = float(-np.sum(o_nz * np.log(o_nz))) / np.log(8)
    orient_conc = float(ohist.max())

    feats = [
        n_kp / area * 1e4, float(resp.mean()), float(resp.std()),
        float(resp.max()), float(scales.mean() / max_scale),
        float(scales.std() / max_scale),
        float(np.mean(np.pi * scales ** 2) / area),
        float(np.hypot(cx - w / 2, cy - h / 2) / np.hypot(w / 2, h / 2)),
        float(dists.std() / np.hypot(w / 2, h / 2)),
        orient_conc, orient_entropy,
    ] + oct_means
    return feats[:14], cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
