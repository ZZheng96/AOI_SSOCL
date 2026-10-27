"""传统底层视觉特征提取器（从 Demo2 AOI_dev 移植，172 维 → 200 维）

14 组特征 + ⑮ SURF（多尺度 Hessian 斑点，+14 维），向量化实现。
单张 256x256 ~30ms（不含 PC/SURF 时更快的常驻路径已向量化）。
"""
import cv2
import numpy as np
from skimage import feature, filters, color, transform, morphology
from skimage.measure import shannon_entropy, moments_hu
from skimage.feature import local_binary_pattern, hog, graycomatrix, graycoprops
import pywt

def _cv2_canny(gray, sigma=1, low=0.1, high=0.3):
    """cv2 Canny（sigma 对应高斯模糊），替代 skimage feature.canny（快 ~50 倍）"""
    blur = cv2.GaussianBlur(gray, (0, 0), sigma)
    lo = max(1, int(255 * low))
    hi = min(255, int(255 * high))
    return cv2.Canny(blur, lo, hi)

def _surf_features(gray, n_octaves=3, n_scales=3, base_sigma=1.6,
                   sigma_step=1.26, rel_thresh=0.01):
    """⑮ SURF（自研简化版，与 AOI_dev 一致）：多尺度 Hessian 斑点 + 统计特征

    OpenCV 5 移除 xfeatures2d，用标准 cv2 自研：高斯滤波 → Hessian 行列式
    det 度量局部弯曲/凹凸，尺度空间极值即关键点。每八度降采样一次，
    方向用峰值点梯度方向（向量化）。返回 14 维图像级特征 + 可视化图。
    """
    h, w = gray.shape
    img = gray.astype(np.float32) / 255.0
    kps = []             # (x, y, scale, response, orient) 映射回原图坐标
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
                # 3x3 局部极大值（阈值过滤 + 边界裁剪 + 峰值数上限）
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
    view = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    rmax = max(resp.max(), 1e-9)
    for (x, y, s, r, _) in kps:
        c = int(255 * (0.3 + 0.7 * r / rmax))
        cv2.circle(view, (int(x), int(y)), max(2, int(s * 2)),
                   (c, c, c), 1, lineType=cv2.LINE_AA)
    return feats[:14], view


class TraditionalFeatureExtractor:
    """200 维传统特征提取器（186 基础 + ⑮ SURF 14）

    分组:
    ① GLCM 纹理 (4)
    ② 区域自相似 (5)
    ③ 边缘梯度 (28)
    ④ HOG (11)
    ⑤ 形状/几何 (11)
    ⑥ 灰度统计 (12)
    ⑦ 材质反光 (4)
    ⑧ 色彩 HSV+矩 (36)
    ⑨ FFT 频域 (13)
    ⑩ Haar 小波 (40)
    ⑪ 逻辑关系 (6)
    ⑫ Gabor (8)
    ⑬ LoG 斑点 (5)
    ⑭ 相位一致性 (3)
    ⑮ SURF 多尺度 Hessian 斑点 (14)
    """
    
    def __init__(self):
        self.dim = 200
        self._fixed_dim = None   # 旧 checkpoint 兼容：截断到该维度（如 172）

    def set_dim(self, d):
        """对齐旧 checkpoint 的 trad_ref 维度（新特征追加在末尾，截断即回退）"""
        self._fixed_dim = int(d) if d else None
    
    def extract(self, img, rng=None):
        """img: ndarray (H, W, 3) RGB, 返回 200 维向量"""
        rng = rng or np.random.default_rng(42)
        # 缩放：大图降采样提速（<=256）；再保证最小边 >=32（HOG 需 32 行/列）
        h, w = img.shape[:2]
        if max(h, w) > 256:
            s = 256.0 / max(h, w)
            img = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))))
        h, w = img.shape[:2]
        if min(h, w) < 32:
            s = 32.0 / min(h, w)
            img = cv2.resize(img, (int(w * s), int(h * s)))
        features = []
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        
        # ① GLCM 纹理 (4)
        glcm = graycomatrix(gray, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], 
                           levels=256, symmetric=True, normed=True)
        for prop in ['contrast', 'correlation', 'energy', 'homogeneity']:
            features.append(graycoprops(glcm, prop).mean())
        
        # ② 区域自相似 (5)
        h, w = gray.shape
        for scale in [4, 8, 16]:
            # 分块匹配（向量化均距替代 pdist：O(N²) 距离由 129ms 降至 ~1ms）
            blocks = []
            for i in range(0, h - scale + 1, scale):
                for j in range(0, w - scale + 1, scale):
                    blocks.append(gray[i:i+scale, j:j+scale].ravel())
            if len(blocks) > 1:
                blocks = np.array(blocks, dtype=np.float32)
                if len(blocks) > 100:
                    idx = rng.choice(len(blocks), 100, replace=False)
                    blocks = blocks[idx]
                # 均距 = sqrt(mean(||bi-bj||²)) 向量化
                bnorm = (blocks ** 2).sum(axis=1)
                dot = blocks @ blocks.T
                dist2 = bnorm[:, None] + bnorm[None, :] - 2 * dot
                np.fill_diagonal(dist2, 0.0)
                nonzero = dist2[dist2 > 0]
                features.append(float(np.sqrt(nonzero).mean()) if nonzero.size else 0.0)
            else:
                features.append(0.0)
        # 加两个自相似统计
        features.append(np.std(gray[:8, :8] - gray[8:16, 8:16]))
        features.append(np.std(gray[::8, ::8]))
        
        # ③ 边缘梯度 (27)
        # Canny 密度（cv2 实现，比 skimage 快 ~50 倍）
        edges = _cv2_canny(gray, sigma=1)
        features.append(edges.mean())
        edges2 = _cv2_canny(gray, sigma=2)
        features.append(edges2.mean())
        edges3 = _cv2_canny(gray, sigma=3)
        features.append(edges3.mean())
        # Sobel 梯度（cv2 实现，比 skimage filters 快 ~10 倍）
        sobel_h = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sobel_v = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        features.append(sobel_h.mean())
        features.append(sobel_v.mean())
        features.append(np.sqrt(sobel_h**2 + sobel_v**2).mean())
        # 梯度方向直方图 (16 bins)
        grad_mag = np.sqrt(sobel_h**2 + sobel_v**2)
        grad_ang = np.arctan2(sobel_v, sobel_h + 1e-8)
        hist, _ = np.histogram(grad_ang, bins=16, range=(-np.pi, np.pi), weights=grad_mag)
        features.extend(hist.tolist())
        # LBP (6 bin)
        lbp = local_binary_pattern(gray, 8, 1, method='uniform')
        lbp_hist, _ = np.histogram(lbp, bins=6, range=(0, 6), density=True)
        features.extend(lbp_hist.tolist())
        
        # ④ HOG (11)
        fd = hog(gray, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2), 
                feature_vector=True, block_norm='L2-Hys')
        # 降采样到 11 维
        idx = np.linspace(0, len(fd)-1, 11, dtype=int)
        features.extend(fd[idx].tolist())
        
        # ⑤ 形状/几何 (11)
        # Hu 矩 (7)
        moments = cv2.moments(gray)
        hu = cv2.HuMoments(moments).flatten()
        features.extend((-np.sign(hu) * np.log10(np.abs(hu) + 1e-10)).tolist())
        # 轮廓统计 (4)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            areas = [cv2.contourArea(c) for c in contours]
            features.append(np.log1p(np.mean(areas)))
            features.append(np.log1p(np.std(areas)))
            features.append(len(contours) / (h * w) * 1e4)
            perims = [cv2.arcLength(c, True) for c in contours]
            features.append(np.mean(perims) / (h + w) if perims else 0.0)
        else:
            features.extend([0, 0, 0, 0])
        
        # ⑥ 灰度统计 (12)
        features.extend([gray.mean(), gray.std(), gray.min(), gray.max(),
                        np.percentile(gray, 25), np.percentile(gray, 50), np.percentile(gray, 75),
                        shannon_entropy(gray),
                        np.mean(gray ** 2), np.mean(np.abs(gray - gray.mean()) ** 3),
                        np.mean(gray ** 4), (gray.max() - gray.min()) / (gray.mean() + 1e-8)])
        
        # ⑦ 材质反光/高光 (4)
        specular = gray > gray.mean() + 2 * gray.std()
        features.append(specular.mean())
        features.append(gray[specular].std() if specular.any() else 0)
        # 高光区域连通性
        _, bw = cv2.threshold(gray, gray.mean() + 2 * gray.std(), 255, cv2.THRESH_BINARY)
        kernel = np.ones((5, 5), np.uint8)
        bw = morphology.remove_small_objects(bw.astype(bool), min_size=50)
        features.append(bw.mean())
        features.append(len(np.unique(bw)) / (h * w) * 1e4)
        
        # ⑧ 色彩 HSV 直方图 + 色彩矩 (39)
        for ch in range(3):
            hist, _ = np.histogram(hsv[:,:,ch], bins=8, range=(0, 256), density=True)
            features.extend(hist.tolist())
            # 色彩矩
            ch_data = hsv[:,:,ch].astype(float)
            features.extend([ch_data.mean(), ch_data.std() / 255,
                           np.mean(((ch_data - ch_data.mean()) / (ch_data.std() + 1e-8)) ** 3)])
        # RGB 相关性 (3)
        for i in range(3):
            for j in range(i+1, 3):
                cc = np.corrcoef(img[:,:,i].ravel(), img[:,:,j].ravel())[0, 1]
                features.append(cc if not np.isnan(cc) else 0.0)
        
        # ⑨ FFT 频域 (13)
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude = np.abs(fshift)
        # 高频能量
        cy, cx = h // 2, w // 2
        radius = min(h, w) // 4
        mask = np.zeros_like(gray, dtype=bool)
        cv2.circle(mask, (cx, cy), radius, True, -1)
        low_energy = magnitude[mask].sum()
        total_energy = magnitude.sum() + 1e-8
        features.append(low_energy / total_energy)
        features.append(1 - low_energy / total_energy)
        # 频谱熵
        prob = magnitude / total_energy
        features.append(-np.sum(prob * np.log(prob + 1e-10)))
        # 径向分布 (10 bins)
        y, x = np.ogrid[:h, :w]
        r = np.sqrt((x - cx)**2 + (y - cy)**2)
        r_bins = np.linspace(0, r.max(), 11)
        for i in range(10):
            mask_i = (r >= r_bins[i]) & (r < r_bins[i+1])
            features.append(magnitude[mask_i].mean() / total_energy)
        
        # ⑩ Haar 小波 (24)
        coeffs = pywt.wavedec2(gray, 'haar', level=3)
        for i, c in enumerate(coeffs):
            if i == 0:  # LL
                features.extend([c.mean(), c.std(), c.min(), c.max()])
            else:
                for detail in c:  # LH, HL, HH
                    features.extend([detail.mean(), detail.std(), 
                                   np.mean(detail**2), np.mean(np.abs(detail))])
        
        # ⑪ 逻辑关系代理 (6)
        # 连通域分析
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thresh, connectivity=8)
        features.append(num_labels / (h * w) * 1e4)
        if num_labels > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            features.append(np.mean(areas) / (h * w))
            features.append(np.std(areas) / (h * w))
            features.append(len(areas[areas > areas.mean()]) / max(1, len(areas)))
        else:
            features.extend([0, 0, 0])
        # 对称性
        diff_h = np.mean(np.abs(gray[:, :w//2] - gray[:, :w//2][:, ::-1]))
        diff_v = np.mean(np.abs(gray[:h//2, :] - gray[:h//2, :][::-1, :]))
        features.extend([diff_h, diff_v])
        
        # ⑫ Gabor (8)
        for theta in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
            for freq in [0.1, 0.3]:
                gabor_kernel = cv2.getGaborKernel((21, 21), 4, theta, 1/freq, 0.5, 0)
                filtered = cv2.filter2D(gray, cv2.CV_32F, gabor_kernel)
                features.append(filtered.mean())
        
        # ⑬ LoG 斑点 (5)（cv2 替代 skimage filters，快 ~10 倍）
        for sigma in [1, 2, 4, 8, 16]:
            blur = cv2.GaussianBlur(gray, (0, 0), sigma)
            log = cv2.Laplacian(blur, cv2.CV_32F)
            features.append(np.abs(log).mean())
        
        # ⑭ 相位一致性 (3)
        pc = self._phase_congruency(gray)
        features.extend([pc.mean(), pc.std(), pc.max()])
        
        # ⑮ SURF 多尺度 Hessian 斑点 (14)
        surf_vals, _ = _surf_features(gray)
        features.extend(surf_vals)
        
        arr = np.nan_to_num(np.array(features, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        # 旧 checkpoint 兼容：trad_ref 是 172 维时截断（SURF 追加在末尾，截断即回退）
        if self._fixed_dim is not None and len(arr) > self._fixed_dim:
            arr = arr[:self._fixed_dim]
        return arr
    
    def _phase_congruency(self, img):
        """简化版相位一致性（cv2 LoG 替代 skimage filters，快 ~10 倍）"""
        nscale = 4
        norient = 4
        # 简化: 用多尺度 LoG 响应近似
        pcs = []
        for s in range(1, nscale + 1):
            sigma = 2 ** s
            blur = cv2.GaussianBlur(img, (0, 0), sigma)
            log = cv2.Laplacian(blur, cv2.CV_32F)
            pcs.append(np.abs(log))
        pc = np.mean(pcs, axis=0)
        return pc / (pc.max() + 1e-8)