"""blob 槽位：DoG 斑点检测（data_local 微缺陷的域不变槽位，§8 机理防线）

v0.2 修正：弃用 skimage blob_log（打分取 sigma² 导致常数饱和，且每 tile 数百毫秒）。
改用 cv2 向量化高斯差分（DoG）：
  - 两个尺度带（σ 1→2、2→4）覆盖细小斑点；
  - 双极性（亮点/暗点都算异物）；
  - 分数 = 响应图 top-10 均值（对强小斑点敏感，对大尺度纹理不敏感）。
无网络、CPU 原生。

U22（2026-08-15 实测）：blob 保持 CPU cv2 全分辨率不动——
  * 降采样（max_side=2048）伤检出：anomaly 前20张 3/20 vs 全分辨率 6/20；
  * GPU 化（conv2d 手工高斯核）伤精度：blob AUROC 0.7075→0.6112、fused→0.3529、
    检出 5/20→3/20，且无提速收益。cv2 DoG 是 gold_finger 微缺陷的关键特征，
    任何实现变更（降采样/fp16/手工核）都改变微小响应排序。320ms CPU 成本是精度代价。
"""
import os
import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor
from .base import Slot

# 外部（backend 治理配置）注入的 GPU 开关读取函数；algo 不反向依赖 backend
_GPU_OVERRIDE = None


def set_gpu_override(fn):
    """注册无参可调用对象，返回真值表示治理配置要求启用 blob GPU；None 清除。"""
    global _GPU_OVERRIDE
    _GPU_OVERRIDE = fn


def _gpu_override_enabled() -> bool:
    if _GPU_OVERRIDE is None:
        return False
    try:
        return bool(_GPU_OVERRIDE())
    except Exception:  # noqa: BLE001 配置读取失败按默认关
        return False


class BlobSlot(Slot):
    name = "blob"
    needs_dino = False

    def __init__(self, cfg, device="cpu"):
        self.topk = cfg.get("topk", 10)
        self.with_heatmap = cfg.get("with_heatmap", False)
        self.max_side = cfg.get("max_side", 0)   # 默认 0=不降采样（U22 实测伤精度）
        # U56（2026-08-16）：两个 DoG 尺度带相互独立，并行计算提速
        # （大图 3000×4000 单带 ~155ms，并行后 blob 整体 ~160ms），
        # np.maximum 组合顺序无关，数值与串行完全一致。
        # R3-speed：σ=2 原先算两次，去重为 3 个独立 blur 并行；
        # cv2.absdiff/cv2.max 与 np.abs(a-b)/np.maximum 逐位相同（float32 逐元素）。
        self._exec = ThreadPoolExecutor(max_workers=3)
        # R3-speed 有损开关（默认关）：GPU DoG。核取 cv2.getGaussianKernel 同核、
        # reflect101 同边界，与 CPU 仅 float32 求和顺序差（实测相对差 ≤4e-7）；
        # 4096×3000 208→48ms。显存不足/小图/CPU 设备自动回落 CPU。
        # R3-coverage：开关来源 = 环境变量 AOI_BLOB_GPU / 槽位 cfg.gpu / 治理配置
        # governance_decisions.blob_gpu_enabled（由 backend 经 set_gpu_override 注入），
        # 任一为真即启用；AOI_BLOB_GPU 显式设为 0/空/false 时强制关闭（保持原语义）。
        env = os.environ.get("AOI_BLOB_GPU")
        if env is None:
            want = bool(cfg.get("gpu", False)) or _gpu_override_enabled()
        else:
            want = env not in ("0", "", "false")
        self._dev = None
        if want and str(device).startswith("cuda"):
            import torch
            if torch.cuda.is_available():
                self._dev = torch.device(device)
        self._gpu_min_px = int(cfg.get("gpu_min_px", 1 << 20))
        self._kern = {}

    def fit(self, ctx):
        return self  # 无训练，尺度由 CDF 校准统一

    def _dog_response(self, g, sigma_scale=1.0):
        """g: float32 [0,1] 灰度 -> 响应图（双极性 DoG max）
        sigma_scale：降采样后 σ 按比例缩放，保持斑点尺度一致。"""
        fs = [self._exec.submit(cv2.GaussianBlur, g, (0, 0), s * sigma_scale)
              for s in (1, 2, 4)]
        b1, b2, b4 = (f.result() for f in fs)
        return cv2.max(cv2.absdiff(b1, b2), cv2.absdiff(b2, b4))

    def _gpu_ok(self, h, w, sigma_scale):
        """显存按设备自适应：峰值 ≈ 10×H×W×4B（输入/填充/3 blur/2 diff），不足回落 CPU。"""
        if self._dev is None or h * w < self._gpu_min_px:
            return False
        if min(h, w) <= int(round(4 * sigma_scale * 8 + 1)) // 2 + 1:   # reflect pad 需 < 边长
            return False
        import torch
        try:
            free, _ = torch.cuda.mem_get_info(self._dev)
        except Exception:
            return False
        return h * w * 4 * 10 < free * 0.8

    def _gpu_kernel(self, s):
        import torch
        if s not in self._kern:
            k = int(round(s * 4 * 2 + 1)) | 1          # 与 cv2 createGaussianKernels(CV_32F) 同尺寸
            g = cv2.getGaussianKernel(k, s, cv2.CV_32F).ravel()
            self._kern[s] = torch.from_numpy(g).to(self._dev)
        return self._kern[s]

    def _gpu_score(self, gu, k, sigma_scale):
        """gu: uint8 灰度（或已缩放 float32 [0,1]）→ (top-k 均值, 响应图或 None)。"""
        import torch
        import torch.nn.functional as F

        def blur(x, s):
            kk = self._gpu_kernel(s)
            r = kk.numel() // 2
            x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="reflect"), kk.view(1, 1, 1, -1))
            return F.conv2d(F.pad(x, (0, 0, r, r), mode="reflect"), kk.view(1, 1, -1, 1))

        with torch.inference_mode():
            g = torch.from_numpy(np.ascontiguousarray(gu)).to(self._dev)
            g = (g.float().div_(255.0) if gu.dtype == np.uint8 else g)[None, None]
            b1, b2, b4 = (blur(g, s * sigma_scale) for s in (1, 2, 4))
            r = torch.maximum((b1 - b2).abs_(), (b2 - b4).abs_())
            del b1, b2, b4, g
            top = torch.topk(r.flatten(), k).values.cpu().numpy()
            hm = r[0, 0].cpu().numpy() if self.with_heatmap else None
            del r
        return float(np.partition(top, -k)[-k:].mean()), hm

    @staticmethod
    def _topk_mean(r, k):
        """精确 top-k 均值：步长子集的第 k 大 ≤ 全集第 k 大，故 flat>=thr
        必含全集 top-k，在候选上 partition 结果与全量 partition 相同。"""
        flat = r.ravel()
        stride = 64
        if flat.size >= stride * k * 4:
            thr = np.partition(flat[::stride], -k)[-k]
            flat = flat[flat >= thr]
        return float(np.partition(flat, -k)[-k:].mean())

    def score_tiles(self, tile_feats=None, tile_imgs=None):
        scores, heatmaps = [], []
        for t in tile_imgs:
            # tile 已是 512² 局部块：全分辨率计算，微斑点不被降采样抹掉
            gu = cv2.cvtColor(t, cv2.COLOR_RGB2GRAY) if t.ndim == 3 else t
            h, w = gu.shape[:2]
            scale = 1.0
            if self.max_side > 0 and max(h, w) > self.max_side:
                scale = self.max_side / max(h, w)
            if gu.dtype == np.uint8 and scale == 1.0 and self._gpu_ok(h, w, scale):
                s, hm = self._gpu_score(gu, min(self.topk, h * w), scale)
                scores.append(s)
                heatmaps.append(hm)
                continue
            g = gu.astype(np.float32)
            np.divide(g, np.float32(255.0), out=g)
            if scale != 1.0:
                g = cv2.resize(g, (max(1, int(w * scale)), max(1, int(h * scale))))
            r = self._dog_response(g, sigma_scale=scale)
            k = min(self.topk, r.size)
            scores.append(self._topk_mean(r, k))
            heatmaps.append(r if self.with_heatmap else None)
        return np.asarray(scores), heatmaps
