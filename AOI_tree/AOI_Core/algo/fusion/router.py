"""线性软路由（§5）：逐样本槽位权重，MIL 排序损失训练 + 上线门控

设计要点（§5/§5.1）：
- 线性结构：输入=各槽位校准分 → softmax 权重，禁止 MLP（130 样本过拟合防护）；
- MIL 排序损失：缺陷图融合分应 > 正常图融合分（图像分=block top-k 聚合已在槽位内完成，
  故直接对图像级校准分排序，语义等价于设计文档的 tile 级 MIL）；
- 负载均衡正则：熵惩罚防权重坍缩（全部压到单一槽位）；
- 门控：路由在评估集 AUC ≥ 保底 + margin 才启用，否则回退保底（§5.1）。

U15 教训（如实记录）：路由用 init_defect(30) 训练、门控在 init_defect 评估，
有标签的槽位选择在域差下不可外推——因此路由是否真正增准必须看 test 侧消融，
门控只是保守防线（防在线退化），不构成"路由已生效"的证据。
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearRouter(nn.Module):
    """输入 (B, n_slots) 校准分 -> (B, n_slots) softmax 权重。
    参数量 n_slots²+n_slots，6 槽位时 ~42 个，130 样本完全可控。"""

    def __init__(self, n_slots, init_equal=True):
        super().__init__()
        self.n_slots = n_slots
        # W 初始化为对角（每个槽位分自己决定权重），避免随机初始化带来的偏差
        self.W = nn.Parameter(torch.eye(n_slots) if init_equal
                              else torch.randn(n_slots, n_slots) * 0.01)
        self.b = nn.Parameter(torch.zeros(n_slots))

    def forward(self, s):
        """s: (B, n_slots) 校准分 [0,1] -> (B, n_slots) 权重（softmax，和为 1）"""
        logits = s @ self.W + self.b
        return F.softmax(logits, dim=-1)

    def fused(self, s):
        """s: (B, n_slots) -> (B,) 加权融合分"""
        w = self.forward(s)
        return (w * s).sum(dim=-1), w


def entropy_reg(w):
    """返回平均熵 H(w)≥0；熵低=坍缩，训练时 loss 应减 λH（即惩罚 -H）。"""
    return -(w * (w + 1e-8).log()).sum(dim=-1).mean()


def mil_ranking_loss(fused_pos, fused_neg, margin=0.1):
    """MIL 排序损失：缺陷融合分至少高出正常分 margin。
    批量内所有正负对 hinge：mean(max(0, margin - (pos - neg)))"""
    d = fused_pos.unsqueeze(1) - fused_neg.unsqueeze(0)  # (P, N)
    return F.relu(margin - d).mean()


def train_router_on_matrix(router, pos_feats, neg_feats, epochs=50, lr=1e-3,
                           margin=0.1, lam_entropy=0.1, seed=42):
    """pos_feats/neg_feats: np.ndarray (P, n_slots) / (N, n_slots) 校准分。
    返回 (router, {"loss": float, "pos_mean": float, "neg_mean": float})"""
    from algo.ssocl.learning_state import candidate_rng_active
    if not candidate_rng_active.get():
        torch.manual_seed(seed)
    # 在线重训时 router 可能已被移到 cuda（推理侧），训练张量须同设备
    dev = router.W.device
    P = torch.tensor(pos_feats, dtype=torch.float32, device=dev)
    N = torch.tensor(neg_feats, dtype=torch.float32, device=dev)
    if P.ndim != 2 or N.ndim != 2 or not len(P) or not len(N):
        raise ValueError("router training requires non-empty positive and negative matrices")
    if not torch.isfinite(P).all() or not torch.isfinite(N).all():
        raise ValueError("router training scores must be finite")
    opt = torch.optim.Adam(router.parameters(), lr=lr, weight_decay=1e-3)
    from algo.ssocl.candidate_budget import charge
    for _ in range(epochs):
        charge("steps", point="router_step")
        fp, wp = router.fused(P)
        fn, _ = router.fused(N)
        # entropy_reg 返回正熵 H；防坍缩需最大化 H，故 loss 减 λH（CDEF6：原为 +λH，
        # 实为鼓励坍缩，与注释相反）。路由仅追溯不参与融合分，修正不改变判定。
        loss = mil_ranking_loss(fp, fn, margin) - lam_entropy * (
            entropy_reg(wp) + entropy_reg(router.forward(N))) / 2
        if not torch.isfinite(loss):
            raise ValueError("router training loss is non-finite")
        opt.zero_grad(); loss.backward(); opt.step()
        if any(not torch.isfinite(p).all() for p in router.parameters()):
            raise ValueError("router training parameters are non-finite")
    router.eval()
    with torch.no_grad():
        fp, wp = router.fused(P)
        fn, _ = router.fused(N)
        if not torch.isfinite(fp).all() or not torch.isfinite(fn).all() or not torch.isfinite(wp).all():
            raise ValueError("router training output is non-finite")
        enough_samples = len(P) > 1
        return router, {"loss": float(loss.item()),
                        "pos_mean": float(fp.mean().item()),
                        "neg_mean": float(fn.mean().item()),
                        "weight_std": float(wp.std(dim=0).mean().item()) if enough_samples else None,
                        "availability": {"weight_std": {
                            "status": "available" if enough_samples else "insufficient",
                            "reason": None if enough_samples else "样本标准差至少需要两个缺陷样本",
                            "n_samples": len(P),
                        }}}


def router_gate(base_auroc, router_auroc, margin=0.005):
    """§5.1 上线门控：路由 AUC ≥ 保底 + margin 才启用"""
    return bool(router_auroc >= base_auroc + margin)
