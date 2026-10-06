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
    """负载均衡正则：熵低=坍缩，目标最大化熵 -> 惩罚 -H(w)"""
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
    torch.manual_seed(seed)
    P = torch.tensor(pos_feats, dtype=torch.float32)
    N = torch.tensor(neg_feats, dtype=torch.float32)
    opt = torch.optim.Adam(router.parameters(), lr=lr, weight_decay=1e-3)
    for _ in range(epochs):
        fp, wp = router.fused(P)
        fn, _ = router.fused(N)
        loss = mil_ranking_loss(fp, fn, margin) + lam_entropy * (
            entropy_reg(wp) + entropy_reg(router.forward(N))) / 2
        opt.zero_grad(); loss.backward(); opt.step()
    router.eval()
    with torch.no_grad():
        fp, wp = router.fused(P)
        fn, _ = router.fused(N)
        return router, {"loss": float(loss.item()),
                        "pos_mean": float(fp.mean().item()),
                        "neg_mean": float(fn.mean().item()),
                        "weight_std": float(wp.std(dim=0).mean().item())}


def router_gate(base_auroc, router_auroc, margin=0.005):
    """§5.1 上线门控：路由 AUC ≥ 保底 + margin 才启用"""
    return bool(router_auroc >= base_auroc + margin)
