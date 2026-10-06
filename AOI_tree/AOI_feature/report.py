"""报告落盘：JSON + Markdown + PNG 图表（matplotlib Agg，中文字体）

组集合与顺序完全由评估结果（即当前特征配置）决定，不写死 15 组。

每品类输出：
  {category}_contribution.json   贡献档案（机器可读）
  {category}_standalone_auc.png  独立 AUROC 条形图
  {category}_loo_drop.png        留一消融损失条形图（负值=该组在拖累，如实绘制）
  {category}_perm_drop.png       排列重要性条形图
汇总输出：
  summary.md                     跨品类汇总 + 特征选择理由附录（可交付的"选择依据"）
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def _bar(keys, values, errors, title, path, ref_line=None):
    fig, ax = plt.subplots(figsize=(max(9, len(keys) * 0.6), 5))
    colors = ["#d62728" if v < 0 else "#1f77b4" for v in values]
    ax.bar(keys, values, yerr=errors, color=colors, capsize=3)
    ax.axhline(0, color="k", lw=0.8)
    if ref_line is not None:
        ax.axhline(ref_line, color="g", ls="--", lw=1, label=f"全向量={ref_line:.3f}")
        ax.legend()
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_category_report(result: dict, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    cat = result["category"]
    groups = result["groups"]
    keys = list(groups)

    with open(os.path.join(out_dir, f"{cat}_contribution.json"), "w",
              encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    _bar(keys, [groups[k]["standalone_auroc"] for k in keys], None,
         f"{cat} 各组独立 AUROC（不翻转，<0.5 即该组单独不可用）",
         os.path.join(out_dir, f"{cat}_standalone_auc.png"),
         ref_line=result["full_auroc"])
    _bar(keys, [groups[k]["loo_drop"] for k in keys], None,
         f"{cat} 留一消融 AUROC 损失（全向量={result['full_auroc']:.3f}）",
         os.path.join(out_dir, f"{cat}_loo_drop.png"))
    _bar(keys, [groups[k]["perm_drop_mean"] for k in keys],
         [groups[k]["perm_drop_std"] for k in keys],
         f"{cat} 排列重要性 AUROC 损失（{result['meta']['n_perm']} 次均值±std）",
         os.path.join(out_dir, f"{cat}_perm_drop.png"))


def save_summary(results: list, out_dir: str, groups_meta: list | None = None):
    """跨品类 Markdown 汇总：组 × 品类矩阵 + 特征选择理由附录

    groups_meta: FeatureSet.probe() 输出，用于附录的选择理由；为 None 时省略附录。
    """
    os.makedirs(out_dir, exist_ok=True)
    lines = ["# 特征贡献评估汇总", ""]
    lines.append("> 口径：AUROC 不翻转；fit 仅用 train 域；负贡献如实登记。")
    lines.append("> 打分：U27 播种 / U36 z-clip / U99 跳过自身最近邻（trad 槽位同口径）。")
    if results:
        lines.append(f"> 特征集：{results[0]['meta'].get('feature_set')}。")
    lines.append("")
    for res in results:
        cat, g = res["category"], res["groups"]
        lines.append(f"## {cat}（全向量 AUROC={res['full_auroc']:.4f}, "
                     f"ref={res['n_ref']}, test={res['n_test']}，缺陷={res['n_defect']}）")
        lines.append("")
        lines.append("| 组 | 维 | 独立AUROC | 留一损失 | 排列损失 |")
        lines.append("|---|---|---|---|---|")
        for k in g:
            r = g[k]
            lines.append(f"| {r['name']} | {r['dim']} | {r['standalone_auroc']:.4f} "
                         f"| {r['loo_drop']:+.4f} "
                         f"| {r['perm_drop_mean']:+.4f}±{r['perm_drop_std']:.4f} |")
        lines.append("")
    if groups_meta:
        lines.append("## 附：特征选择理由（当前特征集注册表）")
        lines.append("")
        lines.append("| 组 | 维 | 选择理由 | 目标缺陷 |")
        lines.append("|---|---|---|---|")
        for m in groups_meta:
            lines.append(f"| {m['name']} | {m['dim']} | {m['rationale']} "
                         f"| {m['targets']} |")
        lines.append("")
    with open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
