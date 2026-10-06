---
name: mm-submit-check
description: 提交前检查与打包。当需要核对论文格式合规（封面/个人信息/摘要）、生成提交文件（命名/MD5/附件打包 ≤50MB）、按时间窗口规划提交或做最终检查时使用本 skill。
whenToUse: 竞赛提交前 1 天与提交当天
---

# 提交前检查与打包

## 检查步骤

1. **★完篇审核（必须先过）**：`python code/scripts/audit_paper.py contests/<年>-<赛>/problem-<X>`——输出 audit_report.md（AI 全自动自查表 6 维度），**零 FAIL 才能进提交流程**（匿名泄漏/AI 未披露/摘要训练指标冒充为一票否决）；
2. **格式合规**：`paper/checklists/paper-checklist.md` 逐项打勾（封面 logo、个人信息、摘要五要素、AI 声明、参考文献）；
3. **图表检查**：`python code/scripts/check_figures.py paper/main.tex`；
4. **打包**：`python code/scripts/package_submit.py --problem A --team 25000010001 --pdf paper/main.pdf --attach-dir code`：
   - 论文自动改名 `A25000010001.pdf` 并计算 **MD5**；
   - 附件自动打包 zip 并校验 ≤50MB；
5. **时间窗口**：按 `paper/checklists/submit-checklist.md` 的官方节点提交：MD5 → PDF → 附件；
6. **纪律确认**：MD5 提交后论文不可修改；除封面外无个人信息；未与队外交流。

## 常见失败点

- 附件 >50MB（压缩数据/删中间产物）；
- 论文命名或封面信息错误 → 判无效；
- MD5 与 PDF 不一致（提交 MD5 后又改 PDF）；
- 错过窗口（提前 2 小时完成提交）。

## 资源

- 检查清单：`paper/checklists/paper-checklist.md`、`paper/checklists/submit-checklist.md`
- 打包脚本：`code/scripts/package_submit.py`
