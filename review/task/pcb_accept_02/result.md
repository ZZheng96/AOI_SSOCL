# PCB-ACCEPT-02 SMT/TH 真实分支与坐标契约覆盖（部分通过）

## 结论：部分 PASS；多个分支 BLOCKED（缺真实样本）

## 覆盖矩阵（稳定版实测 + PCB-ACCEPT-05 的 6-case paired）

| 分支 | 样本 | 结果 | 证据 |
|---|---|---|---|
| SMT 多锡（excess） | excess_pair/pair4 真实图 | 源/适配状态框一致（NG/NG，boxes_equal=True）；桌面复测 NG 多锡 316,181 | pcb_accept_05 raw + pcb_accept_01 |
| SMT 连锡（bridge） | bridge_pair/pair4 | NG/NG 一致 | pcb_accept_05 raw |
| SMT 少锡（insufficient） | insufficient_pair/pair4 | NG/NG 一致 | pcb_accept_05 raw |
| SMT 冷焊（cold_solder，正常图） | cold_solder_pair/pair1 | OK/OK 一致 | pcb_accept_05 raw |
| SMT raw 输出图等价 | 同上 3 例 | **FAIL 0/3**：源算法先在工作图绘制再逆缩放，raw 渲染不等价；规范化契约 3/3 PASS（尺寸/边界/状态/框/非 output metadata 全过） | 已知差异，不回退正确坐标 |
| Shift 多 ROI | 真实 Shift 图 1 例 | 状态/框/metadata 一致 | pcb_accept_05 raw |
| TH scalar/batch | roi/2-8.png、3-2.png | 逐项精确一致 | pcb_accept_05 raw |
| SMT 尺寸门禁 | 941×620 vs 1317×545 | ERROR 已拦截，未执行检测（桌面实测） | pcb_accept_01 |
| SMT toe/rim 差集 | — | **BLOCKED**：无真实配套样本 | — |
| SMT Crack 可执行分支 | cold_solder_pair/pair1 真实图 | **PASS（执行复核）**：源+适配 Crack 诊断实际执行（非"不可用"）且未检出，code=0、0 parts；NG 正例检出仍 **BLOCKED**（需人工确认正例） | raw/crack_mismatch_check.json（2026-10-14 补充） |
| TH 自动轮廓/单框/多框真实模式 | templ_2/templ_3 + 2-1/2-8/3-1/3-2 | **PASS（由 PCB-ACCEPT-01 第三轮覆盖，真实桌面+API）**；清空复活缺陷见 V§101 问题① | pcb_accept_01 raw/th_round |
| 配准失败错配构造（真实图片构造错配） | templ_2 vs 3-1（跨板错配） | **FAIL：未 fail-closed**——孔洞/多锡包锡/不出脚报 OK（少锡报 NG），错配 align `resp=0.311`（匹配对 0.819）、coverage=0.995、review_reason=null。根因：`phase_min_response=0.12` 阈值远低于错配实测 resp；coverage 是自身 mask 重合度不能识别跨板错配；`check_reliability` 只门禁位移幅度与 coverage、不门禁 resp/低置信（align.py"相位低置信"仅写文案仍放行） | raw/crack_mismatch_check.json（2026-10-14 补充）；匹配样本 templ_2 vs 2-1：align_method/coverage 键存在、resp=0.819 |
| 自动缩放+非零 ROI 偏移 | 合成 fixture | 输出图原图尺寸、框落原图、绘制同坐标 | test_pcbdual04 |

## 边界

- 无人工 GT，上述为源对照/契约一致性，不是精度结论。
- "不以规范化 PASS 掩盖 raw FAIL"原则已遵守：raw 0/3 单列。
- 6-case 对照 raw 工件（json/npy）在 `pcb_accept_05/raw/`。
