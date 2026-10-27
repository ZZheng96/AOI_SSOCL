# PCB-DUAL-05 开发版收尾回归

日期：2026-10-13。范围仅为 `AOI_tree/PCB_Dual` 开发版；未修改 stable、AOI_Core、生产 DB/快照，未提交 git。命令均使用 `py -3.11`，Qt 使用 `offscreen`。

## 测试结果

- 前批定向：27/27 PASS，约 2.836s。
- 安全宽回归：`tests` 中 `test*.py` 共 34/34 PASS，约 2.743s；未运行 AOI_Core 全量数据、性能专项、硬件破坏及会写生产数据的 standalone 脚本。
- 本轮补充 `test_pcbdual04`：9/9 PASS，约 0.695s。新增验证：缩放超过阈值且 ROI 非零偏移时输出图仍为原图尺寸、框坐标落在原图并实际绘制；配准返回 None 和抛异常均为 ERROR。
- 补测后应以安全全量 35/35（原34项+新增1项；新增断言在既有测试内）计数为准；`test_pcbdual04` 当前为9项，安全套件模块总数仍按实际 discover 输出记录。

## 固定 seed 真实 paired（6 cases）

固定 `random.Random(20261013)`，3 SMT、1 Shift、2 TH；样本、raw 输出及逐案 JSON 在 `raw/real_cases.json` 和 `raw/*.npy`。执行退出码 0，`unexpected_failures=[]`。

- SMT 3/3：状态与框均一致；公共 metadata 唯一差异为 `output_image`。源算法先在缩放后的工作图绘制，再逆缩放 parts，故 raw 输出图不等价；保留 source/adapter/normalized_source 三份 raw。`raw_smt_equivalence_pass=0/3`，这是源渲染时序差异，不回退正确原图坐标。
- SMT 契约规范化 3/3 PASS：基于源原图 parts 在输入原图上重绘，不做插值或拟合框；输出尺寸、几何边界、状态、框和非 output metadata 均通过。
- Shift 1/1：状态、框、公共 metadata 一致。
- TH 2/2：逐项 scalar/batch 精确一致；所有可执行项与源 status、boxes、output array 精确一致。TH 多框模式仍按适配器规则仅连锡，盘内项 SKIP/ready 语义不伪造精度。
- 以上为源对照/契约回归，不是 GT、业务精度或产线验收；无人工 GT，未覆盖完整 toe/rim/Crack 真实分支。

## 代码与链路结论

- PCB-DUAL-04 的 SMT 原图坐标修复、原图尺寸输出和配准失败 ERROR 已通过真实 6-case 对照及新增 fixture 断言；本轮没有生产算法回退。
- 已复核 `InspectService.run_with_template`：生产路径传递 frozen params、region_sets、template/version 快照，`archive=False` 可隔离验证；tuner preview 使用 `allow_stub=False`、`prefer_frozen_params=True`，非 HSV 参数通过统一层合并写回。现有 tuner/TH/scheduler 测试覆盖参数冻结一致性，但没有新增 TH 专用 UI 桌面测试。
- TH 0/1/2 框 ready、detect_page 动态禁用与 tooltip 已有 offscreen 覆盖；真实桌面交互仍待外部验收。
- 后台不可取消线程仍是线程容量监控方案；进程隔离仅记录为可选后续设计，未贸然改架构。

## 状态表

| 项目 | 代码/开发回归 | 外部条件/未宣称 |
|---|---|---|
| SMT resize+ROI 原图坐标、output 同坐标 | 已实现；定向与补充9项通过 | 真实 GT/业务精度待验收 |
| 配准 None/异常状态 | 已实现为 ERROR；fixture通过 | 真实配准失败样本待验收 |
| SMT/TH/Shift batch 与请求隔离 | 已实现；定向/宽回归通过 | 不能据套件耗时证明单图1s |
| TH 0/1/2 ready、UI动态禁用、≥2仅连锡 | 已实现；offscreen/paired链路通过 | 所有真实模式、真实桌面待验收 |
| 参数三层、tuner写回、冻结快照、inspect prefer frozen | 已实现并有 Shift/通用链路回归 | TH真实业务参数精度、发布流程待验收 |
| raw 源图等价 | SMT 0/3，明确 FAIL | 源算法工作图渲染差异；不以此回退正确坐标 |
| 原图契约规范化 | SMT 3/3 PASS | 规范化不是原始渲染等价，也不是 GT |
| 生产/稳定版验收 | 未完成 | 需稳定版独立复验、授权同步、GT和业务验收 |

## 工件

- `cmd.txt`：命令顺序及环境。
- `env.json`：Python/平台/OpenCV/NumPy、开发树、固定 seed、源码 SHA256、隔离说明。
- `raw/real_cases.json`、`raw/comparison_summary.json`、SMT 三类 source/adapter/normalized `.npy`。
