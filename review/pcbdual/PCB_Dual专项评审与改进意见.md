# PCB_Dual 专项评审与改进意见

> 评审对象：`D:\CGAIC\AOI_SSOCL\AOI_tree\PCB_Dual`
>
> 参考对象：`D:\CGAIC\AOI_CV` 下的成熟传统 CV 算法仓库，以及现有 AOI_tree 总体评审文档。
>
> 评审方式：源码静态审查，重点覆盖 UI 操作流、算法适配、参数/ROI 传递、状态语义、迁移完整性和性能风险。本次未运行全量数据集，也不以静态审查替代小样本验收。

## 1. 总体结论

PCB_Dual 已经不是传统算法占位页面，而是建立了较完整的模板、标定、参数、调度、后台运行和人工复判链路。Shift、SMT 焊锡、插件焊点等算法均通过适配器接入统一的 `DetectRequest` / `AlgorithmResult` 协议，并使用隔离导入避免不同算法仓库的同名 `core`、`algorithms` 包互相污染。

但目前只能得出“架构上已经接入，具备继续验收条件”，还不能得出“已证明与 AOI_CV 成熟版本完全等价、检测结果已正确、单图已满足 1 秒红线”。主要原因是：

1. 没有形成 AOI_CV 源版本到 PCB_Dual 运行版本的逐算法迁移基线和版本差异记录；
2. 多 ROI、四类 SMT 缺陷、五类插件焊点缺陷可能重复执行完整流水线，存在明显速度风险；
3. UI 中 `SKIP`、`REVIEW`、`ERROR`、stub 结果的可见性还不够强，容易误判结果可信度；
4. 缺陷表合并框后主要展示类型、位置和尺寸，算法来源、ROI、参数快照、单项耗时和复判原因不完整；
5. Shift 的适配器、内部合并版算法和 AOI_CV 源版之间仍缺少自动化差异验证；
6. 需要用少量随机图片分别完成正常、缺件、移位、边界、配准失败和异常输入验收。

**建议结论：**暂不判定为“迁移完成”；可判定为“功能接入完成，进入迁移等价性、结果正确性和性能验收阶段”。

## 2. 现有评审文档覆盖情况

现有文档已经涉及 PCB_Dual，但主要集中于：

- 与 AOI Core 的 HTTP 调用和双引擎并行；
- 反馈闭环、资产追踪、稳定 ID、任务状态和系统级流程；
- AOI_tree 的总体改进任务台账。

现有内容尚未形成 PCB_Dual 专项的以下检查：

- UI 标定和调参流程是否能防止错误配置；
- AOI_CV 算法逐项迁移是否完整；
- 参数默认值是否被 UI 错误覆盖；
- ROI、缩放、配准和原图坐标是否闭环；
- `REVIEW` / `SKIP` / `ERROR` 是否正确进入最终判定；
- 单图耗时、重复前处理和不可取消线程是否满足工程约束。

本文件作为已有系统级意见的补充，不替代总体改进清单。

## 3. UI 专项评审

### 3.1 已有优点

`app/ui/detect_page.py` 已提供：

- 模板选择、标准图自动加载、测试图打开；
- 标准图与测试图/结果并排显示；
- 按算法大类折叠、全选/全不选和选中摘要；
- “去建模改区域”和“去预处理调参”入口；
- 调参防抖预览，默认间隔 350ms；
- 结果画布与缺陷表互相定位；
- 默认不归档，用户显式勾选后归档。

`calibration_dialog.py` 对 SMT 的 `pads/toe/rim`、插件焊点单框/多框/自动轮廓、金手指种子点等已有明确说明。`inspect_page.py` 支持后台检测、批量进度、取消、NG/ERROR 筛选、下一张 NG 和人工复判。这些设计具备实际使用基础。

### 3.2 P0/P1 改进项

#### P0-UI-01：禁止或强提示“不可直接相信”的调试结果

调试页调用调度器时使用 `allow_stub=True`。参数面板在适配器缺失时也明确写出“对照检测将返回 stub OK”。这会导致用户看到 OK，却没有足够醒目地知道该结果不是实际算法结果。

建议：

- 结果 HUD 增加“真实算法 / stub / 跳过 / 待复检 / 错误”标签；
- stub 结果禁止使用绿色 OK 样式，改为灰色“未执行”；
- 产线模式继续保持 `allow_stub=False`；
- 调试页在首次出现 stub 或未就绪算法时弹出一次阻断式提示；
- 结果归档中写入 `execution_mode`。

验收：适配器缺失、ROI 未标定、算法未就绪时，UI 不得只显示绿色 OK。

#### P1-UI-02：显式展示标定前置条件和当前 ROI 状态

目前标定说明分散在建模页和提示文字中，运行前虽然会调用 `is_ready()`，但用户仍不容易快速知道：当前模板有多少个 ROI、哪些 ROI 对应哪个算法、哪些算法会因模式限制被跳过。

建议在算法选择区增加：

- ROI 数量和有效性；
- 算法与 ROI 图层对应关系；
- SMT 是否存在 `pads/toe/rim`；
- 插件焊点当前是 0、1 还是多个焊点；
- Shift 当前 base 框数量；
- “当前选择将只运行连锡”等模式提示。

验收：用户不打开源码即可判断当前选择是否满足运行条件。

#### P1-UI-03：区分 `REVIEW`、`SKIP` 和 `ERROR`

插件焊点适配器将算法返回的 `REVIEW` 转成 `ok=True, skipped=True`。调度器默认将 skipped 结果补成 `SKIP`。这保证了不可靠结果不会被判为 NG，但 UI 上仍可能把“算法未执行”“结果不可靠待复检”“系统错误”混在不同消息中。

建议：

- `AlgorithmResult.status` 保留并展示 `REVIEW`；
- `REVIEW` 必须显示“待人工复检”和原因；
- `SKIP` 显示“未执行/条件不满足”；
- `ERROR` 显示错误码和是否建议重试；
- 总体结果增加 `requires_review`，不要只依赖 `GRAY` 或 `skipped` 推断。

验收：配准覆盖率不足、缺少 ROI、算法异常三种情况必须显示三种不同状态。

#### P1-UI-04：增强缺陷表的信息透明度

`defect_table.py` 当前列为“#、类型、位置、尺寸”，缺陷框还会经过 IoU 合并。这样便于浏览，但可能隐藏多个算法对同一位置的原始判断。

建议：

- 增加算法来源、ROI、分数/置信度、单项耗时、状态列；
- 合并框支持展开查看原始算法结果；
- 没有缺陷框但存在 ERROR/REVIEW/SKIP 时，显示状态、错误码、原因和建议动作；
- 明确标注“置信度不可用”而不要把固定值当成真实置信度。

## 4. AOI_CV 迁移矩阵

| 算法 | AOI_CV 成熟实现 | PCB_Dual 接入位置 | 当前判断 | 必须验收 |
|---|---|---|---|---|
| Shift 移位/缺件 | `Dev_shift_smt_Alg/.../shift_smt_all_alg.py` 及 `_shift_core.py` | `alg_repo/shift_smt/.../shift_smt_all_alg.py` + `app/detect/adapters/shift_adapter.py` | 已复制并通过 `roi_bbox` 适配；尚未证明与源版逐函数等价 | 参数、定位框、缺件框、旋转、LUT、AutoGamma、多 ROI |
| SMT 多锡/少锡/连锡/虚焊 | `Dev_solder_smt_Alg/.../solder_smt_all_alg.py` 及 `_solder_core.py` | `app/detect/adapters/smt_adapter.py` 调用隔离仓库 | 参数分层和默认配置底座较好；需核实坐标和执行次数 | `pad_frames`、`toe_frames`、`rim_frames`、缩放映射、差分、四类开关 |
| 插件焊点五类缺陷 | Through-Hole 仓库算法 | `app/detect/adapters/through_hole_adapter.py` | 模式限制和 REVIEW 处理较清楚 | 单框/多框/自动轮廓、标准图、坐标、总体状态 |
| 元件本体/金手指/褶皱等 | AOI_CV 对应算法仓库 | PCB_Dual 各适配器和 Catalog | 已进入统一体系，但本次未逐项完成等价核验 | 按相同矩阵补齐源版本、版本号、参数和样本结果 |

迁移完成的定义应改为：源版本、PCB_Dual 版本、适配器版本、参数 schema、输入输出协议和小样本结果全部可追溯，而不是仅能 import 或能跑通。

## 5. Shift 迁移审查

### 已确认的正确做法

- 使用 `IsolatedImport` 加载 `algorithms.component_detect.shift_smt_all_alg`，避免同名包污染；
- 模板 ROI 优先从 `req.rois` 的 `shift/shift_base` 提取，缺失时回退标定存储；
- 算法默认配置先建立，再叠加生效参数；
- 缺件使用 base 框作为缺陷定位框，符合“应有元件但缺失”的展示语义；
- 移位优先使用算法返回的 `base_box`；
- 所有 ROI 执行失败时返回 `ERROR`。

### 重点风险

1. AOI_CV 源默认值中 `feature_s_low=85、feature_s_high=255、feature_v_high=155`，而 Shift UI 参数默认值为 `feature_s_low=0、feature_s_high=147、feature_v_high=36`。这可能是产品化黑色底座配置，也可能改变成熟算法默认行为，必须用基准样本证明；
2. 每个 base ROI 都调用一次 `run()`，算法内部可能重复 HSV、掩膜、积分图和滑窗，ROI 多时耗时近似线性增长；
3. UI 未暴露 `grid_size、mode、use_edges、edge_ratio、whole_weight、LUT` 等深层参数，需记录为“固定默认”并证明不会影响目标产品；
4. 开启旋转时，搜索角度数量约为 `2 * rotation_range / rotation_step + 1`，需要设上单图预算；
5. 算法返回的 `confidence=1.0` 不应作为真实置信度；
6. PCB_Dual 内部算法是将源版多个文件整合成单文件，尚未建立差异测试，无法仅凭文件存在证明迁移完整。

### 建议验收

- 从 AOI_CV 选取正常、缺件、移位、轻微移位、旋转、颜色边界和多 ROI 样本；
- 对源版和 PCB_Dual 使用完全相同图像、ROI、参数；
- 对比 `code、label、is_shifted、base_box、缺件框、输出图`；
- 允许接口字段不同，但必须建立字段映射表；
- 记录单 ROI、5 ROI、10 ROI 和旋转开关的耗时。

## 6. SMT 迁移审查

### 已确认的正确做法

- `smt_adapter.py` 使用算法 `default_config()` 作为底配置，避免丢失成熟算法的深层默认参数；
- `pad_frames` 来自模板标定，使用 floor/int 统一矩形坐标；
- 共用参数和缺陷专属参数分层保存，并过滤旧参数，避免历史参数覆盖共用层；
- 四类缺陷开关映射明确；
- 虚焊可传递 `toe_frames` 和 `rim_frames`；
- 算法返回非零 code 时输出 `ERROR`。

### 重点风险

1. 四个缺陷检测项分别进入调度循环，每项都调用一次 `SolderSmtAllAlg.run()`，即使成熟算法一次 run 支持四类缺陷，也可能重复执行焊锡提取、配准、缩放和模板差分；
2. 自动缩放打开时，必须证明 `pad_frames、toe_frames、rim_frames` 和缺陷框均正确映射回原图；
3. 模板差分失败、覆盖率不足等状态需要明确转为 `REVIEW` 或 `ERROR`，不能静默变成 OK；
4. 共用参数预览会一次打开四类缺陷，和产线逐项检测的耗时/结果路径不同，UI 必须明确说明“预览路径”；
5. 当前适配器输出框未显式携带原始算法置信度和 ROI 标识，复核追踪能力不足。

### 建议改进

- 增加“同一 SMT 流水线一次运行、四类结果拆分”的可选执行模式；
- 至少缓存标准图配准、缩放图、焊锡掩膜和焊盘分类结果；
- 在结果 metadata 中保存 `scale、offset、mapping_version、pad_count、alignment_status`；
- 用源算法测试样本逐类对比 `excess/insufficient/bridge/cold_solder`，并专门测试边界 1px。

## 7. Through-hole 迁移审查

### 已确认的正确做法

- 通过隔离导入加载算法和元数据；
- 缺陷 ID 12~16 与中文检测项映射明确；
- 少于两个焊点框时不允许连锡；
- 多于一个焊点框时盘内缺陷被判定为未就绪，避免错误运行；
- 标准图和测试图均传给算法；
- 算法返回 `REVIEW` 时不判 NG，而是产生待复检结果。

### 重点风险

1. `REVIEW` 被适配成 `skipped=True`，最终是否进入整板 `GRAY/REVIEW` 需要端到端确认；
2. 算法输出框是否已经是原图坐标需要用自动缩放和配准样本验证；
3. UI 文案虽然说明了 0/1/多框的模式，但算法选择区没有看到足够强的动态禁用/警告机制；
4. 每个缺陷项可能重新创建算法实例并在隔离环境内运行，五项全开时需要测量初始化和算法重复成本。

## 8. AOI_CV tuner 与 PCB_Dual 调参能力对比

### 8.1 AOI_CV tuner 的可取之处

本次重点考察了 AOI_CV 中的 Shift 和 SMT tuner。其价值不只是提供 HSV 滑条，而是把“参数变化—算法中间结果—最终判定”放在同一个闭环中：

- 参数改变后立即调用生产算法或生产算法核心函数，而不是只更新一张颜色掩膜；
- 主图直接叠加 ROI、检测框、缺陷区域和判定颜色，用户可以看到参数变化对结果的实际影响；
- 提供原图、Mask、H/S/V 通道、模板图、差分图、Crack 等辅助视图，帮助判断问题来自取色、配准、阈值还是几何区域；
- SMT tuner 能实时显示 pad 覆盖率、多锡环、rim-toe 区域、连通域和四类缺陷结果；
- Shift tuner 除最终框外，还提供特征图热力图、GT 框、IoU、峰值分数、dx/dy 和移位距离，可以解释“为什么判定为移位”；
- 图片浏览、ROI 编辑、参数保存和生产配置使用同一套数据结构，调参结果具备复现条件。

因此，AOI_CV tuner 的成熟点是“所见即所得的算法诊断”，而不是“控件数量多”。

### 8.2 PCB_Dual 已经具备的能力

PCB_Dual 已经部分吸收了上述设计，当前不是没有 tuner：

- `app/ui/tuner_dialog.py` 提供通用 PySide6 调参窗口；
- `app/tools/tuner.py` 提供自动抽色、手动色板、HSV 掩膜、H/S/V 通道和预处理视图；
- 可以叠加模板 ROI，并把 HSV 六参数写回模板参数；
- 已支持灰度、Otsu、自适应二值和自动 Gamma 等基础辅助视图；
- `AlgorithmParamPanel` 具备防抖实时预览，并会尝试调用真实适配器；
- Shift 已有独立 tuner 入口和独立子进程，具备比通用 tuner 更接近 AOI_CV 的专项调参能力；
- 调参入口、参数面板和模板冻结参数已经形成了基本的“调整—应用—保存”链路。

因此，PCB_Dual 当前可以做到：用户拖动 HSV 参数时，实时观察颜色区域、通道和二值化效果，并在部分算法参数面板中观察真实算法预览。

### 8.3 当前尚未达到 AOI_CV tuner 的部分

当前通用 `TunerDialog` 主要刷新 HSV/预处理辅助视图，尚不能对所有算法统一做到“参数变化后实时展示完整真实检测结果”。主要差距如下：

1. **真实算法结果不统一。** HSV 调参页面和算法参数预览是两条能力链路，尚未形成统一的“当前参数—真实算法—结果状态”视图；必须确认每个算法都走生产适配器，不能用通用掩膜或 stub OK 代替。
2. **Shift 诊断信息未统一集成。** 通用页面没有稳定展示特征图热力图、峰值、IoU、dx/dy、移位距离和检测框/GT 框对照。
3. **SMT 中间结果不足。** 缺少焊锡 Mask、pad 覆盖率、多锡 ring、rim-toe 差集、连通域、差分和 Crack 等可直接定位问题的视图。
4. **Through-hole 中间结果不足。** 缺少配准状态、焊点覆盖率、标准图/测试图差分、轮廓结果和 `REVIEW` 原因的统一展示。
5. **执行透明度不足。** 调参时没有统一显示当前生效参数快照、ROI 快照、真实算法/stub/跳过/待复检/错误状态和本次耗时，用户难以判断看到的结果是否可用于生产。
6. **调试路径与产线路径可能不同。** 调试页仍存在 `allow_stub=True`，且 SMT 等成熟算法可能按检测项重复执行；调参页面必须明确“预览执行模式”，不能把预览结果直接解释为产线性能。

### 8.4 评审结论

PCB_Dual 的 HSV 调参、自动抽色、色板和基础辅助视图已经具备可用基础，Shift 也有专项 tuner；但目前还不能说已经完全达到 AOI_CV 专用 tuner 的算法可解释性。下一步不应简单增加更多滑条，而应优先建立统一的算法预览输出协议：每次参数变化都返回最终结果、执行状态、耗时、ROI/参数快照和可选中间视图。

### 8.5 建议优先级和验收标准

**P0：保证调参结果可信。**

- 调参预览必须优先调用生产适配器/生产算法核心；若未执行真实算法，界面必须明确标记 stub、SKIP 或未就绪，不得显示绿色 OK；
- 显示当前生效参数、ROI、执行模式、算法版本和耗时；
- 应用到模板后，用同一图片和同一参数再次运行，结果必须与预览一致或给出明确的路径差异说明。

**P1：补齐算法解释视图。**

- Shift：特征图、检测框/GT 框、峰值、IoU、dx/dy、移位距离；
- SMT：焊锡 Mask、pad 覆盖率、rim-toe、差分、连通域、Crack 和四类缺陷结果；
- Through-hole：配准/覆盖率/轮廓/差分、`REVIEW` 原因和人工复核入口；
- 统一显示 OK、NG、REVIEW、SKIP、ERROR，并将中间视图与算法结果绑定。

**P2：保证拖动调参的响应。**

- 缓存不随阈值变化的 HSV、灰度、配准和几何中间结果；
- 对滑条使用防抖，避免每个鼠标事件都重复整条流水线；
- 预估调参预览耗时，目标仍按单图不超过 1 秒控制；若真实算法超过预算，应限制 ROI/图片尺寸或采用缓存，而不是只放宽超时。

验收时至少应对正常、缺件、移位、轻微移位、少锡、多锡、连锡、虚焊、配准失败各抽取少量图片，逐项记录：参数快照、ROI、最终状态、缺陷框、中间视图、执行模式和耗时。

## 9. 状态、参数和结果追踪

### 当前优点

- 产线批量路径使用 `allow_stub=False`；
- 参数有模板专用和全局算法默认两个保存层级；
- 模板冻结参数可直接生效；
- 结果结构已有 `algorithm_version、elapsed_ms、error_code、metadata、template_version` 等字段；
- 调度器有单项耗时统计和 30 秒超时保护。

### 必须补齐

- `status` 枚举统一为 `OK/NG/ERROR/REVIEW/SKIP`，并规定总体状态映射；
- 保存实际生效参数快照，而不是只保存参数来源字符串；
- 保存 ROI 快照、ROI 坐标系、缩放/配准元数据；
- 保存算法仓库版本或 commit；
- 保存预览与产线执行模式；
- 结果表和归档记录中显示单项耗时。

## 10. 性能评估与 1 秒红线

目前代码具备预处理缓存，但算法适配器层仍存在重复工作风险：

- Shift：base ROI 数量 × 单次算法完整处理，旋转会进一步放大耗时；
- SMT：四个检测项可能分别执行完整 run；
- Through-hole：五个缺陷项可能分别创建实例和执行；
- 超时线程是后台 daemon 线程，超时后不能强制终止，长时间异常会占用 `_MAX_INFLIGHT_THREADS` 配额；
- 30 秒超时是故障保护，不是满足单图 1 秒的性能证明。

建议采用小样本、带进度的基准矩阵，不运行全量数据集：

| 场景 | 样本量 | 记录指标 |
|---|---:|---|
| 单 ROI 正常图 | 5~10 张 | 总耗时、预处理、算法、绘制 |
| 多 ROI 正常图 | 5~10 张 | ROI 数量与耗时曲线 |
| 每类缺陷 | 每类 5 张左右 | 耗时、检出、误报 |
| SMT 四类全开 | 5~10 张 | 一次流水线与逐项运行对比 |
| Through-hole 五类全开 | 5~10 张 | 初始化和算法耗时 |
| 配准失败/异常图 | 5 张左右 | 是否正确 REVIEW/ERROR、线程是否释放 |

验收建议：以 P95 单图耗时不超过 1 秒为目标；若某算法确实无法达到，应明确限制 ROI 数量、关闭旋转或采用缓存/批处理，而不是只提高超时阈值。

## 11. 优先级任务清单

### P0：影响检测可信度

1. 建立 AOI_CV 与 PCB_Dual 的逐算法 golden case 对比；
2. 明确并修正 `REVIEW/SKIP/ERROR/GRAY` 总体状态映射；
3. 禁止 stub 结果以绿色 OK 呈现；
4. 验证 SMT 自动缩放、配准和缺陷框原图坐标；
5. 验证 Shift UI 默认值与源算法默认值差异是否有产品依据。

### P1：影响实际使用

1. UI 显示 ROI 数量、有效性、模式限制和未就绪原因；
2. 缺陷表增加算法来源、ROI、单项耗时、状态和复判原因；
3. 增加参数/ROI/算法版本快照；
4. 增加 Shift 多 ROI、旋转和 LUT/AutoGamma 的小样本基准；
5. 增加 Through-hole 单框、多框、自动轮廓的交互验收。

### P2：影响维护和性能

1. SMT 共用流水线缓存并拆分四类结果；
2. Through-hole 多缺陷项复用可复用的预处理/配准结果；
3. Shift 多 ROI 复用整图 HSV、掩膜和积分图；
4. 为不可取消线程增加监控、资源告警和后续进程隔离方案。

## 13. PCB-DUAL-03 逐条状态证据（2026-10-13）

| 级别 | 原评审项 | 本批状态 | 证据/边界 |
|---|---|---|---|
| P0 | 统一请求/结果链路与状态语义 | 已实现/已验证 | 沿用现有 `AlgorithmResult`；TH 多焊点不可执行盘内项返回 `SKIP`，`REVIEW` 不降级；11 项定向回归通过。 |
| P1 | ROI/参数/算法版本及结果透明度 | 已部分实现 | SMT/TH `run_many` 严格按图像、模板、ROI、预处理、参数兼容分组；未扩大缺陷表字段，本批不覆盖 PCB-DUAL-02 状态链路。 |
| P1 | Through-hole 单框、多框、自动轮廓验收 | 未测试/真实阻塞 | AOI_CV 未找到配套 TH paired source/标准图/测试图/ROI/params；已有合成/单元路径不能替代真实验收。 |
| P1 | Shift 多 ROI/旋转/LUT/AutoGamma | 已实现/已验证 | 既有 `run_many` 与 `tests/test_shift_batch.py` 4 项精确等价/隔离测试，本批仅复核，未重复宣称算法修复。 |
| P2 | SMT 四类流水线复用 | 已实现/已验证 | SMT 单次生产 run 多开关、按 label 拆分；不兼容配置退回逐项，不跨请求缓存，diagnostics 保留。 |
| P2 | Through-hole 预处理/配准复用 | 已实现/受语义限制 | 同批次单焊点合并 `enabled_defects`；多焊点遵循源算法只执行连锡，其他项 `SKIP`，不复制结果。 |
| P2 | 不可取消线程监控/资源告警 | 未实现/并发范围冲突 | PCB-DUAL-02 已认领 scheduler 状态链路；本批未修改 scheduler。现有 30s 等待保护和 4 线程上限仍非结构化快照/告警，待其交付后验证。 |
| P2 | 单图 1 秒性能 | 未测试/不作结论 | 仅保留架构设计估算，未做专项基准；测试套件用时不作为性能证据。 |

真实样本无 GT，结果只用于源/适配器协议与输出一致性核查，不构成精度通过。

### PCB-DUAL-03 本轮补充状态（上表旧批次结论以下述为准）
- P0 状态 / P1 结果追踪：已有 PCB-DUAL-02 job_status、stub SKIP及缺陷表10列经离屏验证；REVIEW/SKIP/ERROR不得整体OK，算法、ROI、耗时与复核原因可见。算法未返roi_id时显示未提供，快照完整保留ROI，不伪造对应。
- P2 SMT复用：纠正旧四flags合并（cold依赖insufficient，不等价），scheduler已真实调用adapter.run_many，独立判定/渲染，仅请求局部共享提取。默认4选项extract_solder 4→1，core.run仍逐项，不虚报run次数。
- P2 TH复用：按实际阈值分组，默认5选项单框4可执行项inspect 4→2；同配置4→1，多框仅连锡1→1。不同有效参数隔离，REVIEW/code/就绪沿用单项契约，单项重新渲染。
- P2线程：已实现inflight_snapshot结构化不可取消任务状态、超时/重入/容量告警及批量超时metadata；0.01秒超时、无重试、释放回收已测，不冒称强杀。
- P1真实TH阻塞解除：alg_repo/through_hole/detect/input本地paired 2-8与3-2及templ_2/3已验证，源默认+paired solder_mode对照状态/框/输出图一致；未覆盖全部模式或注入整份config.yaml。固定seed真实总数3SMT+1Shift+2TH；SMT/TH批量对逐项字段/数组精确一致，Shift源协议本轮一致。
- 定向回归19/19 PASS，证据review/task/pcbdual03。单图速度仍仅0.95s设计估算，测试时间非性能证据；无GT、真实toe/rim/Crack全分支与稳定版/桌面验收仍为外部条件，不勾生产验收。

## 12. 后续小样本验收记录模板

每次验收至少记录：

```text
模板 ID / 版本：
算法仓库版本或 commit：
PCB_Dual 适配器版本：
输入图片：
标准图与测试图尺寸：
ROI 数量与坐标：
实际生效参数：
预处理与配准配置：
算法状态：OK / NG / ERROR / REVIEW / SKIP
缺陷类型和框：
源算法结果：
PCB_Dual 结果：
结果差异：
单项耗时：
整板耗时：
是否满足 1 秒目标：
复核结论：
```

## 13. 重要发现与设计记录

本次评审的重要发现如下，后续 demo 开发和调优应持续维护：

1. “能加载外部算法”不等于“完成算法迁移”；必须有源版本、适配版本、参数、坐标和 golden case 的可追溯记录；
2. 统一适配器解决了接口问题，但会引入标签、状态、ROI 坐标、默认参数和版本同步风险；
3. 一次成熟算法支持多类缺陷时，按 UI 检测项拆分不应自动等价为多次完整算法运行；
4. `REVIEW` 不能用 `SKIP` 隐式代替，否则用户无法知道是条件不满足还是结果不可靠；
5. 固定 `confidence=1.0` 不能当作检测置信度展示；
6. 单项 30 秒超时只能防止等待无限阻塞，不能证明满足单图 1 秒工程要求；
7. 调参页面必须同时展示“当前生效参数”和“结果执行模式”，否则用户很难复现实验结果。

---

**评审结论：**PCB_Dual 目前具备继续产品化的结构基础，传统 CV 已实际接入，但应在宣称“正确迁移”和“满足产线速度”前，先完成 P0 golden case、状态语义、坐标闭环和性能基准验收。

## 14. PCB-DUAL-04 完整状态/证据（2026-10-13，开发待验收）

本轮新增8测试+指定既有19测试=27 PASS。证据 `review/task/pcbdual04/{result.md,env.json,cmd.txt,raw/unittest.txt}`。本轮真实paired=0、不重复前批6例；合成图fixture=4，另有容量、TH几何和离屏交互。不把生产核心运行的合成测试称为真实样本精度。

|优先级条目|当前状态/代码位置|证据与未验收条件|
|---|---|---|
|P0-1 golden逐算法|部分真实已验证；adapters|沿用pcbdual03 3SMT+1Shift+2TH；全面GT/业务精度待验收|
|P0-2 状态映射|已实现；contract.job_status/types/defect_table|adapter_batch非OK汇总/状态表PASS|
|P0-3 stub不绿色|已实现；scheduler execution_mode/stub SKIP|既有状态链路，本轮真实适配器allow_stub=False|
|P0-4 SMT缩放配准坐标|本轮修复；solder_smt_all_alg.py run|resize先还原后加ROI、原图输出；alignment None明确ERROR；新增2项PASS，真实配准精度待验收|
|P0-5 Shift默认值|已核对；shift_adapter _SHARED_SPECS/default_config|bool归一逐键差异=[]，不存在需要产品决策的当前默认差异；冻结参数仍优先|
|P1-1 ROI有效性/模式限制|已实现；calibration_dialog/detect_page|TH实际ready动态禁用/tooltip；0→1→2→1离屏PASS；真实桌面待验收|
|P1-2 缺陷表透明|已实现；defect_table|10列算法/ROI/耗时/状态/原因离屏PASS；未返回roi_id不伪造|
|P1-3 参数ROI算法版本快照|已实现；scheduler request_snapshot|params/rois/algorithm_version/param_schema_version/version_id既有batch冻结回归PASS|
|P1-4 Shift多ROI旋转LUT/Gamma|已实现；shift_adapter/core|test_shift_batch4项与tuner shift PASS；业务精度仍待验收|
|P1-5 TH三模式交互|本轮补齐；detect_page+adapter is_ready|checkbox和就绪2项PASS，既有TH诊断/REVIEW/SKIP回归PASS；本地paired旧阻塞已解除，所有真实模式与桌面待验收|
|P2-1 SMT复用|已实现；smt_adapter.run_many|extract4→1、独立flags/判断/渲染/配置隔离PASS|
|P2-2 TH复用|已实现；through_hole_adapter.run_many|默认inspect4→2、兼容配置4→1，多框仅连锡PASS|
|P2-3 Shift复用|已实现；shift run_many|整图提取/积分复用和请求隔离PASS|
|P2-4 不可取消线程监控|已实现/本轮容量补测；scheduler|真实4槽满，第5拒绝WARNING，释放恢复/不同key隔离、batch一次超时无重试、最终count=0；进程隔离只是后续选项|

SMT rim-toe精确差集、rule1提前判定不可用crack、实际Python Hough调用及缺输入诊断本轮PASS（_solder_core.py生产函数，合成图）。未证实AOI_CV toe/rim完整paired，不宣称真实Crack精度。输出diagnostic mask保留工作图scale/offset，缺陷框和output_image归原图。GT、业务数据、真实桌面和生产验收是外部缺口，不等于上述代码未做；没有勾生产验收。27测试2.356秒不是单图性能实测，速度只沿用设计估算。
