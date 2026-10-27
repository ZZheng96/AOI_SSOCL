# PCB-TUNER-02 验证结果

日期：2026-10-13

## 已完成

- `TunerDialog` 复用 `AlgorithmParamPanel` / `ParamSpec`，支持 shared 与当前算法的非 HSV int/float/布尔枚举/旋转参数编辑；编辑经既有 250ms 防抖触发真实 `DetectScheduler.run` 预览，不重开窗口。
- 预览与应用使用同一参数层合并规则；应用写入当前模板 ParamStore，后续冻结继续使用 resolved 参数。
- 修复 ParamSpec enum 对 Python bool 的显示归一化（`false`/`true`）。
- 未实现请求级完整结果缓存：SMT/TH 各检测项 cfg、ROI/参数和输出语义仍未证明可安全复用；保持逐检测项生产执行，避免跨请求串结果。
- Shift 源算法 `code=1` 为检测失败、`code=2` 才是算法错误；适配器保持该契约，真实对照脚本修正错误归一化。

## 定向回归

命令：

```powershell
$env:QT_QPA_PLATFORM='offscreen'; py -3.11 -m unittest tests.test_tuner_preview -v
```

结果：7/7 PASS，约 1.9s；包含非 HSV 实时编辑/应用、冻结一致、Shift/SMT/TH 适配器诊断、状态 REVIEW、取消/关闭和代际丢弃。该耗时不是性能验收。

## 真实 AOI_CV paired cases

固定种子 20261013，脚本 `tests/tuner02_real_cases.py`，TEMP cwd，原图/配套 ROI/参数/模板只读：

1. SMT bridge：源/适配器 NG/NG，box 相同。
2. SMT insufficient：源/适配器 NG/NG，box 相同。
3. SMT cold solder：源/适配器 OK/OK，box 均为空。
4. Shift：源/适配器 NG/NG，但 box 不同；源 `[0,2,419,426]`，适配器 `[148,64,419,426]`，dx/dy/距离/rotation/mask/output_image 均不同。该 case 不等价通过，待独立定位源/开发版核心差异。

完整原始记录：`raw/real_cases.json`；环境：`env.json`。

## 未完成 / 边界

- AOI_CV 未找到可直接配套的 through-hole 源入口、paired ROI/参数/模板，TH 真实对照 BLOCKED；现有 TH 仅离屏定向测试。
- SMT 当前仅覆盖 bridge/insufficient/cold-solder 各一例；缺 toe/rim 配套，不能覆盖全部几何/Crack 分支。
- 无 GT，以上是源算法与适配器等价性核查，不是精度通过；不宣称生产验收或性能达标。
- 稳定版、AOI_Core、其他 agent 状态链路未修改；未提交 Git。
