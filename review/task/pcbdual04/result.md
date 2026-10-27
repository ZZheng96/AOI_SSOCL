# PCB-DUAL-04 开发验证（2026-10-13）

结论：PASS（开发定向回归），待验收（业务精度/真实桌面/生产条件）。27项全部通过：新增8项，按要求复跑既有19项；既有真实6例未重复运行。原始终端结果见 raw/unittest.txt，命令见 cmd.txt，环境见 env.json。

## 修复与新增覆盖
- SMT核心 solder_smt_all_alg.py：模板存在但align_template返回None时明确失败，适配器映射ERROR，禁止虚焊空结果伪OK。缩放先恢复局部坐标再加原图ROI偏移，修复偏移被倍乘；output_image使用原图尺寸并在原图坐标绘制缺陷。
- DetectPage._on_template_changed：按TH实际is_ready动态禁用/取消不适用项并保留原因tooltip。0/1框禁连锡，≥2框只可连锡；重新绑定回单框恢复，标定说明与scheduler使用同一规则。
- tests/test_pcbdual04.py：4真实后台线程容量饱和、第5拒绝及WARNING、同key拒绝/不同key隔离、Event释放恢复/最终回收；scheduler批量超时实际一次调用且每项RUN_TIMEOUT/inflight；rim-toe精确差集和规则1提前跳过crack；生产Python Hough实际执行与缺输入不可用；resize+ROI偏移框/工作mask/原图输出；配准失败ERROR；TH 0/1/2模式就绪与离屏checkbox切换。

覆盖数量：新增真实paired=0；明确合成图fixture=4（rim/toe黑图、暗线灰图、缩放ROI图、配准失败图），另有2项线程测试和TH几何/离屏模式测试。Hough执行并不等于真实裂纹检出精度。搜索d:/CGAIC/AOI_CV和AOI_SSOCL的toe/rim命名JSON未获得配套资源；未证实真实toe/rim paired，本轮选择明确合成，不声称资源永久不存在。

## P0/P1/P2完整状态
|条目|开发状态与证据|待验收/边界|
|---|---|---|
|P0-1 golden对照|沿用pcbdual03/raw/real_cases.json的3 SMT+1 Shift+2 TH；未重复|全面golden/GT精度待业务样本|
|P0-2 状态映射|contract.job_status、DetectSummary、DefectTable；test_status_trace_table与non_ok通过|真实业务复判待验收|
|P0-3 stub不绿色|scheduler allow_stub执行模式/跳过既有实现，真实算法测试allow_stub=False|非生产调试路径不构成生产验收|
|P0-4 SMT缩放/配准坐标|本轮核心修复+resize_roi_offset/alignment_failure测试通过|真实配准/toe/rim/Crack精度待样本|
|P0-5 Shift默认差异|ShiftAdapter共享ParamSpec与生产default_config逐键比较，bool归一后差异=[]|无需新增产品默认值决策；模板旧冻结值仍优先|
|P1-1 ROI模式限制|calibration_dialog说明；detect_page按TH ready禁用并tooltip；offscreen 0→1→2→1通过|真实桌面未验收|
|P1-2 结果透明|defect_table10列、状态追踪既有测试通过|源算法未返roi_id时不伪造对应|
|P1-3 参数ROI算法版本|scheduler request_snapshot params/rois/algorithm_version/param_schema_version/version_id；冻结参数和batch测试通过|源版本升级需维护manifest|
|P1-4 Shift多ROI旋转LUT/Gamma|沿用test_shift_batch四项及tuner shift用例通过|未重复真实精度，无专项性能实测|
|P1-5 TH交互|本轮readiness+checkbox离屏，已有TH实际diagnostics与SKIP/REVIEW测试通过|真实自动轮廓各种形状/桌面待验收；pcbdual03本地paired已解除旧阻塞|
|P2-1 SMT复用|test_adapter_batch实际extract4→1通过|保持独立flags判断，无跨请求缓存|
|P2-2 TH复用|实际inspect4→2/同配置4→1既有测试通过|多框只连锡，不强行合并不兼容阈值|
|P2-3 Shift复用|test_shift_batch精确输出/提取积分复用/请求配置隔离通过|无需重复实现|
|P2-4 不可取消线程|本轮4容量/告警/拒绝/释放恢复/batch无重试，最终count=0|线程不能强杀；进程隔离仅后续设计，未实现且非本轮硬要求|

测试套件2.356s是回归用时，不是单图性能指标；保留既有0.95s设计估算，不作速度达标。没有操作真实桌面，不改stable/AOI_Core/生产DB/生产快照，不git提交。当前局部明确代码缺口已修复；仍有外部验收及未来进程隔离选项，不能宣称全面生产完成。
