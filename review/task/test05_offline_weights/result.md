# TEST-05 离线/干净缓存加载验证（AC-075）

## 结论：部分通过 / 存在 FAIL 子项（实测）

| 场景 | 环境 | 结果 | 判定 |
|---|---|---|---|
| A 离线+vendor 完整 | HF_HUB_OFFLINE=1、空 HOME | 3.78s 加载成功，走随包权重 | **PASS** |
| B 离线+DINOV2_LOCAL_DIR 指向空目录 | 同上 | 仍 2.77s 成功：`_vendored_dir` 有随包默认搜索路径，找到 `AOI_tree/assets/dinov2` | PASS（且证明随包兜底不依赖环境变量） |
| C 离线+vendor 有 hubconf 缺权重文件+空缓存 | 同上 | **180s 超时无输出（挂起）** | **FAIL** |

## FAIL 子项详情（转开发修复）

- C 场景日志路径：vendor 缺权重 → 打印“回退 hub 缓存/在线下载” → 缓存缺失 →
  `torch.hub.load("facebookresearch/dinov2", ...)` 直连 GitHub，**无超时控制**，
  180s 内无任何输出与错误（进程被外部超时杀掉）。
- 违反 AC-075 通过条件：“缺权重时明确报错，不能静默联网或卡住”。
  本机有网时该路径会静默联网下载（违反“不静默联网”）；网络不通/被墙时挂起（违反“不卡住”）。
- 建议修复方向（供开发参考）：离线模式（HF_HUB_OFFLINE=1 或配置项）下禁止 torch.hub
  在线回退并立即 raise；或给 hub.load 加显式超时与清晰错误消息。

## 口径与边界

- “断网”为 HF_HUB_OFFLINE=1/TRANSFORMERS_OFFLINE=1 模拟；torch.hub 不受其约束
  （C 场景正是借此暴露联网行为）。未物理断网。
- 干净缓存用全新空 HOME/USERPROFILE/TORCH_HOME/HF_HOME（TEMP 目录），**未清用户真实缓存**。
- A 场景 stderr 的 84.2M 进度条为 `file:///` 本地权重读取，非网络下载。
- 被测加载器为开发版 EXPs（`AOI_tree/EXPs/src/backbone/dino.py`，只读引用，未修改）；
  稳定版不含 EXPs。随包权重 `AOI_tree/assets/dinov2/dinov2_vits14_pretrain.pth`（84.2MiB）。
