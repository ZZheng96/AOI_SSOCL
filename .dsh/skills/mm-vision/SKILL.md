---
name: mm-vision
description: 按需调用火山方舟（豆包/Seedream/Seedance）大模型做读图与生图。当用户需要看懂一张图片（论文配图检查、CV 训练结果分析、截图/图表解读、把图片内容转成文字）、生成论文配图或示意图（文生图、图生图）、或生成短视频时使用本 skill。DeepSeek 主模型是纯文本，读图必须经本 skill 的视觉模型转成文字后才能继续。
whenToUse: 读图/看图/图片分析/图片转文字、生成图片/配图/示意图/流程图配图、图生图、生成短视频
---

# 火山方舟 读图 + 生图 + 生视频（mm-vision）

主模型（DeepSeek）是纯文本模型，**图片内容一律经本 skill 的脚本转发给火山方舟**处理，拿到文字结果后再继续。

## 前置条件（一次性）

1. **API Key**：设置环境变量或写入文件（二选一）：
   - `setx ARK_API_KEY "ark-xxxxx"`（Windows 全局，新开终端生效）
   - 或写入 `C:\Users\<你>\.ark_api_key`（脚本自动读取，推荐，不进环境变量）
2. **Base URL（重要）**：脚本默认 `https://ark.cn-beijing.volces.com/api/plan/v3`（套餐 Plan 版）。
   ⚠️ **不要用 `/api/v3` 路径**（接入会产生额外费用）。
3. **模型**（脚本默认值，可用环境变量覆盖）：
   - 视觉理解：`doubao-seed-2-0-mini-260428`（⚠️ 当前视觉模型不支持通过 Auto 及控制台切换使用）
   - 生图：`doubao-seedream-5.0-lite`（用户当前套餐指定模型）
   - 生视频：`doubao-seedance-1.5-pro`（⚠️ 即将下线，当前不支持新增接入；Medium 套餐暂不支持 Seedance 2.0 系列）
4. 脚本零依赖（纯标准库），用项目 venv 或系统 python 均可。

## 脚本入口

```powershell
python .dsh/skills/mm-vision/scripts/doubao.py <子命令> [参数]
```

## 三个子命令

### 1. 读图（视觉理解）→ `vision`

```powershell
python .dsh/skills/mm-vision/scripts/doubao.py vision "figures/result.png" "这张图展示什么？纵轴单位是什么？峰值在哪？"
python .dsh/skills/mm-vision/scripts/doubao.py vision "C:/x/cv_pred.png" "逐像素检查：预测掩膜和真值有哪些差异区域？" --detail high
```

- 支持本地路径或 URL；`--out 文件.txt` 可把结果落盘。
- 典型用途：论文配图内容核对、CV 结果图解读、截图/图表转文字、报错信息 OCR。

### 2. 生图（Seedream，同步秒回）→ `generate`

```powershell
# 文生图
python .dsh/skills/mm-vision/scripts/doubao.py generate "论文示意图：灰色背景、蓝色折线表示精度随 epoch 上升，带误差带" --ratio 16:9 --out-dir figures --prefix acc
# 图生图（风格迁移/改图）
python .dsh/skills/mm-vision/scripts/doubao.py generate "把背景改成白色，标题改为中文" --image figures/sketch.png --out-dir figures --prefix v2
# 指定分辨率
python .dsh/skills/mm-vision/scripts/doubao.py generate "..." --size 2048x2048 --out-dir figures
```

- 必须给 `--out-dir` 才会落盘（按 `前缀_序号.png` 保存），否则只打印 URL。
- 论文配图建议 `--ratio 16:9` 或 `--size 1920x1080`，生成后仍需人工复核（AI 生图可能有文字错误）。

### 3. 生视频（Seedance，异步）→ `video`

```powershell
# 提交任务，立即返回 task_id
python .dsh/skills/mm-vision/scripts/doubao.py video "一段数据可视化动画：折线从左下到右上" --resolution 720p
# 阻塞等待完成（最长 10 分钟）
python .dsh/skills/mm-vision/scripts/doubao.py video "..." --wait
```

## 使用流程（对 agent 的约束）

1. 用户提到"看图/读图/分析这张图" → 用 `vision` 子命令，**必须**把返回文字完整转述给用户。
2. 用户要求"画一张 XX 图/配图" → 用 `generate`，先确认目标用途（论文/演示）与宽高比，生成后**落盘到项目 `figures/` 或用户指定目录**，并告诉用户保存路径。
3. 生图/生视频结果如有问题（文字错误、构图不对），重新生成或改用 `--image` 图生图迭代。
4. 涉及论文图片时，遵守 `mm-plotting` 的规范（分辨率 300dpi、图题引用），AI 生图通常只作示意图/装饰图，**不得**作为论文核心结果图。

## 故障排查

- `未找到 ARK_API_KEY` → 按前置条件配置 key。
- `火山方舟 API 错误 401/403` → key 无效或未开通对应模型。
- `错误 429` → 免费额度耗尽或限流，稍后重试。
- **额度用完/欠费** → 控制台充值或等待额度恢复；`/api/plan/v3` 计费路径已默认配置，勿手动改成 `/api/v3`。
- 读图结果为空 → 换 `--detail high` 或把 prompt 写得更具体。
- Windows 路径含中文 → 用正斜杠或加引号。
- Seedance 视频不可用 → 该模型即将下线且不支持新增接入，改用 Seedream 生图或换可用模型。
