---
name: screen-record
description: 在 Windows 上录制屏幕/窗口/区域为视频（用于项目演示视频、操作演示、Bug 复现等）。支持指定目标录制、按时长停止、校验视频、常见剪辑（裁剪/拼接/加速/缩放/加水印）。基于已安装的 ffmpeg（gdigrab）与 screencast-mcp（若已注册 MCP）。
license: Apache-2.0
---

# 屏幕录制（Windows）

本工作区已安装以下工具（2026-08-31 安装）：

- **ffmpeg / ffprobe**：`winget install Gyan.FFmpeg` 全局安装。
  ⚠️ IDE 终端不继承系统新 PATH，统一用 shim 调用：
  - ffmpeg → `d:\CGAIC\.trae\tools\bin\ffmpeg.cmd`
  - ffprobe → `d:\CGAIC\.trae\tools\bin\ffprobe.cmd`
  下文命令均以 `ffmpeg`/`ffprobe` 简写，实际执行时替换为上述全路径（或先 `$env:PATH = "$env:PATH;d:\CGAIC\.trae\tools\bin"`）。
- **screencast-mcp**：`npm install -g @tmhs/screencast-mcp`，MCP server（stdio，Windows gdigrab）。
  若已在 Trae「MCP 管理」注册，可直接用 `start_recording` / `stop_recording` / `sample_frames` / `trim` 等工具；
  未注册时用下方 ffmpeg 直连命令（同样可靠，推荐）。
  注：screencast-mcp 需在 ffmpeg/ffprobe 的 PATH 或 `FFMPEG_PATH`/`FFPROBE_PATH` 环境变量中能找到工具。

## 一、录制（ffmpeg 直连，推荐）

所有录制都是显式命令，绝不自动启动。`gdigrab` 是 Windows 专有的桌面捕获设备。

### 1.1 全屏录制（按时长自动停止）

```powershell
# 录 60 秒全屏，30fps，H.264，存到指定路径
ffmpeg -y -f gdigrab -framerate 30 -video_size 1920x1080 -offset_x 0 -offset_y 0 `
  -i desktop -t 60 -c:v libx264 -preset veryfast -crf 23 -pix_fmt yuv420p output.mp4
```

关键参数：
- `-f gdigrab -i desktop`：整屏；`-i title="窗口标题"`：按窗口标题（不含 `-video_size`）。
- `-t 秒`：录制时长，到点自动结束（推荐，简单可控）。
- 想手动控制结束（后台任务）：去掉 `-t`，用 `ffmpeg ... output.mp4` 启动为后台进程，结束用
  `StopCommand` 终止对应命令（发送 q 使文件正常封口）。

### 1.2 只录主显示器或指定区域

```powershell
# 单显示器（多屏时指定 index）
ffmpeg -y -f gdigrab -framerate 30 -i monitor:1 -t 30 -c:v libx264 -preset veryfast -pix_fmt yuv420p mon.mp4

# 指定区域：左上角(100,80) 起 1280x720
ffmpeg -y -f gdigrab -framerate 30 -video_size 1280x720 -offset_x 100 -offset_y 80 -i desktop `
  -t 30 -c:v libx264 -preset veryfast -pix_fmt yuv420p region.mp4
```

### 1.3 按窗口标题录制（推荐用于演示，不含无关区域）

```powershell
# 窗口必须可见、置顶、未最小化；标题大小写不敏感，取最上层匹配窗口
ffmpeg -y -f gdigrab -framerate 30 -i title="火眼工坊" -t 30 -c:v libx264 -preset veryfast -pix_fmt yuv420p win.mp4
```

### 1.4 录制 + 系统声音（可选）

```powershell
# 需要虚拟音频回环设备（如 VB-CABLE）；麦克风不支持
ffmpeg -y -f gdigrab -framerate 30 -i desktop -f dshow -i audio="virtual-audio-capturer" `
  -t 30 -c:v libx264 -preset veryfast -c:a aac -b:a 128k -pix_fmt yuv420p with_sound.mp4
```

## 二、校验录制结果

```powershell
ffprobe -v error -show_entries format=duration,size -show_entries stream=codec_name,width,height,r_frame_rate -of default=noprint_wrappers=1 output.mp4
```

确认：duration 符合预期、有视频流、可解码。失败常见原因：窗口被遮挡/最小化（gdigrab 抓黑帧）、
时长参数写错、路径含中文与特殊字符（用引号包裹）。

## 三、常用剪辑（ffmpeg）

- **剪片段**（快，流复制）：`ffmpeg -y -i in.mp4 -ss 5 -to 15 -c copy out.mp4`
- **拼接**：先写 `list.txt`（`file 'a.mp4'`、`file 'b.mp4'`，同编码），
  `ffmpeg -y -f concat -safe 0 -i list.txt -c copy out.mp4`
- **加速**：`ffmpeg -y -i in.mp4 -filter:v "setpts=PTS/8" -an out8x.mp4`（8 倍，去音轨）
- **统一缩放 1080p**：`ffmpeg -y -i in.mp4 -vf "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2" -c:v libx264 -preset veryfast -crf 23 -pix_fmt yuv420p out.mp4`
- **加文字标题**（drawtext，中文字体需指定字体文件路径）：
  `ffmpeg -y -i in.mp4 -vf "drawtext=text='火眼工坊':fontfile=C:/Windows/Fonts/msyh.ttc:fontsize=72:fontcolor=white:x=(w-text_w)/2:y=(h-text_h)/2" -c:v libx264 -preset veryfast -pix_fmt yuv420p out.mp4`
- **压缩到 200M 以内**：`ffmpeg -y -i in.mp4 -c:v libx264 -crf 28 -preset veryfast -pix_fmt yuv420p small.mp4`

## 四、参赛演示视频录制流程（参考 commit/03_项目视频/录制脚本.md）

1. 清空无关窗口/通知，桌面壁纸中性，路径避免出现用户名。
2. 启动演示系统（如 `python main.py` 或 `python tests\_ui_demo_flow.py`）。
3. 按分镜表逐段录制：每段用 `-t` 限制时长，录完立即 `ffprobe` 校验。
4. 等待段（如预训练）单独录短片段后期加速 8-16×。
5. 用剪辑命令拼接/裁剪/加字幕，最后压缩到 mp4 ≤200M。
6. 命名按规范：`XXX团队_XXX项目_项目视频.mp4`。
