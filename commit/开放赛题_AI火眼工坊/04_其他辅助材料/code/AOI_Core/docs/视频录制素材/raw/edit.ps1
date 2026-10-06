# 项目视频剪辑流水线：seg1/seg2/stats/pcb + 静态卡 → final_raw.mp4
# 统一 1440x900 / 30fps / h264 / 无音轨
$ff = "d:\CGAIC\.trae\tools\bin\ffmpeg.cmd"
$fp = "d:\CGAIC\.trae\tools\bin\ffprobe.cmd"
$raw = "d:\CGAIC\commit\03_项目视频\raw"
$cl = "$raw\clips"
New-Item -ItemType Directory -Force -Path $cl | Out-Null

# 1. 静态卡 -> 静帧视频
& $ff -y -loop 1 -i "$raw\cards\title.png" -t 12 -r 30 -c:v libx264 -preset veryfast -crf 20 -pix_fmt yuv420p -an "$cl\00_title.mp4" 2>$null
& $ff -y -loop 1 -i "$raw\cards\pain.png" -t 15 -r 30 -c:v libx264 -preset veryfast -crf 20 -pix_fmt yuv420p -an "$cl\01_pain.mp4" 2>$null
& $ff -y -loop 1 -i "$raw\cards\metrics.png" -t 12 -r 30 -c:v libx264 -preset veryfast -crf 20 -pix_fmt yuv420p -an "$cl\11_metrics.mp4" 2>$null
& $ff -y -loop 1 -i "$raw\cards\end.png" -t 8 -r 30 -c:v libx264 -preset veryfast -crf 20 -pix_fmt yuv420p -an "$cl\12_end.mp4" 2>$null

# 2. 主流程片段（seg1 时间线）: 截取 + 变速 + 统一分辨率
& $ff -y -ss 6 -t 28 -i "$raw\seg1.mp4" -vf "setpts=PTS/0.8,scale=1440:900:force_original_aspect_ratio=decrease,pad=1440:900:(ow-iw)/2:(oh-ih)/2" -r 30 -c:v libx264 -preset veryfast -crf 20 -pix_fmt yuv420p -an "$cl\02_src.mp4" 2>$null
& $ff -y -ss 36 -t 5.5 -i "$raw\seg1.mp4" -vf "setpts=PTS/0.5,scale=1440:900:force_original_aspect_ratio=decrease,pad=1440:900:(ow-iw)/2:(oh-ih)/2" -r 30 -c:v libx264 -preset veryfast -crf 20 -pix_fmt yuv420p -an "$cl\03_wo.mp4" 2>$null
& $ff -y -ss 41.5 -t 10.5 -i "$raw\seg1.mp4" -vf "scale=1440:900:force_original_aspect_ratio=decrease,pad=1440:900:(ow-iw)/2:(oh-ih)/2" -r 30 -c:v libx264 -preset veryfast -crf 20 -pix_fmt yuv420p -an "$cl\04_prep.mp4" 2>$null
& $ff -y -ss 52 -t 9 -i "$raw\seg1.mp4" -vf "setpts=PTS/0.7,scale=1440:900:force_original_aspect_ratio=decrease,pad=1440:900:(ow-iw)/2:(oh-ih)/2" -r 30 -c:v libx264 -preset veryfast -crf 20 -pix_fmt yuv420p -an "$cl\05_mon.mp4" 2>$null
& $ff -y -ss 61 -t 4.3 -i "$raw\seg1.mp4" -vf "setpts=PTS/0.5,scale=1440:900:force_original_aspect_ratio=decrease,pad=1440:900:(ow-iw)/2:(oh-ih)/2" -r 30 -c:v libx264 -preset veryfast -crf 20 -pix_fmt yuv420p -an "$cl\06_fb.mp4" 2>$null
& $ff -y -ss 65.3 -t 12.7 -i "$raw\seg1.mp4" -vf "setpts=PTS/0.7,scale=1440:900:force_original_aspect_ratio=decrease,pad=1440:900:(ow-iw)/2:(oh-ih)/2" -r 30 -c:v libx264 -preset veryfast -crf 20 -pix_fmt yuv420p -an "$cl\07_rv.mp4" 2>$null
& $ff -y -ss 78 -t 5 -i "$raw\seg1.mp4" -vf "setpts=PTS/0.5,scale=1440:900:force_original_aspect_ratio=decrease,pad=1440:900:(ow-iw)/2:(oh-ih)/2" -r 30 -c:v libx264 -preset veryfast -crf 20 -pix_fmt yuv420p -an "$cl\08_learn.mp4" 2>$null
& $ff -y -ss 83 -t 3.7 -i "$raw\seg1.mp4" -vf "setpts=PTS/0.5,scale=1440:900:force_original_aspect_ratio=decrease,pad=1440:900:(ow-iw)/2:(oh-ih)/2" -r 30 -c:v libx264 -preset veryfast -crf 20 -pix_fmt yuv420p -an "$cl\09_req.mp4" 2>$null

# 3. 统计页 + 双引擎
& $ff -y -ss 0 -t 25 -i "$raw\stats.mp4" -vf "scale=1440:900:force_original_aspect_ratio=decrease,pad=1440:900:(ow-iw)/2:(oh-ih)/2" -r 30 -c:v libx264 -preset veryfast -crf 20 -pix_fmt yuv420p -an "$cl\10_stats.mp4" 2>$null
& $ff -y -ss 6 -t 20 -i "$raw\pcb.mp4" -vf "scale=1440:900:force_original_aspect_ratio=decrease,pad=1440:900:(ow-iw)/2:(oh-ih)/2" -r 30 -c:v libx264 -preset veryfast -crf 20 -pix_fmt yuv420p -an "$cl\10b_pcb.mp4" 2>$null

# 4. 拼接
$list = "$cl\list.txt"
@("00_title.mp4","01_pain.mp4","02_src.mp4","03_wo.mp4","04_prep.mp4","05_mon.mp4","06_fb.mp4","07_rv.mp4","08_learn.mp4","09_req.mp4","10_stats.mp4","10b_pcb.mp4","11_metrics.mp4","12_end.mp4") | ForEach-Object { "file '$cl\$_'" } | Set-Content -Encoding ASCII $list
& $ff -y -f concat -safe 0 -i $list -c copy "$raw\final_raw.mp4" 2>&1 | Select-Object -Last 2

# 5. 校验
& $fp -v error -show_entries format=duration,size -show_entries stream=codec_name,width,height -of default=noprint_wrappers=1 "$raw\final_raw.mp4"
