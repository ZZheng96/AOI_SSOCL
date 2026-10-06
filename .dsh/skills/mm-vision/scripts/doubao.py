#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
火山方舟 (Volcengine Ark) 豆包/Seedream/Seedance 命令行封装。
按需调用：读图(视觉理解) / 生图(Seedream) / 生视频(Seedance)。

依赖：仅 Python 标准库（urllib/json/base64），无需 pip 安装。

环境变量：
  ARK_API_KEY        必填，火山方舟 API Key（ark-xxxxx）
  DOUBAO_BASE_URL    可选，默认 https://ark.cn-beijing.volces.com/api/v3
  DOUBAO_MODEL       可选，视觉模型，默认 doubao-seed-2-0-mini-260428
  SEEDREAM_MODEL     可选，生图模型，默认 doubao-seedream-5-0-lite-260128
  SEEDREAM_I2I_MODEL 可选，图生图模型，默认 doubao-seededit-3-0-i2i-250628
  SEEDANCE_MODEL     可选，生视频模型，默认 doubao-seedance-2-0-260128
"""
import argparse
import base64
import json
import mimetypes
import os
import sys
import time
import urllib.error
import urllib.request

BASE_URL = os.environ.get("DOUBAO_BASE_URL", "https://ark.cn-beijing.volces.com/api/plan/v3")
DEFAULT_VISION_MODEL = os.environ.get("DOUBAO_MODEL", "doubao-seed-2-0-mini-260428")
DEFAULT_SEEDREAM_MODEL = os.environ.get("SEEDREAM_MODEL", "doubao-seedream-5.0-lite")
DEFAULT_SEEDREAM_I2I_MODEL = os.environ.get("SEEDREAM_I2I_MODEL", "doubao-seededit-3-0-i2i-250628")
DEFAULT_SEEDANCE_MODEL = os.environ.get("SEEDANCE_MODEL", "doubao-seedance-1.5-pro")


def api_key() -> str:
    key = os.environ.get("ARK_API_KEY", "").strip()
    if key:
        return key
    # 兜底：从用户目录的 .ark_api_key 读取（避免每次 export）
    for path in (os.path.expanduser("~/.ark_api_key"),
                 os.path.expanduser("~/.config/dsh/ark_api_key")):
        try:
            with open(path, encoding="utf-8") as fh:
                key = fh.read().strip()
            if key:
                return key
        except OSError:
            continue
    raise SystemExit("错误：未找到 ARK_API_KEY。请设置环境变量 ARK_API_KEY=ark-xxxxx "
                     "或写入 ~/.ark_api_key")


def ark_request(path: str, body: dict | None = None, timeout: int = 180):
    """POST/GET 火山方舟 API，返回解析后的 JSON。"""
    url = BASE_URL + path
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, method="POST" if body is not None else "GET")
    req.add_header("Authorization", f"Bearer {api_key()}")
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise SystemExit(f"火山方舟 API 错误 {e.code}: {detail[:800]}") from e
    except urllib.error.URLError as e:
        raise SystemExit(f"网络错误: {e.reason}") from e


def resolve_image_source(source: str) -> str:
    """本地路径 -> data URL；URL 原样返回。"""
    if source.startswith(("http://", "https://")):
        return source
    path = os.path.abspath(os.path.expanduser(source))
    if not os.path.isfile(path):
        raise SystemExit(f"图片文件不存在: {path}")
    mime = mimetypes.guess_type(path)[0] or "image/png"
    with open(path, "rb") as fh:
        b64 = base64.b64encode(fh.read()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def download(url: str, out_path: str):
    req = urllib.request.Request(url, headers={"User-Agent": "dsh-mm-vision"})
    with urllib.request.urlopen(req, timeout=180) as resp, open(out_path, "wb") as fh:
        fh.write(resp.read())


# ── 视觉理解（读图）────────────────────────────────────────────
def cmd_vision(args):
    image = resolve_image_source(args.image)
    content = [
        {"type": "text", "text": args.prompt},
        {"type": "image_url", "image_url": {"url": image, "detail": args.detail}},
    ]
    body = {
        "model": os.environ.get("DOUBAO_MODEL", DEFAULT_VISION_MODEL),
        "messages": [{"role": "user", "content": content}],
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "stream": False,
    }
    data = ark_request("/chat/completions", body)
    text = data["choices"][0]["message"]["content"] or ""
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(f"已写入 {args.out}")
    else:
        print(text)


# ── Seedream 生图 ──────────────────────────────────────────────
def cmd_generate(args):
    body = {"model": os.environ.get("SEEDREAM_MODEL", DEFAULT_SEEDREAM_MODEL),
            "prompt": args.prompt, "n": args.n}
    if args.image:
        body["model"] = os.environ.get("SEEDREAM_I2I_MODEL", DEFAULT_SEEDREAM_I2I_MODEL)
        body["image"] = resolve_image_source(args.image)
        body["response_format"] = "url"
    if args.size:
        body["size"] = args.size
    elif args.ratio:
        body["ratio"] = args.ratio

    data = ark_request("/images/generations", body, timeout=300)
    images = data.get("data") or []
    if not images:
        raise SystemExit("未返回任何图片。")
    saved = []
    for i, img in enumerate(images, 1):
        url = img.get("url") or ""
        b64 = img.get("b64_json") or ""
        if args.out_dir:
            os.makedirs(args.out_dir, exist_ok=True)
            out = os.path.join(args.out_dir, f"{args.prefix}_{i}.png")
            if url:
                download(url, out)
            elif b64:
                with open(out, "wb") as fh:
                    fh.write(base64.b64decode(b64))
            saved.append(out)
        else:
            saved.append(url or f"<b64 {len(b64)} chars>")
    for s in saved:
        print(s)


# ── Seedance 生视频（异步）─────────────────────────────────────
def cmd_video(args):
    content = [{"type": "text", "text": args.prompt}]
    if args.image:
        content.insert(0, {"type": "image_url", "image_url": {"url": resolve_image_source(args.image)}})
    body = {
        "model": os.environ.get("SEEDANCE_MODEL", DEFAULT_SEEDANCE_MODEL),
        "content": content,
    }
    if args.resolution:
        body["resolution"] = args.resolution
    if args.duration:
        body["duration"] = args.duration
    data = ark_request("/contents/generations/tasks", body, timeout=120)
    task_id = data.get("id") or data.get("task_id")
    if not task_id:
        raise SystemExit(f"未返回 task_id: {json.dumps(data, ensure_ascii=False)[:500]}")
    print(f"task_id: {task_id}")
    if args.wait:
        _poll_until_done(task_id, args.poll_interval, args.timeout)


def _poll_until_done(task_id: str, interval: int, timeout: int):
    deadline = time.time() + timeout
    while time.time() < deadline:
        data = ark_request(f"/contents/generations/tasks/{task_id}", timeout=60)
        status = data.get("status") or data.get("state") or "unknown"
        print(f"status: {status}")
        if status == "succeeded":
            url = (data.get("video_url") or data.get("url")
                   or data.get("content", {}).get("video_url") or "")
            if url:
                print(url)
            return
        if status in ("failed", "error"):
            raise SystemExit(f"生成失败: {data.get('error') or data.get('message') or '未知错误'}")
        time.sleep(interval)
    raise SystemExit("等待超时，请稍后用查询命令重试。")


def main():
    p = argparse.ArgumentParser(description="火山方舟 豆包/Seedream/Seedance 命令行")
    sub = p.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser("vision", help="视觉理解（读图）")
    v.add_argument("image", help="本地图片路径或 URL")
    v.add_argument("prompt", help="对图片的指令，越具体越好")
    v.add_argument("--detail", default="auto", choices=["auto", "low", "high"])
    v.add_argument("--max-tokens", type=int, default=4096)
    v.add_argument("--temperature", type=float, default=1.0)
    v.add_argument("--out", help="把结果写入文件")
    v.set_defaults(func=cmd_vision)

    g = sub.add_parser("generate", help="Seedream 生图（同步）")
    g.add_argument("prompt", help="图片描述（中英文皆可）")
    g.add_argument("--image", help="参考图（本地路径/URL），传了即图生图")
    g.add_argument("--size", help="分辨率，如 2048x2048、1920x1080")
    g.add_argument("--ratio", choices=["1:1", "3:4", "4:3", "16:9", "9:16", "2:3", "3:2", "21:9"],
                   help="宽高比（设置 size 时忽略）")
    g.add_argument("--n", type=int, default=1)
    g.add_argument("--out-dir", help="保存目录（缺省只打印 URL）")
    g.add_argument("--prefix", default="gen", help="输出文件名前缀")
    g.set_defaults(func=cmd_generate)

    vid = sub.add_parser("video", help="Seedance 生视频（异步）")
    vid.add_argument("prompt", help="视频描述")
    vid.add_argument("--image", help="参考图（图生视频）")
    vid.add_argument("--resolution", help="如 720p、1080p")
    vid.add_argument("--duration", type=int, help="时长秒数（4~30）")
    vid.add_argument("--wait", action="store_true", help="阻塞轮询直到完成")
    vid.add_argument("--poll-interval", type=int, default=10)
    vid.add_argument("--timeout", type=int, default=600)
    vid.set_defaults(func=cmd_video)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
