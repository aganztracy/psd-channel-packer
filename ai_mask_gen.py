"""
ai_mask_gen.py — 调用腾讯内部 Gemini API 生成 mask，写入 PSD 新图层
"""

import base64
import json
import time
import urllib.error
import urllib.request
from io import BytesIO
from pathlib import Path

from PIL import Image
import numpy as np

# ── API Key 混淆存储 ──
# 简单 XOR + base64，防止明文被直接看到
_XOR_KEY = 0x5A  # 混淆用

def _encode_key(raw_key: str) -> str:
    """编码 API Key（开发时用，把输出粘贴到 _ENCODED_KEY）"""
    xored = bytes([b ^ _XOR_KEY for b in raw_key.encode('utf-8')])
    return base64.b64encode(xored).decode('ascii')

def _decode_key(encoded: str) -> str:
    """运行时解码 API Key"""
    xored = base64.b64decode(encoded)
    return bytes([b ^ _XOR_KEY for b in xored]).decode('utf-8')

# 编码后的 key（通过 _encode_key() 生成）
_ENCODED_KEY = "bjAvGRA3KjwDKQprKhUDOBw0Fxg7MDAOLyAcbyISPQMQPT0uKjFtLg=="

# ── API 配置 ──
API_BASE = "http://api.timiai.woa.com"
I2I_PATH = "/ai_api_manage/llmproxy/chat/completions"
MODEL = "gemini-3-pro-image-preview"
TIMEOUT = 300


def get_api_key() -> str:
    """获取解码后的 API Key"""
    return _decode_key(_ENCODED_KEY)


def _post_api(url: str, payload: dict, api_key: str) -> dict:
    """POST JSON to API"""
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Authorization": api_key,
        "Accept": "application/json",
        "User-Agent": "PSDChannelPacker/1.0",
    }

    req = urllib.request.Request(url, method="POST", headers=headers, data=body)
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        resp_body = resp.read()
        if not resp_body:
            raise RuntimeError(f"API 返回空响应 (HTTP {resp.status})")
        return json.loads(resp_body)


def _closest_aspect_ratio(w: int, h: int) -> str:
    """Map image dimensions to nearest supported aspect ratio."""
    ratio = w / h if h else 1.0
    options = ["1:1", "4:3", "16:9", "21:9", "9:16", "4:5"]
    best = "1:1"
    best_diff = float("inf")
    for ar in options:
        a, b = ar.split(":")
        target = int(a) / int(b)
        diff = abs(ratio - target)
        if diff < best_diff:
            best_diff = diff
            best = ar
    return best


def generate_mask(
    input_image_path: str,
    prompt: str,
    progress_cb=None,
) -> Image.Image:
    """
    调用 AI 生成 mask。
    
    Args:
        input_image_path: 输入图片路径
        prompt: 描述生成什么 mask
        progress_cb: 进度回调
        
    Returns:
        PIL Image (RGBA)
    """
    def log(msg):
        if progress_cb:
            progress_cb(msg)

    api_key = get_api_key()
    url = f"{API_BASE}{I2I_PATH}"

    # 读取并压缩输入图
    log("读取输入图片...")
    img = Image.open(input_image_path)
    w, h = img.size
    max_dim = max(w, h)
    if max_dim > 2048:
        scale = 2048 / max_dim
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    buf = BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=85)
    b64_img = base64.b64encode(buf.getvalue()).decode("ascii")
    image_data_url = f"data:image/jpeg;base64,{b64_img}"

    aspect_ratio = _closest_aspect_ratio(w, h)
    log(f"调用 AI API... (模型: {MODEL}, 比例: {aspect_ratio})")

    payload = {
        "model": MODEL,
        "stream": False,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_data_url}},
            ],
        }],
        "image_config": {
            "aspect_ratio": aspect_ratio,
        },
        "response_modalities": ["IMAGE", "TEXT"],
    }

    result = _post_api(url, payload, api_key)

    # 检查错误
    if result.get("error"):
        raise RuntimeError(f"API 错误: {result['error']}")

    # 提取返回图片
    images = None
    try:
        images = result["choices"][0]["message"]["images"]
    except (KeyError, IndexError, TypeError):
        pass
    if not images:
        raise RuntimeError(f"API 未返回图片: {json.dumps(result, ensure_ascii=False)[:300]}")

    img_url = images[0].get("image_url", {}).get("url", "")
    if not img_url:
        raise RuntimeError("返回图片无 URL")

    # 解码图片
    if img_url.startswith("data:"):
        header, data = img_url.split(",", 1)
        img_bytes = base64.b64decode(data)
    else:
        with urllib.request.urlopen(img_url, timeout=60) as resp:
            img_bytes = resp.read()

    log("AI 生成完成!")
    result_img = Image.open(BytesIO(img_bytes)).convert("RGBA")
    return result_img


def generate_mask_and_create_psd(
    input_image_path: str,
    prompt: str,
    output_psd_path: str = None,
    layer_name: str = "AI_mask",
    progress_cb=None,
) -> str:
    """
    生成 mask，创建 PSD（原图层 + mask 图层）。
    
    PSD 结构：
      - 原图（底层）
      - AI_mask（顶层）
    
    Returns:
        输出 PSD 路径
    """
    from psd_tools import PSDImage

    def log(msg):
        if progress_cb:
            progress_cb(msg)

    # 生成 mask
    mask_img = generate_mask(input_image_path, prompt, progress_cb)

    # 读取原图
    log("读取原图...")
    orig_img = Image.open(input_image_path).convert('RGBA')
    orig_w, orig_h = orig_img.size

    # resize mask 到原图尺寸
    if mask_img.size != (orig_w, orig_h):
        log(f"Resize mask: {mask_img.size} → ({orig_w}, {orig_h})")
        mask_img = mask_img.resize((orig_w, orig_h), Image.LANCZOS)

    # 创建 PSD
    log("创建 PSD...")
    psd = PSDImage.new(mode='RGBA', size=(orig_w, orig_h), depth=8)

    # 先添加原图层（底层）— create_pixel_layer 放到最顶部，所以先创建的在底部
    psd.create_pixel_layer(orig_img, name="原图", top=0, left=0, opacity=255)
    # 再添加 mask 层（顶层）
    psd.create_pixel_layer(mask_img, name=layer_name, top=0, left=0, opacity=255)

    # 保存
    if not output_psd_path:
        stem = Path(input_image_path).stem
        parent = Path(input_image_path).parent
        output_psd_path = str(parent / f"{stem}_DO.psd")

    Path(output_psd_path).parent.mkdir(parents=True, exist_ok=True)
    psd.save(output_psd_path)
    log(f"PSD 保存: {output_psd_path}")

    return output_psd_path


# ── 初始化编码 key ──
# 运行一次生成编码值
if __name__ == "__main__":
    raw = "4juCJmpfYsP1pOYbFnMBajjTuzF5xHgYJggtpk7t"
    encoded = _encode_key(raw)
    print(f"Encoded key: {encoded}")
    # 验证
    decoded = _decode_key(encoded)
    assert decoded == raw, "Decode mismatch!"
    print(f"Verified OK: {decoded}")
