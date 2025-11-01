# image_embed_dedup.py
from __future__ import annotations
import os
import base64
import logging
from typing import List, Optional, TypedDict, Tuple
import aiohttp
import asyncio

from PIL import Image
import io

from tool.cosin import cosine_similarity
from tool.embedding import get_embeddings
from utils.token_tracker import TokenTracker   # 你自己实现的 Token 计数器

log_info = logging.getLogger("img").info
log_warning = logging.getLogger("img").warning
log_error = logging.getLogger("img").error
log_debug = logging.getLogger("img").debug

# ------------------ 类型定义 ------------------
class ImageObject(TypedDict):
    url: str
    embedding: List[List[float]]   # 与 TS 侧保持一致，外层 List 是因为可能批处理


class ImageReference(TypedDict):
    url: str
    # 你可以按需扩展 caption / alt 等字段


# ------------------ download ------------------
async def download_file(uri: str) -> Tuple[bytes, str]:
    async with aiohttp.ClientSession() as session:
        async with session.get(uri) as resp:
            if resp.status != 200 or resp.content_type is None:
                raise ValueError(f"Unexpected response {resp.status}")
            content_length = int(resp.headers.get("content-length", 0))
            if content_length > 100 * 1024 * 1024:          # 100 MB
                raise ValueError("File too large")
            if not resp.content_type.startswith("image/"):
                raise ValueError(f"Invalid content-type {resp.content_type}")
            buff = await resp.read()
            return buff, resp.content_type


# ------------------ load image ------------------
async def load_image(inp: str | bytes) -> Tuple[bytes, str]:
    buff: bytes
    content_type: str = ""

    if isinstance(inp, str):
        if inp.startswith("data:"):
            header, data = inp.split(",", 1)
            content_type = header.split(";")[0].split(":")[1]
            if header.split(";")[1].startswith("base64"):
                buff = base64.b64decode(data)
            else:
                buff = bytes.fromhex(data)   # 极少用，兼容 TS
        elif inp.startswith("http"):
            if inp.endswith(".svg"):
                raise ValueError("Unsupported image type")
            buff, content_type = await download_file(inp)
        else:
            raise ValueError("Invalid input")
    else:
        buff = inp

    if len(buff) > 20 * 1024 * 1024:        # 20 MB
        raise ValueError("Image too large")
    return buff, content_type


# ------------------ resize ------------------
# 使用 sharp-pyio（libvips 绑定）;  如想用 PIL，见下方注释版
# import pyvips
#
#
# def fit_image_to_square_box(image_buffer: bytes,
#                               content_type: str,
#                               size: int = 1024) -> str:
#     try:
#         img = pyvips.Image.new_from_buffer(image_buffer, "")
#     except Exception as e:
#         raise ValueError("Invalid image buffer") from e
#
#     width = img.width
#     height = img.height
#     if width < 256 or height < 256:
#         raise ValueError("Image must be at least 256x256")
#
#     image_type = content_type.split("/")[-1]
#     if image_type == "jpeg":
#         image_type = "jpg"
#
#     # 计算缩放
#     if width > size or height > size:
#         scale = min(size / width, size / height)
#         width = int(width * scale)
#         height = int(height * scale)
#         img = img.resize(scale)
#
#     out_buffer = img.write_to_buffer(f".{image_type}")
#     return base64.b64encode(out_buffer).decode()


# -------------- PIL 版（无需 libvips）--------------
def fit_image_to_square_box(image_buffer: bytes, content_type: str, size: int = 1024) -> str:
    with Image.open(io.BytesIO(image_buffer)) as im:
        im = im.convert("RGB") if im.mode not in ("RGB", "L") else im
        w, h = im.size
        if w < 256 or h < 256:
            raise ValueError("Image must be at least 256x256")
        if w > size or h > size:
            scale = min(size/w, size/h)
            w, h = int(w*scale), int(h*scale)
            im = im.resize((w, h), Image.LANCZOS)
        out = io.BytesIO()
        im.save(out, format=content_type.split('/')[-1].upper())
        return base64.b64encode(out.getvalue()).decode()


# ------------------ 生成 embedding ------------------
async def process_image(url: str, tracker: TokenTracker) -> Optional[ImageObject]:
    try:
        buff, content_type = await load_image(url)
        base64_data = fit_image_to_square_box(buff, content_type, 256)

        # 调用统一 embedding 工具（内部会统计 token）
        emb_resp = await get_embeddings([{"image": base64_data}],
                                        tracker,
                                        dimensions=512,
                                        model="jina-clip-v2")
        embedding: List[List[float]] = emb_resp["embeddings"]
        return ImageObject(url=url, embedding=embedding)

    except Exception as e:
        log_error(f"process_image error: {e}")
        return None


# ------------------ 去重 ------------------
def dedup_images_with_embeddings(
        new_images: List[ImageObject],
        existing_images: List[ImageObject],
        similarity_threshold: float = 0.86) -> List[ImageObject]:
    try:
        if not new_images:
            log_warning("No new images provided for deduplication")
            return []

        if len(new_images) == 1 and not existing_images:
            return new_images

        unique: List[ImageObject] = []
        used_idx = set()

        for i, img_new in enumerate(new_images):
            is_unique = True

            # 与库中已有图片比对
            for ex in existing_images:
                sim = cosine_similarity(img_new["embedding"][0], ex["embedding"][0])
                if sim >= similarity_threshold:
                    is_unique = False
                    break

            # 与本轮已接受的图片比对
            if is_unique:
                for j in used_idx:
                    sim = cosine_similarity(img_new["embedding"][0],
                                            new_images[j]["embedding"][0])
                    if sim >= similarity_threshold:
                        is_unique = False
                        break

            if is_unique:
                unique.append(img_new)
                used_idx.add(i)

        return unique
    except Exception as e:
        log_error(f"dedup error: {e}")
        # 降级：全部返回
        return new_images


# ------------------ 过滤引用 ------------------
def filter_images(image_references: List[ImageReference],
                  deduped_images: List[ImageObject]) -> List[ImageReference]:
    if not image_references:
        log_info("No image references provided for filtering")
        return []
    if not deduped_images:
        log_info("No deduplicated images provided for filtering")
        return image_references

    url_map = {ref["url"]: ref for ref in image_references if ref.get("url")}
    filtered = [url_map[img["url"]]
                for img in deduped_images
                if img["url"] in url_map]
    return filtered


# ------------------ 简单自测 ------------------
async def _test():
    tracker = TokenTracker()
    urls = [
        "https://picsum.photos/seed/a/300/300.jpg",
        "https://picsum.photos/seed/b/600/600.jpg",
        "https://picsum.photos/seed/a/400/400.jpg"  # 与第一张相似
    ]
    tasks = [process_image(u, tracker) for u in urls]
    imgs: List[ImageObject] = [x for x in await asyncio.gather(*tasks) if x]

    unique = dedup_images_with_embeddings(imgs, [], 0.86)
    print("unique images:", [u["url"] for u in unique])

