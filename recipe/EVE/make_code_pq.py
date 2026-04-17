"""
Initial code priority queue generation for EVE Challenger-Solver framework.

This script generates initial code examples for the priority queue, including
various image editing functions with different parameter sets.
"""

import json

# Example code templates for initial priority queue
# These serve as seed examples for the Challenger model

code_str1 = """from PIL import Image, ImageDraw
def edit_image(img: Image.Image, bbox_2d: list) -> Image.Image:
    draw = ImageDraw.Draw(img)
    draw.rectangle(bbox_2d, outline="red", width=5)
    return img

args_list = [
    {"bbox_2d": [205, 220, 335, 422]},
    {"bbox_2d": [103, 94, 378, 510]},
    {"bbox_2d": [452, 610, 556, 850]},
    {"bbox_2d": [154, 750, 357, 958]},
]"""

code_str2 = """from PIL import Image

def edit_image(img: Image.Image, bbox_2d: list) -> Image.Image:
    return img.crop(bbox_2d)

args_list = [
    {"bbox_2d": [205, 220, 335, 422]},
    {"bbox_2d": [103, 94, 378, 510]},
    {"bbox_2d": [452, 610, 556, 850]},
    {"bbox_2d": [154, 750, 357, 958]},
]"""

code_str3 = """from PIL import Image

def edit_image(img: Image.Image, angle: float) -> Image.Image:
    return img.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)

args_list = [
    {'angle': 15},
    {'angle': 45},
    {'angle': 90},
    {'angle': 180},
]"""

code_str4 = """from PIL import Image

def edit_image(img: Image.Image, N: int, order: list) -> Image.Image:
    width, height = img.size
    block_w = width // N
    block_h = height // N
    adjusted_size = (N * block_w, N * block_h)
    img = img.resize(adjusted_size)
    out_img = Image.new("RGB", adjusted_size)
    for new_idx in range(N * N):
        new_row = new_idx // N
        new_col = new_idx % N
        orig_idx = order[new_idx]
        orig_row = orig_idx // N
        orig_col = orig_idx % N
        left = orig_col * block_w
        upper = orig_row * block_h
        right = left + block_w
        lower = upper + block_h
        block = img.crop((left, upper, right, lower))
        out_img.paste(block, (new_col * block_w, new_row * block_h))
    return out_img

args_list = [
    {'N': 2, 'order': [3, 0, 1, 2]},
    {'N': 2, 'order': [1, 0, 3, 2]},
    {'N': 2, 'order': [0, 3, 2, 1]},
    {'N': 2, 'order': [2, 1, 0, 3]},
]"""

code_list = [code_str1, code_str2, code_str3, code_str4]

# Save in format: {"code_str": "...", "diff_reward": 100.0}
p = "recipe/EVE/topk_codestrs_qwen2_5vl.jsonl"

with open(p, "w") as f:
    for i, code in enumerate(code_list):
        f.write(json.dumps({
            "code_str": code,
            "diff_reward": 100.0
        }, ensure_ascii=False) + "\n")
