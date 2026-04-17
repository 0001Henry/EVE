SYSTEM_PROMPT_1 = """# Task
Write a simple Python function named `edit_image` that edit the user's image, and then design 4 different parameter sets. 

# Requirements
1. The `edit_image` function must accept a PIL Image object and specific parameters, returning a modified PIL Image object.
2. The Python code must include necessary imports, the `edit_image` function, and a list of dictionaries named `args_list`.
3. Ensure that the 4 sets of parameters in `args_list` produce 4 visually distinct editing results.
4. No comments must be added to the `edit_image` function.
5. The examples below are for format reference only. The parameters must be designed according to the specific content of the user's image; do not copy them directly.

# Examples

## Example 1 (make a 2*2 jigsaw)
```python
from PIL import Image

def edit_image(img: Image.Image, N: int, order: list[int]) -> Image.Image:
    w, h = img.size
    bw, bh = w // N, h // N
    new_w, new_h = bw * N, bh * N
    src = img.resize((new_w, new_h))
    
    dst = Image.new(src.mode if src.mode != "P" else "RGB", (new_w, new_h))

    def box_of(idx: int) -> tuple[int, int, int, int]:
        r, c = divmod(idx, N)
        x0, y0 = c * bw, r * bh
        return (x0, y0, x0 + bw, y0 + bh)

    for new_idx, orig_idx in enumerate(order):
        block = src.crop(box_of(orig_idx))
        r, c = divmod(new_idx, N)
        dst.paste(block, (c * bw, r * bh))

    return dst.convert("RGB")

args_list = [
    {'N': 2, 'order': [3, 0, 1, 2]},
    {'N': 2, 'order': [1, 0, 3, 2]},
    {'N': 2, 'order': [0, 3, 2, 1]},
    {'N': 2, 'order': [2, 1, 0, 3]},
]
```

## Example 2 (crop four different regions)
```
from PIL import Image

def edit_image(img: Image.Image, bbox_2d: list) -> Image.Image:
    return img.crop(bbox_2d)

args_list = [
    {"bbox_2d": [205, 220, 335, 422]},
    {"bbox_2d": [103, 94, 378, 510]},
    {"bbox_2d": [452, 610, 556, 850]},
    {"bbox_2d": [154, 750, 357, 958]},
]
```
"""

SYSTEM_PROMPT_PQ_1 = """# Task
Write a simple Python function named `edit_image` that edit the user's image, and then design 4 different parameter sets. 

# Requirements
1. The `edit_image` function must accept a PIL Image object and specific parameters, returning a modified PIL Image object.
2. The Python code must include necessary imports, the `edit_image` function, and a list of dictionaries named `args_list`.
3. Ensure that the 4 sets of parameters in `args_list` produce 4 visually distinct editing results.
4. No comments must be added to the `edit_image` function.
5. The examples below are for format reference only. The parameters must be designed according to the specific content of the user's image; do not copy them directly.

# Examples

## Example 1
```python
{code_str1}
```

## Example 2
```python
{code_str2}
```
"""

SYSTEM_PROMPT_2 = """# Task
Write a simple Python function named `edit_image` that edit the user's image, and then design 4 different parameter sets. 

# Requirements
1. The `edit_image` function must accept a PIL Image object and specific parameters, returning a modified PIL Image object.
2. The Python code must include necessary imports, the `edit_image` function, and a list of dictionaries named `args_list`.
3. Ensure that the 4 sets of parameters in `args_list` produce 4 visually distinct editing results.
4. No comments must be added to the `edit_image` function.
5. If coordinates are used, you must use relative coordinates (range 0-1000) in `args_list`. Inside `edit_image`, retrieve the actual image dimensions and convert these relative values to absolute pixel coordinates.
6. The examples below are for format reference only. The parameters must be designed according to the specific content of the user's image; do not copy them directly.

# Examples

## Example (make a 2*2 jigsaw)
```python
from PIL import Image

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
]
```
"""

# ours 原始
SYSTEM_PROMPT_PQ_2 = """# Task
Write a simple Python function named `edit_image` that edit the user's image, and then design 4 different parameter sets. 

# Requirements
1. The `edit_image` function must accept a PIL Image object and specific parameters, returning a modified PIL Image object.
2. The Python code must include necessary imports, the `edit_image` function, and a list of dictionaries named `args_list`.
3. Ensure that the 4 sets of parameters in `args_list` produce 4 visually distinct editing results.
4. No comments must be added to the `edit_image` function.
5. If coordinates are used, you must use relative coordinates (range 0-1000) in `args_list`. Inside `edit_image`, retrieve the actual image dimensions and convert these relative values to absolute pixel coordinates.
6. The examples below are for format reference only. The parameters must be designed according to the specific content of the user's image; do not copy them directly.

# Examples

## Example 1
```python
{code_str1}
```

## Example 2
```python
{code_str2}
```
"""

# 0331
# SYSTEM_PROMPT_PQ = """# Task
# Write a simple Python function named `edit_image` that edit the user's image, and then design 4 different parameter sets. 

# # Requirements
# 1. The `edit_image` function must accept a PIL Image object and specific parameters, returning a modified PIL Image object.
# 2. The Python code must include necessary imports, the `edit_image` function, and a list of dictionaries named `args_list`.
# 3. Ensure that the 4 sets of parameters in `args_list` produce 4 visually distinct editing results.
# 4. No comments must be added to the `edit_image` function.
# 5. Using relative coordinates (the parameter value should be between 0 and 1), representing the ratio relative to the width and height of the image.
# 6. The examples below are for format reference only. The parameters must be designed according to the specific content of the user's image; do not copy them directly.

# # Examples

# ## Example 1
# ```python
# {code_str1}
# ```

# ## Example 2
# ```python
# {code_str2}
# ```
# """


import os
QWEN_MODEL_TYPE = os.environ.get("QWEN_MODEL_TYPE", "qwen2_5vl")
if QWEN_MODEL_TYPE == "qwen2_5vl": 
    SYSTEM_PROMPT = SYSTEM_PROMPT_1
    SYSTEM_PROMPT_PQ = SYSTEM_PROMPT_PQ_1
    USER_PROMPT = """Observe the given image, design the code with reference to the image size and image content. Output your reasoning process inside <think>...</think> tags, followed by the final Python code in ```python\n...\n```"""
    # S_prompt = """\nYou FIRST think about the reasoning process as an internal monologue and then provide the final answer. The final answer MUST BE put within \\boxed{}."""
    # S_prompt = """\nThink step by step and then give your final answer. The final answer MUST BE put within \\boxed{}."""
    S_prompt = """\nPlease put your final answer within \\boxed{}."""
else:
    SYSTEM_PROMPT = SYSTEM_PROMPT_2
    SYSTEM_PROMPT_PQ = SYSTEM_PROMPT_PQ_2      
    USER_PROMPT = """Observe the given image, design the code with reference to the image content. Output your reasoning process inside <thinking>...</thinking> tags, followed by the final Python code in ```python\n...\n```"""

    S_prompt = """\nPlease put your final answer within \\boxed{}."""


SYSTEM_PROMPT_old = """# Task
Write a simple Python function named `edit_image` that edit the user's image, and then design 4 different parameter sets. 

# Requirements
1. The `edit_image` function must accept a PIL Image object and specific parameters, returning a modified PIL Image object.
2. The Python code must include necessary imports, the `edit_image` function, and a list of dictionaries named `args_list`.
3. Possible editing methods include, but are not limited to: cropping objects, drawing bounding boxes, rotating images, creating jigsaw puzzles, and more. You can use your imagination and try other editing methods.
4. Ensure that the 4 sets of parameters in `args_list` produce 4 visually distinct editing results.
5. If coordinates are used, you must use relative coordinates (range 0-1000) in `args_list`. Inside `edit_image`, retrieve the actual image dimensions and convert these relative values to absolute pixel coordinates.
6. The below provided examples are for reference only; design the logic and parameters based on the specific content of the user's image.

# Examples

## Example 1:
```python
from PIL import Image

# rotate the image
def edit_image(img: Image.Image, angle: float) -> Image.Image:
    return img.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)

args_list = [
    {'angle': 15},
    {'angle': 45},
    {'angle': 90},
    {'angle': 180},
]
```

## Example 2:
```python
from PIL import Image

# Crop out 4 objects from the image
def edit_image(img: Image.Image, bbox: list) -> Image.Image:
    # Get the actual image size
    real_width, real_height = img.size
    # Convert relative coordinates to absolute coordinates
    bbox = [int(b * real_width / 1000) if i % 2 == 0 else int(b * real_height / 1000) for i, b in enumerate(bbox)]
    return img.crop(bbox)

# Relative coordinates in the range 0~1000
args_list = [
    {'bbox': [50, 50, 200, 200]},
    {'bbox': [0, 0, 150, 175]},
    {'bbox': [100, 100, 200, 250]},
    {'bbox': [750, 750, 900, 900]},
]
```

## Example 3:
```python
from PIL import Image

# make jigsaw puzzles
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
]
```
"""
