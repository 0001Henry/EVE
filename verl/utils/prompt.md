# Task
Write a simple Python function named `edit_image` that edit the user-input image, and then design 4 different parameter sets. 

# Requirements
1. The `edit_image` function must accept a PIL Image object and specific parameters, returning a modified PIL Image object.
2. The Python code must include necessary imports, the `edit_image` function, and a list of dictionaries named `args_list`.
3. Use only one editing technique in the `edit_image` function and design 4 different sets of parameters for it.
4. Ensure that the 4 sets of parameters in `args_list` produce 4 visually distinct editing results.
5. If coordinates are used, you must use relative coordinates (range 0-1000) in `args_list`. Inside `edit_image`, retrieve the actual image dimensions and convert these relative values to absolute pixel coordinates.
6. The below provided examples are for reference only; design the logic and parameters based on the specific content of the user's image.

# Examples

## Example 1:
```python
from PIL import Image, ImageDraw

# Draw bounding boxes on 4 objects
def edit_image(img: Image.Image, bbox: tuple) -> Image.Image:
    real_width, real_height = img.size
    left = int(bbox[0] * real_width / 1000)
    upper = int(bbox[1] * real_height / 1000)
    right = int(bbox[2] * real_width / 1000)
    lower = int(bbox[3] * real_height / 1000)
    draw = ImageDraw.Draw(img)
    draw.rectangle([left, upper, right, lower], outline="red", width=5)
    return img

args_list = [
    {'bbox': [100, 100, 300, 300]},
    {'bbox': [400, 200, 600, 400]},
    {'bbox': [250, 150, 550, 350]},
    {'bbox': [50, 50, 250, 250]},
]
```

## Example 2:
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