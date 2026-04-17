from verl.utils.reward_score.math_reward import is_equiv, strip_string
import os
import re
import math
import random
import os
import regex as re
from collections import defaultdict
from math_verify import parse, verify
from mathruler.grader import extract_boxed_content
import re
import time
import random
from zai import ZhipuAiClient
from openai import OpenAI
import os
from PIL import Image
import imagehash
from typing import List
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import numpy as np

def get_bleu_score(ref, hyp):
    smoother = smoother = SmoothingFunction().method1
    ref = [ref.split()]
    hyp = hyp.split()
    return sentence_bleu(ref, hyp, smoothing_function=smoother)

# def has_visually_same_images(paths: List[str]) -> bool:
#     """
#     判定是否存在视觉上相似的图片（即使尺寸或格式不同）。
#     """
#     seen_hashes = set()
    
#     for path in paths:
#         try:
#             with Image.open(path) as img:
#                 # 使用 pHash (感知哈希)，对缩放和色彩变化不敏感
#                 v_hash = imagehash.phash(img)
                
#                 if v_hash in seen_hashes:
#                     return True
#                 seen_hashes.add(v_hash)
#         except Exception as e:
#             print(f"无法处理图片 {path}: {e}")
            
#     return False

def has_visually_same_images(paths: List[str], similarity_threshold: float = 0.95) -> bool:
    """
    判定是否存在视觉相似的图片。
    策略：先对比尺寸，若长宽相似度均 >= 95%，才计算感知哈希进行深度对比。
    """
    # 存储已处理图片的信息：(width, height, path)
    processed_images: List[Tuple[int, int, str]] = []
    # 缓存已计算的哈希值，避免重复计算耗时操作
    hash_cache: Dict[str, imagehash.ImageHash] = {}

    def get_phash(image_path: str):
        if image_path not in hash_cache:
            with Image.open(image_path) as img:
                hash_cache[image_path] = imagehash.phash(img)
        return hash_cache[image_path]

    for path in paths:
        try:
            # 1. 快速读取头信息（不解码整张图）
            with Image.open(path) as img:
                curr_w, curr_h = img.size
            
            # 2. 与之前记录的图片进行尺寸比对
            for prev_w, prev_h, prev_path in processed_images:
                # 计算宽度和高度的相似度
                w_ratio = min(curr_w, prev_w) / max(curr_w, prev_w)
                h_ratio = min(curr_h, prev_h) / max(curr_h, prev_h)

                # 3. 只有尺寸足够接近（默认 > 95%），才进行哈希对比
                if w_ratio >= similarity_threshold and h_ratio >= similarity_threshold:
                    if get_phash(path) == get_phash(prev_path):
                        return True
            
            # 记录当前图片信息
            processed_images.append((curr_w, curr_h, path))
            
        except Exception as e:
            print(f"处理失败 {path}: {e}")
            continue

    return False

def normalize_answer(answer):
    if answer is None: return answer
    if 'dfrac' in answer: answer = answer.replace("dfrac", "frac")
    if '%' in answer: answer = answer.replace(r'\%',"").replace('%',"")
    if 'text' in answer: answer = answer.replace("\\text","")
    if "\\varnothing" in answer: answer = answer.replace("\\varnothing","\\emptyset")
    if "minutes" in answer: answer = answer.replace("minutes","")
    if "cm" in answer: answer = answer.replace("cm","")
    if "^\\circ" in answer: answer = answer.replace("^\\circ","")
    if "a.m." in answer: answer = answer.replace("a.m.","")
    return answer 

# client = ZhipuAiClient(api_key=os.environ.get("ZHIPU_KEY", ""))
client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "glm-4.6")
def llm_judge_zhipu(
    solution_str,
    ground_truth,
    question: str,
    model: str = JUDGE_MODEL,
    clip_n: int = 100,
    max_retries: int = 3,          # ✅ 最大重试次数
    base_delay: float = 1.0,       # ✅ 初始等待秒数
) -> float:

    # -------- 0) 安全转字符串 --------
    solution_str = "" if solution_str is None else str(solution_str)
    ground_truth = "" if ground_truth is None else str(ground_truth)

    # -------- 1) \boxed 抽取 --------
    sol = extract_boxed_content(solution_str) if r"\boxed" in solution_str else solution_str
    gt  = extract_boxed_content(ground_truth) if r"\boxed" in ground_truth else ground_truth

    # -------- 2) 轻量归一化 --------
    def _norm(s: str) -> str:
        s = s.strip()
        s = re.sub(r"\s+", " ", s)
        return s

    sol_n = _norm(sol)
    gt_n  = _norm(gt)

    if sol_n == gt_n and sol_n != "":
        return 1.0

    # -------- 3) 裁剪 --------
    def _clip(x: str, n: int) -> str:
        return x if len(x) <= n else (" ...[truncated]" + x[-n:])

    q_c  = _clip(question or "", clip_n)
    gt_c = _clip(gt_n, clip_n)
    sol_c = _clip(sol_n, clip_n)

    # -------- 4) Prompt --------
    system_prompt = "You are an expert evaluator."

    user_prompt = (
        "I will provide a question, a standard answer, and a model's answer."
        "Judge whether the model's answer is essentially correct. "
        """Reply with a single character only: "Y" or "N". Do not output any explanation or other text.\n"""
        "Ignore differences in formatting or wording and focus on whether the key answer is correct. Consider the model's answer is correct if: Different phrasings of the same answer (e.g., 5 people vs five people); Slight variations in description that refer to the same entity Minor differences in precision that don't change the core answer (such as two colors that are similar: e.g., brown and dark brown, white and off-white)\n"
        "**Example 1:**\n"
        "[Question]: What is the number on the page?\n"
        "[Standard Answer]: 67,200\n"
        "[Model's Answer]: The number is 67200.\n"
        "[Your Judgement]: Y\n"
        "**Example 2:**\n"
        "[Question]: What is the number on the page?\n"
        "[Standard Answer]: 67,200\n"
        "[Model's Answer]: 210\n"
        "[Your Judgement]: N\n"
        "**Task:**\n"
        f"[Question]: {question}\n"
        f"[Standard Answer]: {ground_truth}\n"
        f"[Model's Answer]: {solution_str}\n"
        "[Your Judgement]:"
    )

    # -------- 5) 重试机制 --------
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                # thinking={"type": "disabled"},
                temperature=0.0,
                timeout=30,  # ✅ 30秒超时保护
            )

            out = (resp.choices[0].message.content or "").strip()

            m = re.search(r"^(Y|N)$", out, re.IGNORECASE)
            if not m:
                return 0.0

            score = 1.0 if m.group(1).upper() == "Y" else 0.0
            print(f"LLM judge result: {score}, gt: {gt_n}, sol: {sol_n}, question: {question}")
            return score

        except Exception as e:
            print(f"LLM judge error on attempt {attempt}: {e}")
            # 最后一次失败直接返回
            if attempt == max_retries - 1:
                return 0.0
            sleep_time = base_delay * (attempt + 1)  # ✅ 线性退避
            sleep_time += random.uniform(0, 0.5)
            time.sleep(sleep_time)
    return 0.0


def acc_score(solution_str, ground_truth, question=""):
    if isinstance(ground_truth, list):
        return max([acc_score(solution_str, gt, question) for gt in ground_truth])

    if len(solution_str) > 1000:
        solution_str = solution_str[-1000:]
    if "<answer>" in solution_str and "</answer>" in solution_str:
        try:
            solution_str = solution_str.split("<answer>", 1)[1].split("</answer>", 1)[0]
        except:
            return 0.0
    elif "\\boxed" in solution_str:
        solution_str = extract_boxed_content(solution_str)
    
    if "\\boxed" in ground_truth:
        gt = extract_boxed_content(ground_truth)
    elif "<answer>" in ground_truth and "</answer>" in ground_truth:
        gt = ground_truth.split("<answer>", 1)[1].split("</answer>", 1)[0]
    else:
        gt = ground_truth
        
    solution_str = normalize_answer(solution_str)
    gt = normalize_answer(gt)
        
    # 选择或排序题特殊处理
    if gt == "A" or gt == "B" or gt == "C" or gt == "D" or ("A" in gt and "B" in gt and "C" in gt and "D" in gt):
        solution_str = my_clean_answer(solution_str)
        if gt == solution_str:
            return 1.0
        else:
            return 0.0
    
    # ground_truth = f"\\boxed{{{gt}}}"
    # solution_str = f"\\boxed{{{solution_str}}}"
    # verify_result = verify(parse(solution_str, parsing_timeout=None), parse(ground_truth, parsing_timeout=None), timeout_seconds=None)
    # if not verify_result:
    #     verify_result = verify(parse(solution_str.lower(), parsing_timeout=None), parse(ground_truth.lower(), parsing_timeout=None), timeout_seconds=None)
    # if verify_result:
    #     return 1.0
    if os.environ.get("USE_LLM_JUDGE", "0") == "1":
        return llm_judge_zhipu(solution_str, ground_truth, question)
    return 0.0
    

def acc_verifier(student_answer: str, gt: str, problem: str = "") -> float:
    # return 0.0 ~ 1.0
    student_answer = extract_boxed_content(student_answer) if "boxed" in student_answer else student_answer
    # print(f"Student answer: {student_answer}, GT: {gt}")
    if 'A' in gt and 'B' in gt and 'C' in gt and 'D' in gt and len(gt) == 4: # 排序题
        res = ''.join([c for c in student_answer if c in 'ABCD'])
        # if "A" not in res or "B" not in res or "C" not in res or "D" not in res: # 不重复
        #     return 0.0
        if res.count("A") > 2 or res.count("B") > 2 or res.count("C") > 2 or res.count("D") > 2: # 至多重复1次
            return 0.0
        if len(res) != 4:
            return 0.0
        # 计算和gt匹配的比例，按照顺序
        matched = sum(1 for i in range(4) if res[i] == gt[i])
        return matched / 4.0

    ret_score = 0.0
    student_answer = my_clean_answer(student_answer)
    gt = my_clean_answer(gt)
    if student_answer == gt:
        ret_score = 1.0
    elif 'A' in problem and 'B' in problem and 'C' in problem and 'D' in problem: # 选择题
        # 从左到右提取字符串中出现的第一个 A/B/C/D（不能忽略大小写）
        def _first_abcd(s: str):
            m = re.search(r"[ABCD]", s)
            return m.group(0).upper() if m else None

        first_student = _first_abcd(student_answer)
        first_gt = _first_abcd(gt)

        if first_student is not None and first_gt is not None:
            ret_score = 1.0 if first_student == first_gt else 0.0
        else:
            ret_score = 0.0
    else:
        try:
            student_answer_float = float(student_answer)
            gt_float = float(gt)
            ret_score = 1.0 if math.isclose(student_answer_float, gt_float, rel_tol=1e-5) else 0.0
        except:
            ret_score = 1.0 if is_equiv(student_answer, gt) else 0.0

    if random.random() < 0.01 or os.environ.get("DEBUG_MODE") == "true": 
        print(f'[debug] problem: {problem} student_answer: {student_answer} gt: {gt} score: {ret_score}')

    return ret_score

def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None
    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]

    return retval

def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[: len(left)] == left
        assert s[-1] == "}"
        return s[len(left) : -1]
    except Exception:
        return None

def remove_text_answer(solution: str) -> str:
    """
    remove \\text{} from the answer string
    Example :
    \\text{42} apples -> 42 apples
    how about \\text{the answer is 100} -> the answer is 100
    """
    left = "\\text{"
    try:
        assert solution.startswith(left)

        i = len(left)
        num_left_braces_open = 1
        right_brace_idx = None

        while i < len(solution):
            if solution[i] == "{":
                num_left_braces_open += 1
            elif solution[i] == "}":
                num_left_braces_open -= 1
                if num_left_braces_open == 0:
                    right_brace_idx = i
                    break
            i += 1

        assert right_brace_idx is not None
        inner = solution[len(left) : right_brace_idx]
        rest = solution[right_brace_idx + 1 :]
        return inner + rest
    except Exception:
        return None

def extract_boxed_answer(solution: str) -> str:
    """Extract the answer from inside a LaTeX \\boxed{} command"""
    solution = last_boxed_only_string(solution)
    if solution is None:
        return None
    solution = remove_boxed(solution)
    tmp = remove_text_answer(solution)
    if tmp is not None:
        solution = tmp
    return solution

def my_clean_answer(answer: str) -> str:
    """
    如果answer中有A/B/C/D选项，则提取第一个出现的选项字母作为答案返回
    如果answer为数字，转换为浮点数后返回
    如果answer中包含yes/no，返回"Yes"/"No"
    否则返回原始字符串
    """
    answer = str(answer)
    m = re.search(r"[ABCD]", answer)
    if m:
        return m.group(0).upper()
    answer = strip_string(answer)
    try:
        answer_float = float(answer)
        return str(answer_float)
    except:
        if "yes" in answer.lower():
            return "Yes"
        elif "no" in answer.lower():
            return "No"
        return answer



from PIL import Image
MAX_PIXELS = int(os.environ.get("MAX_PIXELS", 16*1024*32*32))
def smart_resize_image(image: Image.Image, max_pixels: int = MAX_PIXELS) -> Image.Image:
    original_width, original_height = image.size
    current_pixels = original_width * original_height

    if current_pixels <= max_pixels:
        return image

    # Calculate new dimensions while maintaining aspect ratio
    scale_factor = (max_pixels / current_pixels) ** 0.5
    new_width = int(original_width * scale_factor)
    new_height = int(original_height * scale_factor)

    # Ensure at least 1 pixel in each dimension
    new_width = max(1, new_width)
    new_height = max(1, new_height)
    print(f"[debug] Resizing image from ({original_width}, {original_height}) to ({new_width}, {new_height}) to meet pixel limit.")

    return image.resize((new_width, new_height), Image.LANCZOS)


import os
import random
import re
from PIL import Image
import datasets
from io import BytesIO
import numpy as np
import math

from qwen_vl_utils import process_vision_info, smart_resize
MAX_PIXELS = int(os.environ.get("MAX_PIXELS", 2000000))  # Qwen max pixels
MIN_PIXELS = int(os.environ.get("MIN_PIXELS", 20000))  # Qwen min pixels
IMAGE_PATCH_SIZE = int(os.environ.get("IMAGE_PATCH_SIZE", 14))
def my_process_image(image: dict, max_pixels: int = MAX_PIXELS, min_pixels: int = MIN_PIXELS, return_original_image=False, factor=IMAGE_PATCH_SIZE*2) -> Image.Image:
    if isinstance(image, dict):
        image = Image.open(BytesIO(image['bytes']))
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image).convert('RGB')
    elif isinstance(image, Image.Image):
        image = image.convert('RGB')
    elif isinstance(image, str):
        assert os.path.exists(image), f"Image path {image} not exists."
        try:
            image = Image.open(image).convert('RGB')
        except Exception as exn:
            print(f"Failed to open image {image}. Exception:", exn)
            raise exn

    if return_original_image:
        return image

    width, height = image.size
    resized_height, resized_width = smart_resize(
            height,
            width,
            factor=factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,  
        )
    resized_height = max(factor, resized_height)
    resized_width = max(factor, resized_width)
    image = image.resize((resized_width, resized_height), resample=Image.Resampling.LANCZOS)

    assert image.width >= factor and image.height >= factor, f"Qwen3VL image size should be larger than {factor} * {factor}."

    if image.mode != 'RGB':
        image = image.convert('RGB')

    return image


from PIL import Image
import numpy as np
from typing import List, Tuple, Optional, Dict
import hashlib

def images_equal_strict(path1: str, path2: str) -> bool:
    img1 = Image.open(path1).convert("RGBA")
    img2 = Image.open(path2).convert("RGBA")

    if img1.size != img2.size:
        return False

    arr1 = np.asarray(img1)
    arr2 = np.asarray(img2)
    return np.array_equal(arr1, arr2)

def _pixel_fingerprint(path: str) -> Tuple[Tuple[int, int], str]:
    """
    返回 (size, digest)，digest 是像素字节的 sha1（很快，用于分桶）。
    """
    img = Image.open(path).convert("RGBA")
    data = img.tobytes()  # RGBA 原始像素
    digest = hashlib.sha1(data).hexdigest()
    return img.size, digest

def has_same_images(paths: List[str]) -> bool:
    """
    严格判定：列表中是否存在两张像素完全相同的图片。
    返回 True/False。
    """
    buckets: Dict[Tuple[Tuple[int, int], str], List[str]] = {}

    for p in paths:
        try:
            key = _pixel_fingerprint(p)  # (size, sha1)
        except Exception:
            continue  # 坏图/打不开就跳过

        # 如果桶里已有候选，做严格确认
        if key in buckets:
            for prev in buckets[key]:
                if images_equal_strict(prev, p):
                    return True
            buckets[key].append(p)
        else:
            buckets[key] = [p]

    return False


import re
def has_image_processing_functions(code_str: str, return_detail: bool = False) -> bool | dict:
    """
    检测代码字符串中是否使用了亮度、对比度、饱和度、锐度、滤波相关函数
    核心逻辑：只要使用了任意一种，就返回 True
    
    Args:
        code_str: 待检测的代码字符串
        return_detail: 是否返回详细检测结果（默认False，仅返回True/False）
    
    Returns:
        若 return_detail=False（默认）：bool，True=使用了任意一种相关函数，False=未使用
        若 return_detail=True：dict，包含详细检测结果（同之前的结构）
    """
    # 定义各处理类型的匹配规则（覆盖主流图像处理库）
    match_rules = {
        "brightness": [
            r"ImageEnhance\.Brightness", r"cv2\.convertScaleAbs\(.*?beta=",
            r"exposure\.adjust_gamma", r"brightness", r"Brightness"
        ],
        "contrast": [
            r"ImageEnhance\.Contrast", r"cv2\.convertScaleAbs\(.*?alpha=",
            r"exposure\.equalize_hist", r"exposure\.rescale_intensity", r"contrast", r"Contrast"
        ],
        "saturation": [
            r"ImageEnhance\.Color", r"hsv.*?[:.]s[:,]?", r"HueSaturationValue",
            r"saturation", r"Saturation"
        ],
        "sharpness": [
            r"ImageEnhance\.Sharpness", r"ImageFilter\.SHARPEN", r"ImageFilter\.UnsharpMask",
            r"filter2D.*?kernel.*?锐化", r"unsharp_mask", r"sharpness", r"Sharpness"
        ],
        "filter": [
            r"ImageFilter\.", r"\.filter\(", r"cv2\.GaussianBlur", r"cv2\.medianBlur",
            r"cv2\.bilateralFilter", r"cv2\.filter2D", r"cv2\.Canny", r"cv2\.Sobel",
            r"filters\.gaussian", r"filters\.sobel", r"滤波", r"blur", r"edge", r"kernel"
        ]
    }
    
    # 遍历匹配规则，检测各类型
    for process_type, patterns in match_rules.items():
        for pattern in patterns:
            regex = re.compile(pattern, re.IGNORECASE)
            found = regex.findall(code_str)
            if found:
                return True
    
    return False


if __name__ == "__main__":
    # test acc_score
    # print(acc_score("""\\boxed{"BCAD"}""", "BCAD", "排序题"))
    # print(acc_score("\\boxed{ACDB}", "CADB", "排序题"))
    # print(acc_score("two people", "2", "How many people are there?"))
    # print(acc_score("\\boxed{1}", ["1"]))
    # print(acc_score("<answer>1</answer>", ["<answer>1</answer>", "3"]))
    # print(acc_score("<answer>I think is A.</answer>", ["A", "3"]))

    x = "The bottom part again has the figure in yellowish dress.\n\nSo we can match:\n- Image 1 matches the paddings in A and the color in B.\n- Image 2 matches the paddings and color in A.\n- Image 3 matches the paddings and color in B.\n- Image 4 matches the paddings and color in C.\n\nThus, the matching sequence is: B, A, B, C. \n\n\\boxed{BABC}"
    print(extract_boxed_answer(x))

    # test acc_verifier
    print(acc_verifier(x, "BADC", "排序题"))
    print(acc_verifier("""'BCAD'""", "BCAD", "排序题"))
    print(acc_verifier("""'CCAD'""", "BCAD", "排序题"))
    print(acc_verifier("""'CBAD'""", "BCAD", "排序题"))
    print(acc_verifier("""'BACD'""", "BCAD", "排序题"))
    print(acc_verifier("""\\boxed{\\text{BCDA}}""", "BCAD", "排序题"))
    print(acc_score("""\\boxed{"B"}""", "B", "选择题"))
    print(acc_score("\\boxed{ACDB}", "CADB", "排序题"))
    print(acc_score("two people", "2", "How many people are there?"))
    print(acc_score("\\boxed{1}", ["1"]))
    print(acc_score("<answer>1</answer>", ["<answer>1</answer>", "3"]))
    print(acc_score("<answer>B</answer>", ["A", "3"]))