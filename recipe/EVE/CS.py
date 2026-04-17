"""
EVE: Executable Visual transformation based self-Evolution

Challenger-Solver dual-policy framework for MLLM self-evolution via code execution.
"""

from verl.utils.cs_sandbox import execute_code_in_sandbox
import json
import re
import logging
import os
import random
import time
import torch
import shutil
from io import BytesIO

import requests
from openai import OpenAI, RateLimitError, AsyncOpenAI
from PIL import Image
import datasets
import numpy as np
from qwen_vl_utils import process_vision_info, smart_resize

import verl.utils.torch_functional as verl_F
from verl.utils.dataset.rl_dataset import RLHFDataset
from verl.utils.model import compute_position_id_with_mask
from verl.utils.cs_prompts import SYSTEM_PROMPT, USER_PROMPT, SYSTEM_PROMPT_PQ, S_prompt
from verl.utils.cs_tools import acc_verifier, acc_score, my_process_image, has_visually_same_images, has_image_processing_functions, get_bleu_score
from mathruler.grader import extract_boxed_content

logger = logging.getLogger(__name__)

ROLLOUT_DATA_DIR = os.environ.get("ROLLOUT_DATA_DIR", "./output/EVE/rollout_data")
TEMP_PROCESSED_DIR = os.path.join(ROLLOUT_DATA_DIR, "temp_processed")
os.makedirs(TEMP_PROCESSED_DIR, exist_ok=True)

MAX_PIXELS = int(os.environ.get("MAX_PIXELS", 2000000))
MIN_PIXELS = int(os.environ.get("MIN_PIXELS", 64 * 64))
VLLM_MAX_LEN_S = int(os.environ.get("VLLM_MAX_LEN_S", 4096))
CSV_ROLE = os.environ.get("CSV_ROLE")
SAMPLE_NUM = int(os.environ.get("SAMPLE_NUM", 6))
SEED = int(os.environ.get("SEED", 42))
random.seed(SEED)

client_list = []
if CSV_ROLE == "C":
    VLLM_URL_S = os.environ.get("VLLM_URL_S")
    VLLM_MODEL_NAME_S = os.environ.get("VLLM_MODEL_NAME_S")
    for i in range(4):
        client_list.append(
            AsyncOpenAI(api_key="EMPTY", base_url=f"http://127.0.0.1:800{i}/v1", timeout=1200)
        )

PQ_K = int(os.environ.get("PQ_K", 50))
ROLLOUT_DATA_DIR = os.environ.get("ROLLOUT_DATA_DIR", "./output/EVE/rollout_data")
ROLLOUT_DATA_DIR_RAW = ROLLOUT_DATA_DIR
ROLLOUT_DATA_DIR = os.path.dirname(ROLLOUT_DATA_DIR[:-2])
PQ_PATH = os.path.join(ROLLOUT_DATA_DIR, "topk_codestrs.jsonl")

QWEN_MODEL_TYPE = os.environ.get("QWEN_MODEL_TYPE", "qwen2_5vl")
if QWEN_MODEL_TYPE == "qwen2_5vl":
    copy_path = "recipe/EVE/topk_codestrs_qwen2_5vl.jsonl"
else:
    copy_path = "recipe/EVE/topk_codestrs.jsonl"
if not os.path.exists(PQ_PATH):
    shutil.copy(copy_path, PQ_PATH)
else:
    backup_path = os.path.join(ROLLOUT_DATA_DIR_RAW, f"topk_codestrs_backup_{int(time.time())}.jsonl")
    shutil.copy(PQ_PATH, backup_path)
    

class CustomRLHFDataset(RLHFDataset):
    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict: str = self.dataframe[item]
        question = row_dict[self.prompt_key][-1]["content"]
        question = question.replace("<image>", "").strip()
        row_dict_images = row_dict.get(self.image_key) # 我们要把图片路径一直存着
        row_dict['extra_info']['image_paths'] = row_dict_images # 存储图片路径以便后续使用
        

        if CSV_ROLE == "S":
            if "options" in question.lower() and "Answer with the option's letter from the given choices" not in question:
                question = question.strip() + random.choice(["\nAnswer with the option's letter from the given choices.","\nPlease select the correct answer from the options above.", "\nAnswer with the option's letter from the given choices directly."])
            if "boxed{" not in question:
                question = question + S_prompt
            images = [my_process_image(image) for image in row_dict_images]
            num_images = len(row_dict_images)
            # 在question前加入n个<image>标记
            question = ("<image>" * num_images) + "\n" + question
            row_dict[self.prompt_key] = [
                {
                    "role": "user",
                    "content": question,
                },
            ]
        elif CSV_ROLE == "C":
            images = [my_process_image(image) for image in row_dict_images] 
            # 目前出题者只支持单张图片
            image = images[0]
            image_size = f"{image.width}x{image.height}"
            QWEN_MODEL_TYPE = os.environ.get("QWEN_MODEL_TYPE", "qwen2_5vl")
            if QWEN_MODEL_TYPE == "qwen2_5vl":
                full_user_prompt = f"\nThe input image size is {image_size}.\n" + USER_PROMPT
            else:
                full_user_prompt = USER_PROMPT
            num_images = len(row_dict_images)
            if random.randint(1, 100) <= 100: 
                # 改为从PQ_PATH 中选2个: code_str1和 code_str2
                with open(PQ_PATH, "r") as f:
                    lines = f.readlines()
                    random.shuffle(lines)
                    selected_line1 = lines[0]
                    item1 = json.loads(selected_line1.strip())
                    code_str1 = item1["code_str"]
                    selected_line2 = lines[1]
                    item2 = json.loads(selected_line2.strip())
                    code_str2 = item2["code_str"]
                    
                system_prompt = SYSTEM_PROMPT_PQ.replace("{code_str1}", code_str1).replace("{code_str2}", code_str2)
                question = ("<image>" * num_images) + "\n" + full_user_prompt

                # 0331 只保留 ## Example 2 之前的内容
                if random.randint(1, 100) <= 50:
                    system_prompt = system_prompt.split("## Example 2")[0]
            else:
                system_prompt = SYSTEM_PROMPT
                question = ("<image>" * num_images) + "\n" + full_user_prompt

            row_dict[self.prompt_key] = [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": question,
                },
            ]

        messages = self._build_messages(row_dict)
        model_inputs = {}

        if self.processor is not None:
            from verl.utils.dataset.vision_utils import process_image, process_video

            raw_prompt = self.processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False, **self.apply_chat_template_kwargs
            )
            multi_modal_data = {}

            # images = None
            # # row_dict_images = row_dict.pop(self.image_key, None)
            if row_dict_images:
                # images = [my_process_image(image) for image in row_dict_images]

                # due to the image key is "image" instead of "images" in vllm, we need to use "image" here
                # link: https://github.com/vllm-project/vllm/blob/3c545c0c3b98ee642373a308197d750d0e449403/vllm/multimodal/parse.py#L205
                multi_modal_data["image"] = images

            videos = None
            videos_kwargs = {}
            row_dict_videos = row_dict.pop(self.video_key, None)
            if row_dict_videos:
                videos, video_metadata = zip(
                    *[
                        process_video(video, image_patch_size=self.image_patch_size, return_video_metadata=True)
                        for video in row_dict_videos
                    ],
                    strict=True,
                )
                videos = list(videos)
                video_metadata = list(video_metadata)
                videos_kwargs = {"video_metadata": video_metadata, "do_sample_frames": False}

                # due to the video key is "video" instead of "videos" in vllm, we need to use "video" here
                # link: https://github.com/vllm-project/vllm/blob/3c545c0c3b98ee642373a308197d750d0e449403/vllm/multimodal/parse.py#L205
                multi_modal_data["video"] = [
                    (video.numpy(), metadata) for video, metadata in zip(videos, video_metadata, strict=True)
                ]

            model_inputs = self.processor(
                text=[raw_prompt], images=images, videos=videos, videos_kwargs=videos_kwargs, return_tensors="pt"
            )

            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

            if "second_per_grid_ts" in model_inputs:
                model_inputs.pop("second_per_grid_ts")

            # There's a trap here, multi_modal_inputs has to be a dict, not BatchFeature
            row_dict["multi_modal_data"] = multi_modal_data

            # We will do batch.union() in the trainer,
            # so we cannot have "multi_modal_inputs" in row_dict if rollout generates new multi_modal_inputs
            if self.return_multi_modal_inputs:
                row_dict["multi_modal_inputs"] = dict(model_inputs)

                # second_per_grid_ts isn't used for training, just for mrope
                row_dict["multi_modal_inputs"].pop("second_per_grid_ts", None)

        else:
            if self.apply_chat_template_kwargs.get("chat_template") is None:
                assert hasattr(self.tokenizer, "chat_template"), (
                    "chat_template should be provided in apply_chat_template_kwargs or tokenizer config, "
                    "models like GLM can copy chat_template.jinja from instruct models"
                )
            raw_prompt = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False, **self.apply_chat_template_kwargs
            )
            model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
            # qwen-vl mrope
            if "Qwen3VLProcessor" in self.processor.__class__.__name__:
                from verl.models.transformers.qwen3_vl import get_rope_index
            else:
                from verl.models.transformers.qwen2_vl import get_rope_index

            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids[0],
                image_grid_thw=model_inputs.get("image_grid_thw"),
                video_grid_thw=model_inputs.get("video_grid_thw"),
                second_per_grid_ts=model_inputs.get("second_per_grid_ts"),
                attention_mask=attention_mask[0],
            )  # (3, seq_length)
            valid_mask = attention_mask[0].bool()
            text_position_ids = torch.ones((1, len(input_ids[0])), dtype=torch.long)
            text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
            position_ids = [torch.cat((text_position_ids, vision_position_ids), dim=0)]  # (1, 4, seq_length)
        elif self.processor is not None and "Glm4vImageProcessor" in self.processor.image_processor.__class__.__name__:
            from verl.models.transformers.glm4v import get_rope_index

            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids[0],
                image_grid_thw=model_inputs.get("image_grid_thw"),
                video_grid_thw=model_inputs.get("video_grid_thw"),
                attention_mask=attention_mask[0],
            )  # (3, seq_length)
            valid_mask = attention_mask[0].bool()
            text_position_ids = torch.ones((1, len(input_ids[0])), dtype=torch.long)
            text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
            position_ids = [torch.cat((text_position_ids, vision_position_ids), dim=0)]  # (1, 4, seq_length)
        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        row_dict["input_ids"] = input_ids[0]
        row_dict["attention_mask"] = attention_mask[0]
        row_dict["position_ids"] = position_ids[0]

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "middle":
                left_half = self.max_prompt_length // 2
                right_half = self.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        row_dict["raw_prompt_ids"] = raw_prompt_ids
        # encode prompts without chat template
        if self.return_raw_chat:
            row_dict["raw_prompt"] = messages

        # get prompts with chat template
        if self.return_full_prompt:
            row_dict["full_prompts"] = raw_prompt  # array of strings

        # add index for each prompt
        if "extra_info" not in row_dict or row_dict["extra_info"] is None:
            row_dict["extra_info"] = dict()
        index = row_dict.get("extra_info", {}).get("index", 0)
        tools_kwargs = row_dict.get("extra_info", {}).get("tools_kwargs", {})
        interaction_kwargs = row_dict.get("extra_info", {}).get("interaction_kwargs", {})
        need_tools_kwargs = row_dict.get("extra_info", {}).get("need_tools_kwargs", self.need_tools_kwargs)
        if need_tools_kwargs and not tools_kwargs:
            logger.warning("tools_kwargs is empty for index {}, data source: {}", index, row_dict["data_source"])
        row_dict["index"] = index
        row_dict["tools_kwargs"] = tools_kwargs
        row_dict["interaction_kwargs"] = interaction_kwargs

        return row_dict

    def _read_files_and_tokenize(self):
        dataframes = []
        # breakpoint()
        for f in self.data_files:
            # read parquet files and cache
            if "parquet" in f:
                dataframe = datasets.load_dataset("parquet", data_files=f)["train"]
            elif "json" in f:
                dataframe = datasets.load_dataset("json", data_files=f)["train"]
            else:
                raise ValueError(f"Unsupported file format: {f}")
            dataframes.append(dataframe)
        self.dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes)

        total = len(self.dataframe)
        print(f"dataset len: {len(self.dataframe)}")

        if self.max_samples > 0 and self.max_samples < total:
            if self.shuffle:
                rngs_args = (self.seed,) if self.seed is not None else ()
                rng = np.random.default_rng(*rngs_args)
                indices = rng.choice(total, size=self.max_samples, replace=False)
            else:
                indices = np.arange(self.max_samples)
            self.dataframe = self.dataframe.select(indices.tolist())
            print(f"selected {self.max_samples} random samples out of {total}")

        self.dataframe = self.maybe_filter_out_long_prompts(self.dataframe)


def local_file_to_base64_url(image_input: Union[str, Image.Image]):
    """
    将本地图片文件或PIL图片对象转换为 Base64 编码的 Data URL。
    """
    if isinstance(image_input, str):
        image_path = image_input
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found at: {image_path}")

        ext = os.path.splitext(image_path)[1].lower()
        if ext in ['.jpg', '.jpeg']:
            mime_type = 'image/jpeg'
        elif ext == '.png':
            mime_type = 'image/png'
        else:
            print(f"Warning: Assuming image is a JPEG for MIME type: {ext}")
            mime_type = 'image/jpeg'

        with open(image_path, "rb") as f:
            encoded_string = base64.b64encode(f.read()).decode('utf-8')
        return f"data:{mime_type};base64,{encoded_string}"
    
    elif isinstance(image_input, Image.Image):
        buffered = io.BytesIO()
        if image_input.mode == 'RGBA' or 'A' in image_input.getbands():
            image_input.save(buffered, format="PNG")
            mime_type = 'image/png'
        else:
            image_input.save(buffered, format="JPEG")
            mime_type = 'image/jpeg'
        
        encoded_string = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return f"data:{mime_type};base64,{encoded_string}"
    
    else:
        raise TypeError(f"Unsupported input type: {type(image_input)}. Expected str or PIL.Image.Image.")

async def get_vllm_response(client_vllm, image_paths: list, query: str, model_name: str, max_response_length: int = 4096, max_retries: int = 3, temperature: float = 0.7, messages: list = None, resize_rate: float = 1.0, add_noise_rate: float = 0.0, seed: int = 42) -> str:
    """
    Sends a request with multiple images to a VLLM endpoint and retries on failure.

    Args:
        client_vllm: The OpenAI client instance for VLLM.
        image_paths: A list of local file paths for the images.
        query: The text query to send with the images.
        model_name: The name of the model to use.
        max_response_length: The maximum number of tokens for the response.
        max_retries: The maximum number of retry attempts.

    Returns:
        The content of the model's response, or None if all retries fail.
    """
    if messages is None:
        content_parts = []
        for i, image_path in enumerate(image_paths):
            try:
                image = my_process_image(image_path) 
                base64_image_url = local_file_to_base64_url(image)
                content_parts.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": base64_image_url},
                    }
                )
            except FileNotFoundError as e:
                print(f"Error processing image: {e}")
                return None

        content_parts.append({"type": "text", "text": query})

        messages = [{"role": "user", "content": content_parts}]

    for attempt in range(max_retries):
        try:
            start_time = time.time()
            QWEN_MODEL_TYPE = os.environ.get("QWEN_MODEL_TYPE", "qwen2_5vl")
            if QWEN_MODEL_TYPE == "qwen2_5vl":
                extra_body={}
            else:
                extra_body={
                    "sampling_params": {
                        "top_k": 40,
                        "top_p": 1.0,
                        "presence_penalty": 2.0
                    }
                }
            response = await client_vllm.chat.completions.create(  # keep await, async client
                model=model_name,
                messages=messages,
                max_tokens=max_response_length,
                temperature=temperature,
                extra_body=extra_body
            )
            if random.random() < 0.01:  # Debug log for 0.1% of requests
                end_time = time.time()
                print(f"Response received in {end_time - start_time:.2f}s")
                print(f"VLLM Response\n: {response.choices[0].message.content}")
            # sleep for a short time to avoid rate limit
            await asyncio.sleep(random.uniform(0, 0.05))
            return response.choices[0].message.content

        except Exception as e:
            print(f"An error occurred on attempt {attempt + 1}/{max_retries}: {e}")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"Retrying in {wait_time} seconds...")
                await asyncio.sleep(wait_time)  # was time.sleep
            else:
                print("All retry attempts failed.")
                return None
    return None

def update_pq(code_str, diff_reward):
    # 维护一个优先队列，保存top-K的code_str，根据diff_reward排序，diff_reward越大越靠前，diff_reward相同则后加入的越靠前
    def _code_prefix_for_dedup(s: str) -> str:
        if "args_list =" in s:
            return s.split("args_list =", 1)[0].strip()
        if "args_list=" in s:
            return s.split("args_list=", 1)[0].strip()
        return s.strip()

    pq = []
    if os.path.exists(PQ_PATH):
        with open(PQ_PATH, "r") as f:
            for line in f:
                item = json.loads(line.strip())
                pq.append((item["code_str"], item["diff_reward"]))
    flag = True
    for ss in pq:
        if get_bleu_score(_code_prefix_for_dedup(ss[0]), _code_prefix_for_dedup(code_str)) > 0.25:
            flag = False
            print("[debug] Duplicate found, skipping...")
            break
    if flag:
        pq.append((code_str, diff_reward))
        # 根据diff_reward排序，diff_reward越大越靠前，diff_reward相同则后加入的越靠前
        indexed_pq = list(enumerate(pq))
        indexed_pq = sorted(indexed_pq, key=lambda t: (-t[1][1], -t[0]))
        pq = [item for _, item in indexed_pq]
        # 去重
        seen = set()
        unique_pq = []
        for item in pq:
            dedup_key = _code_prefix_for_dedup(item[0])
            if dedup_key not in seen:
                seen.add(dedup_key)  # 只根据 code_str 中 args_list 前的部分去重，避免因为参数不同而认为是不同的代码
                unique_pq.append(item)
        pq = unique_pq
        # 只保留top-K
        pq = pq[:PQ_K]
        with open(PQ_PATH, "w") as f:
            for item in pq:
                f.write(json.dumps({"code_str": item[0], "diff_reward": item[1]}) + "\n")

async def get_diff_reward(question, answer, image_paths, sample_num=5, resize_rate=1.0, add_noise_rate=0.0, noise_seed=42):
    # 利用Solver的反馈计算差异奖励，r = 1 - 2*|avg_S_score - 0.5|
    avg_S_score = 0.0
    S_scores = []
    # 获得Solver的回答
    if "A" in question and "B" in question and "C" in question and "D" in question:
        question += "\nAnswer with the option's letter from the given choices."
    else:
        question += "\nAnswer the question using a single word."
    client_S = random.choice(client_list)
    for _ in range(sample_num):
        try:
            S_response = await get_vllm_response(
                client_S,
                image_paths=image_paths,
                query=question + S_prompt,
                model_name=VLLM_MODEL_NAME_S,
                max_response_length=VLLM_MAX_LEN_S,
                temperature=1.0,
                resize_rate=resize_rate,
                add_noise_rate=add_noise_rate,
                seed=noise_seed
            )
            S_answer = extract_boxed_content(S_response)
            if S_answer is None:
                S_score = 0.0
            else:
                S_score = acc_verifier(S_answer, answer, question)
        except Exception as e:
            # print(S_response)
            print(f"Error extracting S answer: {e}")
            S_score = 0.0
        S_scores.append(S_score)
    avg_S_score = sum(S_scores) / len(S_scores)
    # avg_S_score = min(0.95, avg_S_score) 
    diff_reward = 1 - 2 * abs(avg_S_score - 0.5)
    return diff_reward, avg_S_score
    

def get_new_vqa(code_func_str, args_list, input_image_path, processed_file_paths, question_type):
    if question_type == 0:
        idx = random.randint(0, len(args_list) - 1)
        args = args_list[idx]
        question = (
            "The given images are image_0, image_1, image_2, image_3, and image_4, respectively. "
            "Images image_1 through image_4 are the results of applying the `edit_image` function "
            "to image_0 with different arguments."
            + "\n```python\n" + code_func_str
            + "\n```\n"
            + "Question: "
            + f"After applying the `edit_image` function to image_0 with `{args}`, which candidate image will be produced?\n"
        )
        # Provide A/B/C/D options
        question += "Options:\n"
        for i, path in enumerate(processed_file_paths):
            question += f"{chr(65 + i)}. image_{i + 1}\n"
        gt = chr(65 + idx)
        img_lists = [input_image_path] + processed_file_paths
    elif question_type == 1:
        idx = random.randint(0, len(args_list) - 1)
        question = (
            "The given images are image_0 and image_1, respectively. "
            "image_1 is produced by applying the `edit_image` function to image_0 using a specific set of arguments."
            + "\n```python\n" + code_func_str
            + "\n```\n"
            + "Question: "
            + "Which set of arguments, when passed to `edit_image` for image_0, produces image_1?\n"
        )
        # Provide A/B/C/D options
        question += "Options:\n"
        for i, args in enumerate(args_list):
            question += f"{chr(65 + i)}. {args}\n"
        gt = chr(65 + idx)
        img_lists = [input_image_path, processed_file_paths[idx]]
    elif question_type == 2:
        # 假设 processed_file_paths[i] 是对 image_0 使用 args_list[i] 得到的结果（严格一一对应）
        if len(args_list) != len(processed_file_paths):
            raise ValueError("args_list and processed_file_paths must have the same length for question_type == 2.")

        # 1) 打乱 args_list，生成选项
        perm = list(range(len(args_list)))
        random.shuffle(perm)
        shuffled_args = [args_list[i] for i in perm]

        # 2) 建立 original_index -> option_letter 的映射
        #    perm[pos] = original_index，所以 original_index 的选项字母是 chr(65+pos)
        original_to_letter = {perm[pos]: chr(65 + pos) for pos in range(len(perm))}

        # 3) gt: 对 image_1..image_4（即 processed_file_paths[0..3]）输出对应字母序列
        #    image_{k+1} 对应 original_index = k
        gt = "".join(original_to_letter[i] for i in range(len(processed_file_paths)))

        question = (
            "The given images are image_0, image_1, image_2, image_3, and image_4. "
    "Images image_1 to image_4 are generated by applying the `edit_image` function to image_0, "
    "with each image using one unique set of input arguments (no repeats).\n"
            "\n```python\n" + code_func_str
            + "\n```\n"
        )
        question += "The candidate argument sets (labeled A, B, C, D) are as follows:\n"
        for i, args in enumerate(shuffled_args):
            question += f"{chr(65 + i)}. {args}\n"

        question += "Task:\nMatch each image (image_1, image_2, image_3, image_4) to the correct candidate argument set (A/B/C/D).\nOutput requirement: Provide a 4-letter sequence in the order of image_1, image_2, image_3, image_4. Example output: BDAC (meaning image_1=B, image_2=D, image_3=A, image_4=C)."

        img_lists = [input_image_path] + processed_file_paths

    else:
        raise ValueError(f"Unknown question_type: {question_type}")
    guidelines = """\nGuidelines: First, understand the functionality and parameter meanings of the code, then carefully observe each image, and solve the problem based on the key elements or details in the images."""
    question += guidelines

    if QWEN_MODEL_TYPE == "qwen2_5vl":
        input_image_pil = Image.open(input_image_path)
        question = f"The image_0 size is {input_image_pil.width}x{input_image_pil.height}.\n" + question

    return question, gt, img_lists



from PIL import Image, ImageOps
import numpy as np

def is_image_ok(
    image_path: str,
    MAX_PIXELS: int,
    black_ratio_threshold: float = 0.75,
    black_luma_threshold: int = 5,
    min_alpha: int = 10,
    code_str: str = ""
) -> bool:
    """
    返回 True 表示图片合格；返回 False 表示：
    - 近黑像素占比超过 black_ratio_threshold
    - 或者 总像素数(宽*高) > MAX_PIXELS

    参数说明：
    - black_luma_threshold: 亮度阈值，越大越“宽松”地认为是黑色（建议 10~30）
    - min_alpha: 透明像素(或几乎透明)不参与黑色统计，避免 PNG 透明背景误判
    """
    # 如果不是旋转操作，过滤更严格
    if ".rotate(" not in code_str:
        black_ratio_threshold = 0.1
    try:
        with Image.open(image_path) as im:
            # 1) 分辨率限制（先做，避免后续大图转数组成本过高）
            w, h = im.size
            if w * h > MAX_PIXELS:
                return False
            if w * h < MIN_PIXELS:
                return False

            # 2) 统一方向（处理 EXIF 旋转）
            im = ImageOps.exif_transpose(im)

            # 3) 转 RGBA，方便处理透明度
            im_rgba = im.convert("RGBA")
            arr = np.asarray(im_rgba, dtype=np.uint8)  # (H, W, 4)

            rgb = arr[..., :3].astype(np.float32)
            alpha = arr[..., 3].astype(np.uint8)

            # 4) 计算亮度（感知加权）
            luma = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]

            # 5) 只统计“足够不透明”的像素
            valid = alpha >= min_alpha
            valid_count = int(valid.sum())
            if valid_count < w * h * 0.1:  # If less than 1% of pixels are valid (non-transparent)
                # 全透明/几乎全透明，按你的业务可以认为不合格或合格
                # 这里保守返回 False（避免把透明图当成合格）
                return False

            # 6) 判定近黑像素比例
            black = valid & (luma <= black_luma_threshold)
            black_ratio = float(black.sum()) / float(valid_count)
            # print(black_ratio)
            return black_ratio <= black_ratio_threshold

    except Exception:
        # 文件不存在/损坏/不是图片等
        return False


async def compute_score(data_source: str, solution_str: str, ground_truth: str, extra_info=None, **kwargs):
    if CSV_ROLE != "S":
        # 随机延迟
        rd = random.uniform(0, 0.1)
        print(f"[Debug] compute_score random delay: {rd:.3f} seconds")
        await asyncio.sleep(rd)
        
    # breakpoint()
    image_paths = extra_info.get("image_paths")
    question = extra_info.get("question", "")

    raw_solution_str = solution_str
    raw_ground_truth = ground_truth

    my_log = False
    if random.random() < 0.01 or os.environ.get("DEBUG_MODE") == "true":   
        my_log = True

    solution_str = solution_str.strip()
    if "</think>" in solution_str:
        solution_str = solution_str.split("</think>")[-1]

    score = 0.0
    if CSV_ROLE == "C":
        format_reward = 0.0
        code_valid_reward = 0.0
        diff_reward = 0.0
        acc_rate = 0.0
        code_str = ""
        code_func_str = ""

        if_rm = True
        img_dir = None
        # 尝试提取code，提取失败则认为格式错误
        try:
            code_str = solution_str.split("```python")[-1].split("```")[0].strip()
            if len(code_str) == 0:
                raise ValueError("Code string is empty.")
            format_reward = 1.0
            if (not solution_str.count("<thinking>") == solution_str.count("</thinking>") == 1) and (not solution_str.count("<think>") == solution_str.count("</think>") == 1):
                format_reward = 0
            # # 禁止注释
            # comment_cnt = code_str.count("#")
            # comment_cnt += code_str.count('"""')
            # comment_cnt += code_str.count("'''") 
            # format_reward -= 0.1 * min(comment_cnt, 5)
            # 处理图片，保存到临时目录
            image_path = image_paths[0]
            input_image = my_process_image(image_path)
            # Create a unique ID from the image path for directory naming
            path_prefix = os.environ.get("IMG_PATH_PREFIX", "/data/kcl/hyr/")
            img_item_id = image_path.replace(path_prefix, "").replace("/", "_").replace("\\", "_").replace(".", "_").replace(".", "_")
            img_dir = os.path.join(TEMP_PROCESSED_DIR, img_item_id, f"input_{int(time.time()*10000)}_{random.randint(1000,9999)}")
            os.makedirs(img_dir, exist_ok=True)
            input_image_path = os.path.join(img_dir, "input_image.png")
            input_image.save(input_image_path)

            if ".save" in code_str:
                raise ValueError("Code contains .save(), which is not allowed.")
            if "import random" in code_str or "random." in code_str:
                raise ValueError("Code contains random module, which is not allowed.")
            
            # if QWEN_MODEL_TYPE == "qwen2_5vl":
            #     if has_image_processing_functions(code_str):
            #         raise ValueError("Code contains OpenCV, which is not allowed.")

            # 在沙箱中执行code
            POST_CODE = f"""\nimage_path = "test.png"
from PIL import Image
for i, args in enumerate(args_list):
    img = Image.open(image_path)
    result = edit_image(img, **args)
    processed_path = f"{img_dir}/case{{i}}.png"
    result.save(processed_path)
    print(processed_path)
"""
            # processed_file_paths, print_output, error_message, _ = execute_code_in_sandbox(
            #     code_to_execute=code_str + POST_CODE,
            #     input_image_path=input_image_path,
            #     item_id=img_item_id,
            #     temp_output_dir=img_dir,
            #     previous_execution_context=None,
            # )           

            processed_file_paths, print_output, error_message, _ = await asyncio.to_thread(
                execute_code_in_sandbox,
                code_to_execute=code_str + POST_CODE,
                input_image_path=input_image_path,
                item_id=img_item_id,
                temp_output_dir=img_dir,
                previous_execution_context=None,
            )
            await asyncio.sleep(random.uniform(0, 0.2))

            if error_message is not None:
                raise RuntimeError(f"Code execution error: {error_message}")

            # processed_file_paths 从小到大排序
            processed_file_paths.sort()

            # 把code_str分为code_func和args_list两部分
            args_list_str = code_str.split("args_list =")[-1] if "args_list =" in code_str else code_str.split("args_list=")[-1]
            args_list_str = args_list_str.strip()
            args_list = eval(args_list_str)
            if not isinstance(args_list, list):
                raise ValueError("args_list is not a list.")
            if len(args_list) != len(processed_file_paths):
                raise ValueError("args_list length does not match processed_file_paths length.")
            if len(processed_file_paths) != 4:
                raise ValueError("processed_file_paths length is not 4.")
            # 如果 args_list 中有重复的参数，则认为无效
            if len(args_list) != len(set([str(args) for args in args_list])):
                raise ValueError("args_list contains duplicate arguments.")
            # TODO 遍历 args_list, 如果参数在 SYSTEM 中则认为无效

            # 逐条检查 processed_file_paths
            for i, path in enumerate(processed_file_paths):
                if not is_image_ok(path, MAX_PIXELS=MAX_PIXELS, code_str=code_str):
                    raise ValueError(f"Processed image {i} is not valid.")
            
            if has_visually_same_images(processed_file_paths):
                print("[debug] Processed images are the same, which is not allowed.")
                raise ValueError("Processed images are the same, which is not allowed.")

            code_func_str = code_str.split("args_list =")[0] if "args_list =" in code_str else code_str.split("args_list=")[0]
            code_func_str = code_func_str.strip()

            if len(code_func_str) > 2000:
                raise ValueError("Code function string is too long.")

            comment_cnt = code_func_str.count("#")
            comment_cnt += code_func_str.count('"""')
            comment_cnt += code_func_str.count("'''") 
            format_reward = 0 if comment_cnt > 2 else format_reward
            
            if os.environ.get("USE_S", "true") == "true":
                diff_reward_list = []
                # question_type = 2
                for idx in range(2):
                    question_type = idx  # 0, 1, 2
                    question_new, gt_new, img_lists = get_new_vqa(code_func_str, args_list, input_image_path, processed_file_paths, question_type=question_type)
                    diff_reward, acc_rate = await get_diff_reward(
                        question=question_new,
                        answer=gt_new,
                        image_paths=img_lists,
                        sample_num=SAMPLE_NUM
                    )
                    diff_reward_list.append(diff_reward)
                    # 把所有东西记录在img_dir下的log.json中
                    log_dict = {
                        "input_image_path": img_lists[0],
                        "processed_file_paths": img_lists[1:],
                        "question_new": question_new,
                        "gt_new": gt_new,
                        "code_str": code_str,
                        "acc_rate": acc_rate,
                    }
                    log_path = os.path.join(img_dir, f"log_{idx}.json") 
                    if acc_rate >= 0.25 and acc_rate < 1 and comment_cnt <= 4:
                        with open(log_path, "w") as f:
                            json.dump(log_dict, f, indent=4)
                            if_rm = False
                            
                if if_rm:
                    # 把文件夹 img_dir 删除
                    shutil.rmtree(img_dir)
                    diff_reward = sum(diff_reward_list) / len(diff_reward_list)
            else:
                shutil.rmtree(img_dir)

            code_valid_reward = 1.0            
            if os.environ.get("DEBUG_MODE") == "true" or (diff_reward > 0.5 and comment_cnt <= 2 and len(code_func_str) < 2000):
                if PQ_K != 0:
                    update_pq(code_str, diff_reward)
            
        except Exception as e:
            if my_log:
                print(f"[DEBUG] Error in code execution or processing: {e}")
        finally:
            # 可选：如果你希望“异常时也删临时目录”，打开下面逻辑
            if img_dir and os.path.isdir(img_dir) and if_rm:
                shutil.rmtree(img_dir, ignore_errors=True)
            pass
        score = format_reward * 0.2 + code_valid_reward * 0.4 + diff_reward * 0.4


    elif CSV_ROLE == "S":
        format_reward = 0.0
        acc_reward = 0.0
        veri_reward = 0.0
        stu_answer = extract_boxed_content(solution_str)
        if stu_answer is not None:
            format_reward = 1.0
            if solution_str.strip().startswith(r"\boxed{"):
                format_reward = 0.5
            acc_reward = acc_verifier(stu_answer, ground_truth, question)
            # acc_reward = acc_score(stu_answer, ground_truth, question)
        # else:
        #     # 过长样本超过了max_len，放弃
        #     # 其他截取原文末尾
        #     if (len(solution_str) < 10000 and "answer" in solution_str.lower()[-50:]) or ("yes" in ground_truth.lower() or "no" in ground_truth.lower()):
        #         # 截取"answer"或"Answer"后面的内容作为答案
        #         stu_answer = solution_str.split("answer")[-1].split("Answer")[-1].replace(":","").strip()
        #         acc_reward = acc_verifier(stu_answer, ground_truth, question) 

        score = format_reward * 0.2 + acc_reward * 0.8


    res_dict = {}
    if CSV_ROLE == "C":
        res_dict = {
            "score": score,
            "format_reward": format_reward,
            "diff_reward": diff_reward,
            "acc_rate": acc_rate,
            "code_valid_reward": code_valid_reward, 
            "code_str": code_str,
            "code_func_str": code_func_str
        }
    elif CSV_ROLE == "S":
        res_dict = {
            "score": score,
            "format_reward": format_reward,
            "acc_reward": acc_reward,
        }

    if my_log:
        print(f"[DEBUG] Solution: {raw_solution_str}")
        # # print(f"[DEBUG] Question: {question}")
        # print(f"[DEBUG] Ground Truth: {raw_ground_truth}")
        print(f"[DEBUG] Detailed Rewards: {res_dict}")

    return res_dict
    # return {
    #     "score":score,
    #     "format_reward": format_reward,
    # }


if __name__ == "__main__":
    # Test case 1: Original test case
    test_incorrect = '''
    asfdfasrqewrew</think>
    {
        "errors_list": [
            {
                "quote": "The answer is 42",
                "critique": "This is incorrect because..."
            },
            {
                "quote": "The answer is 42",
                "critique": "This is incorrect because...",
                "related_bbox": [10, 20, 30]
            }
        ],
        "if_final_answer_correct": "no"
    }
    '''
    result1 = compute_score("test_source", test_incorrect, "42", extra_info={"question": "What is the answer to life?"})
    