# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from mathruler.grader import extract_boxed_content, grade_answer
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from collections import Counter
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import time

from collections import defaultdict
from typing import Any

import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager
import os

def _bleu_distance_matrix(sentences):
    n = len(sentences)
    dist = np.zeros((n, n))
    smoother = SmoothingFunction().method1
    for i in range(n):
        for j in range(i, n):
            if i == j:
                score = 1.0
            else:
                ref = [sentences[j].split()]
                hyp = sentences[i].split()
                score = sentence_bleu(ref, hyp, smoothing_function=smoother)
            dist[i, j] = dist[j, i] = 1 - score
    return dist

def cluster_share_per_problem(
        problems,
        distance_threshold: float = 0.5,
        linkage: str = "average"):
    if not problems:
        return []
    print('start clustering')
    start_time = time.time()
    dist_mat = _bleu_distance_matrix(problems)

    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric="precomputed",
        linkage=linkage
    )
    labels = clustering.fit_predict(dist_mat)
    print(f'end clustering, time: {time.time() - start_time}')
    total = len(problems)
    cluster_size = Counter(labels)
    cluster_ratio = {lab: sz / total for lab, sz in cluster_size.items()}

    proportions = [cluster_ratio[lab] for lab in labels]
    return proportions



@register("naive")
class NaiveRewardManager(AbstractRewardManager):
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source") -> None:
        """
        Initialize the NaiveRewardManager instance.

        Args:
            tokenizer: The tokenizer used to decode token IDs into text.
            num_examine: The number of batches of decoded responses to print to the console for debugging purpose.
            compute_score: A function to compute the reward score. If None, `default_compute_score` will be used.
            reward_fn_key: The key used to access the data source in the non-tensor batch data. Defaults to
                "data_source".
        """
        self.tokenizer = tokenizer  # Store the tokenizer for decoding token IDs
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key  # Store the key for accessing the data source

    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        CSV_ROLE = os.environ.get("CSV_ROLE","")
        if "rm_scores" in data.batch.keys() and CSV_ROLE != "C":
            if return_dict:
                reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
                reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
                return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        # reward_extra_info = defaultdict(list)
        reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
        reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
    
        # 用code_str计算惩罚, 最大长度 1000, 截取右侧
        if "code_func_str" in data.non_tensor_batch:
            code_str_list = [str(code_str)[-1000:] for code_str in data.non_tensor_batch["code_func_str"]]
        else:
            code_str_list = [str(code_str)[-1000:] for code_str in data.non_tensor_batch["question_gen"]]
        
        print(f"[debug] code_str_list[0]: {code_str_list[0]}")
        
        # 方案1
        # penalty = cluster_share_per_problem(code_str_list)
        # for i in range(len(code_str_list)):
        #     code_str = code_str_list[i]
        #     if not code_str:
        #         penalty[i] = 1.0  # Set penalty to 1.0 if code_str is empty
        #     penalty[i] = max(0.0, min(0.1, penalty[i]))

        # 方案2
        penalty = cluster_share_per_problem(code_str_list)
        # 把 penalty 归一化到 [0, 1]
        max_penalty = max(penalty) if penalty else 1.0
        min_penalty = min(penalty) if penalty else 0.0
        if max_penalty > min_penalty:
            penalty = [(p - min_penalty) / (max_penalty - min_penalty) for p in penalty]
        for i in range(len(code_str_list)):
            code_str = code_str_list[i]
            if not code_str:
                penalty[i] = 0.0  # Set penalty to 0.0 if code_str is empty
        print(f"[debug] penalty: {penalty}")
        # 整体 * 0.3 ?
        penalty = [p * 0.3 for p in penalty]

        reward_extra_info["penalty"] = penalty

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            score = reward_extra_info["score"][i]- 1.0 * penalty[i]
            reward_tensor[i, valid_response_length - 1] = score
            reward_extra_info["score"][i] = score

        # breakpoint()

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
