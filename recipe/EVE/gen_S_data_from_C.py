"""
Generate Solver training data from Challenger outputs.

This script collects log files from Challenger training runs and converts them
into training data for the Solver model.
"""

import os
import json
import glob
from tqdm import tqdm
import sys
import random

def collect_all_log_dicts(root_dir: str, num: int):
    """
    Collect all log dictionaries from log_*.json files under root_dir.

    Each log file contains execution results from the Challenger model.
    """
    pattern = os.path.join(root_dir, "**", "log_*.json")
    all_dicts = []
    for fp in glob.glob(pattern, recursive=True):
        try:
            with open(fp, "r", encoding="utf-8") as f:
                d = json.load(f)
            if len(d["processed_file_paths"]) > 4:
                continue
            acc_rate = d['acc_rate']
            d["diff_reward"] = 1 - 2 * abs(acc_rate - 0.5)
            if acc_rate > 0 and acc_rate < 1:
                d["_source_file"] = fp
                all_dicts.append(d)
        except Exception as e:
            print(f"[WARN] failed to load {fp}: {e}")

    all_dicts.sort(key=lambda x: x["diff_reward"], reverse=True)
    print(f"need {num} samples from {len(all_dicts)} total samples")
    if num >= len(all_dicts):
        return all_dicts

    final_all_dicts = all_dicts[:num]
    return final_all_dicts


data_path = sys.argv[1]
output_path = sys.argv[2]
c_model_path = sys.argv[3]
s_model_path = sys.argv[4]
gen_num = int(sys.argv[5])

print(f"data_path: {data_path}")
print(f"output_path: {output_path}")
print(f"c_model_path: {c_model_path}")
print(f"s_model_path: {s_model_path}")
print(f"gen_num: {gen_num}")

VLLM_MODEL_NAME_S = s_model_path

MAX_PIXELS = int(os.environ.get("MAX_PIXELS", 2000000))
MIN_PIXELS = int(os.environ.get("MIN_PIXELS", 20000))
VLLM_MAX_LEN_S = int(os.environ.get("VLLM_MAX_LEN_S", "4096"))
CSV_ROLE = os.environ.get("CSV_ROLE")
SAMPLE_NUM = int(os.environ.get("SAMPLE_NUM", 6))
SEED = int(os.environ.get("SEED", 42))
random.seed(SEED)
ROLLOUT_DATA_DIR = os.environ.get("ROLLOUT_DATA_DIR")

ps = ROLLOUT_DATA_DIR.replace("S_iter", "C_iter")

all_dicts = collect_all_log_dicts(ps, gen_num)
print(all_dicts[0])

res_list = []
pid = 0
for d in all_dicts:
    try:
        question = d["question_new"]
        answer = d["gt_new"]
        image_paths = [d['input_image_path']] + d["processed_file_paths"]
        new_example = {
            "data_source": "C",
            "prompt": [
                {
                    "role": "user",
                    "content": f"<image>\n{question}"
                }
            ],
            "images": image_paths,
            "reward_model": {
                "style": "model",
                "ground_truth": answer
            },
            "extra_info": {
                "split": "train",
                "pid": pid,
                "answer": answer,
                "question": question,
            }
        }
        pid += 1
        res_list.append(new_example)
    except Exception as e:
        print(f"Error processing dict: {e}")

print("Total samples collected:", len(res_list))
random.shuffle(res_list)
res_list = res_list[:gen_num]

with open(output_path, "w") as f:
    for item in res_list:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"Saved {len(res_list)} examples to {output_path}")
