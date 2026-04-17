"""
Prepare RL training data by sampling from source datasets.

This script samples data from source files for RL training, combining
a base number of samples with additional randomly sampled examples.
"""

import json
import random
import os
from tqdm import tqdm
import sys

BASE_DIR = os.environ.get("BASE_DIR", "./")
experiment_name = sys.argv[1]
gen_num = int(sys.argv[2])

try:
    seed = int(experiment_name[-1])
except ValueError:
    seed = 42
random.seed(seed)


def prepare_rl_data(src_path: str, out_path: str, base_num: int = 2000, sample_num: int = 2000) -> None:
    """
    Prepare training data by combining base samples with randomly sampled examples.

    Output data = first base_num lines + random sample_num from remaining,
    then shuffled together.
    """
    if not os.path.exists(src_path):
        raise FileNotFoundError(src_path)

    with open(src_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    n = len(lines)
    if n == 0:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        open(out_path, "w", encoding="utf-8").close()
        print(f"Prepared 0 lines for {out_path} (empty source)")
        return

    base_num = max(0, min(base_num, n))
    rest = lines[base_num:]
    sample_num = max(0, min(sample_num, len(rest)))

    final_lines = lines[:base_num] + (random.sample(rest, sample_num) if sample_num else [])
    random.shuffle(final_lines)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.writelines(final_lines)

    print(
        f"Prepared {len(final_lines)} lines for {out_path} "
        f"(base={base_num}, sampled={sample_num}, total_src={n})"
    )

p1 = os.path.join(BASE_DIR, "recipe/EVE/rl_data/train_47k_p1.jsonl")
p1_out = os.path.join(BASE_DIR, "recipe/EVE/rl_data/train_47k_p1_tmp.jsonl")
prepare_rl_data(p1, p1_out, base_num=gen_num, sample_num=gen_num*2)

p2 = os.path.join(BASE_DIR, "recipe/EVE/rl_data/train_47k_p2.jsonl")
p2_out = os.path.join(BASE_DIR, "recipe/EVE/rl_data/train_47k_p2_tmp.jsonl")
prepare_rl_data(p2, p2_out, base_num=gen_num, sample_num=gen_num*2)
