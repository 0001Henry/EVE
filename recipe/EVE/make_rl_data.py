"""
Convert dataset to RL training format.

This script transforms raw Vision-SR1 datasets into the format required
for EVE Challenger-Solver RL training.
"""

import os
import random
import json
import datasets
from datasets import load_dataset
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

IMG_DIR = os.environ.get("IMG_DIR", "./data/Vision-SR1-47K-images/")
random.seed(1234)


def transfer(example):
    """Transform dataset example from raw format to RL training format."""
    question = example['problem']
    image = example['images']
    img_path = example['path']
    img_path = os.path.join(IMG_DIR, img_path.lstrip("./"))
    os.makedirs(os.path.dirname(img_path), exist_ok=True)
    image.save(img_path)

    answer = example['answer']
    pid = example['problem_id']
    data_source = example.get('data_source', 'unknown')

    new_example = {
        "data_source": data_source,
        "prompt": [
            {
                "role": "user",
                "content": f"<image>\n{question}"
            }
        ],
        "images": [img_path],
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
    return new_example


DATA_DIR = os.environ.get("DATA_DIR", "LMMs-Lab-Turtle/Vision-SR1-47K")
OUTPUT_PATH = os.environ.get("OUTPUT_PATH", "./recipe/EVE/rl_data/train_47k.jsonl")
CACHE_DIR = os.environ.get("CACHE_DIR", "./tmp/")

dataset = load_dataset("parquet", data_dir=DATA_DIR, split="train", cache_dir=CACHE_DIR)
print(f"Original dataset size: {len(dataset)}")

final_dataset = dataset.map(
    transfer,
    num_proc=16,
    remove_columns=dataset.column_names
)

print(f"Final dataset size: {len(final_dataset)}")

final_dataset.to_json(OUTPUT_PATH, orient='records', lines=True, force_ascii=False)
dataframe = datasets.load_dataset("json", data_files=OUTPUT_PATH)["train"]
print(f"Saved dataset size: {len(dataframe)}")
print(dataframe[0])
print("Processing completed!")
