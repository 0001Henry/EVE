"""
Split dataset into training and validation sets.

This script splits the RL training data into training and validation partitions.
"""

import os
import datasets

DATA_PATH = os.environ.get("DATA_PATH", "./recipe/EVE/rl_data/train_47k.jsonl")
TRAIN_OUTPUT = os.environ.get("TRAIN_OUTPUT", "./recipe/EVE/rl_data/train_47k_p1.jsonl")
VAL_OUTPUT = os.environ.get("VAL_OUTPUT", "./recipe/EVE/rl_data/train_47k_p2.jsonl")

dataframe = datasets.load_dataset("json", data_files=DATA_PATH)["train"]
print(f"Original dataset size: {len(dataframe)}")

split_dataset = dataframe.train_test_split(test_size=0.5, seed=42)

train_dataset = split_dataset['train']
validation_dataset = split_dataset['test']

print(f"Training set size: {len(train_dataset)}")
print(f"Validation set size: {len(validation_dataset)}")

train_dataset.to_json(TRAIN_OUTPUT, orient='records', lines=True, force_ascii=False)
print(f"Training set saved to: {TRAIN_OUTPUT}")

validation_dataset.to_json(VAL_OUTPUT, orient='records', lines=True, force_ascii=False)
print(f"Validation set saved to: {VAL_OUTPUT}")

validation_loaded = datasets.load_dataset("json", data_files=VAL_OUTPUT)["train"]
print(f"Reloaded validation set size: {len(validation_loaded)}")
print(validation_loaded[0])
