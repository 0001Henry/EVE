#!/bin/bash
# Solver (S) training script for EVE
# Usage: bash S_train.sh <challenger_model> <solver_model> <experiment_name>

rm -rf /tmp/ray/
rm -rf ~/.cache/

echo "experiment_name: $experiment_name"

# Generate data
data_path=${BASEDIR}/recipe/EVE/rl_data/train_47k_p2_tmp.jsonl
output_path=${OUTPUT_DIR}/gen_data/${PROJECT_NAME}/${experiment_name}_gen_data.jsonl
gen_num=$((ROLL_BATCH_SIZE*TOTAL_TRAINING_STEPS))

rm "$output_path"
mkdir -p $(dirname $output_path)

export ROLLOUT_DATA_DIR=${OUTPUT_DIR}/rollout_data/${PROJECT_NAME}/${experiment_name}

if [ -f "$output_path" ]; then
    echo "Generated data file $output_path already exists, skipping data generation..."
else
    echo "Generating data to $output_path ..."
    python recipe/EVE/gen_S_data_from_C.py $data_path $output_path $challenger_model $solver_model $gen_num
    echo "Data generation finished: $output_path"
fi

# Start Solver training
echo "Start training solver: $solver_model -> $experiment_name"

# Role configuration
export CSV_ROLE="S"
export ROLLOUT_DATA_DIR=${OUTPUT_DIR}/rollout_data/${PROJECT_NAME}/${experiment_name}

# Training configuration
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
DATASET_TRAIN1=$output_path
DATASET_VAL1="recipe/EVE/rl_data/mmstar_mini_200.jsonl"

python3 -m verl.trainer.main_ppo \
    --config-path=${BASEDIR}/recipe/EVE/configs \
    --config-name='grpo' \
    data.train_files=[${DATASET_TRAIN1}] \
    data.val_files=[${DATASET_VAL1}] \
    data.train_batch_size=${ROLL_BATCH_SIZE} \
    data.max_prompt_length=8192 \
    data.image_patch_size=${IMAGE_PATCH_SIZE} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${BATCH_SIZE} \
    actor_rollout_ref.model.path=${solver_model} \
    actor_rollout_ref.rollout.n=8 \
    trainer.test_freq=5 \
    trainer.save_freq=5 \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.total_training_steps=${TOTAL_TRAINING_STEPS} \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${experiment_name} \
    trainer.default_local_dir=${SAVE_CHECKPOINT_DIR}/${PROJECT_NAME}/${experiment_name}

sleep 5
pkill -f verl.trainer.main_ppo
pkill -f vllm
pkill -9 -f VLLM
pkill -f sglang
pkill -9 -f ray
sleep 5
pkill -f verl.trainer.main_ppo
pkill -f vllm
pkill -9 -f VLLM
pkill -f sglang
pkill -9 -f ray
sleep 10

echo "Solver training finished"
