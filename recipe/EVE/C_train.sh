#!/bin/bash
# Challenger (C) training script for EVE
# Usage: bash C_train.sh <challenger_model> <solver_model> <experiment_name>

rm -rf /tmp/ray/
rm -rf ~/.cache/

echo "experiment_name: $experiment_name"
echo "Start training challenger: $challenger_model -> $experiment_name"

# Prepare data
gen_num=$((ROLL_BATCH_SIZE*TOTAL_TRAINING_STEPS))
python recipe/EVE/prepare_rl_data.py $experiment_name $gen_num
echo "Data preparation finished."

# Start vllm services
bash recipe/EVE/vllm_S.sh $solver_model
echo "vLLM services starting..."
sleep 180
echo "vLLM services started"

# Role configuration
export CSV_ROLE="C"
export VLLM_MODEL_NAME_S=$solver_model
export ROLLOUT_DATA_DIR=${OUTPUT_DIR}/rollout_data/${PROJECT_NAME}/${experiment_name}
export USE_S="true"

# Training configuration
export CUDA_VISIBLE_DEVICES=4,5,6,7
DATASET_TRAIN1=${BASEDIR}/recipe/EVE/rl_data/train_47k_p1_tmp.jsonl
DATASET_VAL1=$DATASET_TRAIN1

python3 -m verl.trainer.main_ppo \
    --config-path=${BASEDIR}/recipe/EVE/configs \
    --config-name='grpo' \
    data.train_files=${DATASET_TRAIN1} \
    data.val_files=[${DATASET_VAL1}] \
    data.train_batch_size=${ROLL_BATCH_SIZE} \
    data.image_patch_size=${IMAGE_PATCH_SIZE} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${BATCH_SIZE} \
    actor_rollout_ref.model.path=${challenger_model} \
    actor_rollout_ref.rollout.n=4 \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.total_training_steps=${TOTAL_TRAINING_STEPS} \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${experiment_name} \
    trainer.default_local_dir=${SAVE_CHECKPOINT_DIR}/${PROJECT_NAME}/${experiment_name} \
    trainer.val_only=${VAL_ONLY} \
    trainer.val_before_train=${VAL_ONLY}

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

echo "Challenger training finished"
