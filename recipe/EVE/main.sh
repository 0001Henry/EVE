# 全局配置
pkill -f verl.trainer.main_ppo
pkill -f vllm
pkill -9 -f VLLM
pkill -f sglang
pkill -9 -f ray
sleep 2

export CUDA_LAUNCH_BLOCKING=1
export HYDRA_FULL_ERROR=1
export VLLM_USE_V1=1
export HF_HUB_DISABLE_ASYNC_WARNING=1
export PYTHONUNBUFFERED=1
export VLLM_URL_S="http://127.0.0.1:22001/v1"


export BASEDIR="your_path_here"
export OUTPUT_DIR=${BASEDIR}/myoutput
export SAVE_CHECKPOINT_DIR=${OUTPUT_DIR}/checkpoints
export SEED=42
export SAMPLE_NUM=6
export VLLM_MAX_LEN_S=2048


export TOTAL_TRAINING_STEPS=10
export ROLL_BATCH_SIZE=128
export BATCH_SIZE=128
export ITER_NUM=5
export DEBUG_MODE=false
export PQ_K=50

model_abbr=Qwen3-VL-8B-Instruct
base_model=/data/kcl/hyr/ckpt/Qwen/Qwen3-VL-8B-Instruct
export IMAGE_PATCH_SIZE=16
export MAX_PIXELS=$((1024*32*32))
export MIN_PIXELS=$((4*32*32))
export QWEN_MODEL_TYPE="qwen3vl"


export PROJECT_NAME=${model_abbr}_0326


log_file=${BASEDIR}/myoutput/logs/${PROJECT_NAME}.log
mkdir -p $(dirname ${log_file})
echo "====================" >> ${log_file} 2>&1
echo "Start time: $(date)" >> ${log_file} 2>&1
echo "project_name: $PROJECT_NAME" >> ${log_file} 2>&1
echo "base_model: $base_model" >> ${log_file} 2>&1


# Initialize first iteration with base model

# Challenger model
if [ -d "${SAVE_CHECKPOINT_DIR}/${PROJECT_NAME}/${model_abbr}_C_iter1/global_step_${TOTAL_TRAINING_STEPS}/actor/huggingface" ]; then
    echo "Challenger model ${model_abbr}_C_iter1 already exists, skipping training..." >> ${log_file} 2>&1
else
    echo "Training Challenger model ${model_abbr}_C_iter1 ..." >> ${log_file} 2>&1
    bash recipe/EVE/C_train.sh \
        $base_model \
        $base_model \
        ${model_abbr}_C_iter1
    echo "Challenger model ${model_abbr}_C_iter1 training finished." >> ${log_file} 2>&1
fi

# Solver model
if [ -d "${SAVE_CHECKPOINT_DIR}/${PROJECT_NAME}/${model_abbr}_S_iter1/global_step_${TOTAL_TRAINING_STEPS}/actor/huggingface" ]; then
    echo "Solver model ${model_abbr}_S_iter1 already exists, skipping training..." >> ${log_file} 2>&1
else
    echo "Training Solver model ${model_abbr}_S_iter1 ..." >> ${log_file} 2>&1
    bash recipe/EVE/S_train.sh \
        ${SAVE_CHECKPOINT_DIR}/${PROJECT_NAME}/${model_abbr}_C_iter1/global_step_${TOTAL_TRAINING_STEPS}/actor/huggingface \
        $base_model \
        ${model_abbr}_S_iter1
    echo "Solver model ${model_abbr}_S_iter1 training finished." >> ${log_file} 2>&1
fi


for i in $(seq 2 $ITER_NUM); do

    prev=$((i-1))
    
    # Check if Challenger model already exists
    if [ -d "${SAVE_CHECKPOINT_DIR}/${PROJECT_NAME}/${model_abbr}_C_iter${i}/global_step_${TOTAL_TRAINING_STEPS}/actor/huggingface" ]; then
        echo "Challenger model ${model_abbr}_C_iter${i} already exists, skipping training..." >> ${log_file} 2>&1
    else
        # Train challenger
        echo "Training Challenger model ${model_abbr}_C_iter${i} ..." >> ${log_file} 2>&1
        bash recipe/EVE/C_train.sh \
            ${SAVE_CHECKPOINT_DIR}/${PROJECT_NAME}/${model_abbr}_C_iter${prev}/global_step_${TOTAL_TRAINING_STEPS}/actor/huggingface \
            ${SAVE_CHECKPOINT_DIR}/${PROJECT_NAME}/${model_abbr}_S_iter${prev}/global_step_${TOTAL_TRAINING_STEPS}/actor/huggingface \
            ${model_abbr}_C_iter${i}
        echo "Challenger model ${model_abbr}_C_iter${i} training finished." >> ${log_file} 2>&1
    fi

    # Check if solver model already exists
    if [ -d "${SAVE_CHECKPOINT_DIR}/${PROJECT_NAME}/${model_abbr}_S_iter${i}/global_step_${TOTAL_TRAINING_STEPS}/actor/huggingface" ]; then
        echo "Solver model ${model_abbr}_S_iter${i} already exists, skipping training..." >> ${log_file} 2>&1
    else
        # Train solver
        echo "Training Solver model ${model_abbr}_S_iter${i} ..." >> ${log_file} 2>&1
        bash recipe/EVE/S_train.sh \
            ${SAVE_CHECKPOINT_DIR}/${PROJECT_NAME}/${model_abbr}_C_iter${i}/global_step_${TOTAL_TRAINING_STEPS}/actor/huggingface \
            ${SAVE_CHECKPOINT_DIR}/${PROJECT_NAME}/${model_abbr}_S_iter${prev}/global_step_${TOTAL_TRAINING_STEPS}/actor/huggingface \
            ${model_abbr}_S_iter${i}
        echo "Solver model ${model_abbr}_S_iter${i} training finished." >> ${log_file} 2>&1
    fi

done

pkill -f verl.trainer.main_ppo
pkill -f vllm
pkill -9 -f VLLM
pkill -f sglang
pkill -9 -f ray
