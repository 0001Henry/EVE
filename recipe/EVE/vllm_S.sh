
model_path=$1
pid_file=${2:-vllm_S.pid}
log_file=/dev/null


# 方案 1
# nohup env CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve $model_path \
#   --data-parallel-size 4 \
#   --limit-mm-per-prompt.video 0 \
#   --max-model-len 16000 \
#   --gpu-memory-utilization 0.6 \
#   --host 0.0.0.0 \
#   --port 22001 \
#   >"$log_file" 2>&1 &

# echo $! > "$pid_file"



# 方案 2
hostname=`hostname -i`
BASE_PORT=8000
NUM_INSTANCES=4

for ((i=0; i<NUM_INSTANCES; i++)); do
    LOG_FILE="./logs/${hostname}_gpu${i}.log"
    PORT=$((BASE_PORT + i))
    CUDA_VISIBLE_DEVICES=$i nohup vllm \
        serve \
        $model_path \
        --limit-mm-per-prompt.video 0 \
        --max-model-len 16000 \
        --gpu-memory-utilization 0.7 \
        --host 0.0.0.0 \
        --port $PORT \
        > "$LOG_FILE" 2>&1 &
done