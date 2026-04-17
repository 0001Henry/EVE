# export HF_ENDPOINT="https://hf-mirror.com"
export HF_HOME="your_path_to_data/VLMEvalKit/huggingface"
export LMUData="your_path_to_data/VLMEvalKit/LMUData"
export CUDA_VISIBLE_DEVICES=0,1,2,3


export VLM_EVAL_MODEL_PATH="your_model_path_here"
export VLM_EVAL_MODEL="Qwen3-VL-8B-Instruct"

export USE_POST_PROCESSING=false
python run.py --data MMVet --model $VLM_EVAL_MODEL --verbose  --reuse
export USE_POST_PROCESSING=true
python run.py --data MMStar HallusionBench MathVista_MINI BLINK VisuLogic --model $VLM_EVAL_MODEL --verbose --reuse