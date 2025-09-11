#!/bin/bash

# HELMET LLaMA3.1-8B-Instruct FlexPrefill 评估脚本 - gamma 0.95, tau 0.1
echo "Running HELMET with LLaMA3.1-8B-Instruct FlexPrefill (gamma=0.95, tau=0.1)"

# 切换到HELMET根目录
cd "$(dirname "$0")/.."

# 🎯 设置自定义缓存路径到项目目录下（避免占用home空间）
export HF_HOME="/home/scratch.sarawang_ent/project/HELMET/.hf_cache"
export TRANSFORMERS_CACHE="/home/scratch.sarawang_ent/project/HELMET/.hf_cache/transformers"
export HF_DATASETS_CACHE="/home/scratch.sarawang_ent/project/HELMET/.hf_cache/datasets"
export HF_HUB_CACHE="/home/scratch.sarawang_ent/project/HELMET/.hf_cache/hub"
export TORCH_HOME="/home/scratch.sarawang_ent/project/HELMET/.torch_cache"

# 🎯 设置ModelScope魔塔镜像
export MODELSCOPE_CACHE="/home/scratch.sarawang_ent/modelscope_cache"
export USE_MODELSCOPE_HUB=1

echo "💡 重定向所有缓存到项目目录，避免填满 ~/.cache"

# 创建缓存目录
mkdir -p "$HF_HOME"
mkdir -p "$TRANSFORMERS_CACHE"
mkdir -p "$HF_DATASETS_CACHE"
mkdir -p "$HF_HUB_CACHE"
mkdir -p "$TORCH_HOME"
mkdir -p "$MODELSCOPE_CACHE"

echo "🗂️ Cache directories set to:"
echo "  HF_HOME: $HF_HOME"
echo "  HF_DATASETS_CACHE: $HF_DATASETS_CACHE"
echo "  HF_HUB_CACHE: $HF_HUB_CACHE"
echo "  TORCH_HOME: $TORCH_HOME"
echo "  MODELSCOPE_CACHE: $MODELSCOPE_CACHE"

# 设置LLaMA3.1-8B-Instruct模型路径
MODEL_NAME=${1:-"/home/scratch.sarawang_ent/modelscope_cache/LLM-Research/Meta-Llama-3.1-8B-Instruct"}

# FlexPrefill参数
GAMMA=0.95
TAU=0.1

# 设置输出目录
export OUTPUT_DIR="llama_output/flex_gamma${GAMMA}_tau${TAU}"
mkdir -p $OUTPUT_DIR

echo "Running 128k versions with LLaMA3.1-8B-Instruct FlexPrefill (gamma=$GAMMA, tau=$TAU)"
for task in "rerank" "cite"; do
    echo "Running task: $task with LLaMA3.1 FlexPrefill (gamma=$GAMMA, tau=$TAU)"
    mkdir -p $OUTPUT_DIR/$task
    python eval.py \
        --config configs/${task}.yaml \
        --model_name_or_path $MODEL_NAME \
        --attn_metric flex \
        --attn_gamma $GAMMA \
        --attn_tau $TAU \
        --tag flex_gamma${GAMMA}_tau${TAU} \
        --output_dir $OUTPUT_DIR/$task
done

echo "LLaMA3.1 FlexPrefill (gamma=$GAMMA, tau=$TAU) evaluation completed! Results in $OUTPUT_DIR"
