#!/bin/bash

# HELMET GLM-4-9B-Chat Full FlashAttention 8k-64k 评估脚本
echo "Running HELMET with GLM-4-9B-Chat Full FlashAttention (8k-64k contexts)"

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

# 设置GLM-4-9B-Chat模型路径
MODEL_NAME=${1:-"/home/scratch.sarawang_ent/modelscope_cache/GLM/glm-4-9b-chat"}

# 设置输出目录
export OUTPUT_DIR="glm4_output/full_flashattention"
mkdir -p $OUTPUT_DIR

echo "Running 8k-64k versions with GLM-4-9B-Chat Full FlashAttention"
for task in "recall_short" "rag_short" "longqa_short" "summ_short" "icl_short" "rerank_short" "cite_short"; do
    echo "Running task: $task with GLM-4 Full FlashAttention"
    mkdir -p $OUTPUT_DIR/$task
    python eval.py \
        --config configs/${task}.yaml \
        --model_name_or_path $MODEL_NAME \
        --attn_metric full \
        --tag glm4_full_flashattention \
        --output_dir $OUTPUT_DIR/$task
done

echo "GLM-4 Full FlashAttention 8k-64k evaluation completed! Results in $OUTPUT_DIR"
