#!/bin/bash

# HELMET LLaMA3.1-8B-Instruct XFlex v6 评估脚本 - 8k到64k版本 (threshold 0.95, score_ratio 0.001)
echo "Running HELMET with LLaMA3.1-8B-Instruct XFlex v6 (8k-64k versions, threshold=0.95, score_ratio=0.001)"
echo "💡 v6 = golden ratio selection + temperature"

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

# XFlex v6参数
THRESHOLD=0.95
STRIDE=8
SCORE_RATIO=0.001
USE_SIMPLE=6  # v6版本：golden ratio selection + temperature

# 设置输出目录
export OUTPUT_DIR="llama_output/xflex_v6_threshold${THRESHOLD}_score${SCORE_RATIO}_short"
mkdir -p $OUTPUT_DIR

echo "Running 8k-64k versions with LLaMA3.1-8B-Instruct XFlex v6 (threshold=$THRESHOLD, stride=$STRIDE, score_ratio=$SCORE_RATIO, use_simple=$USE_SIMPLE)"
for task in "recall" "rag" "longqa" "summ" "icl" "rerank" "cite"; do
    echo "Running task: $task with LLaMA3.1 XFlex v6 (threshold=$THRESHOLD, score_ratio=$SCORE_RATIO, short version)"
    mkdir -p $OUTPUT_DIR/${task}_short
    python eval.py \
        --config configs/${task}_short.yaml \
        --model_name_or_path $MODEL_NAME \
        --attn_metric xflex \
        --attn_threshold $THRESHOLD \
        --attn_stride $STRIDE \
        --attn_score_ratio $SCORE_RATIO \
        --attn_use_simple $USE_SIMPLE \
        --tag llama_xflex_v6_threshold${THRESHOLD}_score${SCORE_RATIO}_short \
        --output_dir $OUTPUT_DIR/${task}_short
done

echo "LLaMA3.1 XFlex v6 short versions (threshold=$THRESHOLD, score_ratio=$SCORE_RATIO) evaluation completed! Results in $OUTPUT_DIR"
