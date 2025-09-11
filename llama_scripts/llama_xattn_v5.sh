#!/bin/bash

# HELMET LLaMA3.1-8B-Instruct XAttention v5 评估脚本 - threshold 0.95
echo "Running HELMET with LLaMA3.1-8B-Instruct XAttention v5 (threshold=0.95)"
echo "💡 v5 = head robin"

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

# XAttention v4参数
THRESHOLD=0.95
STRIDE=8
USE_SIMPLE=5  # v5版本：head robin

# 设置输出目录
export OUTPUT_DIR="llama_output/xattn_v5_threshold${THRESHOLD}"
mkdir -p $OUTPUT_DIR

echo "Running 8k to 64k versions with LLaMA3.1-8B-Instruct XAttention v5 (threshold=$THRESHOLD, stride=$STRIDE, use_simple=$USE_SIMPLE)"
for task in "recall" "rag" "longqa" "summ" "icl" "rerank" "cite"; do
    echo "Running task: $task (short) with LLaMA3.1 XAttention v5 (threshold=$THRESHOLD)"
    mkdir -p $OUTPUT_DIR/$task
    python eval.py \
        --config configs/${task}_short.yaml \
        --model_name_or_path $MODEL_NAME \
        --attn_metric xattn \
        --attn_threshold $THRESHOLD \
        --attn_stride $STRIDE \
        --attn_use_simple $USE_SIMPLE  \
        --tag xattn_v5_threshold${THRESHOLD} \
        --output_dir $OUTPUT_DIR/$task
done


echo "Running 128k versions with LLaMA3.1-8B-Instruct XAttention v5 (threshold=$THRESHOLD, stride=$STRIDE, use_simple=$USE_SIMPLE)"
for task in "recall" "rag" "longqa" "summ" "icl" "rerank" "cite"; do
    echo "Running task: $task with LLaMA3.1 XAttention v5 (threshold=$THRESHOLD)"
    mkdir -p $OUTPUT_DIR/$task
    python eval.py \
        --config configs/${task}.yaml \
        --model_name_or_path $MODEL_NAME \
        --attn_metric xattn \
        --attn_threshold $THRESHOLD \
        --attn_stride $STRIDE \
        --attn_use_simple $USE_SIMPLE \
        --tag xattn_v5_threshold${THRESHOLD} \
        --output_dir $OUTPUT_DIR/$task
done

echo "LLaMA3.1 XAttention v5 (threshold=$THRESHOLD) evaluation completed! Results in $OUTPUT_DIR"
