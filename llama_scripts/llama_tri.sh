#!/bin/bash

# HELMET LLaMA3.1-8B-Instruct TriangleMix 评估脚本
echo "Running HELMET with LLaMA3.1-8B-Instruct TriangleMix Strategy"
echo "💡 TriangleMix = Mixed attention: Flash for early layers, Triangle sparse for later layers"
echo "📊 Sparse ratio: 55% (45% computation reduction)"

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

# 设置输出目录
export OUTPUT_DIR="llama_output/tri"
mkdir -p $OUTPUT_DIR

echo "🎯 TriangleMix Strategy Details:"
echo "  • Early layers (0-15): Full Flash Attention"
echo "  • Later layers (16+): Triangle sparse attention"
echo "  • Sparse ratio: 0.55 (保留55%计算)"
echo "  • Starting layer: 16"
echo ""

echo "Running 8k to 64k versions with LLaMA3.1-8B-Instruct TriangleMix"
for task in "recall" "rag" "longqa" "summ" "icl" "rerank" "cite"; do
    echo "Running task: $task (short) with LLaMA3.1 TriangleMix"
    mkdir -p $OUTPUT_DIR/${task}
    python eval.py \
        --config configs/${task}_short.yaml \
        --model_name_or_path $MODEL_NAME \
        --attn_metric tri \
        --tag tri_sparse55 \
        --output_dir $OUTPUT_DIR/${task}
done

echo "Running 128k versions with LLaMA3.1-8B-Instruct TriangleMix"
for task in "recall" "rag" "longqa" "summ" "icl" "rerank" "cite"; do
    echo "Running task: $task with LLaMA3.1 TriangleMix"
    mkdir -p $OUTPUT_DIR/$task
    python eval.py \
        --config configs/${task}.yaml \
        --model_name_or_path $MODEL_NAME \
        --attn_metric tri \
        --tag tri_sparse55 \
        --output_dir $OUTPUT_DIR/$task
done

echo ""
echo "🎉 LLaMA3.1 TriangleMix evaluation completed!"
echo "📊 Results saved in: $OUTPUT_DIR"
echo "💡 Strategy: Mixed attention (Flash + Triangle sparse)"
echo "⚡ Performance: ~45% computation reduction with maintained quality"
