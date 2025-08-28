#!/bin/bash

# HELMET Qwen3-30B-A3B-Instruct MoE FlashInfer Full Attention 评估脚本
echo "Running HELMET with Qwen3-30B-A3B-Instruct MoE Full Attention"

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

# 🚨 MoE模型特殊配置
export CUDA_VISIBLE_DEVICES=0  # 根据GPU配置调整
export OMP_NUM_THREADS=4       # 控制CPU线程数
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512  # 内存分片优化

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

# 设置Qwen3-30B-A3B-Instruct模型路径
MODEL_NAME=${1:-"/home/scratch.sarawang_ent/modelscope_cache/Qwen/Qwen3-30B-A3B-Instruct-2507"}

# 设置输出目录
export OUTPUT_DIR="moe_qwen3_output/full_flashinfer"
mkdir -p $OUTPUT_DIR

echo "🔧 GPU Memory check: $(nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits | head -1)"
echo "📍 Model: Qwen3-30B-A3B-Instruct MoE"
echo "🎯 Attention: Full FlashInfer"

echo "Running 8k-64k versions with Qwen3-30B-A3B-Instruct MoE Full Attention"
for task in "recall" "rag" "longqa" "summ" "icl" "rerank" "cite"; do
    echo "Running task: $task (8k-64k) with Qwen3 MoE full attention"
    python eval.py \
        --config configs/${task}_short.yaml \
        --model_name_or_path $MODEL_NAME \
        --attn_metric full \
        --tag qwen3_moe_full_flashinfer \
        --output_dir $OUTPUT_DIR/$task \
        --max_test_samples 50 \
        --num_workers 2
done

echo "Running 128k versions with Qwen3-30B-A3B-Instruct MoE Full Attention"
for task in "recall" "rag" "longqa" "summ" "icl" "rerank" "cite"; do
    echo "Running task: $task with Qwen3 MoE full attention"
    python eval.py \
        --config configs/${task}.yaml \
        --model_name_or_path $MODEL_NAME \
        --attn_metric full \
        --tag qwen3_moe_full_flashinfer \
        --output_dir $OUTPUT_DIR/$task \
        --max_test_samples 50 \
        --num_workers 2
done

echo "Qwen3-30B MoE Full attention evaluation completed! Results in $OUTPUT_DIR"
