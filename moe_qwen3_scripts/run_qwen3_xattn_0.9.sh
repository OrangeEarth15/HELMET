#!/bin/bash

# HELMET Qwen3-30B-A3B-Instruct MoE XAttention (threshold=0.9) 评估脚本
echo "Running HELMET with Qwen3-30B-A3B-Instruct MoE XAttention (threshold=0.9)"

# 切换到HELMET根目录
cd "$(dirname "$0")/.."

# 🎯 设置缓存和环境变量
export HF_HOME="/home/scratch.sarawang_ent/project/HELMET/.hf_cache"
export TRANSFORMERS_CACHE="/home/scratch.sarawang_ent/project/HELMET/.hf_cache/transformers"
export HF_DATASETS_CACHE="/home/scratch.sarawang_ent/project/HELMET/.hf_cache/datasets"
export HF_HUB_CACHE="/home/scratch.sarawang_ent/project/HELMET/.hf_cache/hub"
export TORCH_HOME="/home/scratch.sarawang_ent/project/HELMET/.torch_cache"
export MODELSCOPE_CACHE="/home/scratch.sarawang_ent/modelscope_cache"
export USE_MODELSCOPE_HUB=1

# 🚨 MoE模型特殊配置
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# 设置Qwen3-30B-A3B-Instruct模型路径
MODEL_NAME=${1:-"/home/scratch.sarawang_ent/modelscope_cache/Qwen/Qwen3-30B-A3B-Instruct-2507"}

# 设置输出目录
export OUTPUT_DIR="moe_qwen3_output/xattn_threshold0.9"
mkdir -p $OUTPUT_DIR

echo "🔧 GPU Memory check: $(nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits | head -1)"
echo "📍 Model: Qwen3-30B-A3B-Instruct MoE"  
echo "🎯 Attention: XAttention (threshold=0.9, stride=16)"

echo "Running 8k-64k versions with Qwen3-30B-A3B-Instruct MoE XAttention"
for task in "recall" "rag" "longqa" "summ" "icl" "rerank" "cite"; do
    echo "Running task: $task (8k-64k) with Qwen3 MoE xattn"
    python eval.py \
        --config configs/${task}_short.yaml \
        --model_name_or_path $MODEL_NAME \
        --attn_metric xattn \
        --attn_threshold 0.9 \
        --attn_stride 16 \
        --tag qwen3_moe_xattn_threshold0.9 \
        --output_dir $OUTPUT_DIR/$task \
        --max_test_samples 50 \
        --num_workers 2
done

echo "Running 128k versions with Qwen3-30B-A3B-Instruct MoE XAttention"
for task in "recall" "rag" "longqa" "summ" "icl" "rerank" "cite"; do
    echo "Running task: $task with Qwen3 MoE xattn"
    python eval.py \
        --config configs/${task}.yaml \
        --model_name_or_path $MODEL_NAME \
        --attn_metric xattn \
        --attn_threshold 0.9 \
        --attn_stride 16 \
        --tag qwen3_moe_xattn_threshold0.9 \
        --output_dir $OUTPUT_DIR/$task \
        --max_test_samples 50 \
        --num_workers 2
done

echo "Qwen3-30B MoE XAttention evaluation completed! Results in $OUTPUT_DIR"
