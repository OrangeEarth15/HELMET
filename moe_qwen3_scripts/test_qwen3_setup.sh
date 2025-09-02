#!/bin/bash

# HELMET Qwen3-30B-A3B-Instruct MoE 配置测试脚本
echo "Testing HELMET Qwen3-30B-A3B-Instruct MoE configuration 🧪"

# 切换到HELMET根目录
cd "$(dirname "$0")/.."

# 🎯 设置环境变量
export MODELSCOPE_CACHE="/home/scratch.sarawang_ent/modelscope_cache"
export MODELSCOPE_HUB_CACHE="/home/scratch.sarawang_ent/modelscope_cache"
export USE_MODELSCOPE_HUB=1

# 🎯 设置自定义缓存路径
export HF_HOME="/home/scratch.sarawang_ent/project/HELMET/.hf_cache"
export TRANSFORMERS_CACHE="/home/scratch.sarawang_ent/project/HELMET/.hf_cache/transformers"
export HF_DATASETS_CACHE="/home/scratch.sarawang_ent/project/HELMET/.hf_cache/datasets"
export HF_HUB_CACHE="/home/scratch.sarawang_ent/project/HELMET/.hf_cache/hub"
export TORCH_HOME="/home/scratch.sarawang_ent/project/HELMET/.torch_cache"

# 设置Qwen3-30B-A3B-Instruct模型路径
MODEL_NAME="/home/scratch.sarawang_ent/modelscope_cache/Qwen/Qwen3-30B-A3B-Instruct-2507"
export OUTPUT_DIR="moe_qwen3_output/test"

echo "🧪 Testing Qwen3-30B-A3B-Instruct MoE with a simple task"
echo "📍 Model path: $MODEL_NAME"
echo "📁 Output directory: $OUTPUT_DIR"
echo "🔧 GPU Memory check: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)MB"

# 创建输出目录
mkdir -p $OUTPUT_DIR/recall_demo

echo "Running recall task with Qwen3-30B MoE full attention (test)"

# 运行简单的recall测试 - 使用MoE专用配置
python eval.py \
    --config configs/qwen3_moe_recall_demo.yaml \
    --model_name_or_path $MODEL_NAME \
    --attn_metric full \
    --tag qwen3_moe_test \
    --output_dir $OUTPUT_DIR/recall_demo

echo "🎉 Qwen3-30B MoE configuration test completed! Check results in $OUTPUT_DIR"
echo "如果没有错误，说明 Qwen3-30B-A3B-Instruct MoE 配置成功！"

