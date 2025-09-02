#!/bin/bash

# HELMET Qwen2.5-7B-Instruct 配置测试脚本
echo "Testing HELMET Qwen2.5-7B-Instruct configuration"

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

# 创建缓存目录
mkdir -p "$HF_HOME"
mkdir -p "$TRANSFORMERS_CACHE"
mkdir -p "$HF_DATASETS_CACHE"
mkdir -p "$HF_HUB_CACHE"
mkdir -p "$TORCH_HOME"
mkdir -p "$MODELSCOPE_CACHE"

# 设置Qwen2.5-7B-Instruct模型路径
MODEL_NAME="/home/scratch.sarawang_ent/modelscope_cache/Qwen/Qwen2.5-7B-Instruct"

# 设置输出目录
export OUTPUT_DIR="qwen_output/test"
mkdir -p $OUTPUT_DIR

echo "🧪 Testing Qwen2.5-7B-Instruct with a simple task"
echo "📍 Model path: $MODEL_NAME"
echo "📁 Output directory: $OUTPUT_DIR"

# 只运行一个简单的任务作为测试
echo "Running recall task with Qwen2.5 full attention (test)"
mkdir -p $OUTPUT_DIR/recall_demo
python eval.py \
    --config configs/recall_demo.yaml \
    --model_name_or_path $MODEL_NAME \
    --attn_metric full \
    --tag qwen_test \
    --output_dir $OUTPUT_DIR/recall_demo

echo "🎉 Qwen2.5 configuration test completed! Check results in $OUTPUT_DIR"
echo "如果没有错误，说明 Qwen2.5-7B-Instruct 配置成功！"
