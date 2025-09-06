#!/bin/bash

# GPT-4评估 - Llama Long QA 任务
echo "🤖 Running GPT-4 evaluation for Llama Long QA tasks..."

# 切换到HELMET根目录
cd "$(dirname "$0")/.."

# 检查 OPENAI_API_KEY
if [ -z "$OPENAI_API_KEY" ]; then
    echo "❌ 错误: 请设置 OPENAI_API_KEY 环境变量"
    echo "   export OPENAI_API_KEY='your-api-key-here'"
    exit 1
fi

echo "✅ OpenAI API Key 已设置"

# 需要评估的Llama模型配置
models=(
    "full_flashinfer"
    "xattn_threshold0.95" 
    "xattn_v6_threshold0.95"
    "flex_gamma0.95_tau0.1"
)

for model in "${models[@]}"; do
    echo "🔗 创建临时软链接 for $model..."
    mkdir -p output
    ln -sfn ../../llama_output/$model output/llama_$model
    
    echo "📊 评估 Llama $model long QA 结果..."
    
    python scripts/eval_gpt4_longqa.py \
        --model_to_check "llama_$model" \
        --tag "$model" \
        --num_shards 1 \
        --shard_idx 0
    
    # 🧹 清理临时软链接
    echo "🧹 清理临时软链接 for $model..."
    rm -f output/llama_$model
done

echo "✅ Llama Long QA GPT-4 评估完成!"
echo "📁 检查结果文件: llama_output/*/longqa/*-gpt4eval_o.json"
