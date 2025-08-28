#!/bin/bash

# GPT-4评估 - Qwen Summarization 任务
echo "🤖 Running GPT-4 evaluation for Qwen Summarization tasks..."

# 切换到HELMET根目录
cd "$(dirname "$0")/.."

# 检查 OPENAI_API_KEY
if [ -z "$OPENAI_API_KEY" ]; then
    echo "❌ 错误: 请设置 OPENAI_API_KEY 环境变量"
    echo "   export OPENAI_API_KEY='your-api-key-here'"
    exit 1
fi

echo "✅ OpenAI API Key 已设置"

# 🔧 临时软链接以适配GPT-4评估脚本的目录结构
echo "🔗 创建临时软链接..."
mkdir -p output
ln -sfn ../../qwen_output/full_flashinfer output/qwen_full_flashinfer

# 运行 GPT-4 Summarization 评估
echo "📊 评估 Qwen summarization 结果..."

python scripts/eval_gpt4_summ.py \
    --model_to_check "qwen_full_flashinfer" \
    --tag "qwen_full_flashinfer" \
    --num_shards 1 \
    --shard_idx 0

# 🧹 清理临时软链接
echo "🧹 清理临时软链接..."
rm -f output/qwen_full_flashinfer

echo "✅ Qwen Summarization GPT-4 评估完成!"
echo "📁 检查结果文件: qwen_output/full_flashinfer/summ/*-gpt4eval_o.json"
