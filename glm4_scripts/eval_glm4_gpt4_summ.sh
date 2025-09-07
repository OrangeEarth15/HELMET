#!/bin/bash

# GLM4 Summarization GPT-4评估脚本
echo "Running GPT-4 evaluation for GLM4 Summarization results"

# 切换到HELMET根目录
cd "$(dirname "$0")/.."

# 🎯 设置自定义缓存路径到项目目录下
export HF_HOME="/home/scratch.sarawang_ent/project/HELMET/.hf_cache"
export TRANSFORMERS_CACHE="/home/scratch.sarawang_ent/project/HELMET/.hf_cache/transformers"
export HF_DATASETS_CACHE="/home/scratch.sarawang_ent/project/HELMET/.hf_cache/datasets"
export HF_HUB_CACHE="/home/scratch.sarawang_ent/project/HELMET/.hf_cache/hub"
export TORCH_HOME="/home/scratch.sarawang_ent/project/HELMET/.torch_cache"

# 🎯 设置ModelScope魔塔镜像
export MODELSCOPE_CACHE="/home/scratch.sarawang_ent/modelscope_cache"
export USE_MODELSCOPE_HUB=1

echo "🔍 Running GPT-4 evaluation for GLM4 Summarization results..."

# GLM4配置列表
declare -a configs=(
    "glm4_output/full_flashattention/summ:glm4_full_flashattention"
    "glm4_output/xattn_v6_threshold0.95/summ:glm4_xattn_v6_threshold0.95"
    "glm4_output/flex_gamma0.95_tau0.1/summ:glm4_flex_gamma0.95_tau0.1"
)

# 遍历每个配置并运行GPT-4评估
for config in "${configs[@]}"; do
    IFS=':' read -r output_dir tag <<< "$config"
    
    echo "📊 Evaluating: $tag"
    echo "📁 Output directory: $output_dir"
    
    if [ -d "$output_dir" ]; then
        python scripts/eval_gpt4_summ.py \
            --output_dir "$output_dir" \
            --tag "$tag"
        echo "✅ Completed evaluation for $tag"
    else
        echo "⚠️ Directory not found: $output_dir"
    fi
    
    echo "---"
done

echo "🎉 GLM4 Summarization GPT-4 evaluation completed!"
