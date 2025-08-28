#!/usr/bin/env python3
"""
收集Qwen3-30B-A3B-Instruct MoE XAT attention方案的HELMET评估结果
专门用于汇总Qwen3-30B MoE的full, xattn, flex, xflex等attention机制的结果
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import yaml
from dataclasses import dataclass, asdict
from tqdm import tqdm

# 添加HELMET根目录到路径
script_dir = os.path.dirname(os.path.abspath(__file__))
helmet_root = os.path.dirname(script_dir)
sys.path.append(helmet_root)

# 导入collect_results的基础类和函数
from scripts.collect_results import arguments, dataset_to_metrics, custom_avgs

def main():
    """收集Qwen3-30B MoE XAT attention结果"""
    
    # 🎯 Qwen3-30B MoE XAT Attention配置
    qwen3_moe_configs = [
        # Full FlashInfer Attention
        {"model": "Qwen3-30B-A3B-Instruct", "tag": "qwen3_moe_full_flashinfer", 
         "output_dir": "moe_qwen3_output/full_flashinfer", "attention": "full"},
        
        # XAttention - 适合30B MoE的threshold
        {"model": "Qwen3-30B-A3B-Instruct", "tag": "qwen3_moe_xattn_threshold0.9", 
         "output_dir": "moe_qwen3_output/xattn_threshold0.9", "attention": "xattn", "threshold": 0.9},
        {"model": "Qwen3-30B-A3B-Instruct", "tag": "qwen3_moe_xattn_threshold0.95", 
         "output_dir": "moe_qwen3_output/xattn_threshold0.95", "attention": "xattn", "threshold": 0.95},
        
        # XFlex - MoE优化配置
        {"model": "Qwen3-30B-A3B-Instruct", "tag": "qwen3_moe_xflex_threshold0.95_scoreratio0.95", 
         "output_dir": "moe_qwen3_output/xflex_threshold0.95_scoreratio0.95", "attention": "xflex", 
         "threshold": 0.95, "score_ratio": 0.95},
        {"model": "Qwen3-30B-A3B-Instruct", "tag": "qwen3_moe_xflex_threshold0.9_scoreratio0.9", 
         "output_dir": "moe_qwen3_output/xflex_threshold0.9_scoreratio0.9", "attention": "xflex", 
         "threshold": 0.9, "score_ratio": 0.9},
    ]

    # 📋 数据集配置文件
    config_files = [
        "configs/recall.yaml", "configs/recall_short.yaml", 
        "configs/rag.yaml", "configs/rag_short.yaml", 
        "configs/longqa.yaml", "configs/longqa_short.yaml", 
        "configs/summ.yaml", "configs/summ_short.yaml", 
        "configs/rerank.yaml", "configs/rerank_short.yaml", 
        "configs/icl.yaml", "configs/icl_short.yaml", 
        "configs/cite.yaml", "configs/cite_short.yaml", 
    ]

    # 解析数据集配置
    dataset_configs = []
    
    for file in config_files:
        config_path = os.path.join(helmet_root, file)
        if not os.path.exists(config_path):
            print(f"⚠️ 配置文件不存在: {config_path}")
            continue
            
        c = yaml.safe_load(open(config_path))
        
        if isinstance(c["generation_max_length"], int):
            c["generation_max_length"] = ",".join([str(c["generation_max_length"])] * len(c["datasets"].split(",")))
        for d, t, l, g in zip(c['datasets'].split(','), c['test_files'].split(','), c['input_max_length'].split(','), c['generation_max_length'].split(',')):
            # 处理空的test_files
            if t.strip() == '':
                test_name = ''
            else:
                test_name = os.path.basename(os.path.splitext(t)[0])
            
            # 包含所有长度配置
            input_len = int(l.strip())
            dataset_configs.append({
                "dataset": d.strip(), 
                "test_name": test_name, 
                "input_max_length": input_len, 
                "generation_max_length": int(g.strip()), 
                "max_test_samples": c['max_test_samples'], 
                'use_chat_template': c['use_chat_template'], 
                'shots': c['shots']
            })

    print(f"📊 找到 {len(dataset_configs)} 个数据集配置")
    print(f"🎯 将处理 {len(qwen3_moe_configs)} 个Qwen3-30B MoE XAT attention配置")

    # 收集结果
    failed_paths = []
    df = []
    
    for config in tqdm(qwen3_moe_configs, desc="收集Qwen3-30B MoE XAT结果"):
        args = arguments()
        args.tag = config["tag"]
        args.output_dir = config["output_dir"]
        args.model = config["model"]
        
        print(f"\n📁 处理配置: {config['attention']} - {config['tag']}")
        print(f"   输出目录: {args.output_dir}")
        
        # 检查输出目录是否存在
        if not os.path.exists(args.output_dir):
            print(f"⚠️ 输出目录不存在: {args.output_dir}")
            continue
        
        config_found_results = 0
        for dataset in dataset_configs:
            args.update(dataset)
            
            metric = args.get_averaged_metric()
            dsimple, mnames = args.get_metric_name()

            if metric is None:
                failed_paths.append(args.get_path())
                continue
                
            config_found_results += 1
            for k, m in metric.items():
                df.append({
                    **asdict(args), 
                    **config,
                    "metric name": k, 
                    "metric": m, 
                    "dataset_simple": dsimple + " " + k, 
                    "test_data": f"{args.dataset}-{args.test_name}-{args.input_max_length}"
                })
        
        print(f"   ✅ 找到 {config_found_results} 个有效结果")

    if not df:
        print("❌ 没有找到任何有效结果！请检查:")
        print("   1. 输出目录是否存在")
        print("   2. tag名称是否正确")
        print("   3. 是否有完成的评估任务")
        print("   4. 是否需要运行GPT-4评估")
        return

    # 生成汇总表格
    print(f"\n📈 生成汇总表格...")
    all_df = pd.DataFrame(df)
    
    # 创建透视表
    lf_df = all_df.pivot_table(
        index=["input_max_length", "attention", "tag"], 
        columns="dataset_simple", 
        values="metric", 
        sort=False
    )
    lf_df = lf_df.reset_index()

    # 计算自定义平均值
    for k, v in custom_avgs.items():
        available_cols = [col for col in v if col in lf_df.columns]
        if available_cols:
            lf_df[k] = lf_df[available_cols].mean(axis=1)
        else:
            print(f"⚠️ 跳过 {k}: 缺少必要的列")

    # 保存结果
    output_file = os.path.join(helmet_root, "qwen3_moe_results_summary.csv")
    lf_df.to_csv(output_file, index=False)
    
    print(f"✅ 结果已保存到: {output_file}")
    print(f"📊 共处理了 {len(df)} 个数据点")
    
    # 显示预览
    print("\n📋 结果预览:")
    available_custom_cols = [col for col in custom_avgs.keys() if col in lf_df.columns]
    if available_custom_cols:
        print(lf_df[['input_max_length', 'attention', 'tag'] + available_custom_cols].to_string(index=False))

    if failed_paths:
        print(f"\n⚠️ 以下 {len(failed_paths)} 个路径的结果未找到:")
        for path in failed_paths[:10]:  # 只显示前10个
            print(f"   {path}")
        if len(failed_paths) > 10:
            print(f"   ... 还有 {len(failed_paths)-10} 个")

    return lf_df, failed_paths

if __name__ == "__main__":
    main()
