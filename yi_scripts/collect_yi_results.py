#!/usr/bin/env python3
"""
收集Yi-9B-200K模型在HELMET基准测试中的结果
支持所有attention机制：full, xattn, xattn_v6, flex, xflex_v6
"""

import os
import json
import csv
from collections import defaultdict
import argparse

def collect_results(base_dir="yi_output"):
    """收集所有Yi-9B-200K评估结果，支持所有attention机制和序列长度"""
    results = defaultdict(dict)
    
    # 支持的attention机制
    supported_methods = [
        'full_short', 'full_128k',
        'xattn_threshold0.95_short', 'xattn_threshold0.95_128k',
        'xattn_v6_threshold0.95_short', 'xattn_v6_threshold0.95_128k',  
        'xflex_v6_threshold0.95_score0.001_short', 'xflex_v6_threshold0.95_score0.001_128k',
        'flex_gamma0.9_tau0.1_short', 'flex_gamma0.9_tau0.1_128k'
    ]
    
    # 遍历所有输出目录
    for method_dir in os.listdir(base_dir):
        method_path = os.path.join(base_dir, method_dir)
        if not os.path.isdir(method_path):
            continue
            
        print(f"Processing {method_dir}...")
        
        # 检查是否是支持的方法
        if method_dir not in [m.replace('_short', '').replace('_128k', '') for m in supported_methods]:
            print(f"  ⚠️  Unknown method: {method_dir}, will still process...")
        
        # 遍历每个任务目录
        for task_dir in os.listdir(method_path):
            task_path = os.path.join(method_path, task_dir)
            if not os.path.isdir(task_path):
                continue
                
            # 提取任务名和序列长度信息
            if task_dir.endswith('_short'):
                task_name = task_dir.replace('_short', '')
                seq_length = '8k-64k'
            elif task_dir.endswith('_128k'):
                task_name = task_dir.replace('_128k', '')
                seq_length = '128k'
            else:
                task_name = task_dir
                seq_length = 'unknown'
            
            # 查找.score文件
            for file in os.listdir(task_path):
                if file.endswith('.json.score'):
                    score_file = os.path.join(task_path, file)
                    try:
                        with open(score_file, 'r', encoding='utf-8') as f:
                            scores = json.load(f)
                        
                        # 存储结果，包含序列长度信息
                        key = f"{method_dir}_{task_name}_{seq_length}"
                        results[key] = scores
                        
                        print(f"  Found scores for {task_name} ({seq_length}): {list(scores.keys())}")
                        
                    except Exception as e:
                        print(f"  Error reading {score_file}: {e}")
    
    return results

def save_results_csv(results, output_file="yi_9b_200k_results_summary.csv"):
    """将结果保存为CSV文件"""
    if not results:
        print("No results found!")
        return
    
    # 获取所有可能的指标
    all_metrics = set()
    for scores in results.values():
        all_metrics.update(scores.keys())
    
    all_metrics = sorted(list(all_metrics))
    
    # 写入CSV
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # 写入表头
        header = ['Method_Task_SeqLength'] + all_metrics
        writer.writerow(header)
        
        # 写入数据
        for key, scores in sorted(results.items()):
            row = [key]
            for metric in all_metrics:
                row.append(scores.get(metric, ''))
            writer.writerow(row)
    
    print(f"Results saved to {output_file}")

def print_summary(results):
    """打印结果摘要"""
    print("\n" + "="*80)
    print("Yi-9B-200K HELMET Evaluation Results Summary")
    print("="*80)
    
    # 按方法和序列长度分组
    methods = defaultdict(list)
    for key in results.keys():
        parts = key.split('_')
        if len(parts) >= 3:
            # 处理复杂的方法名，如 xflex_v6_threshold0.95_score0.001
            method_parts = []
            task_part = None
            seq_length = parts[-1]  # 最后一部分是序列长度
            
            # 找到任务名的位置
            for i, part in enumerate(parts[:-1]):  # 除了最后的序列长度
                if part in ['recall', 'rag', 'longqa', 'summ', 'icl', 'rerank', 'cite']:
                    task_part = part
                    method_parts = parts[:i]
                    break
            
            if task_part and method_parts:
                method = '_'.join(method_parts)
                task_with_seq = f"{task_part}_{seq_length}"
                methods[method].append((task_with_seq, results[key]))
    
    for method, task_results in sorted(methods.items()):
        print(f"\n📊 {method.upper()}")
        print("-" * 60)
        
        for task, scores in sorted(task_results):
            print(f"  {task:15s}: ", end="")
            
            # 显示主要指标
            main_metrics = []
            if 'exact_match' in scores:
                main_metrics.append(f"EM: {scores['exact_match']:.3f}")
            if 'f1' in scores:
                main_metrics.append(f"F1: {scores['f1']:.3f}")
            if 'rouge_l' in scores:
                main_metrics.append(f"ROUGE-L: {scores['rouge_l']:.3f}")
            if 'accuracy' in scores:
                main_metrics.append(f"Acc: {scores['accuracy']:.3f}")
            
            if main_metrics:
                print(" | ".join(main_metrics))
            else:
                # 显示所有可用指标
                metrics_str = " | ".join([f"{k}: {v:.3f}" if isinstance(v, (int, float)) else f"{k}: {v}" 
                                        for k, v in scores.items() if k != 'num_samples'])
                print(metrics_str[:100] + "..." if len(metrics_str) > 100 else metrics_str)

def main():
    parser = argparse.ArgumentParser(description="Collect Yi-9B-200K HELMET evaluation results")
    parser.add_argument("--base_dir", default="yi_output", help="Base directory containing results")
    parser.add_argument("--output_file", default="yi_9b_200k_results_summary.csv", help="Output CSV file")
    
    args = parser.parse_args()
    
    # 收集结果
    results = collect_results(args.base_dir)
    
    if results:
        # 保存CSV
        save_results_csv(results, args.output_file)
        
        # 打印摘要
        print_summary(results)
        
        print(f"\n✅ Total {len(results)} result sets collected")
    else:
        print("❌ No results found in the specified directory")

if __name__ == "__main__":
    main()
