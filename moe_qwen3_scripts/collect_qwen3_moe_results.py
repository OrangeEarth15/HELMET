#!/usr/bin/env python3
"""
HELMET Qwen3 MoE 结果收集和分析脚本

功能:
1. 收集所有Qwen3 MoE实验结果
2. 生成统一的CSV报告
3. 对比不同attention方法的性能
4. 生成MoE特有的分析报告

使用方法:
    python moe_qwen3_scripts/collect_qwen3_moe_results.py [--output output.csv]
"""

import os
import json
import pandas as pd
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def find_result_files(base_dir: str = "moe_qwen3_output") -> List[str]:
    """递归查找所有结果文件"""
    result_files = []
    base_path = Path(base_dir)
    
    if not base_path.exists():
        logger.warning(f"输出目录不存在: {base_dir}")
        return result_files
    
    # 查找所有.json文件（但不包括.score文件）
    for json_file in base_path.rglob("*.json"):
        if not str(json_file).endswith(".score"):
            result_files.append(str(json_file))
    
    logger.info(f"发现 {len(result_files)} 个结果文件")
    return result_files

def parse_filename(filepath: str) -> Dict[str, str]:
    """从文件路径解析实验参数"""
    parts = filepath.split("/")
    filename = parts[-1]
    
    info = {
        "filepath": filepath,
        "method_dir": parts[-3] if len(parts) >= 3 else "unknown",
        "task_dir": parts[-2] if len(parts) >= 2 else "unknown", 
        "filename": filename
    }
    
    # 从目录名解析attention方法和参数
    method_dir = info["method_dir"]
    if "full" in method_dir:
        info["attention_method"] = "full"
        info["threshold"] = None
        info["score_ratio"] = None
        info["gamma"] = None
        info["tau"] = None
        info["stride"] = None
    elif "xattn" in method_dir:
        info["attention_method"] = "xattn"
        # 解析threshold
        if "threshold" in method_dir:
            parts = method_dir.split("threshold")
            if len(parts) > 1:
                info["threshold"] = float(parts[1].replace("_", "."))
        info["stride"] = 16  # 默认值
    elif "xflex" in method_dir:
        info["attention_method"] = "xflex"
        # 解析threshold和score_ratio
        if "threshold" in method_dir and "scoreratio" in method_dir:
            parts = method_dir.split("_")
            for part in parts:
                if part.startswith("threshold"):
                    info["threshold"] = float(part.replace("threshold", ""))
                elif part.startswith("scoreratio"):
                    info["score_ratio"] = float(part.replace("scoreratio", ""))
        info["stride"] = 16  # 默认值
    elif "flex" in method_dir:
        info["attention_method"] = "flex"
        # 解析gamma和tau
        if "gamma" in method_dir and "tau" in method_dir:
            parts = method_dir.split("_")
            for part in parts:
                if part.startswith("gamma"):
                    info["gamma"] = float(part.replace("gamma", ""))
                elif part.startswith("tau"):
                    info["tau"] = float(part.replace("tau", ""))
    
    # 从文件名解析任务和其他信息
    if "_" in filename:
        name_parts = filename.replace(".json", "").split("_")
        for i, part in enumerate(name_parts):
            if part in ["recall", "rag", "longqa", "summ", "icl", "rerank", "cite"]:
                info["task"] = part
                break
        
        # 解析输入长度
        for part in name_parts:
            if part.startswith("in") and part[2:].isdigit():
                info["input_length"] = int(part[2:])
                break
    
    return info

def load_result_data(filepath: str) -> Dict:
    """加载单个结果文件的数据"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        logger.error(f"加载文件失败 {filepath}: {e}")
        return {}

def extract_metrics(data: Dict, file_info: Dict) -> Dict:
    """从结果数据中提取关键指标"""
    metrics = {
        "model": "Qwen3-30B-A3B-Instruct-MoE",
        "attention_method": file_info.get("attention_method", "unknown"),
        "task": file_info.get("task", "unknown"),
        "input_length": file_info.get("input_length", 0),
        "threshold": file_info.get("threshold"),
        "score_ratio": file_info.get("score_ratio"),
        "gamma": file_info.get("gamma"),
        "tau": file_info.get("tau"),
        "stride": file_info.get("stride"),
        "filepath": file_info["filepath"]
    }
    
    # 提取平均指标
    if "averaged_metrics" in data:
        avg_metrics = data["averaged_metrics"]
        for key, value in avg_metrics.items():
            if isinstance(value, (int, float)):
                metrics[f"avg_{key}"] = value
    
    # 提取性能指标
    if "throughput" in data:
        metrics["throughput"] = data["throughput"]
    
    if "memory_usage" in data:
        metrics["memory_usage_gb"] = data["memory_usage"] / (1024**3)
    
    # 计算一些派生指标
    if "avg_input_len" in metrics and "avg_output_len" in metrics:
        metrics["total_tokens"] = metrics["avg_input_len"] + metrics["avg_output_len"]
    
    return metrics

def generate_summary_report(df: pd.DataFrame) -> str:
    """生成汇总报告"""
    report = []
    report.append("=" * 80)
    report.append("HELMET Qwen3-30B-A3B-Instruct MoE 评估结果汇总")
    report.append("=" * 80)
    report.append("")
    
    # 基本统计
    report.append("📊 基本统计:")
    report.append(f"  • 总实验数: {len(df)}")
    report.append(f"  • 注意力方法: {', '.join(df['attention_method'].unique())}")
    report.append(f"  • 评估任务: {', '.join(df['task'].unique())}")
    report.append(f"  • 输入长度范围: {df['input_length'].min()}-{df['input_length'].max()}")
    report.append("")
    
    # 按attention方法分组的性能对比
    if 'avg_exact_match' in df.columns:
        report.append("🎯 各注意力方法性能对比 (Exact Match):")
        method_performance = df.groupby('attention_method')['avg_exact_match'].agg(['mean', 'std', 'count'])
        for method, row in method_performance.iterrows():
            report.append(f"  • {method:10}: {row['mean']:6.2f}% ± {row['std']:5.2f}% (n={row['count']})")
        report.append("")
    
    # 按任务分组的性能
    if 'avg_exact_match' in df.columns:
        report.append("📋 各任务性能对比 (Exact Match):")
        task_performance = df.groupby('task')['avg_exact_match'].agg(['mean', 'std', 'count'])
        for task, row in task_performance.iterrows():
            report.append(f"  • {task:10}: {row['mean']:6.2f}% ± {row['std']:5.2f}% (n={row['count']})")
        report.append("")
    
    # 性能 vs 效率分析
    if 'throughput' in df.columns and 'avg_exact_match' in df.columns:
        report.append("⚡ 性能与效率分析:")
        for method in df['attention_method'].unique():
            method_data = df[df['attention_method'] == method]
            if len(method_data) > 0:
                avg_acc = method_data['avg_exact_match'].mean()
                avg_throughput = method_data['throughput'].mean() if 'throughput' in method_data.columns else 0
                report.append(f"  • {method:10}: 准确率={avg_acc:5.2f}%, 吞吐量={avg_throughput:5.2f} samples/s")
        report.append("")
    
    # MoE特有分析
    report.append("🔧 MoE架构特性分析:")
    report.append("  • 专家数量: 128个专家")
    report.append("  • 激活专家: 每token激活8个专家")
    report.append("  • 专门优化: 使用load_qwen3_moe.py实现")
    if 'memory_usage_gb' in df.columns:
        avg_memory = df['memory_usage_gb'].mean()
        report.append(f"  • 平均内存使用: {avg_memory:.1f}GB")
    report.append("")
    
    # 最佳配置推荐
    if 'avg_exact_match' in df.columns:
        report.append("🏆 推荐配置:")
        best_overall = df.loc[df['avg_exact_match'].idxmax()]
        report.append(f"  • 最佳整体性能: {best_overall['attention_method']} "
                     f"(准确率: {best_overall['avg_exact_match']:.2f}%)")
        
        if 'throughput' in df.columns:
            best_efficiency = df.loc[df['throughput'].idxmax()]
            report.append(f"  • 最佳效率: {best_efficiency['attention_method']} "
                         f"(吞吐量: {best_efficiency['throughput']:.2f} samples/s)")
        report.append("")
    
    return "\n".join(report)

def main():
    parser = argparse.ArgumentParser(description="收集和分析Qwen3 MoE实验结果")
    parser.add_argument("--output", "-o", default="moe_qwen3_results_summary.csv", 
                       help="输出CSV文件路径")
    parser.add_argument("--base_dir", default="moe_qwen3_output",
                       help="结果文件基础目录")
    parser.add_argument("--report", default="moe_qwen3_results_report.txt",
                       help="汇总报告文件路径") 
    args = parser.parse_args()
    
    logger.info("开始收集Qwen3 MoE实验结果...")
    
    # 查找所有结果文件
    result_files = find_result_files(args.base_dir)
    
    if not result_files:
        logger.error("未找到任何结果文件")
        return
    
    # 处理每个结果文件
    all_metrics = []
    for filepath in result_files:
        logger.info(f"处理文件: {filepath}")
        
        file_info = parse_filename(filepath)
        data = load_result_data(filepath)
        
        if data:
            metrics = extract_metrics(data, file_info)
            all_metrics.append(metrics)
    
    if not all_metrics:
        logger.error("未能提取任何有效数据")
        return
    
    # 创建DataFrame
    df = pd.DataFrame(all_metrics)
    
    # 保存CSV
    df.to_csv(args.output, index=False)
    logger.info(f"结果已保存到: {args.output}")
    
    # 生成并保存汇总报告
    report = generate_summary_report(df)
    with open(args.report, 'w', encoding='utf-8') as f:
        f.write(report)
    logger.info(f"汇总报告已保存到: {args.report}")
    
    # 打印基本统计
    print("\n" + "="*50)
    print("Qwen3 MoE结果收集完成!")
    print("="*50)
    print(f"📊 处理文件数: {len(result_files)}")
    print(f"📈 有效实验数: {len(all_metrics)}")
    print(f"💾 CSV输出: {args.output}")
    print(f"📄 报告输出: {args.report}")
    
    if len(df) > 0:
        print(f"🎯 注意力方法: {', '.join(df['attention_method'].unique())}")
        print(f"📋 评估任务: {', '.join(df['task'].unique())}")
        
        if 'avg_exact_match' in df.columns:
            best_method = df.loc[df['avg_exact_match'].idxmax(), 'attention_method']
            best_score = df['avg_exact_match'].max()
            print(f"🏆 最佳方法: {best_method} ({best_score:.2f}%)")

if __name__ == "__main__":
    main()
