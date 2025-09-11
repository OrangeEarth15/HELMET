#!/bin/bash

# 创建日志目录
LOG_DIR="logs/glm4_1_runs_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "开始GLM-4基础任务运行，日志保存到: $LOG_DIR"
echo "基础任务运行开始时间: $(date)" | tee "$LOG_DIR/basic_run_summary.log"

# 运行Full Attention
echo "=== 运行 GLM-4 Full Attention 128k ===" | tee -a "$LOG_DIR/basic_run_summary.log"
bash run_glm4_full.sh 2>&1 | tee "$LOG_DIR/glm4_full.log"
echo "Complete Full Attention 128k! 完成时间: $(date)" | tee -a "$LOG_DIR/basic_run_summary.log"

# 运行XAttention
echo "=== 运行 GLM-4 XAttention 0.95 128k ===" | tee -a "$LOG_DIR/basic_run_summary.log"
bash run_glm4_xattn_0.95.sh 2>&1 | tee "$LOG_DIR/glm4_xattn_0.95.log"
echo "Complete XAttention 128k! 完成时间: $(date)" | tee -a "$LOG_DIR/basic_run_summary.log"

echo "所有基础任务完成时间: $(date)" | tee -a "$LOG_DIR/basic_run_summary.log"
echo "日志文件保存在: $LOG_DIR"