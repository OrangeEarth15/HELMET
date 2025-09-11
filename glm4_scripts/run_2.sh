#!/bin/bash

# 创建日志目录
LOG_DIR="logs/glm4_2_runs_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "开始GLM-4高级任务运行，日志保存到: $LOG_DIR"
echo "高级任务运行开始时间: $(date)" | tee "$LOG_DIR/advanced_run_summary.log"

# 运行XAttention V6
echo "=== 运行 GLM-4 XAttention V6 0.95 128k ===" | tee -a "$LOG_DIR/advanced_run_summary.log"
bash run_glm4_xattn_v6_0.95.sh 2>&1 | tee "$LOG_DIR/glm4_xattn_v6_0.95.log"
echo "Complete XAttention V6 128k! 完成时间: $(date)" | tee -a "$LOG_DIR/advanced_run_summary.log"

# 运行Flex
echo "=== 运行 GLM-4 Flex 0.95 128k ===" | tee -a "$LOG_DIR/advanced_run_summary.log"
bash run_glm4_flex_0.95.sh 2>&1 | tee "$LOG_DIR/glm4_flex_0.95.log"
echo "Complete Flex 128k! 完成时间: $(date)" | tee -a "$LOG_DIR/advanced_run_summary.log"

echo "所有高级任务完成时间: $(date)" | tee -a "$LOG_DIR/advanced_run_summary.log"
echo "日志文件保存在: $LOG_DIR"

