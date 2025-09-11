#!/bin/bash

# 创建日志目录
LOG_DIR="logs/glm4_runs_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "开始GLM-4所有任务运行，日志保存到: $LOG_DIR"
echo "运行开始时间: $(date)" | tee "$LOG_DIR/run_summary.log"

# 运行各个脚本并记录日志
echo "=== 运行 GLM-4 Full ===" | tee -a "$LOG_DIR/run_summary.log"
bash 1_run_glm4_full.sh 2>&1 | tee "$LOG_DIR/glm4_full.log"
echo "GLM-4 Full 完成时间: $(date)" | tee -a "$LOG_DIR/run_summary.log"

echo "=== 运行 GLM-4 XAttn v6 0.95 ===" | tee -a "$LOG_DIR/run_summary.log"
bash 1_run_glm4_xattn_v6_0.95.sh 2>&1 | tee "$LOG_DIR/glm4_xattn_v6_0.95.log"
echo "GLM-4 XAttn v6 0.95 完成时间: $(date)" | tee -a "$LOG_DIR/run_summary.log"

echo "=== 运行 GLM-4 Flex 0.95 ===" | tee -a "$LOG_DIR/run_summary.log"
bash 1_run_glm4_flex_0.95.sh 2>&1 | tee "$LOG_DIR/glm4_flex_0.95.log"
echo "GLM-4 Flex 0.95 完成时间: $(date)" | tee -a "$LOG_DIR/run_summary.log"

echo "=== 运行 GLM-4 XAttn 0.95 ===" | tee -a "$LOG_DIR/run_summary.log"
bash 1_run_glm4_xattn_0.95.sh 2>&1 | tee "$LOG_DIR/glm4_xattn_0.95.log"
echo "GLM-4 XAttn 0.95 完成时间: $(date)" | tee -a "$LOG_DIR/run_summary.log"

echo "=== 运行 GLM-4 Full 8-64k ===" | tee -a "$LOG_DIR/run_summary.log"
bash run_glm4_full_8-64k.sh 2>&1 | tee "$LOG_DIR/glm4_full_8-64k.log"
echo "GLM-4 Full 8-64k 完成时间: $(date)" | tee -a "$LOG_DIR/run_summary.log"

echo "=== 运行 GLM-4 XAttn v6 0.95 8-64k ===" | tee -a "$LOG_DIR/run_summary.log"
bash run_glm4_xattn_v6_0.95_8-64k.sh 2>&1 | tee "$LOG_DIR/glm4_xattn_v6_0.95_8-64k.log"
echo "GLM-4 XAttn v6 0.95 8-64k 完成时间: $(date)" | tee -a "$LOG_DIR/run_summary.log"

echo "=== 运行 GLM-4 Flex 0.95 8-64k ===" | tee -a "$LOG_DIR/run_summary.log"
bash run_glm4_flex_0.95_8-64k.sh 2>&1 | tee "$LOG_DIR/glm4_flex_0.95_8-64k.log"
echo "GLM-4 Flex 0.95 8-64k 完成时间: $(date)" | tee -a "$LOG_DIR/run_summary.log"

echo "所有任务完成时间: $(date)" | tee -a "$LOG_DIR/run_summary.log"
echo "日志文件保存在: $LOG_DIR"