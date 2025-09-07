# GLM4 HELMET 评测框架

本目录包含了GLM-4-9B-Chat模型在HELMET基准测试上的完整评测框架，支持多种注意力机制优化方案。

## 📁 文件结构

```
glm4_scripts/
├── README.md                      # 本说明文件
├── test_glm4_setup.sh            # GLM4配置测试脚本
├── run_glm4_full.sh              # Full FlashAttention评测 (128k)
├── run_glm4_full_8-64k.sh        # Full FlashAttention评测 (8k-64k)
├── run_glm4_xattn_v6_0.95.sh     # XAttention v6评测 (128k)
├── run_glm4_xattn_8-64k.sh       # XAttention评测 (8k-64k)
├── run_glm4_flex_0.95.sh         # FlexPrefill评测 (128k)
├── run_glm4_flex_8-64k.sh        # FlexPrefill评测 (8k-64k)
├── collect_glm4_results.py       # 结果收集脚本
├── eval_glm4_gpt4_longqa.sh      # LongQA GPT-4评估
└── eval_glm4_gpt4_summ.sh        # Summarization GPT-4评估
```

## 🚀 快速开始

### 1. 测试配置
首先运行测试脚本确保GLM4配置正确：
```bash
./glm4_scripts/test_glm4_setup.sh
```

### 2. 运行评测
选择需要的注意力机制运行评测：

#### Full FlashAttention (基线)
```bash
# 128k上下文评测
./glm4_scripts/run_glm4_full.sh

# 8k-64k上下文评测
./glm4_scripts/run_glm4_full_8-64k.sh
```

#### XAttention v6 (推荐)
```bash
# 128k上下文评测 (threshold=0.95)
./glm4_scripts/run_glm4_xattn_v6_0.95.sh

# 8k-64k上下文评测
./glm4_scripts/run_glm4_xattn_8-64k.sh
```

#### FlexPrefill
```bash
# 128k上下文评测 (gamma=0.95, tau=0.1)
./glm4_scripts/run_glm4_flex_0.95.sh

# 8k-64k上下文评测
./glm4_scripts/run_glm4_flex_8-64k.sh
```

### 3. GPT-4评估 (可选)
对于需要GPT-4评估的任务：
```bash
# LongQA任务GPT-4评估
./glm4_scripts/eval_glm4_gpt4_longqa.sh

# Summarization任务GPT-4评估
./glm4_scripts/eval_glm4_gpt4_summ.sh
```

### 4. 收集结果
运行结果收集脚本生成汇总表格：
```bash
python glm4_scripts/collect_glm4_results.py
```

## 📊 支持的评测任务

- **recall**: 信息检索任务
- **rag**: 检索增强生成
- **longqa**: 长文档问答
- **summ**: 文档摘要
- **icl**: 上下文学习
- **rerank**: 重排序任务
- **cite**: 引用生成

## 🔧 注意力机制说明

### XAttention v6
- **特点**: Golden ratio selection + temperature
- **参数**: threshold=0.95, stride=8, use_simple=6
- **适用**: 平衡性能和效率的通用方案

### FlexPrefill
- **特点**: 灵活的稀疏注意力
- **参数**: gamma=0.95, tau=0.1
- **适用**: 需要动态调整注意力模式的场景


## 📈 结果输出

评测结果将保存在以下目录结构中：
```
glm4_output/
├── full_flashattention/          # Full FlashAttention结果
├── xattn_v6_threshold0.95/       # XAttention v6结果
└── flex_gamma0.95_tau0.1/        # FlexPrefill结果
```

每个目录下包含各个任务的详细结果文件。

## 🛠️ 自定义配置

### 修改模型路径
编辑脚本中的`MODEL_NAME`变量：
```bash
MODEL_NAME="/home/scratch.sarawang_ent/modelscope_cache/GLM/glm-4-9b-chat"
```

### 调整注意力参数
在相应脚本中修改参数：
- `THRESHOLD`: XAttention阈值
- `STRIDE`: 注意力步长
- `USE_SIMPLE`: 简化注意力版本
- `GAMMA`, `TAU`: FlexPrefill参数

## 📋 依赖要求

确保已安装以下依赖：
- PyTorch
- Transformers
- Flash Attention
- XAT (XAttention库)
- HELMET评测框架

## 🔍 故障排除

1. **模型加载失败**: 检查模型路径和权限
2. **内存不足**: 调整batch size或使用更小的上下文长度
3. **CUDA错误**: 确保GPU驱动和CUDA版本兼容
4. **导入错误**: 检查XAT库路径配置

## 📞 支持

如有问题，请检查：
1. GLM4模型是否正确下载
2. XAT库是否正确安装
3. 环境变量是否正确设置
4. 缓存目录是否有足够空间
