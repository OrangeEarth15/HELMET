import os
import json
import numpy as np
import pandas as pd
import yaml
from dataclasses import dataclass, asdict
from tqdm import tqdm

dataset_to_metrics = {
    "json_kv": "substring_exact_match",
    "nq": "substring_exact_match",
    "popqa": "substring_exact_match",
    "triviaqa": "substring_exact_match",
    "hotpotqa": "substring_exact_match",
    
    "narrativeqa": ["gpt-4-score"],
    "msmarco_rerank_psg": "NDCG@10",
    
    "trec_coarse": "exact_match",
    "trec_fine": "exact_match",
    "banking77": "exact_match",
    "clinic150": "exact_match",
    "nlu": "exact_match",
    
    "qmsum": "rougeL_recall",
    "multi_lexsum": ["gpt-4-f1"],
    
    "ruler_niah_s_1": "ruler_recall",
    "ruler_niah_s_2": "ruler_recall",
    "ruler_niah_s_3": "ruler_recall",
    "ruler_niah_mk_1": "ruler_recall",
    "ruler_niah_mk_2": "ruler_recall",
    "ruler_niah_mk_3": "ruler_recall",
    "ruler_niah_mq": "ruler_recall",
    "ruler_niah_mv": "ruler_recall",
    "ruler_fwe": "ruler_recall",
    "ruler_cwe": "ruler_recall",
    "ruler_vt": "ruler_recall",
    "ruler_qa_1": "substring_exact_match",
    "ruler_qa_2": "substring_exact_match",
    
    "infbench_qa": ["rougeL_f1"],
    "infbench_choice": ["exact_match"],
    # "infbench_sum": ["gpt-4-f1"],
    "infbench_sum_eng": ["gpt-4-f1"],
    
    "alce_asqa": ["str_em", "citation_rec", "citation_prec"],
    "alce_qampari": ["qampari_rec_top5", "citation_rec", "citation_prec"],
}

dataset_to_metrics = {k: [v] if isinstance(v, str) else v for k, v in dataset_to_metrics.items()}
# 层次化任务分类：七大类 -> 具体任务（使用真实任务名称）
hierarchical_tasks = {
    "Recall": {
        "json_kv": ["json_kv substring_exact_match"],
        "niah_mk_2": ["ruler_niah_mk_2 ruler_recall"],
        "niah_mk_3": ["ruler_niah_mk_3 ruler_recall"],
        "niah_mv": ["ruler_niah_mv ruler_recall"],
    },
    "RAG": {
        "nq": ["nq substring_exact_match"],
        "hotpotqa": ["hotpotqa substring_exact_match"],
        "popqa": ["popqa substring_exact_match"],
        "triviaqa": ["triviaqa substring_exact_match"],
    },
    "ICL": {
        "trec_coarse": ["trec_coarse exact_match"],
        "trec_fine": ["trec_fine exact_match"],
        "banking77": ["banking77 exact_match"],
        "clinic150": ["clinic150 exact_match"],
        "nlu": ["nlu exact_match"],
    },
    "Cite": {
        "asqa_str_em": ["alce_asqa str_em"],
        "asqa_citation_rec": ["alce_asqa citation_rec"],
        "asqa_citation_prec": ["alce_asqa citation_prec"],
        "qampari_rec_top5": ["alce_qampari qampari_rec_top5"],
        "qampari_citation_rec": ["alce_qampari citation_rec"],
        "qampari_citation_prec": ["alce_qampari citation_prec"],
    },
    "Re-rank": {
        "msmarco_psg": ["msmarco_rerank_psg NDCG@10"],
    },
    "LongQA": {
        "narrativeqa": ["narrativeqa gpt-4-score"],
        "infbench_qa": ["infbench_qa rougeL_f1"],
        "infbench_choice": ["infbench_choice exact_match"],
    },
    "Summ": {
        "infbench_sum_eng": ["infbench_sum_eng gpt-4-f1"],
        "multi_lexsum": ["multi_lexsum gpt-4-f1"],
    },
}

# 保持向后兼容的平铺结构
custom_avgs = {
    "Recall": ["json_kv substring_exact_match", "ruler_niah_mk_2 ruler_recall", "ruler_niah_mk_3 ruler_recall", "ruler_niah_mv ruler_recall"],
    "RAG": ['nq substring_exact_match', 'hotpotqa substring_exact_match', 'popqa substring_exact_match', 'triviaqa substring_exact_match',],
    "ICL": ['trec_coarse exact_match', 'trec_fine exact_match', 'banking77 exact_match', 'clinic150 exact_match', 'nlu exact_match'],
    "Cite": ['alce_asqa str_em', 'alce_asqa citation_rec', 'alce_asqa citation_prec', 'alce_qampari qampari_rec_top5', 'alce_qampari citation_rec', 'alce_qampari citation_prec', ],
    "Re-rank": ['msmarco_rerank_psg NDCG@10', ],
    "LongQA": ['narrativeqa gpt-4-score', 'infbench_qa rougeL_f1', 'infbench_choice exact_match', ],
    # "Summ": ['infbench_sum gpt-4-f1', 'multi_lexsum gpt-4-f1', ],
    "Summ": ['infbench_sum_eng gpt-4-f1', 'multi_lexsum gpt-4-f1', ],
    # "RULER": ['ruler_niah_s_1 ruler_recall', 'ruler_niah_s_2 ruler_recall', 'ruler_niah_s_3 ruler_recall', 'ruler_niah_mk_1 ruler_recall', 'ruler_niah_mk_2 ruler_recall', 'ruler_niah_mk_3 ruler_recall', 'ruler_niah_mq ruler_recall', 'ruler_niah_mv ruler_recall', 'ruler_cwe ruler_recall', 'ruler_fwe ruler_recall', 'ruler_vt ruler_recall', 'ruler_qa_1 substring_exact_match', 'ruler_qa_2 substring_exact_match'],
    "Ours": ['Recall', 'RAG', 'ICL', 'Cite', 'Re-rank', 'LongQA', 'Summ'],
}

def flatten_hierarchical_tasks(hierarchical_tasks):
    """将层次化任务结构扁平化，用于计算平均值"""
    flattened = {}
    for major_cat, subtasks in hierarchical_tasks.items():
        for subtask_name, metrics in subtasks.items():
            # 创建层次化的列名：大类|小类
            col_name = f"{major_cat}|{subtask_name}"
            flattened[col_name] = metrics
    return flattened

def create_hierarchical_csv(df, output_file):
    """生成层次化的CSV文件"""
    import pandas as pd
    
    # 首先生成基本的透视表
    lf_df = df.pivot_table(
        index=["input_max_length", "attention", "tag"], 
        columns="dataset_simple", 
        values="metric", 
        sort=False
    ).reset_index()
    
    # 计算层次化的平均值
    hierarchical_avgs = flatten_hierarchical_tasks(hierarchical_tasks)
    
    # 为每个层次化类别计算平均值
    for col_name, metrics in hierarchical_avgs.items():
        available_cols = [col for col in metrics if col in lf_df.columns]
        if available_cols:
            lf_df[col_name] = lf_df[available_cols].mean(axis=1)
    
    # 保持原有的平铺结构计算（向后兼容）
    for k, v in custom_avgs.items():
        available_cols = [col for col in v if col in lf_df.columns]
        if available_cols:
            lf_df[k] = lf_df[available_cols].mean(axis=1)
    
    # 创建层次化的列结构
    base_cols = ["input_max_length", "attention", "tag"]
    
    # 按大类组织列
    organized_cols = base_cols.copy()
    
    # 为每个大类添加其子任务列
    for major_cat in ["Recall", "RAG", "ICL", "Cite", "Re-rank", "LongQA", "Summ"]:
        # 添加该大类的所有task列
        task_cols = [col for col in lf_df.columns if col.startswith(f"{major_cat}|")]
        task_cols.sort()  # 确保task1, task2, task3的顺序
        organized_cols.extend(task_cols)
        
        # 添加该大类的聚合列
        if major_cat in lf_df.columns:
            organized_cols.append(major_cat)
    
    # 添加其他剩余的列
    remaining_cols = [col for col in lf_df.columns if col not in organized_cols]
    organized_cols.extend(remaining_cols)
    
    # 重新排列列顺序
    lf_df = lf_df[organized_cols]
    
    # 按序列长度优先排序，然后按attention类型排序
    lf_df = lf_df.sort_values(['input_max_length', 'attention', 'tag'], ascending=[True, True, True])
    
    # 创建多级表头的CSV（如果需要）
    # 这里我们先保存标准CSV，然后创建一个带多级表头的版本
    lf_df.to_csv(output_file, index=False)
    
    # 创建多级表头版本
    hierarchical_output_file = output_file.replace('.csv', '_hierarchical.csv')
    create_multi_header_csv(lf_df, hierarchical_output_file)
    
    return lf_df

def create_multi_header_csv(df, output_file):
    """创建极其层次化的多级表头CSV文件，只包含层次化的列"""
    import pandas as pd
    
    # 只选择我们想要的列：基础配置列 + 层次化列 + 汇总列
    base_cols = ["input_max_length", "attention", "tag"]
    hierarchical_cols = [col for col in df.columns if "|" in col]
    summary_cols = [col for col in df.columns if col in ["Recall", "RAG", "ICL", "Cite", "Re-rank", "LongQA", "Summ", "Ours"]]
    
    # 选择的列
    selected_cols = base_cols + hierarchical_cols + summary_cols
    selected_df = df[selected_cols].copy()
    
    # 准备三级表头
    header1 = []  # 第一级表头（大类）
    header2 = []  # 第二级表头（子类别）
    header3 = []  # 第三级表头（具体指标）
    
    for col in selected_df.columns:
        if col in ["input_max_length", "attention", "tag"]:
            header1.append("Config")
            header2.append("")
            header3.append(col)
        elif "|" in col:
            major_cat, task = col.split("|", 1)  # 只分割一次，防止task名中有|
            header1.append(major_cat)
            
            # 根据任务名称进一步分类
            if major_cat == "Cite":
                if "asqa" in task:
                    header2.append("ASQA")
                elif "qampari" in task:
                    header2.append("QAMPARI")
                else:
                    header2.append("Summary")
            elif major_cat == "Recall":
                if "niah" in task:
                    header2.append("NIAH")
                elif "json_kv" in task:
                    header2.append("KeyValue")
                else:
                    header2.append("Summary")
            elif major_cat == "LongQA":
                if "narrativeqa" in task:
                    header2.append("Narrative")
                elif "infbench" in task:
                    header2.append("InfBench")
                else:
                    header2.append("Summary")
            elif major_cat == "ICL":
                if "trec" in task:
                    header2.append("TREC")
                elif task in ["banking77", "clinic150", "nlu"]:
                    header2.append("Other")
                else:
                    header2.append("Summary")
            elif major_cat == "RAG":
                header2.append("")  # RAG下面直接是各个数据集
            elif major_cat == "Re-rank":
                header2.append("")  # Re-rank下面直接是msmarco
            elif major_cat == "Summ":
                header2.append("")  # Summ下面直接是各个任务
            else:
                header2.append("")
                
            header3.append(task)
        elif col in ["Recall", "RAG", "ICL", "Cite", "Re-rank", "LongQA", "Summ"]:
            header1.append(col)
            header2.append("Summary")
            header3.append("avg_score")
        elif col == "Ours":
            header1.append("Ours")
            header2.append("")
            header3.append("")
    
    # 直接写入CSV文件，确保列数一致
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        import csv
        writer = csv.writer(f)
        
        # 写入三级表头
        writer.writerow(header1)
        writer.writerow(header2)
        writer.writerow(header3)
        
        # 写入数据行
        for _, row in selected_df.iterrows():
            writer.writerow(row.tolist())
    
    print(f"Created clean hierarchical CSV with {len(selected_cols)} columns: {output_file}")

@dataclass
class arguments:
    tag: str = "v1"
    input_max_length: int = 131072
    generation_max_length: int = 100
    generation_min_length: int = 0
    max_test_samples: int = 100
    shots: int = 2
    do_sample: bool = False
    temperature: float = 0.0
    top_p: float = 1.0
    use_chat_template: bool = False
    seed: int = 42
    test_name: str = ""
    dataset: str = "nq"
    output_dir: str = "output"
    popularity_threshold: float = 3
        
    category: str = "synthetic"
    
    def update(self, new):
        for key, value in new.items():
            if hasattr(self, key):
                setattr(self, key, value)
                
    def get_path(self):
        tag = self.tag
        # path = os.path.join(self.output_dir, "{args.dataset}_{tag}_{args.test_name}_in{args.input_max_length}_size{args.max_test_samples}_shots{args.shots}_samp{args.do_sample}max{args.generation_max_length}min{args.generation_min_length}t{args.temperature}p{args.top_p}_chat{args.use_chat_template}_{args.seed}.json".format(args=self, tag=tag))
        
        # 根据数据集确定子目录
        subdirs = {
            "json_kv": "recall",
            "ruler_niah": "recall", 
            "nq": "rag",
            "triviaqa": "rag", 
            "hotpotqa": "rag",
            "popqa": "rag",
            "narrativeqa": "longqa",
            "infbench_qa": "longqa",
            "infbench_choice": "longqa",
            "infbench_sum": "summ",
            "multi_lexsum": "summ",
            "msmarco_rerank": "rerank",
            "trec": "icl",
            "banking77": "icl",
            "clinic150": "icl", 
            "nlu": "icl",
            "alce": "cite"
        }
        
        # 确定子目录
        subdir = None
        for prefix, dir_name in subdirs.items():
            if prefix in self.dataset:
                subdir = dir_name
                break
        
        if subdir is None:
            # 如果找不到匹配的子目录，尝试根据数据集名称猜测
            if "kv" in self.dataset or "ruler" in self.dataset:
                subdir = "recall"
            elif any(x in self.dataset for x in ["trec", "banking", "clinic", "nlu"]):
                subdir = "icl"
            elif any(x in self.dataset for x in ["qa", "narrativeqa", "infbench"]):
                subdir = "longqa"
            elif "sum" in self.dataset:
                subdir = "summ"
            elif "rerank" in self.dataset:
                subdir = "rerank"
            elif "alce" in self.dataset:
                subdir = "cite"
            else:
                subdir = "rag"  # 默认
        
        filename = "{args.dataset}_{tag}_{args.test_name}_in{args.input_max_length}_size{args.max_test_samples}_shots{args.shots}_samp{args.do_sample}max{args.generation_max_length}min{args.generation_min_length}t{args.temperature}p{args.top_p}_chat{args.use_chat_template}_{args.seed}.json".format(args=self, tag=tag)
        
        # 检查是否为短序列配置（tag包含_short后缀）
        if tag.endswith('_short'):
            subdir = subdir + '_short'
        
        path = os.path.join(self.output_dir, subdir, filename)

        if os.path.exists(path.replace(".json", "-gpt4eval_o.json")):
            return path.replace(".json", "-gpt4eval_o.json")
        if "alce" in self.dataset:
            return path.replace(".json", ".json.score")
        
        if os.path.exists(path + ".score"):
            return path + ".score"
        return path

    def get_metric_name(self):
        for d, m in dataset_to_metrics.items():
            if d in self.dataset:
                return d, m
        return None
    
    def get_averaged_metric(self, _auto_eval_attempted=False):
        path = self.get_path()
        print(path)
        if not os.path.exists(path):
            print("path doesn't exist")
            # 对于ALCE数据集，尝试查找原始JSON文件并自动运行eval_alce.py
            if "alce" in self.dataset and path.endswith(".score") and not _auto_eval_attempted:
                json_path = path.replace(".score", ".json")
                if os.path.exists(json_path):
                    print(f"Score file missing but JSON exists: {json_path}")
                    print("Attempting to run eval_alce.py automatically...")
                    
                    # 尝试自动运行eval_alce.py
                    if self._run_eval_alce(json_path):
                        print("Successfully ran eval_alce.py, checking for score file...")
                        if os.path.exists(path):
                            print("Score file now exists, proceeding with normal evaluation")
                            # 递归调用以使用正常的评估逻辑，标记已经尝试过自动评估
                            return self.get_averaged_metric(_auto_eval_attempted=True)
                    
                    # 如果自动运行失败，直接报错，不退而求其次
                    print("ERROR: Auto eval_alce.py failed, and score file is required for accurate metrics")
                    return None
            else:
                print("ERROR: Required score file not found and cannot be generated automatically")
            return None
        
        try:
            with open(path) as f:
                results = json.load(f)
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            return None
        except Exception as e:
            print(f"Error reading file: {e}")
            return None
        
        _, metric = self.get_metric_name()
        if path.endswith(".score"):
            if any([m not in results for m in metric]):
                print("metric doesn't exist")
                return None
            s = {m: results[m] for m in metric}
        else:
            if any([m not in results["averaged_metrics"] for m in metric]):
                print("metric doesn't exist")
                return None
            s = {m: results['averaged_metrics'][m] for m in metric}
        
        s = {m : v * (100 if m == "gpt-4-f1" else 1) * (100/3 if m == "gpt-4-score" else 1) for m, v in s.items()}
        print("found scores:", s)
        return s
    
    def _extract_alce_metrics_from_json(self, json_path):
        """从ALCE的原始JSON文件中提取基本指标 - 严格模式，要求所有指标都存在"""
        try:
            with open(json_path) as f:
                results = json.load(f)
            
            if "averaged_metrics" not in results:
                print("ERROR: No averaged_metrics found in JSON file")
                return None
            
            # 提取可用的指标
            _, expected_metrics = self.get_metric_name()
            available_metrics = {}
            missing_metrics = []
            
            for metric in expected_metrics:
                if metric in results["averaged_metrics"]:
                    available_metrics[metric] = results["averaged_metrics"][metric]
                else:
                    missing_metrics.append(metric)
            
            # 严格检查：如果有任何指标缺失，直接报错
            if missing_metrics:
                print(f"ERROR: Missing required metrics: {missing_metrics}")
                print("This indicates incomplete evaluation. eval_alce.py with --citations must be run successfully.")
                return None
            
            # 应用相同的缩放因子
            scaled_metrics = {m : v * (100 if m == "gpt-4-f1" else 1) * (100/3 if m == "gpt-4-score" else 1) 
                            for m, v in available_metrics.items()}
            print(f"Successfully extracted all required metrics from JSON: {scaled_metrics}")
            return scaled_metrics
            
        except Exception as e:
            print(f"ERROR extracting metrics from JSON: {e}")
            return None
    
    def _run_eval_alce(self, json_path):
        """尝试自动运行eval_alce.py生成score文件"""
        try:
            import subprocess
            import sys
            
            # 构建eval_alce.py命令
            script_dir = os.path.dirname(os.path.abspath(__file__))
            helmet_root = os.path.dirname(script_dir)
            eval_alce_path = os.path.join(helmet_root, "eval_alce.py")
            
            if not os.path.exists(eval_alce_path):
                print(f"eval_alce.py not found at {eval_alce_path}")
                return False
            
            # 构建命令参数
            cmd = [sys.executable, eval_alce_path, "--f", json_path]
            if not "nocite" in self.dataset:
                cmd.append("--citations")
            
            print(f"Running command: {' '.join(cmd)}")
            
            # 运行命令
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)  # 5分钟超时
            
            if result.returncode == 0:
                print("eval_alce.py completed successfully")
                # 验证score文件是否真的生成了
                score_path = json_path + ".score"
                if os.path.exists(score_path):
                    print(f"Verified: score file created at {score_path}")
                    return True
                else:
                    print(f"ERROR: eval_alce.py returned success but score file not found at {score_path}")
                    return False
            else:
                print(f"ERROR: eval_alce.py failed with return code {result.returncode}")
                print(f"STDOUT: {result.stdout}")
                print(f"STDERR: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print("eval_alce.py timed out after 5 minutes")
            return False
        except ImportError:
            print("subprocess module not available")
            return False
        except Exception as e:
            print(f"Error running eval_alce.py: {e}")
            return False
    
    def get_sparsity_info(self):
        """获取稀疏度信息"""
        path = self.get_path()
        if not os.path.exists(path):
            return None
        
        try:
            with open(path) as f:
                results = json.load(f)
            
            # 检查是否有稀疏度信息
            sparsity_info = {}
            if 'avg_sparse_ratio' in results:
                sparsity_info['avg_sparse_ratio'] = results['avg_sparse_ratio']
            
            return sparsity_info if sparsity_info else None
        except:
            return None
        
    def get_metric_by_depth(self):
        path = self.get_path()
        path = path.replace(".score", '')
        print(path)
        if not os.path.exists(path):
            return None
        with open(path) as f:
            results = json.load(f)

        output = []        
        _, metric = self.get_metric_name()
        metric = metric[0]
        keys = ["depth", "k", metric]
        for d in results["data"]:
            o = {}
            for key in keys:
                if key == "k" and "ctxs" in d:
                    d["k"] = len(d['ctxs'])
                if key not in d:
                    print("no", key)
                    return None
                o[key] = d[key]
            o["metric"] = o.pop(metric)
            output.append(o)
        
        df = pd.DataFrame(output)
        dfs = df.groupby(list(output[0].keys())[:-1]).mean().reset_index()

        return dfs.to_dict("records")

if __name__ == "__main__":
    # comment out the models you don't want to include, or add the new ones 
    models_configs = [
        {"model": "gpt-4-0125-preview", "use_chat_template": True, "training_length": 128000},
        {"model": "gpt-4o-mini-2024-07-18", "use_chat_template": True, "training_length": 128000},
        {"model": "gpt-4o-2024-05-13", "use_chat_template": True, "training_length": 128000},
        {"model": "gpt-4o-2024-08-06", "use_chat_template": True, "training_length": 128000},
        {"model": "claude-3-5-sonnet-20240620", "use_chat_template": True, "training_length": 200000},
        {"model": "gemini-1.5-flash-001", "use_chat_template": True, "training_length": 1048576},
        {"model": "gemini-1.5-pro-001", "use_chat_template": True, "training_length": 2097152},

        # llama 2 based models
        {"model": "Llama-2-7B-32K", "use_chat_template": False, "training_length": 32768},
        {"model": "Llama-2-7B-32K-Instruct", "training_length": 32768},
        {"model": "llama-2-7b-80k", "use_chat_template": False, "training_length": 80000},
        {"model": "Yarn-Llama-2-7b-64k", "use_chat_template": False, "training_length": 65536},
        {"model": "Yarn-Llama-2-7b-128k", "use_chat_template": False, "training_length": 131072},
        
        # llama 3 models
        {"model": "Meta-Llama-3-8B", "use_chat_template": False, "training_length": 8192},
        {"model": "Meta-Llama-3-8B-Instruct", "training_length": 8192},
        {"model": "Meta-Llama-3-8B-Theta16M", "use_chat_template": False, "training_length": 8192},
        {"model": "Meta-Llama-3-8B-Instruct-Theta16M", "training_length": 8192},
        {"model": "Meta-Llama-3-70B-Theta16M", "use_chat_template": False, "training_length": 8192},
        {"model": "Meta-Llama-3-70B-Instruct-Theta16M", "training_length": 8192},
        
        {"model": "Llama-3.1-8B", "use_chat_template": False, "training_length": 131072},
        {"model": "Llama-3.1-8B-Instruct", "training_length": 131072},
        {"model": "Llama-3.1-70B", "use_chat_template": False, "training_length": 131072},
        {"model": "Llama-3.1-70B-Instruct", "training_length": 131072},
        {"model": "Llama-3.3-70B-Instruct", "training_length": 131072},
        
        {"model": "Llama-3.2-1B", "use_chat_template": False, "training_length": 131072},
        {"model": "Llama-3.2-1B-Instruct", "training_length": 131072},
        {"model": "Llama-3.2-3B", "use_chat_template": False, "training_length": 131072},
        {"model": "Llama-3.2-3B-Instruct", "training_length": 131072},
        
        # mistral models
        {"model": "Mistral-7B-v0.1", "use_chat_template": False, "training_length": 8192},
        {"model": "Mistral-7B-Instruct-v0.1", "training_length": 8192},
        {"model": "Mistral-7B-Instruct-v0.2", "training_length": 32768},
        {"model": "Mistral-7B-v0.3", "use_chat_template": False, "training_length": 32768},
        {"model": "Mistral-7B-Instruct-v0.3", "training_length": 32768},
        {"model": "Ministral-8B-Instruct-2410", "training_length": 131072},
        
        {"model": "Mistral-Nemo-Base-2407", "use_chat_template": False, "training_length": 128000},
        {"model": "Mistral-Nemo-Instruct-2407", "training_length": 128000},
        {"model": "MegaBeam-Mistral-7B-512k", "training_length": 524288},
        
        # yi models
        {"model": "Yi-6B-200K", "use_chat_template": False, "training_length": 200000},
        {"model": "Yi-9B-200K", "use_chat_template": False, "training_length": 200000},
        {"model": "Yi-34B-200K", "use_chat_template": False, "training_length": 200000},
        {"model": "Yi-1.5-9B-32K", "use_chat_template": False, "training_length": 32768},
        
        # phi models
        {"model": "Phi-3-mini-128k-instruct", "training_length": 131072},
        {"model": "Phi-3-small-128k-instruct", "training_length": 131072},
        {"model": "Phi-3-medium-128k-instruct", "training_length": 131072},
        {"model": "Phi-3.5-mini-instruct", "training_length": 131072},
        
        # qwen models
        {"model": "Qwen2-7B", "use_chat_template": False, "training_length": 32768},
        {"model": "Qwen2-7B-Instruct", "training_length": 32768},
        {"model": "Qwen2-57B-A14B", "use_chat_template": False, "training_length": 32768},
        {"model": "Qwen2-57B-A14B-Instruct", "training_length": 32768},
        {"model": "Qwen2.5-1.5B", "use_chat_template": False, "training_length": 32768},
        {"model": "Qwen2.5-1.5B-Instruct", "training_length": 32768},
        {"model": "Qwen2.5-3B", "use_chat_template": False, "training_length": 32768},
        {"model": "Qwen2.5-3B-Instruct", "training_length": 32768},
        {"model": "Qwen2.5-7B", "use_chat_template": False, "training_length": 131072},
        {"model": "Qwen2.5-7B-Instruct", "training_length": 131072},
        {"model": "Qwen2.5-72B-Instruct", "training_length": 131072},
        
        # prolong
        {"model": "Llama-3-8B-ProLong-512k-Instruct", "training_length": 524288},
        
        # gemma 2 models
        {"model": "gemma-2-9b", "use_chat_template": False, "training_length": 8192},
        {"model": "gemma-2-9b-it", "training_length": 8192},
        {"model": "gemma-2-9b-it-Theta320K", "training_length": 8192},

        {"model": "gemma-2-27b", "use_chat_template": False, "training_length": 8192},
        {"model": "gemma-2-27b-it", "training_length": 8192},
        {"model": "gemma-2-27b-it-Theta320K", "training_length": 8192},
        
        # others
        {"model": "c4ai-command-r-v01", "training_length": 131072},
        {"model": "Jamba-v0.1", "use_chat_template": False, "training_length": 262144},
        {"model": "AI21-Jamba-1.5-Mini", "training_length": 262144},
    ]

    
    # XAT attention结果配置
    models_configs = [
            # {"model": "Llama-3.1-8B", "use_chat_template": False, "training_length": 131072},
            # {"model": "Llama-3.1-8B-Instruct", "training_length": 131072},
            # {"model": "DeepSeek-R1-Distill-Llama-8B", "training_length": 131072, "do_sample": True, "temperature": 0.6},
            # {"model": "Qwen2-7B", "use_chat_template": False, "training_length": 32768},
            # {"model": "Qwen2-7B-Instruct", "training_length": 32768},
            # {"model": "DeepSeek-R1-Distill-Qwen-7B", "training_length": 131072, "do_sample": True, "temperature": 0.6},
            # 根据你的输出目录结构修改这里
            {"model": "full_flashinfer", "use_chat_template": True, "training_length": 131072},
            {"model": "xattn_threshold0.95", "use_chat_template": True, "training_length": 131072},
            {"model": "xattn_threshold0.9", "use_chat_template": True, "training_length": 131072},
            {"model": "flex_gamma0.95_tau0.1", "use_chat_template": True, "training_length": 131072},
            {"model": "flex_gamma0.9_tau0.1", "use_chat_template": True, "training_length": 131072},
            {"model": "xflex_threshold0.95_scoreratio0.95", "use_chat_template": True, "training_length": 131072},
            {"model": "xflex_threshold0.95_scoreratio0.5", "use_chat_template": True, "training_length": 131072},
    ]

    # set your configs here, only include the ones that you ran
    config_files = [
        "configs/recall.yaml", "configs/recall_short.yaml", 
        "configs/rag.yaml", "configs/rag_short.yaml", 
        "configs/longqa.yaml", "configs/longqa_short.yaml", 
        "configs/summ.yaml", "configs/summ_short.yaml", 
        "configs/rerank.yaml", "configs/rerank_short.yaml", 
        "configs/icl.yaml", "configs/icl_short.yaml", 
        "configs/cite.yaml", "configs/cite_short.yaml", 
        "configs/ruler.yaml", "configs/ruler_short.yaml", 
    ]

    dataset_configs = []
    for file in config_files:
        c = yaml.safe_load(open(file))
        
        if isinstance(c["generation_max_length"], int):
            c["generation_max_length"] = ",".join([str(c["generation_max_length"])] * len(c["datasets"].split(",")))
        for d, t, l, g in zip(c['datasets'].split(','), c['test_files'].split(','), c['input_max_length'].split(','), c['generation_max_length'].split(',')):
            dataset_configs.append({"dataset": d, "test_name": os.path.basename(os.path.splitext(t)[0]), "input_max_length": int(l), "generation_max_length": int(g), "max_test_samples": c['max_test_samples'], 'use_chat_template': c['use_chat_template'], 'shots': c['shots']})
    print(dataset_configs)    

    failed_paths = []
    df = []
    for model in tqdm(models_configs):
        args = arguments()
        # args.tag = "v1" # SET YOUR TAG HERE
        args.tag = model['model']  # 使用模型名称作为tag，匹配输出目录
        args.output_dir = f"output/{model['model']}"
    
        for dataset in dataset_configs:
            args.update(dataset)
            args.update(model)

            metric = args.get_averaged_metric()
            dsimple, mnames = args.get_metric_name()

            if metric is None:
                failed_paths.append(args.get_path())
                continue
                
            for k, m in metric.items():
                df.append({**asdict(args), **model,
                    "metric name": k, "metric": m, 
                    "dataset_simple": dsimple + " " + k, "test_data": f"{args.dataset}-{args.test_name}-{args.input_max_length}"
                })

    all_df = pd.DataFrame(df)
    lf_df = all_df.pivot_table(index=["input_max_length", "model", ], columns="dataset_simple", values="metric", sort=False)
    lf_df = lf_df.reset_index()

    for k, v in custom_avgs.items():
        lf_df[k] = lf_df[v].mean(axis=1)

    print(lf_df.to_csv(index=False))

    print("Warning, failed to get the following paths, make sure that these are correct or the printed results will not be accurate:", failed_paths)
    # import pdb; pdb.set_trace()