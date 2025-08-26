# <img src="assets/logo.jpeg" alt="HELMET" width="30"> HELMET: 如何有效而全面地评估长上下文语言模型

---

<p align="center">
    <a href="https://arxiv.org/abs/2410.02694" target="_blank" rel="noopener noreferrer">
        <img alt="paper" src="https://img.shields.io/badge/paper-paper?logo=arxiv&logoColor=%23B31B1B&labelColor=white&color=%23B31B1B">
    </a>
    <a href="https://princeton-nlp.github.io/HELMET/" target="_blank" rel="noopener noreferrer">
        <img alt="website" src="https://img.shields.io/badge/website-website?logo=safari&logoColor=%23006CFF&labelColor=white&color=%23006CFF">
    </a>
</p>

<img src="assets/logo.jpeg" alt="HELMET" width="30"> HELMET（如何有效而全面地评估长上下文模型）是一个用于长上下文语言模型的综合基准测试套件，涵盖七个不同的任务类别。这些数据集以应用为中心，旨在评估模型在不同长度和复杂性级别上的表现。请查看论文了解更多详细信息，本仓库将详细介绍如何运行评估。

本仓库还支持 [LongProc](https://princeton-pli.github.io/LongProc/)，这是我们新的长上下文过程生成基准测试。请参阅 [longproc_addon](longproc_addon/README.md) 了解如何运行评估。整体结构与 HELMET 相同，但使用了额外的数据、配置和评估指标。

## 快速链接

- [环境设置](#环境设置)
- [数据](#数据)
- [运行评估](#运行评估)
- [添加新任务](#添加新任务)
- [添加新模型](#添加新模型)
- [数据集相关性分析](#数据集相关性分析)
- [其他功能](#其他功能)
- [联系方式](#联系方式)
- [引用](#引用)

## 发布进度

查看 `CHANGELOG.md` 了解更新和更多详细信息。

- [x] HELMET 代码
- [x] HELMET 数据
- [x] VLLM 支持
- [x] 相关性分析笔记本
- [ ] 支持 >128k 输入长度
- [ ] 检索设置

## 环境设置

请使用以下命令安装必要的包（推荐使用虚拟环境，使用 Python 3.11 测试）：
```bash
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

对于在 NVIDIA GPU 上进行评估，请参考 [flash attention 仓库](https://github.com/Dao-AILab/flash-attention) 安装 `flash-attn`。

此外，如果您希望使用 API 模型，需要安装相应的包：
```bash
pip install openai # OpenAI API (GPT)
pip install anthropic==0.42.0 # Anthropic API (Claude)
pip install google-generativeai # Google API (Gemini)
pip install vertexai==1.71.0 # Google API (Gemini)
pip install together # Together API
```
您还需要设置相应的环境变量以便正确进行 API 调用。要查看您需要设置的变量，请查看 `model_utils.py` 和相应的类（例如 `GeminiModel`）。

## 数据

<img width="1354" alt="benchmark_overview" src="assets/benchmark_overview.png">

您可以使用以下脚本下载数据：
```bash
bash scripts/download_data.sh
```
这将首先下载 .tar.gz 文件，然后将其解压到 `data` 目录。

数据托管在此 Huggingface [仓库](https://huggingface.co/datasets/princeton-nlp/HELMET) 上，存储了我们预处理的 jsonl 文件数据，约 34GB。对于回忆、RAG、段落重新排序和 ALCE，我们自己生成数据或进行检索，因此这些存储在 jsonl 文件中，而我们的脚本将为其他任务（LongQA、Summ 和 ICL）从 Huggingface 加载数据。数据还包含为基于模型的评估而提取的摘要关键点。

## 运行评估

要运行评估，只需使用 `configs` 目录中的配置文件之一，您还可以通过命令行覆盖配置文件中的任何参数或添加新参数（参见 `arguments.py`）：
```bash
for task in recall rag rerank cite longqa summ icl; do
  python eval.py --config configs/${task}.yaml \
    --model_name_or_path {本地模型路径或 huggingface 模型名称} \
    --output_dir {输出目录，默认为 output/{model_name}} \
    --use_chat_template False # 仅当您使用非指令调优模型时使用，否则使用默认值。
done
```

这将在输出目录下输出两个结果文件：`.json` 包含所有数据点详细信息，而 `.json.score` 仅包含聚合指标。

对于 slurm 用户，您可能会发现我们的 slurm 脚本很有用：
```bash
# 推荐使用这些 slurm 脚本，因为它们包含更多详细信息（包括所有模型名称）并且可以轻松修改以适合您的设置
# 您也可以通过将 sbatch 替换为 bash 在您的 shell 中运行它们，查看文件了解更多详细信息
sbatch scripts/run_eval_slurm.sh # 128k
sbatch scripts/run_short_slurm.sh # 8k-64k

# 对于 API 模型，请注意由于 API 调用的随机性，API 结果可能会有所不同
bash scripts/run_api.sh 
```

### 在 Intel Gaudi 加速器上运行
如果您想在配有 Intel Gaudi 的 vLLM 上启用评估，可以使用以下命令：
```bash
## 构建 vllm docker 镜像
cd scripts/vllm-gaudi
bash build_image.sh

## 启动 vllm 容器，根据需要更改 `LLM_MODEL_ID` 和 `NUM_CARDS`
bash launch_container.sh

## 评估
cd ../../
bash scripts/run_eval_vllm_gaudi.sh
```

查看脚本文件了解更多详细信息！
参见 [其他功能](#其他功能) 了解 slurm 脚本、轻松收集所有结果和使用 VLLM。

我们评估的完整结果在 [这里](https://docs.google.com/spreadsheets/d/1LBt6dP4UwZwU_CjoYhyAd_rjKhQLvo0Gq4cYUnpi_CA/edit?usp=sharing)。

测试了我们没有测试的模型？
请将结果文件发送给我，我会将它们添加到电子表格中！
参见 [联系方式](#联系方式) 获取我的电子邮件。

### 基于模型的评估

要对 LongQA 和 Summarization 运行基于模型的评估，请确保您已设置 OpenAI 的环境变量以便可以调用 GPT-4o，然后您可以运行：
```bash
# 默认情况下，我们假设所有输出文件都存储在 output/{model_name} 中
python scripts/eval_gpt4_longqa.py --model_name_or_path {本地模型路径或 huggingface 模型名称} --tag {模型标签}
python scripts/eval_gpt4_summ.py --model_name_or_path {本地模型路径或 huggingface 模型名称} --tag {模型标签}

# 或者，如果您想分片处理
bash scripts/eval_gpt4_longqa.sh
bash scripts/eval_gpt4_summ.sh
```

## 添加新模型

现有代码支持使用 HuggingFace 支持的模型和 API 模型（OpenAI、Anthropic、Google 和 Together）。要添加新模型或使用不同的框架（HuggingFace 以外），您可以修改 `model_utils.py` 文件。具体来说，您需要创建一个实现 `prepare_inputs`（如何处理输入）和 `generate` 函数的新类。然后，您可以向 `load_LLM` 添加新的情况。请参考现有的类作为示例。

## 添加新任务

要添加新的任务/数据集，您只需修改 `data.py` 文件：

创建一个指定如何加载数据的函数：
1. 通过 `user_template`、`system_template` 和 `prompt_template`（通常只是两者的串联）为任务指定字符串模板
2. 处理每个样本以适应指定的模板（tokenization 代码将调用 `user_template.format(**test_sample)` 和 `system_template` 同样）。重要的是，每个样本都应有一个 `context` 字段，如果输入过长将自动截断（例如，对于 QA，这是检索的段落；对于 NarrativeQA，这是书籍/脚本）。您应该使用 `question` 和 `answer` 字段来简化评估/打印。
3. 可选地，添加一个 `post_process` 函数来处理模型输出（例如，对于 MS MARCO，我们使用排序解析函数；对于 RULER，我们计算召回率）。还有一个 `default_post_process` 函数可以解析并计算简单指标如 EM 和 F1，您可以使用。此函数应接受模型输出和测试样本并返回 `(metrics, changed_output)` 的元组，`metrics`（例如 EM、ROUGE）在所有样本中聚合，`changed_output` 添加到 test_sample 并保存到输出文件。
4. 函数应返回 `{'data': [数据样本列表], 'prompt_template': prompt_template, 'user_template': user_template, 'system_template': system_template, 'post_process': [可选自定义函数]}`。

最后，只需向 `load_data` 函数添加新的情况，调用您刚刚编写的加载数据的函数。您可以参考现有任务作为示例（例如，`load_json_kv`、`load_narrativeqa` 和 `load_msmarco_rerank`）。

## 数据集相关性分析

<img width="838" alt="task_correlation" src="assets/task_correlation.png">

我们还分析了不同数据集性能之间的相关性。
代码将很快发布。

## 其他功能

<details>

<summary>收集结果</summary>
要快速收集所有结果，您可以使用脚本：

```bash
python scripts/collect_results.py
```

您应该检查脚本了解更多详细信息并修改特定字段以适合您的需求。例如，您可以更改模型、任务配置、输出目录、标签等。

</details>

<details>

<summary>Slurm 脚本</summary>

我还包括了用于运行论文中所有实验的 slurm 脚本。您可以使用以下命令运行脚本：
```bash
sbatch scripts/run_eval_slurm.sh
sbatch scripts/run_short_slurm.sh
sbatch scripts/run_api.sh
```
请注意，您可能需要修改脚本以适合您的集群设置。例如：
 - `--array 0-1` 指定要运行的作业数，此索引对应于数组中的模型索引。
 - 您还可以使用 `MNAME="${S_MODELS[$M_IDX]}"` 或 `MNAME="${L_MODELS[$M_IDX]}"` 分别为短模型和长模型指定要运行的模型集。
 - `--gres=gpu:1` 指定您要使用的 GPU 数量，对于较大的模型，您可能需要更多 GPU（我们使用多达 8x80GB GPU）。
 - `--mail-user` 指定发送作业状态的电子邮件地址。
 - `source env/bin/activate` 指定要使用的虚拟环境。
 - `MODEL_NAME="/path/to/your/model/$MNAME"` 您应该在此处指定模型的路径。

</details>

<details>

<summary>使用 VLLM</summary>

要使用 VLLM 运行评估，您可以简单地在命令行中添加 `--use_vllm` 标志，如下所示：
```bash
python eval.py --config configs/cite.yaml --use_vllm
```
免责声明：
VLLM 可能比使用原生 HuggingFace 生成快得多；但是，我们发现结果可能略有不同，因此我们建议使用原生 HuggingFace 生成进行最终评估。论文中报告的所有结果都来自原生 HuggingFace 生成。对于生成更多标记的任务，加速更加明显（例如，摘要可能看到高达 2 倍的加速），而对于生成较少标记的任务，加速不太明显（例如，JSON KV 可能看到不到 5% 的加速）。

</details>

<details>

<summary>InfiniteBench 加载错误</summary>

如果您在不同模式下（在线 vs. 离线推理）加载 InfiniteBench 数据集时遇到错误，似乎是哈希函数中的错误。要解决此问题，您可以执行以下操作：
```bash
cd {cache_dir}/huggingface/datasets/xinrongzhang2022___infinitebench
ln -s default-819c8cda45921923 default-7662505cb3478cd4
```

</details>

## 联系方式

如果您有任何问题，请发送电子邮件至 `hyen@cs.princeton.edu`。如果您遇到任何问题，您也可以在此处提出问题。请尝试详细说明问题，以便我们能够更好更快地帮助您！

## 引用

如果您发现我们的工作有用，请引用我们：
```
@inproceedings{yen2025helmet,
      title={HELMET: How to Evaluate Long-Context Language Models Effectively and Thoroughly}, 
      author={Howard Yen and Tianyu Gao and Minmin Hou and Ke Ding and Daniel Fleischer and Peter Izsak and Moshe Wasserblat and Danqi Chen},
      year={2025},
      booktitle={International Conference on Learning Representations (ICLR)},
}
```

请同时引用原始数据集创建者，列在下方：
<details>

<summary>引用</summary>

```
@article{Liu2023LostIT,
  title={Lost in the Middle: How Language Models Use Long Contexts},
  author={Nelson F. Liu and Kevin Lin and John Hewitt and Ashwin Paranjape and Michele Bevilacqua and Fabio Petroni and Percy Liang},
  journal={Transactions of the Association for Computational Linguistics},
  year={2023},
  volume={12},
  pages={157-173},
  url={https://api.semanticscholar.org/CorpusID:259360665}
}

@inproceedings{
  hsieh2024ruler,
  title={{RULER}: What{\textquoteright}s the Real Context Size of Your Long-Context Language Models?},
  author={Cheng-Ping Hsieh and Simeng Sun and Samuel Kriman and Shantanu Acharya and Dima Rekesh and Fei Jia and Boris Ginsburg},
  booktitle={First Conference on Language Modeling},
  year={2024},
  url={https://openreview.net/forum?id=kIoBbc76Sy}
}

@inproceedings{mallen-etal-2023-trust,
    title = "When Not to Trust Language Models: Investigating Effectiveness of Parametric and Non-Parametric Memories",
    author = "Mallen, Alex  and
      Asai, Akari  and
      Zhong, Victor  and
      Das, Rajarshi  and
      Khashabi, Daniel  and
      Hajishirzi, Hannaneh",
    editor = "Rogers, Anna  and
      Boyd-Graber, Jordan  and
      Okazaki, Naoaki",
    booktitle = acl,
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-long.546",
    doi = "10.18653/v1/2023.acl-long.546",
    pages = "9802--9822",
}

@inproceedings{yang-etal-2018-hotpotqa,
    title = "{H}otpot{QA}: A Dataset for Diverse, Explainable Multi-hop Question Answering",
    author = "Yang, Zhilin  and
      Qi, Peng  and
      Zhang, Saizheng  and
      Bengio, Yoshua  and
      Cohen, William  and
      Salakhutdinov, Ruslan  and
      Manning, Christopher D.",
    editor = "Riloff, Ellen  and
      Chiang, David  and
      Hockenmaier, Julia  and
      Tsujii, Jun{'}ichi",
    booktitle = "Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing",
    month = oct # "-" # nov,
    year = "2018",
    address = "Brussels, Belgium",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/D18-1259",
    doi = "10.18653/v1/D18-1259",
    pages = "2369--2380",
}

@inproceedings{joshi2017triviaqa,
    title = "{T}rivia{QA}: A Large Scale Distantly Supervised Challenge Dataset for Reading Comprehension",
    author = "Joshi, Mandar  and
      Choi, Eunsol  and
      Weld, Daniel  and
      Zettlemoyer, Luke",
    editor = "Barzilay, Regina  and
      Kan, Min-Yen",
    booktitle = "Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2017",
    address = "Vancouver, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/P17-1147",
    doi = "10.18653/v1/P17-1147",
    pages = "1601--1611",
}

@inproceedings{petroni-etal-2021-kilt,
    title = "{KILT}: a Benchmark for Knowledge Intensive Language Tasks",
    author = {Petroni, Fabio  and Piktus, Aleksandra  and
      Fan, Angela  and Lewis, Patrick  and
      Yazdani, Majid  and De Cao, Nicola  and
      Thorne, James  and Jernite, Yacine  and
      Karpukhin, Vladimir  and Maillard, Jean  and
      Plachouras, Vassilis  and Rockt{\"a}schel, Tim  and
      Riedel, Sebastian},
    booktitle = "Proceedings of the 2021 Conference of the North American Chapter of the Association 
                 for Computational Linguistics: Human Language Technologies",
    month = jun,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.naacl-main.200",
    doi = "10.18653/v1/2021.naacl-main.200",
    pages = "2523--2544",
}

@article{kwiatkowski2019natural,
    title = "Natural Questions: A Benchmark for Question Answering Research",
    author = "Kwiatkowski, Tom  and
      Palomaki, Jennimaria  and
      Redfield, Olivia  and
      Collins, Michael  and
      Parikh, Ankur  and
      Alberti, Chris  and
      Epstein, Danielle  and
      Polosukhin, Illia  and
      Devlin, Jacob  and
      Lee, Kenton  and
      Toutanova, Kristina  and
      Jones, Llion  and
      Kelcey, Matthew  and
      Chang, Ming-Wei  and
      Dai, Andrew M.  and
      Uszkoreit, Jakob  and
      Le, Quoc  and
      Petrov, Slav",
    editor = "Lee, Lillian  and
      Johnson, Mark  and
      Roark, Brian  and
      Nenkova, Ani",
    journal = "Transactions of the Association for Computational Linguistics",
    volume = "7",
    year = "2019",
    address = "Cambridge, MA",
    publisher = "MIT Press",
    url = "https://aclanthology.org/Q19-1026",
    doi = "10.1162/tacl_a_00276",
    pages = "452--466",
}

@inproceedings{gao2023alce,
    title = "Enabling Large Language Models to Generate Text with Citations",
    author = "Gao, Tianyu  and
      Yen, Howard  and
      Yu, Jiatong  and
      Chen, Danqi",
    editor = "Bouamor, Houda  and
      Pino, Juan  and
      Bali, Kalika",
    booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.emnlp-main.398",
    doi = "10.18653/v1/2023.emnlp-main.398",
    pages = "6465--6488",
}

@inproceedings{stelmakh2022asqa,
    title = "{ASQA}: Factoid Questions Meet Long-Form Answers",
    author = "Stelmakh, Ivan  and
      Luan, Yi  and
      Dhingra, Bhuwan  and
      Chang, Ming-Wei",
    booktitle = "Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.emnlp-main.566",
    doi = "10.18653/v1/2022.emnlp-main.566",
    pages = "8273--8288",
}

@inproceedings{fan-etal-2019-eli5,
    title = "{ELI}5: Long Form Question Answering",
    author = "Fan, Angela  and
      Jernite, Yacine  and
      Perez, Ethan  and
      Grangier, David  and
      Weston, Jason  and
      Auli, Michael",
    booktitle = acl,
    year = "2019",
    url = "https://aclanthology.org/P19-1346",
    doi = "10.18653/v1/P19-1346",
    pages = "3558--3567",
}

@article{rubin2022qampari,
  title={{QAMPARI: An Open-domain Question Answering Benchmark for Questions with Many Answers from Multiple Paragraphs}},
  author={Rubin, Samuel Joseph Amouyal Ohad and Yoran, Ori and Wolfson, Tomer and Herzig, Jonathan and Berant, Jonathan},
  journal={arXiv preprint arXiv:2205.12665},
  year={2022},
  url="https://arxiv.org/abs/2205.12665"
}

@misc{bajaj2018ms,
      title={MS MARCO: A Human Generated MAchine Reading COmprehension Dataset}, 
      author={Payal Bajaj and Daniel Campos and Nick Craswell and Li Deng and Jianfeng Gao and Xiaodong Liu and Rangan Majumder and Andrew McNamara and Bhaskar Mitra and Tri Nguyen and Mir Rosenberg and Xia Song and Alina Stoica and Saurabh Tiwary and Tong Wang},
      year={2018},
      eprint={1611.09268},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url="https://arxiv.org/abs/1611.09268"
}

@article{kocisky2018narrativeqa,
    title = "The {N}arrative{QA} Reading Comprehension Challenge",
    author = "Ko{\v{c}}isk{\'y}, Tom{\'a}{\v{s}}  and
      Schwarz, Jonathan  and
      Blunsom, Phil  and
      Dyer, Chris  and
      Hermann, Karl Moritz  and
      Melis, G{\'a}bor  and
      Grefenstette, Edward",
    journal = "Transactions of the Association for Computational Linguistics",
    volume = "6",
    year = "2018",
    address = "Cambridge, MA",
    publisher = "MIT Press",
    url = "https://aclanthology.org/Q18-1023",
    doi = "10.1162/tacl_a_00023",
    pages = "317--328"
}

@inproceedings{
  shen2022multilexsum,
  title={Multi-LexSum: Real-world Summaries of Civil Rights Lawsuits at Multiple Granularities},
  author={Zejiang Shen and Kyle Lo and Lauren Yu and Nathan Dahlberg and Margo Schlanger and Doug Downey},
  booktitle={Thirty-sixth Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
  year={2022},
  url={https://openreview.net/forum?id=z1d8fUiS8Cr}
}

@misc{zhang2024inftybenchextendinglongcontext,
  title={$\infty$Bench: Extending Long Context Evaluation Beyond 100K Tokens}, 
  author={Xinrong Zhang and Yingfa Chen and Shengding Hu and Zihang Xu and Junhao Chen and Moo Khai Hao and Xu Han and Zhen Leng Thai and Shuo Wang and Zhiyuan Liu and Maosong Sun},
  year={2024},
  eprint={2402.13718},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2402.13718}, 
}

@inproceedings{li-roth-2002-learning,
    title = "Learning Question Classifiers",
    author = "Li, Xin  and
      Roth, Dan",
    booktitle = "{COLING} 2002: The 19th International Conference on Computational Linguistics",
    year = "2002",
    url = "https://aclanthology.org/C02-1150",
}

@article{Liu2019BenchmarkingNL,
  title={Benchmarking Natural Language Understanding Services for building Conversational Agents},
  author={Xingkun Liu and Arash Eshghi and Pawel Swietojanski and Verena Rieser},
  journal={ArXiv},
  year={2019},
  volume={abs/1903.05566},
  url={https://api.semanticscholar.org/CorpusID:76660838}
}

@inproceedings{casanueva-etal-2020-efficient,
    title = "Efficient Intent Detection with Dual Sentence Encoders",
    author = "Casanueva, I{\~n}igo  and
      Tem{\v{c}}inas, Tadas  and
      Gerz, Daniela  and
      Henderson, Matthew  and
      Vuli{\'c}, Ivan",
    editor = "Wen, Tsung-Hsien  and
      Celikyilmaz, Asli  and
      Yu, Zhou  and
      Papangelis, Alexandros  and
      Eric, Mihail  and
      Kumar, Anuj  and
      Casanueva, I{\~n}igo  and
      Shah, Rushin",
    booktitle = "Proceedings of the 2nd Workshop on Natural Language Processing for Conversational AI",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2020.nlp4convai-1.5",
    doi = "10.18653/v1/2020.nlp4convai-1.5",
    pages = "38--45",
}

@inproceedings{larson-etal-2019-evaluation,
    title = "An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction",
    author = "Larson, Stefan  and
      Mahendran, Anish  and
      Peper, Joseph J.  and
      Clarke, Christopher  and
      Lee, Andrew  and
      Hill, Parker  and
      Kummerfeld, Jonathan K.  and
      Leach, Kevin  and
      Laurenzano, Michael A.  and
      Tang, Lingjia  and
      Mars, Jason",
    editor = "Inui, Kentaro  and
      Jiang, Jing  and
      Ng, Vincent  and
      Wan, Xiaojun",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)",
    month = nov,
    year = "2019",
    address = "Hong Kong, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/D19-1131",
    doi = "10.18653/v1/D19-1131",
    pages = "1311--1316",
}

@article{ye25longproc,
    title={LongProc: Benchmarking Long-Context Language Models on Long Procedural Generation},
    author={Ye, Xi and Yin, Fangcong and He, Yinghui and Zhang, Joie and Yen, Howard and Gao, Tianyu and Durrett, Greg and Chen, Danqi},
    journal={arXiv preprint},
    year={2025}
}
```

</details>
