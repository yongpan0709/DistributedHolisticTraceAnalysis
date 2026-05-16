# megatron_parallel_analysis

> English documentation: [README.md](README.md)

`megatron_parallel_analysis` 是 HTA 面向 Megatron-LM Pipeline Parallel 训练场景扩展出的分布式 trace 分析模块。它以 Pipeline Parallel Group（PP group）为任务单元，将大规模多 rank trace 切分到不同 MPI 进程处理，在每个 PP group 内构建调用图、识别 pipeline stage、关联相邻 stage 的 P2P 通信，并输出可用于分析 pipeline bubble、stage 负载不均衡和异常耗时的报告。

该目录只包含 Megatron 分布式与 pipeline 专项分析逻辑；基于 PyTorch callstack 模板的模型层级、kernel 层级统计位于 [`pytorch_callstack_analysis/`](../pytorch_callstack_analysis/README.zh-CN.md)。

## 功能概览

- **按 PP group 分发分析任务**：根据 TP / CP / EP / DP / PP 配置生成并行组，把原始 rank trace 软链接或分组到 `workspace/<trace-name>/trace/pp_group_<id>/`。
- **MPI 多进程并行处理**：每个 MPI 进程负责一段 PP group，降低单进程加载和解析大规模 trace 的压力。
- **Pipeline 调度专项分析**：支持 `1f1b`、`1f1b-interleaved`、`1f1b-interleaved-epoverlap` 三类调度。
- **节点内 PP group 分析**：为每个 PP group 构建 HTA `Trace` / `CallGraph`，提取通信相关 trace，设置 micro batch ID，并关联相邻 stage 的 P2P send / recv。
- **报告与 trace 导出**：为每个 PP group 输出 `report-pp<id>.csv` 和 `pp<id>-trace.json`，用于观察 stage 等待、bubble、通信和负载差异。
- **集群级聚合能力**：保留跨 MPI rank 聚合与异常检测逻辑，可进一步启用 PP group 间、layer 间的 straggler 分析。

- **异常节点/GPU/算子检测**
   - 比较同一 rank 上不同 layer 之间的耗时差异，定位时间不稳定的算子。
   - 比较同一时间不同 rank 之间的耗时差异，定位跨机器或跨卡的空间不稳定性。

## 目录结构

```text
megatron_parallel_analysis/
├── distribute_trace_analysis.py
├── run_distributed_megatron_trace_analysis.py
├── megatron_pipeline_group_base.py
├── megatron_pipeline_group_1f1b.py
├── megatron_pipeline_group_1f1b_interleaved.py
├── megatron_pipeline_group_1f1b_interleaved_epoverlap.py
└── utils/
    ├── call_graph_utils.py
    ├── parallel_state.py
    ├── pipeline_parallel_utils.py
    ├── trace_filter_utils.py
    └── utils.py
```

## 典型工作流

```bash
# 1. 获取代码
git clone -b v0.6.1-musa0.0.1 https://sh-code.mthreads.com/ai/HolisticTraceAnalysis
cd HolisticTraceAnalysis

# 如需切换到特定分支，请按实际开发分支执行 git checkout

# 安装单机
pip install -r requirements.txt
pip install -e .

# 构建wheel
pip wheel . --wheel-dir=dist/ --no-deps --use-pep517 --no-build-isolation

# 或直接安装whl
pip install traceinsight-*-py3-none-any.whl -i https://pypi.tuna.tsinghua.edu.cn/simple
# 安装多机(当1000卡甚至更大规模的trace需要分析时，可以支持多机并行分析)
cd musa_examples/
bash install_hta.sh <HolisticTraceAnalysis_Path>  # 需要hostfile

```

## 核心模块

### `distribute_trace_analysis.py`

分布式 Megatron trace 分析的核心编排模块，主要类是 `DistributedMegatronTraceAnalysis`。

初始化时会完成：

1. 记录 trace 目录和 TP / CP / EP / DP / PP / VPP / micro batch 配置；
2. 初始化 `MPI.COMM_WORLD`，获取当前 MPI rank、world size 和节点名；
3. 使用 `RankGenerator` 生成 DP / TP / PP 并行组；
4. 创建 workspace、output、log、stragglers 目录；
5. 按 PP group 将 trace 文件分组；
6. 将 PP group 任务切分给不同 MPI 进程。

每个 PP group 的处理流程是：

1. 根据 `--pp-schedule` 创建对应的 pipeline trace 对象；
2. 解析当前 PP group 内的 rank trace 并构建调用图；
3. 过滤并保留通信相关 trace；
4. 设置 micro batch ID；
5. 建立相邻 pipeline stage 之间的 P2P 链接；
6. 输出通信 trace JSON 和 CSV 报告。

### `run_distributed_megatron_trace_analysis.py`

可配置的命令行入口，直接创建 `DistributedMegatronTraceAnalysis` 并调用 `analyze()`。

常用示例：

```bash
python -m megatron_parallel_analysis.run_distributed_megatron_trace_analysis \
  --trace-dir /path/to/trace-dir \
  --tp 1 \
  --pp 4 \
  --dp 2 \
  --ep 8 \
  --num-bs 16 \
  --pp-schedule 1f1b
```

只分析部分 PP group：

```bash
python -m megatron_parallel_analysis.run_distributed_megatron_trace_analysis \
  --trace-dir /path/to/trace-dir \
  --tp 1 \
  --pp 4 \
  --dp 2 \
  --ep 8 \
  --num-bs 16 \
  --pp-schedule 1f1b \
  --pp-group-id-range 0 3
```

Interleaved / EP overlap 场景可额外指定 VPP：

```bash
python -m megatron_parallel_analysis.run_distributed_megatron_trace_analysis \
  --trace-dir /path/to/trace-dir \
  --tp 1 \
  --pp 2 \
  --dp 1 \
  --ep 8 \
  --num-bs 16 \
  --vpp 2 \
  --pp-schedule 1f1b-interleaved
```

### Pipeline group 分析类

- `megatron_pipeline_group_base.py`：PP group trace 分析基类，提供 trace 解析、通信过滤、micro batch 标注、P2P 链接和报告生成的共用能力。
- `megatron_pipeline_group_1f1b.py`：Megatron `1f1b` 调度分析。
- `megatron_pipeline_group_1f1b_interleaved.py`：Megatron interleaved 1F1B 调度分析。
- `megatron_pipeline_group_1f1b_interleaved_epoverlap.py`：Megatron interleaved + EP overlap 调度分析。

### `utils/`

- `parallel_state.py`：生成 Megatron 风格的 TP / CP / EP / DP / PP rank group。
- `pipeline_parallel_utils.py`：pipeline stage、micro batch、P2P 通信等辅助逻辑。
- `trace_filter_utils.py`：trace 过滤工具。
- `call_graph_utils.py`：从 HTA `CallGraph` 中定位 main stack 等辅助函数。
- `utils.py`：目录准备、trace 文件分组等通用工具。

## 命令行参数

| 参数 | 是否必需 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--trace-dir` | 是 | 无 | 原始 trace 目录。 |
| `--tp` | 否 | `1` | Tensor Parallel size。 |
| `--pp` | 否 | `2` | Pipeline Parallel size。 |
| `--dp` | 否 | `1` | Data Parallel size。 |
| `--ep` | 否 | `8` | Expert Parallel size。 |
| `--pp-schedule` | 否 | `1f1b` | Pipeline 调度方式，可选：`1f1b`、`1f1b-interleaved`、`1f1b-interleaved-epoverlap`。 |
| `--num-bs` | 否 | `16` | Micro batch 数量，传给分析器的 `micro_bs`。 |
| `--vpp` | 否 | `2` | Virtual Pipeline Parallel size，仅 interleaved 类调度使用。 |
| `--pp-group-id-range` | 否 | `None` | 仅分析指定 PP group 闭区间，格式为 `START END`。 |

## MPI 多机/多进程运行

大规模 trace 建议用 MPI 启动，让多个进程并行处理不同 PP group。

```bash
mpirun -allow-run-as-root -np 2 --bind-to none \
  --hostfile ./hostfile \
  --map-by ppr:1:node \
  --wdir /path/to/HolisticTraceAnalysis \
  python -m megatron_parallel_analysis.run_distributed_megatron_trace_analysis \
    --trace-dir /path/to/trace-dir \
    --tp 1 --pp 4 --dp 2 --ep 8 --num-bs 16
```

更多进程示例：

```bash
mpirun -allow-run-as-root -np 16 --bind-to none \
  --hostfile ./hostfile \
  --map-by ppr:8:node \
  --wdir /path/to/HolisticTraceAnalysis \
  python -m megatron_parallel_analysis.run_distributed_megatron_trace_analysis \
    --trace-dir /path/to/trace-dir \
    --tp 1 --pp 31 --dp 3 --ep 8 --num-bs 128
```

`-np` 通常按 `hostfile` 中节点数乘以每节点进程数配置；`--map-by ppr:<cnt>:node` 表示每台机器启动 `<cnt>` 个 MPI 进程。实际进程数不需要等于 PP group 数，分析器会把 PP group 均分给可用 MPI 进程。

## Workspace 与输出结构

运行后会在当前工作目录下生成：

```text
workspace/
└── <trace-dir-name>/
    ├── log/
    │   └── log_mpirun_parallel_<rank>.log
    ├── output/
    │   └── stragglers/
    └── trace/
        ├── pp_group_0/
        │   └── output/
        ├── pp_group_1/
        │   └── output/
        ├── pp0-trace.json
        ├── pp1-trace.json
        ├── report-pp0.csv
        └── report-pp1.csv
```

主要输出：

1. `log/log_mpirun_parallel_<rank>.log`：每个 MPI 进程的分析日志。
2. `trace/pp_group_<id>/`：当前 PP group 对应 rank trace 的分组目录。
3. `trace/pp<id>-trace.json`：保留 P2P 通信信息后的 PP group trace。
4. `trace/report-pp<id>.csv`：当前 PP group 的 pipeline 分析报告。
5. `output/stragglers/`：启用聚合与异常分析后保存 straggler 图表和结果。

## 运行前检查项

1. **Trace 文件完整性**：确认每个 rank 的 trace 文件存在，命名能被 HTA 的 `get_trace_files()` 识别。
2. **并行配置一致性**：`--tp`、`--pp`、`--dp`、`--ep`、`--vpp`、`--num-bs` 必须和训练任务一致。
3. **调度方式匹配**：`--pp-schedule` 需要和 Megatron 实际 pipeline 调度一致。
4. **共享路径可见性**：多机运行时，代码目录、trace 目录和 workspace 所在路径应对所有节点可访问，或提前同步到相同路径。
5. **MPI 环境**：确认 `mpi4py`、`mpirun`、hostfile、SSH 免密和 Python 环境在各节点一致。
6. **资源容量**：大规模 trace 会占用较多内存和磁盘，建议先用 `--pp-group-id-range` 小范围验证。

## 分析建议

1. 先单进程或只选少量 PP group 验证 trace 命名、并行配置和 schedule 是否正确。
2. 再扩大到多进程或多机运行，观察每个 `log_mpirun_parallel_<rank>.log` 是否完成。
3. 优先查看 `report-pp<id>.csv` 中各 stage 的计算、通信、等待与 bubble 相关指标。
4. 若某个 PP group 异常，再用 `pp<id>-trace.json` 或 PyTorch callstack 分析脚本做单 rank / 单 stage 细查。
5. 若要启用全局 straggler 聚合，可在 `DistributedMegatronTraceAnalysis.analyze()` 中恢复 `gather_infos_from_all_ranks()`、`analyze_anomalies()` 和 `post_process()` 调用，并确认对应输出格式满足当前需求。

## 故障排查

- **trace 文件找不到**：检查 `--trace-dir`、rank 文件命名，以及 PP group 分组目录是否生成。
- **PP group 数量不符合预期**：检查 TP / CP / EP / DP / PP 的乘积和 rank order 是否与训练一致。
- **报告为空或 P2P 链接缺失**：检查 trace 是否包含目标 iteration、pipeline send / recv 事件和 GPU kernel 事件。
- **MPI 运行失败**：检查 hostfile、工作目录、Python 环境、`mpi4py` 安装和节点间 SSH 配置。
- **interleaved 结果异常**：确认 `--vpp` 与训练时 virtual pipeline parallel size 一致。
