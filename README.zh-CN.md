# Distributed Holistic Trace Analysis

Distributed Holistic Trace Analysis（DHTA）是一个面向大规模分布式训练的性能分析工具。项目基于 [Facebook Research HolisticTraceAnalysis](https://github.com/facebookresearch/HolisticTraceAnalysis) 开发，面向现代大模型预训练和强化学习系统，扩展了 Megatron-LM 并行分析、PP/VPP 关键路径建模，以及 PyTorch Callstack 和 Kernel 下钻分析能力。

DHTA 的输入是通过 [PyTorch Profiler / Kineto](https://github.com/pytorch/kineto) 采集的 Trace。它将 Trace 级性能统计、分布式并行组分析、模型级调用栈分析和关键路径 What-if 建模整合到一个工具中。

## DHTA 新增功能

### 1. Megatron 分布式并行分析

[`megatron_parallel_analysis/`](megatron_parallel_analysis/README.md) 将 Trace 分析从单个 Rank 扩展到大规模 Megatron-LM 训练任务，主要能力包括：

- **基于 PP Group 的分析**：根据 TP、CP、EP、DP、PP 配置生成 Megatron 风格的 Rank Group，并以 Pipeline Parallel Group 为分析单元。
- **MPI 多进程和多节点执行**：将不同 PP Group 分配给不同 MPI 进程，降低大规模 Trace 分析时的单进程内存和解析压力。
- **Pipeline 调度支持**：支持普通 `1f1b`、交错式 `1f1b-interleaved` 和 `1f1b-interleaved-epoverlap` 调度，并支持 VPP 时间线分析。
- **Pipeline 通信和 Bubble 分析**：关联相邻 Stage 之间的 P2P Send/Recv 事件，统计计算时间、传输时间、等待时间、理论 Bubble、非均衡 Bubble 和 Stage 间负载不均衡。
- **Expert Parallel 分析**：针对支持的 MoE 调度，汇总不同 EP Rank 的 GPU 启动时间和执行时长，并导出合并后的 EP Trace，便于使用 Perfetto 进行对比。
- **Straggler 和异常分析**：比较不同 Layer、Rank 和 EP Rank 的执行时间分布，定位慢 GPU、Expert 负载不均衡、同步等待和异常 Operator。
- **Trace ETL 和解析缓存**：清理原始 Trace 中的噪声事件，并复用经过校验的 Rank 级解析缓存，减少重复解析开销。

详细命令、输出格式、MPI 示例和结果解读请参考 [Megatron 并行分析文档](megatron_parallel_analysis/README.md)。

### 2. PP/VPP 关键路径和优化收益建模

[`megatron_parallel_analysis/pp_critical_path_analysis/`](megatron_parallel_analysis/pp_critical_path_analysis/) 将 PP/VPP 执行过程建模为依赖 DAG，主要能力包括：

- 从 DHTA 生成的 PP Trace 中提取关键时间信息；
- 构建 PP/VPP 节点、依赖边和拓扑执行顺序；
- 对当前关键路径上的指定 Forward 事件设置目标耗时；
- 每轮重新计算最长路径，估算整个 Training Step 的收益；
- 当关键路径转移到其他 Stage 或 Micro-Batch 时，识别下一处瓶颈；
- 输出 DAG、每轮关键路径、机器可读分析数据和 Markdown 报告。

该功能是基于固定依赖图的 What-if 模型，不是对真实执行结果的替代。预测的收益应在完成实际优化后重新采集 Trace 进行验证。模型假设、限制和示例请参考 [PP/VPP 关键路径分析文档](megatron_parallel_analysis/pp_critical_path_analysis/README.md)。

### 3. PyTorch Callstack 和 Kernel 分析

[`pytorch_callstack_analysis/`](pytorch_callstack_analysis/README.md) 用于在分布式分析定位到目标 Rank、Pipeline Stage 或 Operator 后，进行模型级和 Kernel 级的局部下钻分析：

- 使用基于缩进的 Callstack 模板匹配 HTA Call Graph 中的模型区域；
- 支持使用 `@dup@` 标记区分重复函数，使用 `@shape@` 标记提取输入 Shape；
- 输出 Forward/Backward 耗时统计、分位数、最大值、最小值、调用次数和区域占比；
- 提取 Kernel Shape，并估算 MUSA Kernel 的 TFLOPS 或内存带宽；
- 提供 Forward/Backward 匹配工具，以及包含 MPI 支持的计时工具。

该模块既支持单 Rank 分析，也支持分析 Trace 目录中发现的全部 Rank。模板和命令行示例请参考 [PyTorch Callstack 分析文档](pytorch_callstack_analysis/README.md)。

## 保留的 HTA 原生能力

DHTA 保留了原始 HTA 面向 PyTorch Profiler Trace 的核心分析能力，包括：

- GPU 时间和 Kernel Breakdown；
- GPU Idle Time 归因；
- 通信与计算重叠分析；
- 高频 CUDA Kernel 和 Kernel Launch 统计；
- Queue Length 和 Memory Bandwidth 时间序列及汇总统计；
- Trace 对比和可视化；
- 实验性的 CUPTI Counter 和 Roofline 分析。

这些能力可以独立用于快速性能概览，也可以与 DHTA 新增功能组合使用，形成从整体到局部的分析流程：

```text
原始 PyTorch Profiler Trace
        │
        ├── HTA 基础性能分析
        ├── Megatron PP/EP Group 分析
        │       └── 定位异常 Stage、Rank 或 Micro-Batch
        ├── PP/VPP 关键路径 What-if 分析
        │       └── 确定下一步优化目标
        └── PyTorch Callstack / Kernel 下钻
                └── 定位具体模型区域或 Kernel
```

## 安装

DHTA 支持 Linux 和 macOS，要求 Python 3.10 或更高版本。使用分布式 Megatron 分析功能时，还需要安装 `mpi4py`，并准备可用的 MPI 运行时，例如 Open MPI。

### 从源码安装

进入代码库根目录后安装依赖：

```bash
cd DistributedHolisticTraceAnalysis
pip install -r requirements.txt
pip install -e .
```

构建 Wheel：

```bash
pip wheel . --wheel-dir=dist/ --no-deps --use-pep517 --no-build-isolation
```

独立运行的 PP/VPP 关键路径脚本只依赖 Python 标准库。其输入是 DHTA 为一个完整 PP Group 生成的 `ppN-trace.json` 文件。

## 快速开始

### 分析 Megatron Pipeline Group

```bash
python -m megatron_parallel_analysis.run_distributed_megatron_trace_analysis \
  --trace-dir /path/to/trace-dir \
  --tp 1 --pp 4 --dp 16 --ep 8 \
  --num-bs 16 --pp-schedule 1f1b
```

对于大规模 Trace，可以通过 MPI 将 PP Group 分配到多个进程：

```bash
mpirun -np 16 --bind-to none \
  --hostfile ./hostfile \
  --map-by ppr:8:node \
  --wdir /path/to/DistributedHolisticTraceAnalysis \
  python -m megatron_parallel_analysis.run_distributed_megatron_trace_analysis \
    --trace-dir /path/to/trace-dir \
    --tp 1 --pp 31 --dp 24 --ep 8 --num-bs 128
```

对于支持的 MoE 调度，可以通过 `--enable_ep_analysis` 开启 EP 报告。建议先使用较小的 PP Group 范围验证 Trace 发现和并行配置，再扩展到完整任务。

### 分析模型 Callstack 或 Kernel

```bash
python -m pytorch_callstack_analysis.model_level_fwdbwd_statistics \
  --trace-dir /path/to/trace-dir \
  --rank 16 \
  --template kimi_epoverlap \
  --output-dir model_main_stack
```

```bash
python -m pytorch_callstack_analysis.kernel_level_fwdbwd_statistics \
  --trace-dir /path/to/trace-dir \
  --rank 32 \
  --output-dir kernel_fwdbwd_statistics
```

省略 `--rank` 时，将分析 Trace 目录中发现的所有 Rank。

### 执行关键路径建模

```bash
python3 megatron_parallel_analysis/pp_critical_path_analysis/iterative_critical_path.py \
  --trace /path/to/pp36-trace.json \
  --threshold-ms 70 --target-ms 50 --max-rounds 10 \
  --output /tmp/pp-critical-path
```

输出包括 `report.md`、`analysis.json`、`dag.json`，以及每条代表性关键路径对应的 CSV 文件。

## 推荐分析流程

1. 为预训练或强化学习任务采集完整的 PyTorch Profiler Trace。
2. 运行 HTA 原生分析，建立 GPU 利用率、通信重叠、Idle Time 和 Kernel 性能基线。
3. 运行 Megatron PP/EP 分析，从大规模任务中比较不同 Stage、Micro-Batch、Rank 和 Expert Group。
4. 使用 PP/VPP 关键路径模型，估算哪些耗时优化能够最大程度降低完整 Training Step 的执行时间。
5. 使用 Callstack 和 Kernel 分析工具对目标 Rank 或 Stage 进行下钻。
6. 实施优化后重新采集 Trace，验证模型预测的收益。

## 输出目录概览

分布式 Megatron 分析通常会生成包含分组 Trace、日志和报告的工作目录，结构类似于：

```text
workspace/<trace-name>/
├── log/                          # 每个 MPI 进程的日志
├── output/stragglers/            # 可选的异常和 Straggler 结果
└── trace/
    ├── pp_group_<id>/             # 各 PP Group 的 Trace
    ├── pp<id>-trace.json          # 包含 P2P 关联信息的 PP Trace
    ├── report-pp<id>.csv          # PP 汇总报告
    ├── report-pp<id>-detail.csv   # Bubble 详细分解
    ├── ep_group_stats/             # 可选的 EP 汇总统计
    └── ep_trace/                   # 可选的合并 EP Trace
```

## 文档

- [Megatron 并行分析](megatron_parallel_analysis/README.md)
- [PP/VPP 关键路径分析](megatron_parallel_analysis/pp_critical_path_analysis/README.md)
- [PyTorch Callstack 分析](pytorch_callstack_analysis/README.md)
- [原始 HTA 文档](https://hta.readthedocs.io/en/latest/index.html)

## 贡献

欢迎提交 Issue 和代码贡献。报告问题时，建议提供 DHTA 版本或 Commit、Python 和 MPI 版本、相关并行配置、Pipeline 调度方式，以及 Trace 目录结构的最小化描述。请勿上传包含业务机密的 Trace 或其他敏感训练数据。

## 许可证

DHTA 继承上游项目的 MIT 许可证条款。详细信息请参阅 [LICENSE](LICENSE)。
