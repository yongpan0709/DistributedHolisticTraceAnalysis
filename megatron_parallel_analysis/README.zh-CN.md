# DHTA: megatron_parallel_analysis

> English documentation: [README.md](README.md)

`megatron_parallel_analysis` 是 HTA 面向 Megatron-LM Pipeline Parallel 训练场景的分布式 trace 性能分析工具。它以 Pipeline Parallel Group（PP group）为任务单元，将大规模多 rank trace 切分到不同 MPI 进程处理，在每个 PP group 内构建调用图、识别 pipeline stage、关联相邻 stage 的 P2P 通信，并输出可用于分析 pipeline bubble、stage 负载不均衡和异常耗时的报告。


## 功能概览

- **按 PP group 分发分析任务**：根据 TP / CP / EP / DP / PP 配置生成并行组。
- **MPI 多进程并行处理**：每个 MPI 进程负责一段 PP group，降低单进程加载和解析大规模 trace 的压力。
- **Pipeline 调度专项分析**：支持 `1f1b`、`1f1b-interleaved`、`1f1b-interleaved-epoverlap` 三类调度。
- **PP/EP group 分析**：为每个 PP/EP group 构建 HTA `Trace` / `CallGraph`，提取通信相关函数，关联相邻 stage 的 P2P send / recv，计算实际的通信耗时和等待耗时 wait time、fwd/bwd的 start ts 和 duration。
- **PP group 的报告分析与 trace 导出**：为每个 PP group 输出 `report-pp<id>.csv`、`report-pp<id>-detail.csv` 和 `pp<id>-trace.json`，用于分析 stage 等待、pipeline bubble、P2P 通信和负载差异。
- **EP group 的报告分析与 trace 导出**：启用 EP 分析后，为同一 PP stage 上的每个 EP group 输出 `ep_group_<ep-id>-pp_stage_<stage-id>.csv` GPU 时间统计报告，以及 `ep_group_<ep-id>-pp_stage_<stage-id>-trace.json` 合并 trace，能够基于 perfetto 可视化的直观比较各 EP rank 在每个 micro-batch 中的 GPU 起始时间和持续时间。
- **集群级聚合能力**：保留跨 MPI rank 聚合与异常检测逻辑，可进一步启用 PP group 间、layer 间的 straggler 分析。

- **异常节点/GPU/算子检测**
   - 比较同一 rank 上不同 layer 之间的耗时差异，定位时间不稳定的算子。
   - 比较同一时间不同 rank 之间的耗时差异，定位跨机器或跨卡的空间不稳定性。
   - 对比同一 EP group、同一 PP stage 和同一 micro-batch 下各 EP rank 的 GPU duration min/max/mean/std，识别 Expert 计算负载不均衡、慢卡或异常算子导致的 EP straggler。
   - 对比各 EP rank 的 GPU start ts min/max，识别执行起点偏移以及可能由同步、调度或通信等待引起的异常；再结合 EP group 合并 trace 定位具体 rank、micro-batch 和 forward/backward（或 VPP）阶段。

## 目录结构

```text
megatron_parallel_analysis/
├── README.md
├── README.zh-CN.md
├── install_hta.sh
├── distribute_trace_analysis.py
├── run_distributed_megatron_trace_analysis.py
├── trace_etl.py
├── megatron_pipeline_group_base.py
├── megatron_pipeline_group_1f1b.py
├── megatron_pipeline_group_1f1b_interleaved.py
├── megatron_pipeline_group_1f1b_interleaved_epoverlap.py
└── utils/
    ├── __init__.py
    ├── call_graph_utils.py
    ├── parallel_state.py
    ├── pipeline_parallel_utils.py
    ├── trace_filter_utils.py
    ├── trace_parse_cache.py
    └── utils.py
```

其中：

- `distribute_trace_analysis.py`：编排 PP group 和完整 EP group 的 MPI 分布式分析任务，并生成 PP/EP 分析结果。
- `run_distributed_megatron_trace_analysis.py`：PP/EP trace 分析的命令行入口。
- `trace_etl.py`：原始 trace 的并行清洗入口。
- `megatron_pipeline_group_base.py`：pipeline group 分析基类，提供 trace 解析、通信过滤、P2P 关联和报告生成等通用能力。
- `megatron_pipeline_group_1f1b.py`：普通 1F1B 调度分析及 rank 级 forward/backward GPU timeline 提取。
- `megatron_pipeline_group_1f1b_interleaved.py`：interleaved 1F1B 调度分析及 VPP GPU timeline 提取。
- `megatron_pipeline_group_1f1b_interleaved_epoverlap.py`：interleaved + EP overlap 调度的 PP 分析实现。
- `utils/trace_parse_cache.py`：rank 级 trace 解析缓存的读取、校验和写入。
- `install_hta.sh`：多机环境下的 HTA 安装辅助脚本。


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
cd megatron_parallel_analysis/
bash install_hta.sh <HolisticTraceAnalysis_Path>  # 需要hostfile

```

## 命令行参数

| 参数 | 是否必需 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--trace-dir` | 是 | 无 | 原始 trace 目录。 |
| `--tp` | 否 | `1` | Tensor Parallel size。 |
| `--pp` | 否 | `2` | Pipeline Parallel size。 |
| `--dp` | 否 | `8` | 总 Data Parallel size；必须能被 `--ep` 整除。 |
| `--ep` | 否 | `8` | Expert Parallel size。 |
| `--pp-schedule` | 否 | `1f1b` | Pipeline 调度方式，可选：`1f1b`、`1f1b-interleaved`、`1f1b-interleaved-epoverlap`。 |
| `--num-bs` | 否 | `16` | Micro batch 数量，传给分析器的 `micro_bs`。 |
| `--vpp` | 否 | `2` | Virtual Pipeline Parallel size，仅 interleaved 类调度使用。 |
| `--pp-group-id-range` | 否 | `None` | 仅分析指定 PP group 闭区间，格式为 `START END`。 |
| `--enable_ep_analysis` | 否 | 关闭 | 显式启用 MoE 模型的 EP 分析；默认只分析 PP group。`ep=1` 时按 dense 模型处理，不生成 EP 结果。 |
| `--rebuild-parse-cache` | 否 | 关闭 | 忽略并刷新本次分析涉及 ranks 的缓存。 |

`--enable_ep_analysis` 在 `ep>1` 时为 `1f1b` 和 `1f1b-interleaved` 调度启用 MoE EP 分析；`1f1b-interleaved-epoverlap` 当前即使指定该参数也仍只执行 PP 分析。未指定 `--pp-group-id-range` 时，会分析全部 PP groups 和全部完整 EP groups；指定范围时，范围必须完整覆盖一个 EP group 依赖的全部 PP groups。只与某个 group 部分相交时，程序会在处理 trace 前报错退出，不会使用不完整的 rank 集合计算 EP 统计；完整覆盖一个或多个 groups 时，会输出对应的 PP 和 EP 结果。

其中 `--dp` 表示总 Data Parallel size，`--ep` 表示 Expert Parallel size。EP 分析要求两者均为正数且满足 `dp % ep == 0`，内部 Expert Data Parallel size 按 `dp / ep` 计算。EP groups 及其 PP groups 均由 Megatron rank groups 映射得到，不依赖 rank 算术推断。

解析缓存默认启用。每个 rank 的缓存保存原始 trace 解析、符号解码和 CallGraph 构建后得到的 pristine main-stack DataFrame。重复运行（包括再次选择相同的 `--pp-group-id-range`）会加载有效的 rank 缓存，仅构建缺失或失效的 ranks。源 trace 的真实路径或文件元数据发生变化、backward annotation 变化，或 cache/parser schema 更新时，对应缓存会自动失效。通信事件过滤、micro-batch 标注、P2P 关联和报告生成仍会在每次运行中执行。如果修改了当前 parser version 尚未覆盖的解析或 CallGraph 行为，请使用 `--rebuild-parse-cache`。

## MPI 多进程运行示例：
大规模 trace 建议用 MPI 启动，让多个进程并行处理不同 PP group。

* `--map-by ppr:<cnt>:node` 表示每个节点配置的进程数 `<cnt>` 个 MPI 进程;
* `-np` 通常按 `hostfile` 中节点数（ip的个数）乘以每节点配置的进程数；
* 实际进程数不需要等于 PP group 数，DHTA工具会把 PP group 均分给可用 MPI 进程，最后一个 MPI 进程，可能会分到剩余不整除的部分 PP group。例如：启动 -np 3，但是有8个 PP groups时，会分成 3 + 3 + 2 的分配方式。
  
```bash
#两个节点，每个节点上启动8个MPI 进程的配置
mpirun -allow-run-as-root -np 16 --bind-to none \
  --hostfile ./hostfile \
  --map-by ppr:8:node \
  --wdir /path/to/HolisticTraceAnalysis \
  ...
```

### ETL 数据清洗

`trace_etl.py` 用于 trace 预处理，会先过滤原始 trace 中的噪声事件，将清洗后的 trace 输出到同级的 `<trace-dir>-etl` 目录。

单机运行示例：

```bash
python -m megatron_parallel_analysis.trace_etl \
  --trace-dir /path/to/trace-dir \
  --tp 1 \
  --pp 4 \
  --dp 2 \
  --ep 8
```

```bash
mpirun -allow-run-as-root -np 2 --bind-to none \
  --hostfile ./hostfile \
  --map-by ppr:1:node \
  --wdir /path/to/HolisticTraceAnalysis \
  python -m megatron_parallel_analysis.trace_etl \
    --trace-dir /path/to/trace-dir \
    --tp 1 --pp 4 --dp 2 --ep 8
```

不要直接执行 `python megatron_parallel_analysis/trace_etl.py`。该文件使用了包导入，应该在仓库根目录下或已可导入该包的环境中，通过 `python -m megatron_parallel_analysis.trace_etl` 启动。

### Pipeline Parallel Group 分析

```bash
mpirun -allow-run-as-root -np 2 --bind-to none \
  --hostfile ./hostfile \
  --map-by ppr:1:node \
  --wdir /path/to/HolisticTraceAnalysis \
  python -m megatron_parallel_analysis.run_distributed_megatron_trace_analysis \
    --trace-dir /path/to/trace-dir \
    --tp 1 --pp 4 --dp 16 --ep 8 --num-bs 16
```

更多进程示例：

```bash
#两个节点，每个节点上启动8个MPI 进程的配置
mpirun -allow-run-as-root -np 16 --bind-to none \
  --hostfile ./hostfile \
  --map-by ppr:8:node \
  --wdir /path/to/HolisticTraceAnalysis \
  python -m megatron_parallel_analysis.run_distributed_megatron_trace_analysis \
    --trace-dir /path/to/trace-dir \
    --tp 1 --pp 31 --dp 24 --ep 8 --num-bs 128
```



### 分析结果与输出结构

运行后会在当前工作目录下创建workspace目录：

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
        ├── report-pp1.csv
        ├── ep_group_stats/
        │   └── ep_group_<ep-id>-pp_stage_<stage-id>.csv
        └── ep_trace/
            └── ep_group_<ep-id>-pp_stage_<stage-id>-trace.json
```

主要输出：

1. `log/log_mpirun_parallel_<rank>.log`：每个 MPI 进程的分析日志。
2. `trace/pp_group_<id>/`：当前 PP group 对应 rank trace 的分组目录。
3. `trace/pp<id>-trace.json`：保留 P2P 通信信息后的 PP group trace。
4. `trace/report-pp<id>.csv`：当前 PP group 的摘要版 pipeline 分析报告。
5. `trace/report-pp<id>-detail.csv`：同一个 PP group 的 bubble 明细拆分报告。
6. `trace/ep_group_stats/ep_group_<ep-id>-pp_stage_<stage-id>.csv`：仅当 `--enable_ep_analysis` 对 `1f1b` 或 `1f1b-interleaved` 调度生效时，生成每个 EP group 聚合后的 GPU 时间统计数据。
7. `trace/ep_trace/ep_group_<ep-id>-pp_stage_<stage-id>-trace.json`：在相同条件下生成同一个 EP group、同一个 PP stage 的合并 trace，恰好包含该 EP group 的 `ep_size` 个已处理 rank。
8. `output/stragglers/`：启用聚合与异常分析后保存 straggler 图表和结果。
9. EP group statistics 文件中的 `ep-id` 是 `RankGenerator.get_ranks('ep')` 返回列表的索引，`stage-id` 是 rank 在其 PP group 中的位置。每一行表示一个按 `micro_batch_id` 排序的 micro-batch，并汇总该 EP group 中全部 rank 的数据。普通 `1f1b` 的事件为 `forward` 和 `backward`；`1f1b-interleaved` 的事件依次为 `forward_vpp0` 至 `forward_vpp<N-1>`，再到 `backward_vpp0` 至 `backward_vpp<N-1>`。每个事件包含 `<event>_start_min_ms`、`<event>_start_max_ms`、`<event>_duration_min_ms`、`<event>_duration_max_ms`、`<event>_duration_mean_ms` 和 `<event>_duration_std_ms` 六列。start 来自绝对 GPU 时间戳 `first_kernel_start`，duration 来自 `kernel_span`，二者都会从 trace 的微秒单位转换为毫秒。duration 标准差按 EP ranks 的总体标准差计算（`ddof=0`）。CSV 数值保留三位小数。若 rank、micro-batch 或时间区间缺失、重复、格式错误或无效，会校验失败且不生成残缺文件。
10. 使用 `--pp-group-id-range` 时，只有所选范围包含某个 EP group 所需的全部 PP groups，才会生成对应的 EP group statistics CSV 和合并 EP group JSON。若范围与某个 group 相交但未覆盖其全部 PP groups，程序会在处理任何已选 trace 前校验失败。合并 JSON 中的 rank 由 descriptor 的 `ep_group_id` 和 `pp_stage_id` 确定，恰好包含 `ep_size` 个已处理 rank；interleaved trace 中保留 `forward_step_mb<mb>_vpp<vpp>` / `backward_step_mb<mb>_vpp<vpp>` 事件名。`1f1b` 和 `1f1b-interleaved` 均按完整 EP group 分配 MPI 任务，group 内仍逐个解析 PP trace，以控制内存峰值。

### `ep_group_<ep-id>-pp_stage_<stage-id>.csv` EP 聚合统计列说明

[`distribute_trace_analysis.py:491-628`](distribute_trace_analysis.py#L491-L628) 生成的 EP 聚合统计报告以每个 micro-batch 一行的形式，汇总同一个 EP group 中全部 rank 的 GPU 时间数据。文件名中的 `ep-id` 是 `RankGenerator.get_ranks('ep')` 返回列表的索引，`stage-id` 是这些 rank 在各自 PP group 中的 pipeline stage 位置。

| 标识 | CSV 列名 | 含义 |
| --- | --- | --- |
| A | `micro_batch_id` | 当前行对应的 micro-batch 编号，从 `0` 开始。 |
| B | `<event>_start_min_ms` | 当前事件在 EP group 全部 rank 中最早的 GPU 开始时间，即 `first_kernel_start` 的最小值，单位为毫秒。 |
| C | `<event>_start_max_ms` | 当前事件在 EP group 全部 rank 中最晚的 GPU 开始时间，即 `first_kernel_start` 的最大值，单位为毫秒。`C - B` 可用于衡量各 rank 进入该事件的时间偏差。 |
| D | `<event>_duration_min_ms` | 当前事件在 EP group 全部 rank 中最短的 GPU kernel span，单位为毫秒。 |
| E | `<event>_duration_max_ms` | 当前事件在 EP group 全部 rank 中最长的 GPU kernel span，单位为毫秒。`E - D` 可用于衡量各 rank 的耗时极差。 |
| F | `<event>_duration_mean_ms` | 当前事件在 EP group 全部 rank 中 GPU kernel span 的算术平均值，单位为毫秒。 |
| G | `<event>_duration_std_ms` | 当前事件在 EP group 全部 rank 中 GPU kernel span 的总体标准差，使用 `ddof=0` 计算，单位为毫秒。该值越大，表示 rank 间耗时差异越明显。 |

其中，`<event>` 会根据 pipeline 调度方式展开：

| 调度方式 | `<event>` 取值及列顺序 |
| --- | --- |
| `1f1b` | `forward`、`backward` |
| `1f1b-interleaved` | `forward_vpp0` 至 `forward_vpp<N-1>`，然后是 `backward_vpp0` 至 `backward_vpp<N-1>`；`N` 为 VPP size。 |

因此，每个事件都会重复生成 B～G 六类统计列。`start` 是 trace 中的绝对 GPU 时间戳，可用于比较同一份 trace 内各 rank 的事件到达时间；`duration` 使用从首个 kernel 开始到最后一个 kernel 结束的 `kernel_span`，其中可能包含 kernel 之间的空闲或等待时间，并不等同于所有 kernel 执行时长之和。原始 trace 的微秒值在写入 CSV 时统一转换为毫秒，数值保留三位小数。

### `report-pp<id>.csv` 摘要列说明

[`megatron_pipeline_group_base.py:343-432`](megatron_pipeline_group_base.py#L343-L432) 生成的摘要报告以每个 rank 一行的形式输出以下列：

| 标识 | CSV 列名 | 含义 |
| --- | --- | --- |
| A | `Global rank in a pp group, Rank_(i) + pp_size = Rank_(i+1)` | 当前 PP group 内的全局 rank 编号。相邻 pipeline stage 的 rank 相差 `pp_size`。 |
| B | `Elapsed time per iteration` | iteration 的端到端耗时。 |
| C | `Micro-Batch count` | iteration 中的micro-batch 数量。 |
| D | `Sum(Micro-Batch_forward_time + Micro-Batch_backward_time)` | 所有 micro-batch 的 forward 与 backward 计算时间总和。 |
| E | `PP SendRecv time` | PP send/recv 总耗时，包含真实传输时间和 send/recv bubble 时间。 |
| F | `Finalize_model_grads_step_time` | `finalize_model_grads` 的耗时。 |
| G | `Should_run_forward_backward_time` | `should_run_forward_backward` 的耗时。 |
| H | `Logical_and_across_model_parallel_group_time` | `logical_and_across_model_parallel_group` 的耗时。 |
| I | `Optimizer_time` | optimizer 步骤耗时。 |
| J | `Compute time total / Elapsed time per iteration` | 单次 iteration 中计算时间占比，即 `D / B`。 |
| K | `PP SendRecv time / Elapsed time per iteration` | 单次 iteration 中 PP 通信时间占比，即 `E / B`。 |
| L | `Finalize_model_grads_step_time / Elapsed time per iteration` | `F / B`。 |
| M | `Should_run_forward_backward_time / Elapsed time per iteration` | `G / B`。 |
| N | `Logical_and_across_model_parallel_group_time / Elapsed time per iteration` | `H / B`。 |
| O | `Optimizer_time / Elapsed time per iteration` | `I / B`。 |

当前实现中，每个 rank 的 iteration 时间按如下关系建模：

`B ~ D + E + H + I`

这样可以让摘要报告聚焦在计算、PP send/recv、model parallel logical-and 同步和 optimizer 四类主要时间。

`J + K + N + O` 往往小于 100%。缺失的这部分通常来自 iteration 开始阶段的 all-reduce、all-gather 等 collective，它们没有被并入这四个比例项中。

### `report-pp<id>-detail.csv` bubble 明细列说明

与摘要报告同时生成的明细报告，给出了 [`megatron_pipeline_group_1f1b.py:130-257`](megatron_pipeline_group_1f1b.py#L130-L257) 使用的 bubble 拆分结果：

| 标识 | CSV 列名 | 含义 |
| --- | --- | --- |
| A | `Global rank in a pp group, Rank_(i) + pp_size = Rank_(i+1)` | 当前 PP group 内的全局 rank 编号。 |
| B | `Elapsed time per iteration` | 被分析 iteration 的端到端耗时。 |
| C | `Micro-Batch count` | 被分析 iteration 使用的 micro-batch 数量。 |
| D | `SUM(micro_batch_forward_time) / Micro-Batch count` | 单个 micro-batch 的平均 forward 时间。 |
| E | `STD(micro_batch_forward_time)` | 单个 micro-batch 的 forward 时间标准差。 |
| F | `SUM(micro_batch_backward_time) / Micro-Batch count` | 单个 micro-batch 的平均 backward 时间。 |
| G | `STD(micro_batch_backward_time)` | 单个 micro-batch 的 backward 时间标准差。 |
| H | `PP SendRecv time` | PP send/recv 总耗时。 |
| I | `Actual transfer time of send-recv` | send/recv 配对后的真实传输时间的总和，真实传输时间的计算方式为 `min(send, recv)`。 |
| J | `Bubble time of send-recv` | send/recv bubble 时间总和，Bubble 计算方式为 `max(send, recv) - min(send, recv)`。 |
| K | `Theoretical bubble time warmup` | warmup 阶段的理论 bubble。 |
| L | `Theoretical bubble time steady` | steady-state 阶段的理论 bubble。 |
| M | `Theoretical bubble time cooldown` | cooldown 阶段的理论 bubble。为了让 cooldown 阶段在各个 rank 上具有可比性，当前实现使用 `should_run_forward_backward` 的结束时间作为整个 cooldown 区间的结束点。这样做是因为 `should_run_forward_backward` 内包含一次 all-reduce，可以将同一个 PP group 中的所有 rank 做同步，从而保证各 rank 的统计更容易满足 `B ~ D + E + H + I` 这组关系。但这也带来一个限制：在最后一个 backward step 结束之后，到 `should_run_forward_backward` 结束之前，还包含了其他几个模块：`finalize_model_grads` 、`should_run_forward_backward`。其中包含非 PP send/recv 的通信，而当前实现会把这部分时间计入 cooldown bubble。因此，当前 cooldown bubble 存在已知的近似误差。 |
| N | `Non-balanced bubble time warmup` | warmup 阶段超出理论 bubble 的额外等待时间，主要来自 stage 不均衡或额外等待。 |
| O | `Non-balanced bubble time steady` | steady-state 阶段超出理论 bubble 的额外等待时间，主要来自 stage 不均衡或额外等待。 |
| P | `Non-balanced bubble time cooldown` | cooldown 阶段超出理论 bubble 的额外等待时间，主要来自 stage 不均衡或额外等待。 |
| Q | `Theoretical bubble time / elapsed time per iteration` | `(K + L + M) / B`。 |
| R | `Theoretical bubble time / SUM(micro_batch_forward_time + micro_batch_backward_time)` | `(K + L + M) / (compute_time_total + PP SendRecv time)`。 |
| S | `bubble_ratio in paper` | 按论文公式计算的理论 bubble 占比。 |

该明细报告满足以下关系：

- `H = I + J`
- `I = sum(all peer {min(send, recv)})`
- `J = sum(all peer {max(send, recv) - min(send, recv)}) ` + `finalize_model_grads` + `should_run_forward_backward`
- `J = K + L + M + N + O + P`

这个拆分主要用于区分两类流水线性能损失来源：

1. **理论空泡**：由 pipeline 调度模型本身决定，对应 `K + L + M`。
2. **非均衡空泡**：由 stage 负载不均衡或额外等待导致，对应 `N + O + P`。

在实践中，PP 算法优化主要作用于 `K`、`L`、`M`，而 PP overlap 类优化可以降低 `N`、`O`、`P`，以及部分 `I`。


### 下一步优化点

`finalize_model_grads`、`should_run_forward_backward`过程中的非 Pipeline 相关通信，需要进一步拆分为真实传输时间与等待时间。

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

### 基于Pipeline Parallel Group 分析生成的csv数据，问题分析prompt参考建议

workspace/*** 路径下面是大模型预训练过程中，某一次采集的profile traces，通过当前megatron_parallel_analysis路径下，dhta分析 Pipeline Parallel 的统计数据。每一个csv文件，包含的内容信息介绍可以参考 megatron_parallel_analysis/README.md 中，report-pp<id>-detail.csv bubble 明细列说明章节介绍了每个列的数据意义。对比分析一下现在路径下 workspace/*** 下面*个pp group的统计信息，是否存在什么异常点, 形成一个可量化的分析报告，并写入md文件。

### 基于Expert Pipeline 分析生成的csv数据，问题分析prompt参考建议
workspace/***/trace/ep_group_stats 路径下面是大模型预训练过程中，某一次采集的profile traces，通过当前megatron_parallel_analysis路径下，dhta分析 Pipeline+Expert Parallel 的统计数据。每一个csv文件，包含的内容信息介绍可以参考 megatron_parallel_analysis/README.zh-CN.md 中，ep_group_<ep-id>-pp_stage_<stage-id>.csv EP 聚合统计列说明章节介绍了每个列的数据意义。对比分析一下现在路径下 workspace/ 下面*个ep group的统计信息，是否存在什么异常点, 形成一个可量化的分析报告，并写入md文件。

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

### `trace_etl.py`

ETL 数据清洗命令行入口，详见上文“ETL 数据清洗”。

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

