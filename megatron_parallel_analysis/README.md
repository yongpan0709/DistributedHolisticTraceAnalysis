# DHTA: megatron_parallel_analysis

> 中文文档: [README.zh-CN.md](README.zh-CN.md)

`megatron_parallel_analysis` extends HTA for distributed trace analysis of Megatron-LM Pipeline Parallel training. It uses a Pipeline Parallel Group (PP group) as the analysis unit, splits large multi-rank traces across MPI processes, builds call graphs inside each PP group, identifies pipeline stages, links P2P communication between adjacent stages, and exports reports for analyzing pipeline bubbles, stage imbalance, and abnormal latency.

This directory only contains Megatron distributed and pipeline-specific analysis logic. Template-driven PyTorch callstack model-level and kernel-level statistics live in [`pytorch_callstack_analysis/`](../pytorch_callstack_analysis/README.md).

## Capabilities

- **Distribute analysis by PP group**: generate Megatron-style rank groups from TP / CP / EP / DP / PP settings and symlink or group rank traces under `workspace/<trace-name>/trace/pp_group_<id>/`.
- **MPI multi-process execution**: each MPI process handles a subset of PP groups, reducing the memory and parsing pressure of very large traces.
- **Pipeline schedule analysis**: supports `1f1b`, `1f1b-interleaved`, and `1f1b-interleaved-epoverlap`.
- **Per-PP-group processing**: builds HTA `Trace` / `CallGraph` objects, extracts communication-related spans, assigns micro batch IDs, and links P2P send / recv events between adjacent stages.
- **Report and trace export**: writes `report-pp<id>.csv` and `pp<id>-trace.json` for each PP group to inspect stage wait time, bubbles, communication, and workload imbalance.
- **Cluster-level aggregation hooks**: keeps MPI gather and anomaly-detection logic that can be enabled for PP-group and layer-level straggler analysis.

- **Abnormal node / GPU / operator detection**:
  - Compare latency differences across layers on the same rank to locate operators with unstable execution time.
  - Compare latency differences across ranks at the same time to locate spatial instability across machines or GPUs.

## Layout

```text
megatron_parallel_analysis/
├── distribute_trace_analysis.py
├── run_distributed_megatron_trace_analysis.py
├── trace_etl.py
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

## Typical workflow

```bash
# 1. Get the code
git clone -b v0.6.1-musa0.0.1 https://sh-code.mthreads.com/ai/HolisticTraceAnalysis
cd HolisticTraceAnalysis

# If you need a specific branch, run git checkout for the actual development branch.

# Single-node installation
pip install -r requirements.txt
pip install -e .

# Build a wheel
pip wheel . --wheel-dir=dist/ --no-deps --use-pep517 --no-build-isolation

# Or install the wheel directly
pip install traceinsight-*-py3-none-any.whl -i https://pypi.tuna.tsinghua.edu.cn/simple

# Multi-node installation, useful when analyzing traces from 1000 GPUs or larger scales in parallel
cd megatron_parallel_analysis/
bash install_hta.sh <HolisticTraceAnalysis_Path>  # requires hostfile
```

## Command-line arguments

| Argument | Required | Default | Description |
| --- | --- | --- | --- |
| `--trace-dir` | Yes | None | Raw trace directory. |
| `--tp` | No | `1` | Tensor Parallel size. |
| `--pp` | No | `2` | Pipeline Parallel size. |
| `--dp` | No | `1` | Data Parallel size. |
| `--ep` | No | `8` | Expert Parallel size. |
| `--pp-schedule` | No | `1f1b` | Pipeline schedule. Choices: `1f1b`, `1f1b-interleaved`, `1f1b-interleaved-epoverlap`. |
| `--num-bs` | No | `16` | Number of micro batches, passed to the analyzer as `micro_bs`. |
| `--vpp` | No | `2` | Virtual Pipeline Parallel size, used only by interleaved schedules. |
| `--pp-group-id-range` | No | `None` | Inclusive PP group range to analyze, formatted as `START END`. |

## MPI multi-process execution examples

For large traces, run with MPI so multiple processes can analyze different PP groups in parallel.

- `--map-by ppr:<cnt>:node` specifies `<cnt>` MPI processes per node.
- `-np` is usually the number of nodes in `hostfile` multiplied by the number of processes per node.
- The process count does not need to equal the number of PP groups. DHTA divides PP groups across the available MPI processes; the last MPI process may receive the remaining non-evenly-divisible PP groups. For example, with `-np 3` and 8 PP groups, the assignment is `3 + 3 + 2`.

```bash
# Two nodes, with 8 MPI processes started on each node
mpirun -allow-run-as-root -np 16 --bind-to none \
  --hostfile ./hostfile \
  --map-by ppr:8:node \
  --wdir /path/to/HolisticTraceAnalysis \
  ...
```

### ETL data cleaning

`trace_etl.py` is used for trace pre-processing. It filters noisy events from raw traces and writes the cleaned traces into a sibling `<trace-dir>-etl` directory.

Single-process example:

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

Do not run `python megatron_parallel_analysis/trace_etl.py` directly. The file uses package imports and should be started with `python -m megatron_parallel_analysis.trace_etl` from the repository root or an environment where the package is importable.

### Pipeline Parallel Group analysis

```bash
mpirun -allow-run-as-root -np 2 --bind-to none \
  --hostfile ./hostfile \
  --map-by ppr:1:node \
  --wdir /path/to/HolisticTraceAnalysis \
  python -m megatron_parallel_analysis.run_distributed_megatron_trace_analysis \
    --trace-dir /path/to/trace-dir \
    --tp 1 --pp 4 --dp 16 --ep 8 --num-bs 16
```

Example with more processes:

```bash
# Two nodes, with 8 MPI processes started on each node
mpirun -allow-run-as-root -np 16 --bind-to none \
  --hostfile ./hostfile \
  --map-by ppr:8:node \
  --wdir /path/to/HolisticTraceAnalysis \
  python -m megatron_parallel_analysis.run_distributed_megatron_trace_analysis \
    --trace-dir /path/to/trace-dir \
    --tp 1 --pp 31 --dp 24 --ep 8 --num-bs 128
```

### Analysis results and output structure

The run creates a `workspace` directory under the current working directory:

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

Main outputs:

1. `log/log_mpirun_parallel_<rank>.log`: analysis log for each MPI process.
2. `trace/pp_group_<id>/`: grouped trace directory for a PP group.
3. `trace/pp<id>-trace.json`: PP group trace with P2P communication information preserved.
4. `trace/report-pp<id>.csv`: summary pipeline analysis report for the PP group.
5. `trace/report-pp<id>-detail.csv`: detailed bubble breakdown for the same PP group.
6. `output/stragglers/`: straggler charts and result files when aggregation and anomaly analysis are enabled.

### `report-pp<id>.csv` summary columns

The summary report generated by [`megatron_pipeline_group_base.py:343-432`](megatron_pipeline_group_base.py#L343-L432) contains one row per rank with the following columns:

| Label | CSV column | Meaning |
| --- | --- | --- |
| A | `Global rank in a pp group, Rank_(i) + pp_size = Rank_(i+1)` | Global rank id inside the PP group. Adjacent pipeline stages differ by `pp_size`. |
| B | `Elapsed time per iteration` | End-to-end time of the analyzed iteration. |
| C | `Micro-Batch count` | Micro-batch count used for the analyzed iteration. |
| D | `Sum(Micro-Batch_forward_time + Micro-Batch_backward_time)` | Total compute time from all forward and backward micro-batch steps. |
| E | `PP SendRecv time` | Total PP send/recv time, including both actual transfer time and send/recv bubble time. |
| F | `Finalize_model_grads_step_time` | Time spent in `finalize_model_grads`. |
| G | `Should_run_forward_backward_time` | Time spent in `should_run_forward_backward`. |
| H | `Logical_and_across_model_parallel_group_time` | Time spent in `logical_and_across_model_parallel_group`. |
| I | `Optimizer_time` | Time spent in the optimizer step. |
| J | `Compute time total / Elapsed time per iteration` | Compute proportion of one iteration, `D / B`. |
| K | `PP SendRecv time / Elapsed time per iteration` | PP communication proportion of one iteration, `E / B`. |
| L | `Finalize_model_grads_step_time / Elapsed time per iteration` | `F / B`. |
| M | `Should_run_forward_backward_time / Elapsed time per iteration` | `G / B`. |
| N | `Logical_and_across_model_parallel_group_time / Elapsed time per iteration` | `H / B`. |
| O | `Optimizer_time / Elapsed time per iteration` | `I / B`. |

For the current implementation, the per-rank iteration time is modeled approximately as:

`B ~ D + E + H + I`

This keeps the top-level report focused on compute, PP send/recv, model-parallel logical-and synchronization, and optimizer time.

A practical consequence is that `J + K + N + O` is often below 100%. The missing portion usually comes from iteration-start collectives such as all-reduce and all-gather that are not folded into these four ratios.

### `report-pp<id>-detail.csv` bubble breakdown columns

The detailed report generated alongside the summary CSV exposes the bubble decomposition used by [`megatron_pipeline_group_1f1b.py:130-257`](megatron_pipeline_group_1f1b.py#L130-L257):

| Label | CSV column | Meaning |
| --- | --- | --- |
| A | `Global rank in a pp group, Rank_(i) + pp_size = Rank_(i+1)` | Global rank id inside the PP group. |
| B | `Elapsed time per iteration` | End-to-end time of the analyzed iteration. |
| C | `Micro-Batch count` | Micro-batch count used for the analyzed iteration. |
| D | `SUM(micro_batch_forward_time) / Micro-Batch count` | Average forward time per micro batch. |
| E | `STD(micro_batch_forward_time)` | Standard deviation of forward time per micro batch. |
| F | `SUM(micro_batch_backward_time) / Micro-Batch count` | Average backward time per micro batch. |
| G | `STD(micro_batch_backward_time)` | Standard deviation of backward time per micro batch. |
| H | `PP SendRecv time` | Total PP send/recv time. |
| I | `Actual transfer time of send-recv` | Actual transfer time of send/recv pairs, computed as `min(send, recv)`. |
| J | `Bubble time of send-recv` | Send/recv bubble time, computed as `max(send, recv) - min(send, recv)`. |
| K | `Theoretical bubble time warmup` | Theoretical bubble in warmup. |
| L | `Theoretical bubble time steady` | Theoretical bubble in steady state. |
| M | `Theoretical bubble time cooldown` | Theoretical bubble in cooldown. To make cooldown analysis comparable across ranks, the current implementation uses the end timestamp of `should_run_forward_backward` as the end of the whole cooldown region. This is because `should_run_forward_backward` includes an all-reduce that synchronizes all ranks in the same PP group, making it easier for the per-rank accounting to stay close to `B ~ D + E + H + I`. This also introduces a limitation: after the last backward step and before `should_run_forward_backward` finishes, there is still additional work such as `finalize_model_grads` and `should_run_forward_backward` itself. That region can include communication that is not PP send/recv, but the current implementation counts it into cooldown bubble time. As a result, the cooldown bubble has a known approximation error. |
| N | `Non-balanced bubble time warmup` | Extra warmup bubble caused by stage imbalance or waiting beyond the theoretical bubble. |
| O | `Non-balanced bubble time steady` | Extra steady-state bubble caused by stage imbalance or waiting beyond the theoretical bubble. |
| P | `Non-balanced bubble time cooldown` | Extra cooldown bubble caused by stage imbalance or waiting beyond the theoretical bubble. |
| Q | `Theoretical bubble time / elapsed time per iteration` | `(K + L + M) / B`. |
| R | `Theoretical bubble time / SUM(micro_batch_forward_time + micro_batch_backward_time)` | `(K + L + M) / (compute_time_total + PP SendRecv time)`. |
| S | `bubble_ratio in paper` | Theoretical bubble ratio from the paper formula. |

The detailed report follows these identities:

- `H = I + J`
- `I = sum(all peer {min(send, recv)})`
- `J = sum(all peer {max(send, recv) - min(send, recv)}) + finalize_model_grads + should_run_forward_backward`
- `J = K + L + M + N + O + P`

This decomposition is useful when separating two major sources of pipeline inefficiency:

1. **Theoretical bubble** described by the pipeline scheduling model, represented by `K + L + M`.
2. **Non-balanced bubble** caused by stage imbalance or extra waiting between stages, represented by `N + O + P`.

In practice, PP algorithm improvements mainly target `K`, `L`, and `M`, while PP overlap techniques can reduce `N`, `O`, `P`, and sometimes part of `I`.

### Next improvement

A remaining gap in the current implementation is that communication inside `finalize_model_grads` and `should_run_forward_backward` is not yet split into actual transfer time versus waiting time. Improving that decomposition is the next recommended optimization point.

## Pre-run checklist

1. **Trace completeness**: every rank trace exists and can be discovered by HTA `get_trace_files()`.
2. **Parallel configuration**: `--tp`, `--pp`, `--dp`, `--ep`, `--vpp`, and `--num-bs` match the training job.
3. **Schedule match**: `--pp-schedule` matches the actual Megatron pipeline schedule.
4. **Shared paths**: in multi-node runs, the code directory, trace directory, and workspace path should be visible from every node, or synchronized to the same path beforehand.
5. **MPI environment**: `mpi4py`, `mpirun`, hostfile, passwordless SSH, and Python environments should be consistent across nodes.
6. **Resource capacity**: large traces need substantial memory and disk space; validate with `--pp-group-id-range` first.

## Analysis recommendations

1. Start with a single process or a small PP group range to validate trace naming, parallel settings, and schedule selection.
2. Scale to multiple processes or nodes after the small run succeeds, and inspect each `log_mpirun_parallel_<rank>.log`.
3. Check `report-pp<id>.csv` first for stage compute, communication, waiting, and bubble-related metrics.
4. If one PP group looks abnormal, inspect `pp<id>-trace.json` or use the PyTorch callstack analysis scripts for a rank/stage-level drilldown.
5. To enable global straggler aggregation, restore the `gather_infos_from_all_ranks()`, `analyze_anomalies()`, and `post_process()` calls in `DistributedMegatronTraceAnalysis.analyze()` and verify the expected output format for the current workflow.

## Troubleshooting

- **Trace files are not found**: check `--trace-dir`, rank file names, and whether PP group directories were created.
- **Unexpected PP group count**: verify the TP / CP / EP / DP / PP product and rank order against the training job.
- **Empty report or missing P2P links**: confirm the trace contains the target iteration, pipeline send / recv events, and GPU kernel events.
- **MPI launch failure**: check hostfile, working directory, Python environment, `mpi4py`, and SSH configuration.
- **Unexpected interleaved results**: confirm `--vpp` matches the virtual pipeline parallel size used in training.

## Prompt suggestion for CSV-based pipeline analysis

The `workspace/***` path contains profile traces collected from one large-model pretraining run, together with DHTA pipeline-parallel statistics generated by `megatron_parallel_analysis`. The meaning of each CSV column can be found in the `report-pp<id>-detail.csv` bubble breakdown section of `megatron_parallel_analysis/README.md`. Compare the statistics of the PP groups under `workspace/***`, identify any abnormal patterns, produce a quantitative analysis report, and write the result into a Markdown file.

## Core modules

### `distribute_trace_analysis.py`

The main distributed orchestration module. Its primary class is `DistributedMegatronTraceAnalysis`.

During initialization, it:

1. records the trace directory and TP / CP / EP / DP / PP / VPP / micro batch settings;
2. initializes `MPI.COMM_WORLD` and discovers the current MPI rank, world size, and processor name;
3. uses `RankGenerator` to create DP / TP / PP rank groups;
4. creates the workspace, output, log, and straggler directories;
5. partitions trace files by PP group;
6. assigns PP group tasks across MPI processes.

For each PP group, it:

1. creates the schedule-specific pipeline trace object from `--pp-schedule`;
2. parses the rank traces in the PP group and builds call graphs;
3. filters and keeps communication-related traces;
4. assigns micro batch IDs;
5. establishes P2P links between adjacent pipeline stages;
6. writes a communication trace JSON file and a CSV report.

### `run_distributed_megatron_trace_analysis.py`

A configurable command-line entrypoint that creates `DistributedMegatronTraceAnalysis` and calls `analyze()`.

Basic example:

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

Analyze only a subset of PP groups:

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

For interleaved or EP-overlap schedules, pass VPP explicitly:

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

ETL data-cleaning command-line entrypoint. See the “ETL data cleaning” section above.

### Pipeline group analyzers

- `megatron_pipeline_group_base.py`: PP group trace analysis base class; provides shared trace parsing, communication filtering, micro batch labeling, P2P linking, and report generation.
- `megatron_pipeline_group_1f1b.py`: analysis for Megatron `1f1b` scheduling.
- `megatron_pipeline_group_1f1b_interleaved.py`: analysis for Megatron interleaved 1F1B scheduling.
- `megatron_pipeline_group_1f1b_interleaved_epoverlap.py`: analysis for Megatron interleaved scheduling with EP overlap.

### `utils/`

- `parallel_state.py`: generates Megatron-style TP / CP / EP / DP / PP rank groups.
- `pipeline_parallel_utils.py`: helpers for pipeline stages, micro batches, and P2P communication.
- `trace_filter_utils.py`: trace filtering utilities.
- `call_graph_utils.py`: helpers for locating the main stack from an HTA `CallGraph`.
- `utils.py`: general helpers for directory preparation and trace file partitioning.

