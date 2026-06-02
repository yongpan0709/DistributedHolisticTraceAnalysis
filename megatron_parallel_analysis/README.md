# megatron_parallel_analysis

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

### Pipeline group analyzers

- `megatron_pipeline_group_base.py`: base class for PP group trace analysis; provides shared trace parsing, communication filtering, micro batch labeling, P2P linking, and report generation.
- `megatron_pipeline_group_1f1b.py`: analysis for Megatron `1f1b` scheduling.
- `megatron_pipeline_group_1f1b_interleaved.py`: analysis for Megatron interleaved 1F1B scheduling.
- `megatron_pipeline_group_1f1b_interleaved_epoverlap.py`: analysis for Megatron interleaved scheduling with EP overlap.

### `utils/`

- `parallel_state.py`: generates Megatron-style TP / CP / EP / DP / PP rank groups.
- `pipeline_parallel_utils.py`: helpers for pipeline stages, micro batches, and P2P communication.
- `trace_filter_utils.py`: trace filtering utilities.
- `call_graph_utils.py`: helpers for locating the main stack from an HTA `CallGraph`.
- `utils.py`: general helpers for directory preparation and trace file partitioning.

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

## MPI multi-node / multi-process execution

For large traces, run the entrypoint with MPI so multiple processes can analyze different PP groups in parallel.

```bash
mpirun -allow-run-as-root -np 2 --bind-to none \
  --hostfile ./hostfile \
  --map-by ppr:1:node \
  --wdir /path/to/HolisticTraceAnalysis \
  python -m megatron_parallel_analysis.run_distributed_megatron_trace_analysis \
    --trace-dir /path/to/trace-dir \
    --tp 1 --pp 4 --dp 2 --ep 8 --num-bs 16
```

Example with more processes:

```bash
mpirun -allow-run-as-root -np 16 --bind-to none \
  --hostfile ./hostfile \
  --map-by ppr:8:node \
  --wdir /path/to/HolisticTraceAnalysis \
  python -m megatron_parallel_analysis.run_distributed_megatron_trace_analysis \
    --trace-dir /path/to/trace-dir \
    --tp 1 --pp 31 --dp 3 --ep 8 --num-bs 128
```

`-np` is usually the number of nodes in `hostfile` multiplied by the number of processes per node. `--map-by ppr:<cnt>:node` starts `<cnt>` MPI processes per machine. The process count does not need to equal the number of PP groups; the analyzer divides PP groups across the available MPI processes.

## Workspace and outputs

The analyzer writes results under the current working directory:

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
4. `trace/report-pp<id>.csv`: pipeline analysis report for the PP group.
5. `output/stragglers/`: straggler charts and result files when aggregation and anomaly analysis are enabled.

### `report-pp<id>.csv` columns from the base analyzer

The `report-pp<id>.csv` generated by [`megatron_pipeline_group_base.py`](megatron_pipeline_group_base.py#L372-L402) contains one row per rank with the following columns:

| Column | Description |
| --- | --- |
| `rank` | Global rank id of the current trace row. |
| `time_per_iteration` | End-to-end time of one iteration for the rank. |
| `num_microbatch` | Number of micro batches configured or observed for the analyzed iteration. |
| `forward_step_avg_time` | Average time of the forward step. |
| `fwd_step_std_time` | Standard deviation of the forward-step time. |
| `backward_step_avg_time` | Average time of the backward step. |
| `bwd_step_std_time` | Standard deviation of the backward-step time. |
| `compute_time_total` | Total compute time accumulated for the analyzed iteration. |
| `comm_time_total` | Total communication time accumulated for the analyzed iteration. |
| `comm_time_true` | Communication time after separating out overlap-hidden wait or overhead time. |
| `overhead_wait_time_total` | Total wait or overhead time composed of the warmup, steady-state, and cooldown phases; each phase consists of two parts: theoretical bubble time and actual wait time between PP stages. |
| `bubble_time_warmup` | Actual bubble time in the pipeline warmup phase caused by extra factors such as communication between PP stages, excluding the first bubble already counted in the theoretical bubble time. |
| `bubble_time_steady` | Actual bubble time in the pipeline steady-state phase caused by extra factors such as communication between PP stages, excluding the first bubble already counted in the theoretical bubble time. |
| `bubble_time_cooldown` | Actual bubble time in the pipeline cooldown phase caused by extra factors such as communication between PP stages, excluding the first bubble already counted in the theoretical bubble time. |
| `theoretical_bubble_time_warmup` | Theoretical bubble time for the warmup phase. |
| `theoretical_bubble_time_steady` | Theoretical bubble time for the steady-state phase. |
| `theoretical_bubble_time_cooldown` | Theoretical bubble time for the cooldown phase. |
| `overhead_wait_time_ratio` | Ratio of `overhead_wait_time_total` to `time_per_iteration`. |
| `bubble_time_ratio` | Ratio of `overhead_wait_time_total` to `compute_time_total + comm_time_total`. |
| `bubble_time_ratio_theoretical` | Theoretical bubble ratio derived from the micro-batch count. |
| `pipeline_parallel_size` | Configured pipeline parallel size (`PP`). |
| `comm_time_true_ratio` | Ratio of `comm_time_true` to `time_per_iteration`. |
| `comp_time_ratio` | Ratio of `compute_time_total` to `time_per_iteration`. |
| `comm_time_ratio` | Ratio of `comm_time_total` to `time_per_iteration`. |
| `finalize_model_grads_step_time` | Time spent in the `finalize_model_grads` step. |
| `logical_and_across_model_parallel_group_time` | Time spent in the logical-and synchronization across the model-parallel group. |
| `optimizer_time_total` | Total time spent in the optimizer step. |

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
