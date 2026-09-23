# DistributedHolisticTraceAnalysis

> 中文文档：[README.zh-CN.md](README.zh-CN.md)

Distributed Holistic Trace Analysis (DHTA) is a performance analysis toolkit for large-scale distributed training. It is based on [Facebook Research HolisticTraceAnalysis](https://github.com/facebookresearch/HolisticTraceAnalysis) and extends HTA with analysis workflows for modern pre-training and reinforcement-learning systems, especially Megatron-LM pipeline/expert parallelism and PyTorch call stacks.

DHTA consumes traces collected with the [PyTorch Profiler / Kineto](https://github.com/pytorch/kineto). It combines trace-level measurement, distributed parallel-group analysis, model-level call-stack drilldown, and what-if critical-path modeling in one toolkit.

## What DHTA adds

### 1. Distributed Megatron parallel analysis

[`megatron_parallel_analysis/`](megatron_parallel_analysis/README.md) scales trace analysis from a single rank to large Megatron-LM jobs:

- **PP-group-based analysis**: builds Megatron-style TP/CP/EP/DP/PP rank groups and analyzes a pipeline-parallel group as a unit.
- **MPI-distributed execution**: partitions PP groups across MPI processes, allowing very large multi-node trace sets to be analyzed in parallel with bounded per-process memory.
- **Pipeline schedule support**: analyzes regular `1f1b`, interleaved `1f1b-interleaved`, and `1f1b-interleaved-epoverlap` schedules, including VPP timelines.
- **Pipeline communication and bubble analysis**: links P2P send/recv events between adjacent stages and reports compute time, transfer time, waiting time, theoretical bubbles, non-balanced bubbles, and stage imbalance.
- **Expert Parallel (EP) analysis**: for supported MoE schedules, aggregates GPU start-time and duration statistics across EP ranks and exports merged EP traces for direct Perfetto comparison.
- **Straggler and anomaly analysis**: compares operators across layers and ranks, and compares EP-rank timing distributions to locate slow GPUs, imbalanced experts, synchronization delays, and abnormal operators.
- **Trace ETL and parse caching**: cleans noisy raw traces and reuses validated rank-level parse caches to reduce repeated parsing cost.

See the [Megatron parallel analysis guide](megatron_parallel_analysis/README.md) for commands, output formats, MPI examples, and interpretation guidance.

### 2. PP/VPP critical-path and optimization modeling

[`megatron_parallel_analysis/pp_critical_path_analysis/`](megatron_parallel_analysis/pp_critical_path_analysis/) models a PP/VPP execution as a dependency DAG:

- extracts the minimum required timing inputs from a DHTA PP trace;
- constructs PP/VPP nodes, dependencies, and a topological execution order;
- iteratively applies target durations to selected forward events on the current critical path;
- recomputes the longest path after every change to estimate the **whole training-step benefit**;
- identifies the next bottleneck when the critical path moves to another stage or micro-batch; and
- writes the DAG, per-round paths, machine-readable analysis data, and a Markdown report.

This is a what-if model rather than a replacement for measurement: predicted gains should be validated with a new trace after the optimization is implemented. Read the [critical-path analysis guide](megatron_parallel_analysis/pp_critical_path_analysis/README.md) for assumptions, limitations, and examples.

### 3. PyTorch call-stack and kernel analysis

[`pytorch_callstack_analysis/`](pytorch_callstack_analysis/README.md) provides template-driven local drilldown after distributed analysis has identified an interesting rank, stage, or operator:

- matches model regions in HTA call graphs using indentation-based call-stack templates;
- supports `@dup@` annotations for repeated functions and `@shape@` annotations for shape extraction;
- reports forward/backward duration statistics, percentiles, min/max, counts, and contribution by model region;
- extracts kernel shapes and estimates MUSA kernel TFLOPS or memory bandwidth; and
- provides forward/backward matching and reusable timing utilities, including MPI-aware timers.

It supports per-rank analysis as well as analysis of all ranks discovered in a trace directory. See the [PyTorch call-stack analysis guide](pytorch_callstack_analysis/README.md) for templates and command-line examples.

## HTA capabilities retained

DHTA retains the core HTA analyses for PyTorch Profiler traces, including:

- GPU temporal and kernel breakdowns;
- GPU idle-time attribution;
- communication/computation overlap;
- frequent CUDA-kernel and kernel-launch statistics;
- queue-length and memory-bandwidth time series and summaries;
- trace comparison and visualization; and
- experimental CUPTI counter and roofline analysis.

These analyses can be used independently for a quick overview, or together with the DHTA extensions for a top-down-to-bottom-up workflow:

```text
raw PyTorch Profiler traces
        │
        ├── HTA baseline analyses
        ├── Megatron PP/EP group analysis
        │       └── identify a slow stage, rank, or micro-batch
        ├── PP/VPP critical-path what-if analysis
        │       └── prioritize the next optimization
        └── PyTorch call-stack / kernel drilldown
                └── identify the responsible model region or kernel
```

## Installation

DHTA runs on Linux and macOS with Python 3.10 or newer. For the distributed Megatron workflow, install `mpi4py` and provide a working MPI runtime such as Open MPI in addition to the Python dependencies.

### Install from source

Clone this repository and enter its root directory, then install the Python dependencies:

```bash
cd DistributedHolisticTraceAnalysis
pip install -r requirements.txt
pip install -e .
```

For a wheel build:

```bash
pip wheel . --wheel-dir=dist/ --no-deps --use-pep517 --no-build-isolation
```

The standalone PP/VPP critical-path scripts use only the Python standard library. Their input is a DHTA-generated `ppN-trace.json` for one complete PP group.

## Quick start

### Analyze a Megatron pipeline group

```bash
python -m megatron_parallel_analysis.run_distributed_megatron_trace_analysis \
  --trace-dir /path/to/trace-dir \
  --tp 1 --pp 4 --dp 16 --ep 8 \
  --num-bs 16 --pp-schedule 1f1b
```

For a large trace set, distribute PP groups across MPI processes:

```bash
mpirun -np 16 --bind-to none \
  --hostfile ./hostfile \
  --map-by ppr:8:node \
  --wdir /path/to/DistributedHolisticTraceAnalysis \
  python -m megatron_parallel_analysis.run_distributed_megatron_trace_analysis \
    --trace-dir /path/to/trace-dir \
    --tp 1 --pp 31 --dp 24 --ep 8 --num-bs 128
```

Enable EP reporting for a supported MoE schedule with `--enable_ep_analysis`. Start with a small PP-group range to validate trace discovery and parallel configuration before scaling to the full job.

### Analyze a model call stack or kernel

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

Omit `--rank` to analyze every rank discovered in the trace directory.

### Run critical-path modeling

```bash
python3 megatron_parallel_analysis/pp_critical_path_analysis/iterative_critical_path.py \
  --trace /path/to/pp36-trace.json \
  --threshold-ms 70 --target-ms 50 --max-rounds 10 \
  --output /tmp/pp-critical-path
```

The output includes `report.md`, `analysis.json`, `dag.json`, and a CSV file for each representative critical path.

## Recommended analysis workflow

1. Collect complete PyTorch Profiler traces for the training or reinforcement-learning workload.
2. Run the retained HTA analyses to establish GPU utilization, communication overlap, idle time, and kernel baselines.
3. Run Megatron PP/EP analysis to compare stages, micro-batches, ranks, and expert groups at scale.
4. Use the PP/VPP critical-path model to estimate which timing improvements can reduce the complete training step most.
5. Drill into the selected rank or stage with the call-stack and kernel-level tools.
6. Implement the optimization and collect a new trace to verify the predicted improvement.

## Output overview

A distributed Megatron run creates a workspace containing grouped traces, logs, and reports similar to:

```text
workspace/<trace-name>/
├── log/                         # one MPI-process log per worker
├── output/stragglers/           # optional anomaly and straggler results
└── trace/
    ├── pp_group_<id>/            # traces for each PP group
    ├── pp<id>-trace.json         # PP trace with linked P2P events
    ├── report-pp<id>.csv         # per-rank PP summary
    ├── report-pp<id>-detail.csv  # bubble decomposition
    ├── ep_group_stats/            # optional EP aggregate statistics
    └── ep_trace/                  # optional merged EP traces
```

## Documentation

- [Megatron parallel analysis](megatron_parallel_analysis/README.md)
- [PP/VPP critical-path analysis](megatron_parallel_analysis/pp_critical_path_analysis/README.md)
- [PyTorch call-stack analysis](pytorch_callstack_analysis/README.md)
- [Original HTA documentation](https://hta.readthedocs.io/en/latest/index.html)

## Contributing

Issues and contributions are welcome. When reporting a problem, include the DHTA version or commit, Python and MPI versions, the relevant parallel configuration, the pipeline schedule, and a minimal description of the trace layout. Do not upload proprietary traces or other sensitive training data.

## License

DHTA retains the upstream project's MIT licensing terms. See [LICENSE](LICENSE) for the repository license.
