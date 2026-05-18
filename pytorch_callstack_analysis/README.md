# pytorch_callstack_analysis

> 中文文档: [README.zh-CN.md](README.zh-CN.md)

`pytorch_callstack_analysis` provides template-driven callstack and callgraph analysis utilities for HTA-parsed PyTorch Profiler / Kineto traces. It is intended for rank-level or local trace drilldown after a distributed Megatron analysis has identified an interesting rank, pipeline stage, or operator region.

Based on pytorch callstack, the package focuses on model-level forward/backward duration statistics, kernel-level shape extraction, and MUSA kernel TFLOPS / bandwidth estimation. Distributed Megatron PP-group orchestration lives in [`megatron_parallel_analysis/`](../megatron_parallel_analysis/README.md).

## Capabilities

- **Model-level forward/backward statistics**: load HTA traces, build a `CallGraph`, match functions from an indentation-based template, and summarize forward/backward kernel spans by rank.
- **Callstack Template-driven call graph traversal**: describe model regions as callstack templates and use annotations to disambiguate duplicate names or mark nodes that need shape extraction.
- **Kernel-level shape / TFLOPS / bandwidth analysis**: extract `input_dims` and `input_type` from selected nodes and estimate TFLOPS or GB/s for common MUSA kernels.
- **Forward/backward matching helpers**: use HTA call graph links and `fwdbwd_index` relationships to locate backward regions corresponding to selected forward nodes.
- **Timing utilities**: measure analysis phases with decorators, context managers, MPI-aware timers, and a reusable global timer.

## Layout

```text
pytorch_callstack_analysis/
├── __init__.py
├── call_graph_template.py
├── model_level_fwdbwd_statistics.py
├── kernel_level_fwdbwd_statistics.py
└── utils/
    ├── __init__.py
    ├── timing.py
    ├── example_timing_usage.py
    ├── musa_fwdbwd_util.py
    └── musa_basic_kernel_info.py
```

## Model-level forward/backward statistics

`model_level_fwdbwd_statistics.py` is the general command-line entrypoint for model-level forward/backward duration statistics.

### Processing flow

1. Discover rank trace files from `--trace-dir` with HTA `get_trace_files()`.
2. If `--rank` is provided, analyze only that rank; otherwise analyze all discovered ranks in rank order.
3. Select a template from `call_graph_template.py`; the default is `kimi_epoverlap`.
4. For each selected rank:
   - load the trace with HTA `Trace`;
   - decode symbol IDs while preserving full function names;
   - build a `CallGraph`;
   - locate the rank's main stack;
   - parse the template into function names and ancestor chains;
   - match normal functions by name and match `@dup@` functions with ancestor/path context;
   - compute forward and backward statistics from `kernel_span`.
5. Write one text report per rank as `<template>-<rank>-main-stack.txt`.

### Basic usage

```bash
export HTA_DISABLE_NS_ROUNDING=1

python -m pytorch_callstack_analysis.model_level_fwdbwd_statistics \
  --trace-dir /path/to/trace-dir \
  --rank 16 \
  --template kimi_epoverlap \
  --output-dir model_main_stack
```

Output example:

```text
model_main_stack/
└── kimi_epoverlap-16-main-stack.txt
```

### Analyze all ranks

If `--rank` is omitted, the script analyzes every rank discovered under `--trace-dir`:

```bash
python -m pytorch_callstack_analysis.model_level_fwdbwd_statistics \
  --trace-dir /path/to/trace-dir \
  --template kimi_epoverlap \
  --output-dir model_main_stack
```

Output example:

```text
model_main_stack/
├── kimi_epoverlap-0-main-stack.txt
├── kimi_epoverlap-1-main-stack.txt
├── kimi_epoverlap-2-main-stack.txt
└── ...
```

### Arguments

| Argument | Required | Default | Description |
| --- | --- | --- | --- |
| `--trace-dir` | Yes | None | Trace directory. The script discovers rank traces from this directory. |
| `--rank` | No | `None` | Rank to analyze. If omitted, all discovered ranks are analyzed. |
| `--template` | No | `kimi_epoverlap` | Template name. Choices: `default`, `debug`, `kimi`, `kimi_epoverlap`. |
| `--output-dir` / `--output` | No | `model_main_stack` | Output directory. Each rank writes `<template>-<rank>-main-stack.txt`. |

### Supported templates

Templates are mapped by `TEMPLATE_MAP` in `model_level_fwdbwd_statistics.py`:

| Template | Template variable | Typical use |
| --- | --- | --- |
| `default` | `output_template_to_file` | General DeepSeek / MoE-style template. |
| `debug` | `output_template_to_file_debug` | Small-scope callstack debugging. |
| `kimi` | `output_template_to_file_kimi` | Kimi regular pipeline scheduling. |
| `kimi_epoverlap` | `output_template_to_file_kimi_epoverlap` | Kimi fine-grained / EP-overlap traces; current default. |

Templates live in `call_graph_template.py`. If model code, Megatron version, or pipeline schedule changes, first verify that the template function names and ancestor hierarchy still match the trace.

## Output metrics

Model-level reports are indentation-preserving text files. Each template node may include both `fwd` and `bwd` rows.

The main timing metric is `kernel_span`:

```text
kernel_span = last_kernel_end - first_kernel_start
```

It represents the span from the first kernel start timestamp to the last kernel end timestamp inside the selected function region.

Reported columns are converted to milliseconds:

- `mean_percent`: contribution of this node to the root forward or backward total, computed as `mean * count / total_mean`.
- `mean`: average duration.
- `q_25` / `q_50` / `q_75`: 25th, 50th, and 75th percentiles.
- `max` / `min`: maximum and minimum duration.
- `count`: number of matched calls.

Example:

```text
# Rank: 16
pretrain_kimi.py(\d+): <module>
     fwd: mean_percent: 1.00, mean: 1234.56, q_25: 1200.00, q_50: 1230.00, q_75: 1260.00, max: 1300.00, min: 1180.00, count: 1.00
     bwd: mean_percent: 1.00, mean: 1100.00, q_25: 1080.00, q_50: 1100.00, q_75: 1120.00, max: 1150.00, min: 1050.00, count: 1.00
    megatron/core/pipeline_parallel/combined_1f1b.py(\d+): combined_forward_backward_step
         fwd: mean_percent: 0.95, mean: 1170.00, q_25: 1150.00, q_50: 1170.00, q_75: 1190.00, max: 1210.00, min: 1130.00, count: 1.00
         bwd: mean_percent: 0.92, mean: 1012.00, q_25: 990.00, q_50: 1010.00, q_75: 1035.00, max: 1060.00, min: 970.00, count: 1.00
```

## Pytorch Callstack Template annotations

`call_graph_template.py` uses indentation to express call hierarchy. It supports two annotations:

- `@dup@`: marks a function name that appears multiple times. Matching uses ancestor context and previously matched node indices to disambiguate instances.
- `@shape@`: marks a function or kernel whose shape should be extracted. The model-level script recognizes this annotation but does not output shape metrics; it is mainly used by the kernel-level script.

Template example:

```python
output_template_to_file_kimi_epoverlap = r"""
pretrain_kimi.py(\d+): <module>
    musa_patch/training.py(\d+): train_step
        megatron/core/pipeline_parallel/combined_1f1b.py(\d+): combined_forward_backward_step
            megatron/core/models/common/model_chunk_schedule_plan.py(\d+): run
                megatron/core/models/gpt/fine_grained_callables.py(\d+): submodule_attn_forward
                    megatron/core/transformer/transformer_layer.py(\d+): _forward_attention
                        megatron/core/tensor_parallel/random.py(\d+): checkpoint @dup@
                            nn.Module: RMSNorm_0 @shape@
"""
```

## Kernel-level shape / TFLOPS / bandwidth analysis

`kernel_level_fwdbwd_statistics.py` is now a command-line entrypoint for per-rank or all-rank kernel-level shape and MUSA TFLOPS / bandwidth analysis.

### Processing flow

The script:

1. discovers rank trace files from `--trace-dir` with HTA `get_trace_files()`;
2. enables `ParserConfig.ARGS_INPUT_SHAPE` before loading traces so `input_dims` / `input_type` are available;
3. if `--rank` is provided, analyzes only that rank; otherwise analyzes all discovered ranks;
4. for each selected rank:
   - load the trace with HTA `Trace`;
   - decode symbol IDs while preserving full function names;
   - build a `CallGraph` and locate the rank's main stack;
   - parse `kernel_level_template` and resolve normal / `@dup@` functions;
   - fill `shape`, `TFLOPS`, and `GB/s` columns on the matched kernel nodes;
   - compute forward `fwd-0` metrics and matched backward `bwd-0`, `bwd-1`, ... metrics;
5. writes one text report per rank as `kernel-level-rank<rank>-fwdbwd.txt` under `--output-dir`.

### Basic usage

```bash
python -m pytorch_callstack_analysis.kernel_level_fwdbwd_statistics \
  --trace-dir /path/to/trace-dir \
  --rank 32 \
  --output-dir kernel_fwdbwd_statistics
```

Output example:

```text
kernel_fwdbwd_statistics/
└── kernel-level-rank32-fwdbwd.txt
```

### Analyze all ranks

If `--rank` is omitted, the script analyzes every rank discovered under `--trace-dir`:

```bash
python -m pytorch_callstack_analysis.kernel_level_fwdbwd_statistics \
  --trace-dir /path/to/trace-dir \
  --output-dir kernel_fwdbwd_statistics
```

Output example:

```text
kernel_fwdbwd_statistics/
├── kernel-level-rank0-fwdbwd.txt
├── kernel-level-rank1-fwdbwd.txt
├── kernel-level-rank2-fwdbwd.txt
└── ...
```

### Arguments

| Argument | Required | Default | Description |
| --- | --- | --- | --- |
| `--trace-dir` | Yes | None | Trace directory. The script discovers rank traces from this directory. |
| `--rank` | No | `None` | Rank to analyze. If omitted, all discovered ranks are analyzed. |
| `--output-dir` / `--output` | No | `kernel_fwdbwd_statistics` | Output directory. Each rank writes `kernel-level-rank<rank>-fwdbwd.txt`. |

### Kernel output metrics

Kernel-level output includes:

- `shape`;
- `mean_time(us)`;
- mean `TFLOPS` or mean `GB/s`;
- `q_25` / `q_50` / `q_75`;
- `count`.

Example:

```text
nn.Module: RMSNorm_0
fwd-0 shape: [1, 8192],  mean_time(us): 12.34, BW mean: 456.78 GB/s, q_25: 430.00, q_50: 455.00, q_75: 480.00, count: 32
bwd-0 shape: [1, 8192],  mean_time(us): 14.21, BW mean: 410.55 GB/s, q_25: 398.00, q_50: 408.00, q_75: 421.00, count: 32
```

### Kernel analysis recommendations

1. First run the model-level script on a representative rank and confirm the template matches real trace nodes.
2. Run the kernel-level script with the same trace set; shape parsing is enabled automatically inside the script.
3. If model code or kernel names changed, update both `call_graph_template.py` `@shape@` entries and the script's `SHAPE_POSITION_FWD_BWD` / `SHAPE_POSITION_FWD_BWD_OF_FLASH_ATTENTION` mappings.
4. Use kernel-level analysis for targeted operator drilldown, not as a replacement for the model-level report.

## Utility APIs

### Forward/backward span helpers

From `pytorch_callstack_analysis.utils.musa_fwdbwd_util`:

- `get_forward_duration_uniq(df, forward_func_name)`: select matching forward nodes when the function name is unique.
- `get_forward_duration_dup(df, forward_func_name, func_ancestors, cg, rank, func_mapping_node_index)`: disambiguate duplicated function names with ancestor and path-to-root information.
- `get_backward_duration(df, cg, rank, forward_index)`: aggregate backward kernel spans linked to selected forward nodes.

### MUSA kernel estimation helpers

From `pytorch_callstack_analysis.utils.musa_basic_kernel_info`:

- `BYTES_DICT` / `get_num_of_bytes(dtype)`: dtype-to-byte mapping.
- `drop_empty_arrays(...)`: normalize nested shape arrays by removing empty elements before metric calculation.
- `calculate_linear_tflops_or_bw(...)`: TFLOPS or GB/s estimation for Linear-like kernels.
- `calculate_groupedlinear_tflops_or_bw(...)`: estimation for grouped GEMM kernels.
- `calculate_scaled_dot_product_attention_flash_musa_flops(...)`: FlashAttention FLOPS estimation.
- `calculate_CheckpointWithoutOutputFunction(...)`: checkpoint-related volume estimation.

### Timing helpers

From `pytorch_callstack_analysis.utils.timing`:

- `TimingRecord`: single timing record data structure.
- `TimingTracker`: reusable named timer collection.
- `QuickTimer`: lightweight one-off timer.
- `MPITimer`: MPI-aware timer that prints on the root process.
- `get_timer()`, `reset_timer()`: global timer helpers.
- `time_it(name)`: timing decorator.
- `measure_time(name)`: convenience context-manager helper backed by the global timer.

Quick example:

```python
from pytorch_callstack_analysis.utils import get_timer, reset_timer, time_it

@time_it("load_trace")
def load_trace():
    ...

reset_timer()
timer = get_timer()

with timer.measure("analysis"):
    load_trace()

timer.print_summary()
```

Run the timing examples with:

```bash
python -m pytorch_callstack_analysis.utils.example_timing_usage
```

## Typical workflow

1. Generate PyTorch Profiler / Kineto traces.
2. Select one representative rank and run `model_level_fwdbwd_statistics.py` with the closest template.
3. If the output is empty or raises `No statistics were generated`, update the template function names and ancestor hierarchy.
4. Run the same template across selected ranks or all ranks.
5. Compare mean, quartiles, min/max, count, and percent contribution for important model regions.
6. For specific kernels, enable shape parsing and adapt `kernel_level_fwdbwd_statistics.py` for the target rank and kernel mapping.

## Troubleshooting

- **No trace files found**: check `--trace-dir` and rank trace naming.
- **Rank not found**: verify the rank exists in `get_trace_files(trace_dir)` output.
- **No statistics were generated**: the selected template likely does not match the trace callstack.
- **Duplicate function matched incorrectly**: add `@dup@` and make the ancestor chain specific enough to identify the desired instance.
- **Shape fields are missing**: ensure the parser config includes `ParserConfig.ARGS_INPUT_SHAPE` before loading traces.
- **Kernel metrics look wrong**: verify the shape source node, dtype, and formula mapping for the target kernel.
