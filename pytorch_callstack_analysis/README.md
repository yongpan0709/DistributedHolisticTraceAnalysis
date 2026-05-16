# pytorch_callstack_analysis

English documentation is maintained in this file.

- 中文版请见：[README.zh-CN.md](README.zh-CN.md)

`pytorch_callstack_analysis` provides callstack/callgraph-based analysis utilities for HTA-parsed PyTorch Profiler / Kineto traces. It focuses on model-level and kernel-level forward/backward timing, function-template-driven call graph traversal, MUSA kernel shape extraction, and TFLOPS / bandwidth estimation.

## What this package does

- **Model-level forward/backward statistics**: load HTA traces, build call graphs, match functions from a call graph template, and summarize forward/backward kernel spans by rank.
- **Kernel-level performance estimation**: extract input shapes from selected call graph nodes and estimate TFLOPS or GB/s for common MUSA kernels.
- **Callstack template-driven call graph traversal**: define important model regions with indentation-based templates and optional annotations for duplicated function names and shape extraction.
- **Forward/backward duration helpers**: compute forward and backward kernel spans from trace DataFrame fields and call graph links.
- **Timing utilities**: measure analysis phases with decorators, context managers, and a reusable global timer.

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

## Core modules

### `model_level_fwdbwd_statistics.py`

Command-line analysis entrypoint for model-level forward/backward duration statistics.

It:

1. discovers trace files under `--trace-dir`,
2. loads each selected rank with HTA `Trace`,
3. builds a `CallGraph`,
4. extracts the main stack for the rank,
5. matches functions from the selected template,
6. computes forward and backward kernel-span statistics, and
7. writes one text report per rank.

Example:

```bash
python -m pytorch_callstack_analysis.model_level_fwdbwd_statistics \
  --trace-dir /path/to/traces \
  --rank 0 \
  --template kimi_epoverlap \
  --output-dir model_main_stack
```

If `--rank` is omitted, all discovered ranks are analyzed.

Available templates are defined by `TEMPLATE_MAP` in the module and currently include:

- `default`
- `debug`
- `kimi`
- `kimi_epoverlap`

### `kernel_level_fwdbwd_statistics.py`

Kernel-level exploratory analysis for extracting shape-related nodes and estimating TFLOPS / GB/s for selected forward and backward kernels.

This module combines:

- template annotations from `call_graph_template.py`,
- forward/backward node matching from `utils.musa_fwdbwd_util`, and
- kernel formulas from `utils.musa_basic_kernel_info`.

The current script body is oriented toward local trace investigation and may require adapting `trace_dir`, rank selection, template choice, and output path before use.

### `call_graph_template.py`

Stores call graph templates and parsing helpers.

Templates are indentation-based text blocks representing important model regions. Two annotations are supported:

- `@dup@`: marks functions whose names may appear multiple times and need ancestor-based disambiguation.
- `@shape@`: marks functions whose input shapes should be extracted for kernel-level estimation.

Useful helpers:

- `extract_func_name_from_template(template)`: parse template lines into `(function_name, ancestors)` entries.
- `extract_dup_or_shape_func_name_from_template(template)`: collect duplicated and shape-needed functions.
- `set_pandas_display_options()`: configure pandas output for full call graph inspection.

## Duration model

These analyses use `kernel_span` as the primary timing metric:

```text
kernel_span = last_kernel_end - first_kernel_start
```

That is, the span from the first launched kernel start timestamp to the last launched kernel end timestamp for the selected function region.

- Forward spans are computed from matching forward nodes.
- Backward spans use PyTorch autograd links in the call graph: starting from the forward node, descendants are traversed to find linked backward regions, then the earliest `first_kernel_start` and latest `last_kernel_end` are aggregated.

## Utility APIs

### Forward/backward span helpers

From `pytorch_callstack_analysis.utils.musa_fwdbwd_util`:

- `get_forward_duration_uniq(df, forward_func_name)`: select matching forward nodes when the function name is unique.
- `get_forward_duration_dup(df, forward_func_name, func_ancestors, cg, rank, func_mapping_node_index)`: disambiguate duplicated function names with ancestor/path-to-root information.
- `get_backward_duration(df, cg, rank, forward_index)`: aggregate backward kernel spans linked to the selected forward nodes.

### MUSA kernel estimation helpers

From `pytorch_callstack_analysis.utils.musa_basic_kernel_info`:

- `BYTES_DICT` / `get_num_of_bytes(dtype)`: dtype-to-byte mapping.
- `calculate_linear_tflops_or_bw(...)`: TFLOPS or GB/s estimation for Linear-like kernels.
- `calculate_groupedlinear_tflops_or_bw(...)`: estimation for grouped GEMM kernels.
- `calculate_scaled_dot_product_attention_flash_musa_flops(...)`: FlashAttention FLOPS estimation.
- `calculate_CheckpointWithoutOutputFunction(...)`: checkpoint-related volume estimation.

### Timing helpers

From `pytorch_callstack_analysis.utils.timing`:

- `TimingTracker`: reusable named timer collection.
- `QuickTimer`: lightweight one-off timer.
- `MPITimer`: MPI-aware timer that prints on the root process.
- `get_timer()`, `reset_timer()`: global timer helpers.
- `time_it(name)`: timing decorator.
- `measure_time(name)`: context-manager helper.

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
2. Choose or edit a call graph template in `call_graph_template.py` to match the model region of interest.
3. Run the model-level statistics entrypoint to produce per-rank forward/backward summaries.
4. For kernel-level analysis, enable shape extraction in the parser config and adapt the kernel-level script to the target rank/template/output.
5. Inspect generated reports to compare mean, quartiles, min/max, count, and percent contribution for important model regions.
