# pytorch_callstack_analysis

> 中文版文档；英文版见：[README.md](README.md)

`pytorch_callstack_analysis` 是基于 HTA 解析后的 PyTorch Profiler / Kineto trace 做 callstack / callgraph 分析的工具包，重点覆盖模型级与 kernel 级的 forward/backward 耗时统计、基于模板的调用图遍历、MUSA kernel shape 提取，以及 TFLOPS / 带宽估算。

## 功能概览

- **模型级 forward/backward 统计**：加载 HTA trace，构建 call graph，根据调用图模板匹配函数，并按 rank 汇总 forward/backward kernel span。
- **Kernel 级性能估算**：从指定 call graph 节点提取输入 shape，针对常见 MUSA kernel 估算 TFLOPS 或 GB/s。
- **Callstack 模板驱动的调用图遍历**：用缩进文本描述重点模型区域，并通过注解处理重复函数名与 shape 提取需求。
- **Forward/backward 持续时间辅助函数**：基于 trace DataFrame 字段与 call graph 链接计算 forward / backward kernel span。
- **计时工具**：通过装饰器、上下文管理器和全局计时器统计分析阶段耗时。

## 目录结构

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

## 核心模块

### `model_level_fwdbwd_statistics.py`

模型级 forward/backward 耗时统计的命令行入口。

它会：

1. 从 `--trace-dir` 发现 trace 文件；
2. 使用 HTA `Trace` 加载指定 rank；
3. 构建 `CallGraph`；
4. 提取该 rank 的 main stack；
5. 根据选定模板匹配函数；
6. 计算 forward 和 backward 的 kernel-span 统计信息；
7. 为每个 rank 输出一个文本报告。

示例：

```bash
python -m pytorch_callstack_analysis.model_level_fwdbwd_statistics \
  --trace-dir /path/to/traces \
  --rank 0 \
  --template kimi_epoverlap \
  --output-dir model_main_stack
```

如果省略 `--rank`，会分析 trace 目录中发现的所有 rank。

当前模板由该模块中的 `TEMPLATE_MAP` 定义，包括：

- `default`
- `debug`
- `kimi`
- `kimi_epoverlap`

### `kernel_level_fwdbwd_statistics.py`

Kernel 级探索分析脚本，用于提取 shape 相关节点，并对选定 forward / backward kernel 估算 TFLOPS 或 GB/s。

该模块组合使用：

- `call_graph_template.py` 中的模板注解；
- `utils.musa_fwdbwd_util` 中的 forward/backward 节点匹配逻辑；
- `utils.musa_basic_kernel_info` 中的 kernel 估算公式。

当前脚本主体偏向本地 trace 调研场景，使用前通常需要按目标 trace 调整 `trace_dir`、rank 选择、模板和输出路径。

### `call_graph_template.py`

保存调用图模板与模板解析辅助函数。

模板是用缩进表示层级关系的文本块，用于描述需要关注的模型区域。支持两个注解：

- `@dup@`：标记函数名可能重复，需要结合祖先节点进行消歧。
- `@shape@`：标记该函数需要提取输入 shape，用于 kernel 级估算。

常用辅助函数：

- `extract_func_name_from_template(template)`：将模板解析为 `(function_name, ancestors)` 条目。
- `extract_dup_or_shape_func_name_from_template(template)`：收集重复函数名与需要 shape 的函数名。
- `set_pandas_display_options()`：设置 pandas 输出，便于完整查看 call graph 数据。

## Duration 口径

分析中主要使用 `kernel_span` 作为耗时指标：

```text
kernel_span = last_kernel_end - first_kernel_start
```

也就是：所选函数区域内，第一个 kernel 的开始时间戳到最后一个 kernel 的结束时间戳之间的跨度。

- Forward span 直接来自匹配到的 forward 节点。
- Backward span 利用 PyTorch autograd 在 call graph 中建立的关联：从 forward 节点出发向下遍历，找到对应 backward 区域，再聚合最早的 `first_kernel_start` 与最晚的 `last_kernel_end`。

## 工具 API

### Forward/backward span 辅助函数

来自 `pytorch_callstack_analysis.utils.musa_fwdbwd_util`：

- `get_forward_duration_uniq(df, forward_func_name)`：函数名唯一时，直接筛选匹配的 forward 节点。
- `get_forward_duration_dup(df, forward_func_name, func_ancestors, cg, rank, func_mapping_node_index)`：函数名重复时，结合祖先节点与 path-to-root 信息消歧。
- `get_backward_duration(df, cg, rank, forward_index)`：聚合与选定 forward 节点关联的 backward kernel span。

### MUSA kernel 估算辅助函数

来自 `pytorch_callstack_analysis.utils.musa_basic_kernel_info`：

- `BYTES_DICT` / `get_num_of_bytes(dtype)`：dtype 到字节数的映射。
- `calculate_linear_tflops_or_bw(...)`：Linear 类 kernel 的 TFLOPS 或 GB/s 估算。
- `calculate_groupedlinear_tflops_or_bw(...)`：Grouped GEMM 场景估算。
- `calculate_scaled_dot_product_attention_flash_musa_flops(...)`：FlashAttention FLOPS 估算。
- `calculate_CheckpointWithoutOutputFunction(...)`：checkpoint 相关数据量估算。

### 计时工具

来自 `pytorch_callstack_analysis.utils.timing`：

- `TimingTracker`：可复用的命名计时器集合。
- `QuickTimer`：轻量的一次性计时器。
- `MPITimer`：MPI 环境计时器，仅在 root 进程打印。
- `get_timer()`、`reset_timer()`：全局计时器辅助函数。
- `time_it(name)`：计时装饰器。
- `measure_time(name)`：便捷上下文管理器。

快速示例：

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

运行计时示例：

```bash
python -m pytorch_callstack_analysis.utils.example_timing_usage
```

## 典型使用流程

1. 生成 PyTorch Profiler / Kineto trace。
2. 在 `call_graph_template.py` 中选择或编辑调用图模板，使其匹配目标模型区域。
3. 运行模型级统计入口，生成每个 rank 的 forward/backward 汇总报告。
4. 如需 kernel 级分析，在 parser config 中开启 shape 提取，并按目标 rank / 模板 / 输出路径调整 kernel 级脚本。
5. 查看生成报告，对关键模型区域的 mean、分位数、min/max、count 和占比进行对比分析。
