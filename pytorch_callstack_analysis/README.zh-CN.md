# pytorch_callstack_analysis

> English documentation: [README.md](README.md)

`pytorch_callstack_analysis` 是基于 HTA 解析后的 PyTorch Profiler / Kineto trace 做 callstack / callgraph 分析的工具包。它主要用于分布式 Megatron 分析定位到某个 rank、pipeline stage 或算子区域之后，继续做单 rank 或局部 trace 细查。

该工具包重点覆盖模型层级 forward/backward 耗时统计、kernel 层级 shape 提取，以及 MUSA kernel TFLOPS / 带宽估算。Megatron PP group 的分布式编排逻辑位于 [`megatron_parallel_analysis/`](../megatron_parallel_analysis/README.zh-CN.md)。

## 功能概览

- **模型级 forward/backward 统计**：加载 HTA trace，构建 `CallGraph`，根据缩进模板匹配函数，并按 rank 汇总 forward/backward kernel span。
- **Pytorch Callstack 模板驱动的调用图遍历**：用文本模板描述重点模型区域，通过注解处理重复函数名和 shape 提取需求。
- **Kernel 级 shape / TFLOPS / 带宽分析**：从指定节点提取 `input_dims` 和 `input_type`，对常见 MUSA kernel 估算 TFLOPS 或 GB/s。
- **Forward/backward 匹配辅助逻辑**：基于 HTA call graph 链接和 `fwdbwd_index` 关系定位 selected forward 节点对应的 backward 区域。
- **计时工具**：通过装饰器、上下文管理器、MPI root 计时器和全局计时器统计分析阶段耗时。

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

## 模型层级 forward/backward 统计

`model_level_fwdbwd_statistics.py` 是模型层级 forward/backward 耗时统计的通用命令行入口。

### 处理流程

1. 通过 `--trace-dir` 指定 trace 目录，并使用 HTA `get_trace_files()` 自动发现 rank trace 文件。
2. 如果指定 `--rank`，只分析该 rank；如果不指定 `--rank`，默认按 rank 顺序分析全部 rank。
3. 根据 `call_graph_template.py` 中的模板选择调用图模板，默认使用 `kimi_epoverlap`。
4. 对每个 rank：
   - 使用 HTA `Trace` 加载 trace；
   - decode symbol ID，并保留完整函数名；
   - 构建 `CallGraph`；
   - 获取该 rank 的 main stack；
   - 将模板解析为函数名和祖先链；
   - 对普通函数按名称匹配，对 `@dup@` 函数结合祖先和 path 上下文消歧；
   - 基于 `kernel_span` 计算 forward 和 backward 统计。
5. 每个 rank 输出一个文本报告，文件名为 `<template>-<rank>-main-stack.txt`。

### 基本运行

```bash
export HTA_DISABLE_NS_ROUNDING=1

python -m pytorch_callstack_analysis.model_level_fwdbwd_statistics \
  --trace-dir /path/to/trace-dir \
  --rank 16 \
  --template kimi_epoverlap \
  --output-dir model_main_stack
```

输出示例：

```text
model_main_stack/
└── kimi_epoverlap-16-main-stack.txt
```

### 分析全部 rank

如果不传 `--rank`，脚本会分析 `--trace-dir` 中发现的全部 rank：

```bash
python -m pytorch_callstack_analysis.model_level_fwdbwd_statistics \
  --trace-dir /path/to/trace-dir \
  --template kimi_epoverlap \
  --output-dir model_main_stack
```

输出示例：

```text
model_main_stack/
├── kimi_epoverlap-0-main-stack.txt
├── kimi_epoverlap-1-main-stack.txt
├── kimi_epoverlap-2-main-stack.txt
└── ...
```

### 命令行参数

| 参数 | 是否必需 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--trace-dir` | 是 | 无 | trace 目录，脚本会从该目录自动发现 rank trace 文件。 |
| `--rank` | 否 | `None` | 指定要分析的 rank；不指定时分析全部 rank。 |
| `--template` | 否 | `kimi_epoverlap` | 调用图模板名称。可选：`default`、`debug`、`kimi`、`kimi_epoverlap`。 |
| `--output-dir` / `--output` | 否 | `model_main_stack` | 输出目录。每个 rank 输出一个 `<template>-<rank>-main-stack.txt` 文件。 |

### 支持的模板

模板映射由 `model_level_fwdbwd_statistics.py` 中的 `TEMPLATE_MAP` 定义：

| 模板名 | 对应模板变量 | 适用场景 |
| --- | --- | --- |
| `default` | `output_template_to_file` | DeepSeek / MoE 通用模板。 |
| `debug` | `output_template_to_file_debug` | 小范围调用栈调试。 |
| `kimi` | `output_template_to_file_kimi` | Kimi 常规 Pipeline 调度模板。 |
| `kimi_epoverlap` | `output_template_to_file_kimi_epoverlap` | Kimi fine-grained / EP overlap 场景，当前默认模板。 |

模板定义位于 `call_graph_template.py`。如果模型代码、Megatron 版本或调度方式发生变化，应优先检查模板中的函数名和调用层级是否仍能匹配实际 trace。

## 输出指标

模型层级报告是保留模板缩进层级的文本文件。每个模板节点可能包含 `fwd` 和 `bwd` 两行统计。

主要耗时口径是 `kernel_span`：

```text
kernel_span = last_kernel_end - first_kernel_start
```

也就是所选函数区域内，第一个 kernel 的开始时间戳到最后一个 kernel 的结束时间戳之间的跨度。

输出列会转换为毫秒级展示：

- `mean_percent`：该节点总耗时相对根节点 forward/backward 总耗时的比例，计算方式为 `mean * count / total_mean`。
- `mean`：平均耗时。
- `q_25` / `q_50` / `q_75`：25%、50%、75% 分位数。
- `max` / `min`：最大值和最小值。
- `count`：参与统计的调用次数。

示例：

```text
# Rank: 16
pretrain_kimi.py(\d+): <module>
     fwd: mean_percent: 1.00, mean: 1234.56, q_25: 1200.00, q_50: 1230.00, q_75: 1260.00, max: 1300.00, min: 1180.00, count: 1.00
     bwd: mean_percent: 1.00, mean: 1100.00, q_25: 1080.00, q_50: 1100.00, q_75: 1120.00, max: 1150.00, min: 1050.00, count: 1.00
    megatron/core/pipeline_parallel/combined_1f1b.py(\d+): combined_forward_backward_step
         fwd: mean_percent: 0.95, mean: 1170.00, q_25: 1150.00, q_50: 1170.00, q_75: 1190.00, max: 1210.00, min: 1130.00, count: 1.00
         bwd: mean_percent: 0.92, mean: 1012.00, q_25: 990.00, q_50: 1010.00, q_75: 1035.00, max: 1060.00, min: 970.00, count: 1.00
```

## Pytorch Callstack 模板标记

`call_graph_template.py` 使用缩进表达调用层级，支持两个特殊标记：

- `@dup@`：标记存在重复调用的函数。脚本会结合祖先上下文和已匹配节点索引区分不同位置的同名函数。
- `@shape@`：标记需要提取 shape 信息的函数或 kernel。模型层级脚本会识别但不输出 shape 指标；该标记主要被 kernel 层级脚本使用。

模板示例：

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

## Kernel 层级 shape / TFLOPS / 带宽分析

`kernel_level_fwdbwd_statistics.py` 当前更像一个实验性分析脚本，而不是已经封装完成的通用 CLI。它适合作为 kernel 层级定位问题时的分析模板或二次开发入口。

### 处理流程

该脚本可以：

1. 基于调用图模板定位带 `@shape@` 标记的 forward kernel；
2. 从当前节点或父节点提取 `input_dims` / `input_type`；
3. 对以下类型 kernel 计算 shape 相关指标：
   - `general_gemm`；
   - `general_grouped_gemm`；
   - `quantize`；
   - `aten::_scaled_dot_product_attention_flash_musa`；
4. 关联 backward 路径中的对应 kernel，并输出 `bwd-0`、`bwd-1` 等位置统计；
5. 将结果写入文本文件，输出 shape、平均 kernel 时间、TFLOPS 或带宽分位数以及 count。

### 当前脚本假设

与 `model_level_fwdbwd_statistics.py` 不同，该脚本的 `if __name__ == "__main__":` 内仍有固定默认值：

- `base_dir = "../"`；
- `trace_dir` 固定指向 `../perf`；
- 默认只分析 `rank == 32`；
- 默认输出文件名形如 `20260211-<rank>.txt`；
- 启动前会显式把 `ParserConfig.ARGS_INPUT_SHAPE` 加入默认解析配置，以确保 shape 信息被提取出来。

如果要直接使用该脚本，通常需要修改：

- `base_dir`；
- `trace_dir`；
- 目标 rank；
- 输出文件名；
- 模板和 shape-position 映射，如果模型结构或 kernel 名称发生变化。

### Kernel 输出指标

Kernel 层级输出包括：

- `shape`；
- `mean_time(us)`；
- `TFLOPS` 或 `GB/s` 的平均值；
- `q_25` / `q_50` / `q_75`；
- `count`。

### Kernel 分析建议

1. 先用模型层级脚本确认模板能稳定匹配到目标函数，再做 kernel 级 shape / TFLOPS 分析。
2. 运行前确认解析配置已经启用 `ParserConfig.ARGS_INPUT_SHAPE`，否则 `input_dims` 和 `input_type` 字段可能不存在。
3. 如果模型代码或 kernel 名称变化，优先更新 `call_graph_template.py` 中带 `@shape@` 的模板项，以及脚本内的 `SHAPE_POSITION_FWD_BWD` / `SHAPE_POSITION_FWD_BWD_OF_FLASH_ATTENTION` 映射。
4. Kernel 层级分析适合针对特定算子细查，不建议替代模型层级报告。

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
2. 选择一个代表性 rank，使用最接近的模板运行 `model_level_fwdbwd_statistics.py`。
3. 如果输出为空或报 `No statistics were generated`，更新模板函数名和祖先层级。
4. 单 rank 验证通过后，再对一组 rank 或全部 rank 运行同一模板。
5. 对比关键模型区域的 mean、分位数、min/max、count 和占比。
6. 若要分析特定 kernel，启用 shape 解析，并按目标 rank 和 kernel 映射改造 `kernel_level_fwdbwd_statistics.py`。

## 故障排查

- **找不到 trace 文件**：检查 `--trace-dir` 和 rank trace 命名。
- **指定 rank 不存在**：确认该 rank 是否出现在 `get_trace_files(trace_dir)` 结果中。
- **没有生成统计结果**：通常是所选模板与 trace 调用栈不匹配。
- **重复函数匹配错误**：添加 `@dup@`，并保证祖先链足够区分目标实例。
- **缺少 shape 字段**：确保加载 trace 前 parser config 已加入 `ParserConfig.ARGS_INPUT_SHAPE`。
- **Kernel 指标异常**：检查 shape 来源节点、dtype 和目标 kernel 的公式映射是否正确。
