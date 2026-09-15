# PP/VPP 关键路径分析

**给定某些stage和某些 batch forward/backward 能达到的目标耗时，计算它们能带来多少一个 training step 的收益，以及下一处瓶颈在哪里。** Python 3.9+，仅依赖标准库。

核心思路：DHTA 提供事件耗时，PP/VPP 调度规则提供依赖；构建 DAG 后，整体完成时间由最长依赖路径决定。每轮只改节点耗时，保留依赖和此前修改，再用拓扑顺序动态规划重算全图。**整体收益 = 修改前最长路径耗时 − 修改后最长路径耗时**，不能直接累加局部缩短量。

## 三个文件与执行顺序

| 顺序 | 文件 | 职责 |
|---|---|---|
| 1 | [extract_trace_inputs.py](extract_trace_inputs.py) | 精简 DHTA trace，保留原始时间和来源 SHA256，生成 `trace_inputs.json` |
| 2 | [audit.py](audit.py) | `Graph` 构建 PP/VPP 节点、依赖边和拓扑顺序；由主程序自动调用 |
| 3 | [iterative_critical_path.py](iterative_critical_path.py) | 主程序：读取输入、调用构图、逐轮修改 forward、重算关键路径并输出报告 |

**手动执行 1 → 3 即可。** 第 3 个脚本自动调用第 2 个模块，无需单独运行 `audit.py`。

```text
DHTA ppN-trace.json
  → extract_trace_inputs.py → trace_inputs.json
  → iterative_critical_path.py
      → audit.Graph：构图
      → 最长路径 → 设置 forward 目标 → 更新耗时 → 重算最长路径 → …
  → report.md / analysis.json / dag.json / 每轮路径 CSV
```

## 快速运行

先完成 DHTA 导出。输入应为**一个完整 PP group** 的 `ppN-trace.json`，包含 `forward_step_mbN_vppN` / `backward_step_mbN_vppN` 事件以及 `rank`、`ts`、`dur` 字段；原始时间单位为微秒。无需配套 CSV。

在本目录执行，替换 `/path/to/pp36-trace.json`；输出目录需不存在或为空。

```bash
# 1. 精简输入，不修改原始耗时
python3 extract_trace_inputs.py \
  --trace /path/to/pp36-trace.json --key run \
  --output /tmp/trace_inputs.json

# 2. 自动构图，并逐轮将当前关键路径上 >70 ms 的 forward 设为 50 ms
python3 iterative_critical_path.py \
  --input /tmp/trace_inputs.json --key run \
  --threshold-ms 70 --target-ms 50 --max-rounds 10 \
  --output /tmp/pp-critical-path
```

只分析一次时，也可以跳过精简步骤：

```bash
python3 iterative_critical_path.py \
  --trace /path/to/pp36-trace.json \
  --threshold-ms 70 --target-ms 50 --max-rounds 10 \
  --output /tmp/pp-critical-path-direct
```

两个入口都支持 `--ssh HOST --trace /remote/path/to/pp36-trace.json`，读取远程文件并在本地输出结果。

## 常用参数

| 参数 | 含义 |
|---|---|
| `--threshold-ms 70 --target-ms 50` | 当前耗时严格 >70 ms 的 forward 才设为 50 ms；不修改 backward |
| `--max-rounds 10` | 最多执行 10 轮优化，不含初始关键路径计算 |
| `--num-paths 5` | 记录到 5 条不同代表路径后停止，包含初始路径；不保证一定达到 |
| `--iteration 19` | 选择一个 iteration；输入含多个 iteration 时必须指定 |
| `--ranks 804,1252` | 仅允许指定 rank 的 forward 被自动选择 |
| `--later-threshold-ms 50` | 第二轮起改用 >50 ms 的筛选阈值 |
| `--node-targets targets.json` | 按具体事件指定目标，覆盖自动阈值和 rank 筛选 |

`targets.json` 示例：`[{"rank":804,"name":"forward_step_mb2_vpp1","target_ms":50}]`。每轮只修改当前代表关键路径上已指定、且仍高于目标耗时的节点。

## 如何读结果

先看输出目录中的 **`report.md`**：每轮“原路径预计缩短”是节点缩短量之和，“全图模型收益”是重算前后最长路径之差。两者相等，说明本轮路径缩短量全部生效；否则，收益受到其他路径限制。

- `analysis.json`：来源、每轮修改、关键路径节点列表、收益和停止原因。
- `dag.json`：原始节点、固定依赖边和拓扑顺序。
- `round_000_path.csv`：初始关键路径；后续编号对应每轮修改后的路径。

轮数与不同路径数不相等；一轮收益为零时，仍可能需要继续处理并列关键路径。当前路径没有符合规则的可缩短 forward 时停止。程序记录每轮的代表路径，不枚举原始图的前 N 长路径或单轮加速过程中的全部切换点。

**目标耗时是工程假设，输出是固定 DAG 下的模型收益。** 当前模型按 rank 升序连接 PP，VPP 编号从 1 连续开始；保留根起点，依赖边延迟为零，尚未完整建模通信等待、资源竞争和训练尾部同步。实际优化后需用新 trace 和实测耗时验证。

完整的数学解法、模型边界及 0804 案例见 [analysis.md](analysis.md)。
