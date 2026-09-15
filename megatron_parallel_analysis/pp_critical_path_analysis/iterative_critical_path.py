"""Evaluate successive forward-only interventions on a fixed PP/VPP DAG.

Accept a DHTA ppN-trace.json directly, or a trace_inputs.json dataset. Edges
come from audit.Graph: observed per-rank order, ascending-rank PP order,
1-based VPP chunks, and forward/backward dependencies. This is a compute-span
counterfactual model, not a full communication/iteration replay.
"""
import argparse
import collections
import csv
import hashlib
import json
import shlex
import subprocess
import sys
from fractions import Fraction as Q
from pathlib import Path

from audit import Graph, PAT


def number(value, description):
    try:
        return Q(str(value))
    except (ValueError, ZeroDivisionError):
        raise ValueError(f"{description} must be a finite number: {value!r}") from None


def read_source(args):
    """Return one dataset and provenance; no CSV is needed for DAG analysis."""
    path = args.trace or args.input
    if args.ssh:
        if not args.trace or args.ssh.startswith("-"):
            raise ValueError("--ssh requires --trace and a valid SSH host")
        command = ["ssh", "-o", "ConnectTimeout=15", args.ssh,
                   "cat -- " + shlex.quote(str(path))]
        result = subprocess.run(command, capture_output=True)
        if result.returncode:
            raise ValueError(result.stderr.decode(errors="replace").strip())
        raw = result.stdout
        input_path = str(path)
    else:
        path = Path(path).expanduser().resolve()
        raw = path.read_bytes()
        input_path = str(path)
    parsed = json.loads(raw)
    digest = hashlib.sha256(raw).hexdigest()
    if args.trace:
        if not isinstance(parsed, dict) or not isinstance(parsed.get("traceEvents"), list):
            raise ValueError("--trace expects a DHTA JSON object containing traceEvents")
        data = {"events": [e for e in parsed["traceEvents"] if e.get("ph") == "X"]}
        provenance = {"input_kind": "dhta_trace", "path": input_path,
                      "sha256": digest, "ssh_host": args.ssh}
    else:
        if not isinstance(parsed, dict):
            raise ValueError("--input expects a trace_inputs dataset mapping")
        key = args.key
        if key is None and len(parsed) == 1:
            key = next(iter(parsed))
        if key is None or key not in parsed:
            raise ValueError("Use --key to select one of: " + ", ".join(parsed))
        data = parsed[key]
        if not isinstance(data, dict) or not isinstance(data.get("events"), list):
            raise ValueError("Selected dataset must contain an events array")
        provenance = {"input_kind": "compact_snapshot", "input_path": input_path,
                      "input_sha256": digest, "key": key,
                      "path": data.get("path"), "sha256": data.get("sha256")}
    return data, provenance


def build_graph(data, iteration=None):
    events = []
    for event in data["events"]:
        match = PAT.fullmatch(event.get("name", ""))
        if match is None:
            continue
        if iteration is not None:
            value = event.get("args", {}).get("iteration")
            if value is None or number(value, "iteration") != iteration:
                continue
        for field in ("rank", "ts", "dur"):
            if field not in event:
                raise ValueError(f"Event {event['name']} is missing {field}")
        rank = number(event["rank"], "rank")
        if rank.denominator != 1:
            raise ValueError("rank must be an integer")
        if number(event["dur"], "duration") < 0:
            raise ValueError("Event durations must be nonnegative")
        number(event["ts"], "timestamp")
        events.append(dict(event, rank=int(rank)))
    if not events:
        raise ValueError("No forward_step_mbN_vppN/backward_step_mbN_vppN events found")
    iterations = {number(e["args"]["iteration"], "iteration") for e in events
                  if e.get("args", {}).get("iteration") is not None}
    if len(iterations) > 1:
        raise ValueError("Multiple iterations found; select one with --iteration")
    signatures = collections.Counter(
        (e["rank"], PAT.fullmatch(e["name"])[1], int(PAT.fullmatch(e["name"])[2]),
         int(PAT.fullmatch(e["name"])[3])) for e in events
    )
    if any(count != 1 for count in signatures.values()):
        raise ValueError("Duplicate rank/direction/microbatch/VPP events; select one iteration")
    ranks = sorted({key[0] for key in signatures})
    microbatches = sorted({key[2] for key in signatures})
    chunks = sorted({key[3] for key in signatures})
    if chunks != list(range(1, max(chunks) + 1)):
        raise ValueError("This graph builder requires contiguous, 1-based VPP chunk IDs")
    missing = [(r, direction, mb, vpp) for r in ranks
               for direction in ("forward", "backward") for mb in microbatches
               for vpp in chunks if (r, direction, mb, vpp) not in signatures]
    if missing:
        raise ValueError(f"Incomplete PP group/iteration: missing {len(missing)} events; first: {missing[:3]}")
    try:
        graph = Graph({"events": events})
    except AssertionError as error:
        raise ValueError(f"Invalid dependency graph: {error}") from error
    durations = [number(e["dur"], "duration") / 1000 for e in events]
    releases = [number(e["ts"], "timestamp") / 1000 if not graph.pred[i] else Q(0)
                for i, e in enumerate(events)]
    return graph, durations, releases, [str(value) for value in sorted(iterations)]


def longest_path(graph, durations, releases):
    """Topological DP with exact weights and counts of tied source-to-sink paths."""
    finish = [Q(0)] * len(durations)
    start = [Q(0)] * len(durations)
    parent = [None] * len(durations)
    counts = [0] * len(durations)
    for v in graph.order:
        if graph.pred[v]:
            start[v] = max(finish[u] for u in graph.pred[v])
            ties = [u for u in graph.pred[v] if finish[u] == start[v]]
            parent[v] = min(ties)
            counts[v] = sum(counts[u] for u in ties)
        else:
            start[v] = releases[v]
            counts[v] = 1
        finish[v] = start[v] + durations[v]
    sinks = [v for v in graph.order if not graph.succ[v]]
    total = max(finish[v] for v in sinks)
    last = [v for v in sinks if finish[v] == total]
    path = []
    v = min(last)
    while v is not None:
        path.append(v)
        v = parent[v]
    return {"T": total, "path": path[::-1], "start": start, "finish": finish,
            "critical_path_count": sum(counts[v] for v in last)}


def load_targets(path, graph):
    """Explicit per-event priors override automatic threshold/rank rules."""
    if path is None:
        return None
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("--node-targets expects an array of {rank, name, target_ms}")
    lookup = {(node["rank"], node["name"]): i for i, node in enumerate(graph.nodes)}
    targets = {}
    for row in rows:
        key = (row["rank"], row["name"])
        if key not in lookup:
            raise ValueError(f"Target event does not exist in selected trace: {key}")
        v = lookup[key]
        if v in targets:
            raise ValueError(f"Duplicate target event: {key}")
        if graph.nodes[v]["direction"] != "forward":
            raise ValueError(f"Backward optimization is disabled: {key}")
        target = number(row["target_ms"], "target_ms")
        if target < 0:
            raise ValueError("target_ms must be nonnegative")
        targets[v] = target
    return targets


def select_targets(graph, durations, path, threshold, target, ranks=None, explicit=None):
    selected = {}
    for v in path:
        node = graph.nodes[v]
        if node["direction"] != "forward":
            continue
        if explicit is not None:
            if v in explicit and durations[v] > explicit[v]:
                selected[v] = explicit[v]
        elif (ranks is None or node["rank"] in ranks) and durations[v] > threshold and durations[v] > target:
            selected[v] = target
    return selected


def evaluate_rounds(graph, original, releases, max_rounds, threshold, target,
                    num_paths=None, later_threshold=None, ranks=None, explicit=None):
    """Retain prior interventions; a zero-gain round does not stop the search."""
    current = list(original)
    solved = longest_path(graph, current, releases)
    baseline = solved["T"]
    path_ids = {tuple(solved["path"]): "P0"}
    history = []
    states = [(current[:], solved)]
    reason = "round_limit_reached"
    for round_number in range(1, max_rounds + 1):
        if num_paths is not None and len(path_ids) >= num_paths:
            reason = "requested_path_count_reached"
            break
        active_threshold = later_threshold if round_number > 1 and later_threshold is not None else threshold
        selected = select_targets(graph, current, solved["path"], active_threshold,
                                  target, ranks, explicit)
        if not selected:
            reason = "current_critical_path_has_no_eligible_forward"
            break
        previous = solved
        previous_id = path_ids[tuple(previous["path"])]
        changes = [{"node_id": v, "rank": graph.nodes[v]["rank"],
                    "name": graph.nodes[v]["name"], "vpp": graph.nodes[v]["vpp"],
                    "microbatch": graph.nodes[v]["mb"], "before_ms": float(current[v]),
                    "after_ms": float(value), "reduction_ms": float(current[v] - value)}
                   for v, value in selected.items()]
        ideal = sum((current[v] - value for v, value in selected.items()), Q(0))
        for v, value in selected.items():
            current[v] = value
        solved = longest_path(graph, current, releases)
        gain = previous["T"] - solved["T"]
        previous_path_after = releases[previous["path"][0]] + sum(current[v] for v in previous["path"])
        assert Q(0) <= gain <= ideal
        assert previous_path_after == previous["T"] - ideal
        signature = tuple(solved["path"])
        is_new = signature not in path_ids
        if is_new:
            path_ids[signature] = f"P{len(path_ids)}"
        remaining = [v for v in solved["path"] if graph.nodes[v]["direction"] == "forward" and current[v] > target]
        history.append({
            "round": round_number, "previous_path_id": previous_id,
            "path_id": path_ids[signature], "new_distinct_path": is_new,
            "threshold_ms": None if explicit is not None else float(active_threshold),
            "changed_count": len(selected), "changes": changes,
            "before_ms": float(previous["T"]), "after_ms": float(solved["T"]),
            "after_ms_exact": str(solved["T"]),
            "ideal_path_gain_ms": float(ideal), "model_gain_ms": float(gain),
            "model_gain_ms_exact": str(gain), "cumulative_gain_ms": float(baseline - solved["T"]),
            "previous_path_after_update_ms": float(previous_path_after),
            "unrealized_gain_ms": float(ideal - gain),
            "realization_ratio": float(gain / ideal),
            "full_path_gain_realized": solved["T"] == previous_path_after,
            "previous_path_still_critical": solved["T"] == previous_path_after,
            "critical_path_count_before": previous["critical_path_count"],
            "critical_path_count_after": solved["critical_path_count"],
            "path_node_ids": solved["path"],
            "forward_above_global_target_on_new_path": len(remaining),
        })
        states.append((current[:], solved))
    if num_paths is not None and len(path_ids) >= num_paths:
        reason = "requested_path_count_reached"
    backward_changes = sum(current[i] != original[i] for i, node in enumerate(graph.nodes)
                           if node["direction"] == "backward")
    assert backward_changes == 0
    return {
        "baseline_ms": float(baseline), "final_ms": float(solved["T"]),
        "total_gain_ms": float(baseline - solved["T"]),
        "total_gain_ms_exact": str(baseline - solved["T"]),
        "completed_rounds": len(history), "distinct_path_count": len(path_ids),
        "requested_path_count": num_paths, "stop_reason": reason,
        "modified_node_count": sum(a != b for a, b in zip(original, current)),
        "backward_modified_count": backward_changes,
        "baseline_path_node_ids": states[0][1]["path"],
        "baseline_critical_path_count": states[0][1]["critical_path_count"],
        "unique_paths": [{"path_id": label, "node_ids": list(path)} for path, label in path_ids.items()],
        "rounds": history,
    }, states


STOP_TEXT = {
    "round_limit_reached": "达到最大优化轮数",
    "requested_path_count_reached": "达到指定的不同代表路径数量（含初始路径）",
    "current_critical_path_has_no_eligible_forward": "当前关键路径已没有符合本次规则的可缩短 forward",
}


def write_outputs(output, graph, original, releases, result, states):
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("Output directory must be empty; use a new --output directory")
    graph_dump = {
        "nodes": [{"id": i, "rank": n["rank"], "physical_stage": graph.ranks.index(n["rank"]),
                   "name": n["name"], "direction": n["direction"], "vpp": n["vpp"],
                   "microbatch": n["mb"], "original_ms": float(original[i]),
                   "original_ms_exact": str(original[i]), "root_release_ms": float(releases[i])}
                  for i, n in enumerate(graph.nodes)],
        "edges": [[u, v, kind] for v, preds in enumerate(graph.pred) for u, kind in preds.items()],
        "topological_order": graph.order,
    }
    (output / "dag.json").write_text(json.dumps(graph_dump, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "analysis.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    fields = ["sequence", "node_id", "rank", "vpp", "microbatch", "direction", "name",
              "original_ms", "current_ms", "model_start_ms", "model_finish_ms", "incoming_dependency"]
    for k, (durations, state) in enumerate(states):
        with (output / f"round_{k:03d}_path.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for position, v in enumerate(state["path"]):
                n = graph.nodes[v]
                writer.writerow({"sequence": position, "node_id": v, "rank": n["rank"],
                                 "vpp": n["vpp"], "microbatch": n["mb"], "direction": n["direction"],
                                 "name": n["name"], "original_ms": float(original[v]),
                                 "current_ms": float(durations[v]), "model_start_ms": float(state["start"][v]),
                                 "model_finish_ms": float(state["finish"][v]),
                                 "incoming_dependency": "root_release" if position == 0 else graph.pred[v][state["path"][position-1]]})
    report = [
        "# DHTA 连续关键路径评估", "",
        f"原始模型耗时 **{result['baseline_ms']/1000:.9f} s**；最终 **{result['final_ms']/1000:.9f} s**；累计模型收益 **{result['total_gain_ms']/1000:.9f} s**。", "",
        f"完成 {result['completed_rounds']} 轮优化，记录 {result['distinct_path_count']} 条不同代表关键路径，修改 {result['modified_node_count']} 个 forward；backward 修改数为 {result['backward_modified_count']}。", "",
        "停止原因：" + STOP_TEXT[result["stop_reason"]] + "。", "",
        "节点与依赖边固定；每轮保留此前修改。路径编号仅表示节点序列，不表示原始图中路径长度排名。", "",
        "| 轮次 | 路径 | 修改节点数 | 原路径预计缩短 ms | 全图模型收益 ms | 累计收益 ms | 更新后总耗时 s | 全部生效 |",
        "|---:|---|---:|---:|---:|---:|---:|---|",
        f"| 0 | P0 | 0 | — | — | 0 | {result['baseline_ms']/1000:.9f} | — |",
    ]
    for row in result["rounds"]:
        report.append(f"| {row['round']} | {row['previous_path_id']} → {row['path_id']} | {row['changed_count']} | {row['ideal_path_gain_ms']:.6f} | {row['model_gain_ms']:.6f} | {row['cumulative_gain_ms']:.6f} | {row['after_ms']/1000:.9f} | {'是' if row['full_path_gain_realized'] else '否'} |")
    report += ["", "每轮理想收益是沿修改前关键路径累加的节点缩短量；整体收益是全图最长路径重算前后的差值。“全部生效”比较原路径更新后的长度与全图新最长路径长度，不要求返回的路径 vector 相同。", "",
               "若一轮收益为零，程序仍可继续处理接替或并列的关键路径。停止于不可优化的当前关键路径，表示该路径在当前规则下构成无法继续降低的模型完成时间下界。", "",
               "这是固定计算 DAG 的反事实分析。阈值和目标耗时是工程师提供的假设，脚本不证明这些目标可实现；没有完整建模通信等待、资源竞争和训练尾部同步，不能直接视为实测训练迭代收益。", "",
               "输入与图信息：", "", "```json", json.dumps({"source": result["source"], "graph": result["graph"], "policy": result["policy"]}, ensure_ascii=False, indent=2), "```", "",
               "文件：`dag.json` 保存原始节点、固定依赖边和拓扑顺序；`analysis.json` 保存每轮修改、路径、精确收益和停止原因；`round_000_path.csv` 是基线，后续 CSV 是对应轮次更新后的关键路径。", ""]
    (output / "report.md").write_text("\n".join(report), encoding="utf-8")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--trace", help="DHTA traceEvents JSON, directly readable")
    source.add_argument("--input", help="Compact trace_inputs.json")
    parser.add_argument("--ssh", help="Read --trace on an SSH host, e.g. pan")
    parser.add_argument("--key", help="Dataset key for --input")
    parser.add_argument("--iteration", help="Select exactly one trace iteration")
    parser.add_argument("--threshold-ms", default="70", help="Strict forward duration threshold; default 70")
    parser.add_argument("--later-threshold-ms", help="Threshold from round 2 onward; default same as first")
    parser.add_argument("--target-ms", default="50", help="Forward target duration; default 50")
    parser.add_argument("--ranks", help="Optional comma-separated eligible rank IDs")
    parser.add_argument("--node-targets", help="Explicit per-event targets JSON; overrides threshold/rank selection")
    parser.add_argument("--max-rounds", type=int, default=10, help="Maximum optimization rounds, excluding baseline")
    parser.add_argument("--num-paths", type=int, help="Stop at this many distinct representative paths, including baseline")
    parser.add_argument("--output", required=True, help="New/empty local output directory")
    args = parser.parse_args()
    if args.max_rounds < 0 or args.num_paths is not None and args.num_paths < 1:
        parser.error("--max-rounds must be nonnegative and --num-paths must be positive")
    threshold = number(args.threshold_ms, "threshold")
    target = number(args.target_ms, "target")
    later = number(args.later_threshold_ms, "later threshold") if args.later_threshold_ms is not None else None
    if min(threshold, target, later if later is not None else threshold) < 0:
        parser.error("Thresholds and targets must be nonnegative")
    ranks = {int(v) for v in args.ranks.split(",")} if args.ranks else None
    data, provenance = read_source(args)
    graph, original, releases, iterations = build_graph(data, number(args.iteration, "iteration") if args.iteration is not None else None)
    if ranks is not None and not ranks.issubset(set(graph.ranks)):
        raise ValueError("--ranks contains IDs absent from the selected PP group")
    targets = load_targets(args.node_targets, graph)
    result, states = evaluate_rounds(graph, original, releases, args.max_rounds, threshold,
                                     target, args.num_paths, later, ranks, targets)
    violations = graph.violations()
    result.update({"source": provenance,
                   "graph": {"nodes": len(graph.nodes), "edges": sum(map(len, graph.pred)),
                             "rank_order": graph.ranks, "iterations": iterations,
                             "topological_nodes": len(graph.order), "vpp_wrap_edges_included": True,
                             "observed_timestamp_dependency_violations": len(violations),
                             "max_observed_dependency_overlap_ms": max((-v["gap_ms"] for v in violations), default=0)},
                   "policy": {"direction": "forward_only", "threshold_ms": float(threshold),
                              "later_threshold_ms": float(later) if later is not None else float(threshold),
                              "target_ms": float(target), "eligible_ranks": sorted(ranks) if ranks else None,
                              "node_targets_path": str(Path(args.node_targets).resolve()) if args.node_targets else None,
                              "node_targets": [{"node_id": i, "target_ms": float(value)} for i, value in targets.items()] if targets is not None else None,
                              "max_rounds": args.max_rounds, "selection": "one_current_representative_critical_path",
                              "tie_breaking": "smallest node ID", "arithmetic": "exact fractions of input decimals"}})
    output = write_outputs(args.output, graph, original, releases, result, states)
    print(f"Baseline: {result['baseline_ms']/1000:.9f} s")
    for row in result["rounds"]:
        print(f"Round {row['round']}: {row['previous_path_id']} -> {row['path_id']}, {row['changed_count']} fwd, "
              f"ideal={row['ideal_path_gain_ms']:.6f} ms, model={row['model_gain_ms']:.6f} ms, "
              f"total={row['after_ms']/1000:.9f} s")
    print(f"Cumulative model gain: {result['total_gain_ms']/1000:.9f} s; "
          f"distinct paths: {result['distinct_path_count']}; stop: {result['stop_reason']}")
    print(f"Report: {output / 'report.md'}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError) as error:
        print(f"Analysis failed: {error}", file=sys.stderr)
        sys.exit(1)
