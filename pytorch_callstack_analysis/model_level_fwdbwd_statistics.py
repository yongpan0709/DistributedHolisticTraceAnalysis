import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from hta.common.trace import Trace
from hta.common.trace_call_graph import CallGraph
from hta.common.trace_file import get_trace_files
from hta.configs.parser_config import ParserConfig

from pytorch_callstack_analysis.call_graph_template import (
    extract_dup_or_shape_func_name_from_template,
    extract_func_name_from_template,
    output_template_to_file,
    output_template_to_file_debug,
    output_template_to_file_kimi,
    output_template_to_file_kimi_epoverlap,
    set_pandas_display_options,
)
from pytorch_callstack_analysis.utils.musa_fwdbwd_util import get_backward_duration, get_forward_duration_dup, get_forward_duration_uniq
from pytorch_callstack_analysis.utils.timing import time_it
from megatron_parallel_analysis.utils.call_graph_utils import get_main_stack_on_rank

TEMPLATE_MAP = {
    "default": output_template_to_file,
    "debug": output_template_to_file_debug,
    "kimi": output_template_to_file_kimi,
    "kimi_epoverlap": output_template_to_file_kimi_epoverlap,
}


@time_it("calculate_statistics")
def calculate_statistics(df: pd.DataFrame, func_name: str, calculate_col_name: str = "kernel_span"):
    series = df[calculate_col_name] / 1000.0
    series = series[series > 0]
    df_dur = pd.DataFrame(
        {
            "mean": series.mean(),
            "q_25": series.quantile(.25),
            "q_50": series.quantile(.5),
            "q_75": series.quantile(.75),
            "max": series.max(),
            "min": series.min(),
            "var": series.var(),
            "count": series.count(),
        },
        index=[func_name],
        columns=["mean", "q_25", "q_50", "q_75", "max", "min", "var", "count"],
    )
    return df_dur


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze model-level forward/backward statistics from an HTA trace.",
    )
    parser.add_argument("--trace-dir", required=True, help="trace directory")
    parser.add_argument("--rank", type=int, default=None, help="rank to analyze; if not specified, all ranks will be analyzed")
    parser.add_argument(
        "--template",
        choices=sorted(TEMPLATE_MAP.keys()),
        default="kimi_epoverlap",
        help="template name to use for call graph traversal",
    )
    parser.add_argument(
        "--output-dir",
        "--output",
        default="model_main_stack",
        help="output directory; each rank is written to <template>-<rank>-main-stack.txt",
    )
    return parser.parse_args()


# HTA_DISABLE_NS_ROUNDING=1 python call_graph_kernel_dur_sum.py
def analyze_rank(
    rank: int,
    trace_files: Dict[int, str],
    template,
    template_name: str,
    output_path: str,
):
    """Analyze a single rank and write results to file."""
    print(f"Analyzing rank {rank}...")
    
    t = Trace(trace_files={rank: trace_files[rank]}, trace_dir="")
    t.load_traces()
    t.decode_symbol_ids(use_shorten_name=False)
    set_pandas_display_options()

    cg = CallGraph(t)

    _, main_stack = get_main_stack_on_rank(cg, rank)
    df = main_stack.full_df

    dup_func_name, _ = extract_dup_or_shape_func_name_from_template(template)
    func_name = extract_func_name_from_template(template)
    func_mapping_node_index: Dict[str, List[np.float64]] = defaultdict(list)
    stat_info_list = []
    for forward_func_name, func_ancestors in func_name:
        print(f"Processing function: {forward_func_name}")
        base_func_name = forward_func_name.split('@')[0]
        if base_func_name not in dup_func_name:
            fwd_df = get_forward_duration_uniq(df, base_func_name)
        else:
            fwd_df = get_forward_duration_dup(df, base_func_name, func_ancestors, cg, rank, func_mapping_node_index)
        if len(fwd_df) == 0:
            print(f"forward_func_name no data: {forward_func_name}")
            continue

        fwd_dur = calculate_statistics(fwd_df, forward_func_name, "kernel_span")
        func_mapping_node_index[forward_func_name] = fwd_df.index
        bwd_df = get_backward_duration(
            df,
            cg,
            rank,
            fwd_df.index
        )
        bwd_dur = calculate_statistics(bwd_df, forward_func_name + '-bwd', "kernel_span")
        stat_info_list.extend([fwd_dur, bwd_dur])

    if not stat_info_list:
        raise ValueError(f"No statistics were generated for rank {rank} with the selected template")

    stat_info_funcs_grouped = pd.concat(stat_info_list, axis=0)

    root_func_name, _ = func_name[0]
    fwd_total_dur_mean = stat_info_funcs_grouped.loc[root_func_name, 'mean']
    bwd_total_dur_mean = stat_info_funcs_grouped.loc[root_func_name + '-bwd', 'mean']

    is_bwd = stat_info_funcs_grouped.index.str.endswith('-bwd')
    stat_info_funcs_grouped['mean_percent'] = np.where(
        is_bwd,
        stat_info_funcs_grouped['mean'] * stat_info_funcs_grouped['count'] / bwd_total_dur_mean,
        stat_info_funcs_grouped['mean'] * stat_info_funcs_grouped['count'] / fwd_total_dur_mean,
    )

    with open(output_path, "w") as f:
        f.write(f"# Rank: {rank}\n")
        for forward_func_name, func_ancestors in func_name:
            print(f'{"    " * len(func_ancestors)}{forward_func_name}')
            f.write(f'{"    " * len(func_ancestors)}{forward_func_name}\n')
            if forward_func_name not in stat_info_funcs_grouped.index:
                continue

            backward_func_name = forward_func_name + '-bwd'
            fwd_row = stat_info_funcs_grouped.loc[forward_func_name]
            bwd_row = stat_info_funcs_grouped.loc[backward_func_name]
            f.write(
                f'{"    " * (len(func_ancestors)+1)} fwd: mean_percent: {fwd_row["mean_percent"]:.2f}, '
                f'mean: {fwd_row["mean"]:.2f}, q_25: {fwd_row["q_25"]:.2f}, q_50: {fwd_row["q_50"]:.2f}, '
                f'q_75: {fwd_row["q_75"]:.2f}, max: {fwd_row["max"]:.2f}, min: {fwd_row["min"]:.2f}, '
                f'count: {fwd_row["count"]:.2f}\n'
            )
            f.write(
                f'{"    " * (len(func_ancestors)+1)} bwd: mean_percent: {bwd_row["mean_percent"]:.2f}, '
                f'mean: {bwd_row["mean"]:.2f}, q_25: {bwd_row["q_25"]:.2f}, q_50: {bwd_row["q_50"]:.2f}, '
                f'q_75: {bwd_row["q_75"]:.2f}, max: {bwd_row["max"]:.2f}, min: {bwd_row["min"]:.2f}, '
                f'count: {bwd_row["count"]:.2f}\n'
            )
    
    print(f"Results for rank {rank} written to {output_path}")


@time_it("main")
def main():
    args = parse_args()
    trace_dir = args.trace_dir.rstrip("/")
    template_name = args.template
    template = TEMPLATE_MAP[template_name]

    cfg = ParserConfig.get_default_cfg()
    #cfg.add_args(ParserConfig.ARGS_INPUT_SHAPE)
    cfg.set_drop_python_function_events(True)
    ParserConfig.set_default_cfg(cfg)

    trace_files = get_trace_files(trace_dir)
    
    # Determine which ranks to analyze
    if args.rank is not None:
        # Analyze specific rank
        if args.rank not in trace_files:
            raise ValueError(f"Rank {args.rank} not found in trace dir: {trace_dir}")
        ranks_to_analyze = [args.rank]
    else:
        # Analyze all ranks
        ranks_to_analyze = sorted(trace_files.keys())
        print(f"No rank specified, analyzing all ranks: {ranks_to_analyze}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for rank in ranks_to_analyze:
        output_path = output_dir / f"{template_name}-{rank}-main-stack.txt"
        analyze_rank(
            rank=rank,
            trace_files=trace_files,
            template=template,
            template_name=template_name,
            output_path=str(output_path),
        )


if __name__ == "__main__":
    main()
