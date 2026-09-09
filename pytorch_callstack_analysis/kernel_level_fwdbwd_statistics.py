import argparse
from collections import defaultdict
from pathlib import Path
import re
from typing import Dict, List

import numpy as np
import pandas as pd

from hta.common.trace import Trace
from hta.common.trace_call_graph import CallGraph, CallStackIdentity
from hta.common.trace_file import get_trace_files
from hta.configs.parser_config import ParserConfig
from megatron_parallel_analysis.utils.call_graph_utils import get_main_stack_on_rank
from pytorch_callstack_analysis.call_graph_template import (
    SHAPE_POSITION_FWD_BWD,
    SHAPE_POSITION_FWD_BWD_OF_FLASH_ATTENTION,
    extract_dup_or_shape_func_name_from_template,
    extract_func_name_from_template,
    kernel_level_template,
    set_pandas_display_options,
)
from pytorch_callstack_analysis.utils.musa_fwdbwd_util import get_forward_duration_dup, get_forward_duration_uniq


def extract_shape_from_parents_to_tflops_or_bw_in_fwd(func_name, df, func_mapping_node_index, need_shape_func_name, rank, cg, shape_position):
    tflop_bw_mapping_index: Dict[str, pd.Series] = defaultdict(pd.Series)

    for forward_func_name, _ in func_name:
        if forward_func_name.split('@')[0] in need_shape_func_name:
            if forward_func_name.split('@')[0] in shape_position:
                index_array = []
                func_index = func_mapping_node_index[forward_func_name]
                if len(func_index) == 0:
                    continue
                
                for index, row in df[df.index.isin(func_index)].iterrows(): 
                    cur_node_index = index
                    pid, tid = row['pid'], row['tid']
                    parent_index = cg.rank_to_stacks[rank][CallStackIdentity(rank, pid, tid)].get_parent(cur_node_index)
                    while parent_index >= 0:
                        if re.match(shape_position[forward_func_name.split('@')[0]]["ShapeFrom"], df.at[cur_node_index, 's_name']):
                            # print(f"Found shape for {forward_func_name} from parent func {df.at[cur_node_index, 's_name']} at index {cur_node_index} with input dims {df.at[cur_node_index, 'input_dims']}\n")
                            #df.at[index, 'shape'] = df.at[cur_node_index, 'input_dims']
                            _calculate_tflops_or_bw(df, index, cur_node_index, forward_func_name, shape_position, cal_phrase='fwd-0')
                            index_array.append(index)
                            break
                        else:
                            cur_node_index = parent_index
                            pid, tid = df[df.index == cur_node_index][['pid', 'tid']].values[0]
                            parent_index = cg.rank_to_stacks[rank][CallStackIdentity(rank, pid, tid)].get_parent(cur_node_index)
                tflop_bw_mapping_index[forward_func_name] = pd.Series(index_array)
            else:
                # Todo: Assume here the func is unique
                fwd_df = get_forward_duration_uniq(df, forward_func_name.split('@')[0])
                for index, row in df[df.index.isin(fwd_df.index)].iterrows():
                    _calculate_tflops_or_bw(df, index, index, forward_func_name, SHAPE_POSITION_FWD_BWD_OF_FLASH_ATTENTION, cal_phrase='fwd-0')
                tflop_bw_mapping_index[forward_func_name] = fwd_df.index
    return tflop_bw_mapping_index

def _dfs_traverse(cg, rank, node_id, forward_func_name, df, matched_nodes):
    """Depth first traversal on a specific call stack to find matched nodes.
    """
    node = cg.rank_to_nodes[rank].get(node_id)
    if re.match(forward_func_name.split('@')[0], df.at[node_id, 's_name']):
        matched_nodes.append(node_id)
        return
    for child_nid in node.children:
        _dfs_traverse(cg, rank, child_nid, forward_func_name, df, matched_nodes)


def _calculate_tflops_or_bw(df, cur_kernel_node_id, shape_from_node_id, forward_func_name, shape_position, cal_phrase='fwd-0'):
    formula_func = shape_position[forward_func_name.split('@')[0]]["formula"]
    calculate_type = shape_position[forward_func_name.split('@')[0]]["type"] 
    kernel_span = df.at[cur_kernel_node_id, 'kernel_span']/1000.0/1000.0 # convert us to s
    df.at[cur_kernel_node_id, 'shape'], df.at[cur_kernel_node_id, calculate_type] = formula_func(df.at[shape_from_node_id, 'input_dims'], df.at[shape_from_node_id, 'input_type'][0], kernel_span, df.at[shape_from_node_id, 's_name'], calculate_type, cal_phrase)


def extract_shape_from_parents_to_tflops_or_bw_in_bwd(func_name, df, func_mapping_node_index, need_shape_func_name, rank, cg, shape_position):
    tflop_bw_bwd_mapping_index: Dict[str, pd.Series] = defaultdict(pd.Series)
    for forward_func_name, func_ancestors in func_name:
        if forward_func_name.split('@')[0] in need_shape_func_name:
            if forward_func_name.split('@')[0] in shape_position:
                nearest_ancestor_index = func_mapping_node_index[func_ancestors[-1]]
                bwd_index_array = defaultdict(list)
                if len(nearest_ancestor_index) == 0:
                    continue
                # the parent of nodes with label shape have fwd_bwd link to backward
                for index, row in df[df.index.isin(nearest_ancestor_index)].iterrows(): 
                    cur_node_fwdbwd_index = df.at[index, 'fwdbwd_index']
                    if cur_node_fwdbwd_index <= 0:
                        continue
                    # DFS traverse to find backward kernels for keeping the order
                    matched_nodes = []
                    _dfs_traverse(cg, rank, cur_node_fwdbwd_index, forward_func_name, df, matched_nodes)
                    # print(f'matched nodes: {matched_nodes} for forward func name: {forward_func_name}')

                    # idx_for_node: the n-th kernel in bwd
                    for idx_for_node, cur_node in enumerate(matched_nodes):
                        bwd_index_array[idx_for_node].append(cur_node)
                        _calculate_tflops_or_bw(df, cur_node, index, forward_func_name, shape_position, cal_phrase=f'bwd-{idx_for_node}')
                for bwd_pos, bwd_index_list in bwd_index_array.items():
                    tflop_bw_bwd_mapping_index[f'{forward_func_name}-bwd-{bwd_pos}'] = pd.Series(bwd_index_list)
            else:
                if forward_func_name.split('@')[0] in SHAPE_POSITION_FWD_BWD_OF_FLASH_ATTENTION:
                    # Todo: Assume here the fa bwd func is also unique
                    bwd_func_name = SHAPE_POSITION_FWD_BWD_OF_FLASH_ATTENTION[forward_func_name.split('@')[0]]['bwd_func']
                    bwd_df = get_forward_duration_uniq(df, bwd_func_name)
                    for index, row in df[df.index.isin(bwd_df.index)].iterrows():
                        _calculate_tflops_or_bw(df, index, index, forward_func_name, SHAPE_POSITION_FWD_BWD_OF_FLASH_ATTENTION, cal_phrase='bwd-0')
                    tflop_bw_bwd_mapping_index[f'{forward_func_name}-bwd-0'] = bwd_df.index
    return tflop_bw_bwd_mapping_index


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze kernel-level forward/backward TFLOPS or bandwidth statistics from an HTA trace.",
    )
    parser.add_argument("--trace-dir", required=True, help="trace directory")
    parser.add_argument(
        "--rank",
        type=int,
        default=None,
        help="rank to analyze; if not specified, all ranks will be analyzed",
    )
    parser.add_argument(
        "--output-dir",
        "--output",
        default="kernel_fwdbwd_statistics",
        help="output directory; each rank is written to <rank>-kernel-fwdbwd.txt",
    )
    return parser.parse_args()


def analyze_rank(rank: int, trace_files: Dict[int, str], output_path: str):
    print(f"Analyzing rank {rank}...")

    t = Trace(trace_files={rank: trace_files[rank]}, trace_dir="")
    t.load_traces()
    t.decode_symbol_ids(use_shorten_name=False)
    set_pandas_display_options()

    cg = CallGraph(t)
    _, main_stack = get_main_stack_on_rank(cg, rank)
    df = main_stack.full_df

    dup_func_name, need_shape_func_name = extract_dup_or_shape_func_name_from_template(kernel_level_template)
    func_name = extract_func_name_from_template(kernel_level_template)
    func_mapping_node_index: Dict[str, List[np.float64]] = defaultdict(list)

    for forward_func_name, func_ancestors in func_name:
        if forward_func_name.split('@')[0] not in dup_func_name:
            fwd_df = get_forward_duration_uniq(df, forward_func_name.split('@')[0])
        else:
            fwd_df = get_forward_duration_dup(df, forward_func_name.split('@')[0], func_ancestors, cg, rank, func_mapping_node_index)
        if len(fwd_df) == 0:
            print(f"forward_func_name no data: {forward_func_name}")
            continue
        func_mapping_node_index[forward_func_name] = fwd_df.index

    if 'shape' not in df:
        df['shape'] = df['input_dims']
        df['TFLOPS'] = 0.0
        df['GB/s'] = 0.0

    tflop_bw_mapping_index = extract_shape_from_parents_to_tflops_or_bw_in_fwd(func_name, df, func_mapping_node_index, need_shape_func_name, rank, cg, SHAPE_POSITION_FWD_BWD)
    tflop_bw_mapping_index_bwd = extract_shape_from_parents_to_tflops_or_bw_in_bwd(func_name, df, func_mapping_node_index, need_shape_func_name, rank, cg, SHAPE_POSITION_FWD_BWD)

    with open(output_path, "w") as f:
        for forward_func_name, func_ancestors in func_name:
            if forward_func_name in tflop_bw_mapping_index:
                index_series = tflop_bw_mapping_index[forward_func_name]
                bwd_first_index = tflop_bw_mapping_index_bwd[f'{forward_func_name}-bwd-0']
                bwd_second_index = tflop_bw_mapping_index_bwd.get(f'{forward_func_name}-bwd-1', [])
                f.write(f'{"    " * len(func_ancestors)}{forward_func_name}\n')
                for fwd_bwd_key, df_subset_index_series in [('fwd-0', index_series), ('bwd-0', bwd_first_index), ('bwd-1', bwd_second_index)]:
                    if len(df_subset_index_series) == 0:
                        continue

                    df_subset = df[df.index.isin(df_subset_index_series)]
                    f.write(f"{fwd_bwd_key} shape: {df_subset['shape'].iloc[0]}, ")
                    if forward_func_name.split('@')[0] in SHAPE_POSITION_FWD_BWD:
                        calculate_type = SHAPE_POSITION_FWD_BWD[forward_func_name.split('@')[0]]["type"]
                    else:
                        calculate_type = SHAPE_POSITION_FWD_BWD_OF_FLASH_ATTENTION[forward_func_name.split('@')[0]]["type"]
                    tflops_or_bw_mean = df_subset[calculate_type].mean()
                    f.write(f" mean_time(us): {df_subset['kernel_span'].mean():.2f},")
                    f.write(f" {calculate_type if calculate_type == 'TFLOPS' else 'BW'} mean: {tflops_or_bw_mean:.2f} {calculate_type},")
                    f.write(f" q_25: {df_subset[calculate_type].quantile(.25):.2f}, q_50: {df_subset[calculate_type].quantile(.5):.2f}, q_75: {df_subset[calculate_type].quantile(.75):.2f},")
                    f.write(f" count: {len(df_subset)}\n")
            else:
                f.write(f'{"    " * len(func_ancestors)}{forward_func_name}\n')

    print(f"Results for rank {rank} written to {output_path}")


def main():
    args = parse_args()
    trace_dir = args.trace_dir.rstrip("/")

    cfg = ParserConfig.get_default_cfg()
    cfg.add_args(ParserConfig.ARGS_INPUT_SHAPE)
    cfg.set_drop_python_function_events(True)
    ParserConfig.set_default_cfg(cfg)

    trace_files = get_trace_files(trace_dir)

    if args.rank is not None:
        if args.rank not in trace_files:
            raise ValueError(f"Rank {args.rank} not found in trace dir: {trace_dir}")
        ranks_to_analyze = [args.rank]
    else:
        ranks_to_analyze = sorted(trace_files.keys())
        print(f"No rank specified, analyzing all ranks: {ranks_to_analyze}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for rank in ranks_to_analyze:
        output_path = output_dir / f"kernel-level-rank{rank}-fwdbwd.txt"
        analyze_rank(rank=rank, trace_files=trace_files, output_path=str(output_path))


if __name__ == "__main__":
    main()
