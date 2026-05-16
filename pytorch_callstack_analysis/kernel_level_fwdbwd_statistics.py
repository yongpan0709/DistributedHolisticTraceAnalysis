from pathlib import Path
from hta.common.trace import Trace
from hta.common.trace_file import get_trace_files
from hta.configs.parser_config import  ParserConfig
from hta.common.trace_call_graph import CallGraph, CallStackIdentity
from collections import defaultdict
from typing import Dict, List
import numpy as np
import pandas as pd
from pytorch_callstack_analysis.call_graph_template import extract_func_name_from_template, extract_dup_or_shape_func_name_from_template, set_pandas_display_options
import re
from pytorch_callstack_analysis.utils.musa_basic_kernel_info import calculate_groupedlinear_tflops_or_bw, calculate_linear_tflops_or_bw, calculate_scaled_dot_product_attention_flash_musa_flops
from pytorch_callstack_analysis.utils.musa_fwdbwd_util import get_forward_duration_dup, get_forward_duration_uniq
from megatron_parallel_analysis.utils.call_graph_utils import get_main_stack_on_rank

def extract_shape(func_name, df, func_mapping_node_index, need_shape_func_name):
    shape_info: Dict[str, Dict[str, pd.Series]] = defaultdict(lambda: defaultdict(pd.Series))
    for forward_func_name, _ in func_name:
        if forward_func_name.split('@')[0] in need_shape_func_name:
            func_index = func_mapping_node_index[forward_func_name]
            if len(func_index) == 0:
                continue
            shape_array = []
            for index, row in df[df.index.isin(func_index)].iterrows(): 
                shape_array.append(row['input_dims'][0][0])
            shape_info[forward_func_name]['data'] = pd.Series(shape_array)
            shape_info[forward_func_name]['example'] = df.loc[func_index[0], 'input_dims']
    return shape_info

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


SHAPE_POSITION_FWD_BWD = {
    r"transformer_engine/pytorch/cpp_extensions/gemm.py\(\d+\): general_grouped_gemm": {
        "type": "TFLOPS",
        "ShapeFrom": r'_GroupedLinear', #  r"nn.Module: TE(Row|Column)ParallelGroupedLinear_0"
        "formula": calculate_groupedlinear_tflops_or_bw,
    },
    r"transformer_engine/pytorch/cpp_extensions/gemm.py\(\d+\): general_gemm": {
        "type": "TFLOPS",
        "ShapeFrom": r"(_Linear|_LayerNormLinear|RouterGatingLinearFunction)", # _Linear
        "formula": calculate_linear_tflops_or_bw,
    },
    r"LinearWithGradAccumulationAndAsyncCommunication": {
        "type": "TFLOPS",
        "ShapeFrom": r"LinearWithGradAccumulationAndAsyncCommunication", # _Linear
        "formula": calculate_linear_tflops_or_bw,
    },
    r"INVALID": {
        "type": "TFLOPS",
        "ShapeFrom": r"LinearWithGradAccumulationAndAsyncCommunication", # _Linear
        "formula": calculate_linear_tflops_or_bw,
    },
    r"transformer_engine/pytorch/tensor/quantized_tensor.py\(\d+\): quantize": {
        "type": "GB/s",
        "ShapeFrom": r"(_Linear|_LayerNormLinear|RouterGatingLinearFunction)", # _Linear
        "formula": calculate_linear_tflops_or_bw,
    },
    r"<built-in method fused_multi_quantize of PyCapsule object at 0x[0-9a-fA-F]+>":{
        "type": "GB/s",
        "ShapeFrom": r'_GroupedLinear', 
        "formula": calculate_groupedlinear_tflops_or_bw,
    }
}
# In template, funcs with @shape@ label for extracting shape info
# its first parent node is also the entry link to backward func node
# One kernel in forward will generate two backward kernels
output_template_to_file_debug = r"""
nn.Module: MLASelfAttention_0
    nn.Module: TELinear_0
        _Linear @dup@
            transformer_engine/pytorch/tensor/quantized_tensor.py\(\d+\): quantize @dup@ @shape@
            transformer_engine/pytorch/cpp_extensions/gemm.py\(\d+\): general_gemm @dup@ @shape@
    nn.Module: TELinear_1
        _Linear @dup@
            transformer_engine/pytorch/tensor/quantized_tensor.py\(\d+\): quantize @dup@ @shape@
            transformer_engine/pytorch/cpp_extensions/gemm.py\(\d+\): general_gemm @dup@ @shape@
    nn.Module: TELayerNormColumnParallelLinear_0
        _LayerNormLinear @dup@
            transformer_engine/pytorch/tensor/quantized_tensor.py\(\d+\): quantize @dup@ @shape@
            transformer_engine/pytorch/cpp_extensions/gemm.py\(\d+\): general_gemm @dup@ @shape@
    nn.Module: TELayerNormColumnParallelLinear_1
        _LayerNormLinear @dup@
            transformer_engine/pytorch/tensor/quantized_tensor.py\(\d+\): quantize @dup@ @shape@
            transformer_engine/pytorch/cpp_extensions/gemm.py\(\d+\): general_gemm @dup@ @shape@
    nn.Module: TEDotProductAttention_0
        nn.Module: FlashAttention_0
            aten::_scaled_dot_product_attention_flash_musa @shape@
    nn.Module: TERowParallelLinear_0
        _Linear @dup@
            transformer_engine/pytorch/tensor/quantized_tensor.py\(\d+\): quantize @dup@ @shape@
            transformer_engine/pytorch/cpp_extensions/gemm.py\(\d+\): general_gemm @dup@ @shape@
nn.Module: MoELayer_0
    nn.Module: TopKRouter_0
        RouterGatingLinearFunction
            transformer_engine/pytorch/cpp_extensions/gemm.py\(\d+\): general_gemm @dup@ @shape@
    nn.Module: SharedExpertMLP_0
        nn.Module: TEColumnParallelLinear_0 
            _Linear @dup@
                transformer_engine/pytorch/tensor/quantized_tensor.py\(\d+\): quantize @dup@ @shape@
                transformer_engine/pytorch/cpp_extensions/gemm.py\(\d+\): general_gemm @dup@ @shape@
        nn.Module: TERowParallelLinear_1
            _Linear @dup@
                transformer_engine/pytorch/tensor/quantized_tensor.py\(\d+\): quantize @dup@ @shape@
                transformer_engine/pytorch/cpp_extensions/gemm.py\(\d+\): general_gemm @dup@ @shape@
    nn.Module: TEGroupedMLP_0
        nn.Module: TEColumnParallelGroupedLinear_0
            _GroupedLinear @dup@
                <built-in method fused_multi_quantize of PyCapsule object at 0x[0-9a-fA-F]+> @dup@ @shape@
                transformer_engine/pytorch/cpp_extensions/gemm.py\(\d+\): general_grouped_gemm @dup@ @shape@
        nn.Module: TERowParallelGroupedLinear_0
            _GroupedLinear @dup@
                <built-in method fused_multi_quantize of PyCapsule object at 0x[0-9a-fA-F]+> @dup@ @shape@
                transformer_engine/pytorch/cpp_extensions/gemm.py\(\d+\): general_grouped_gemm @dup@ @shape@
megatron/core/models/gpt/gpt_model.py\(\d+\): _postprocess
    nn.Module: ColumnParallelLinear_0 @dup@
        LinearWithGradAccumulationAndAsyncCommunication
            INVALID @dup@ @shape@
"""

SHAPE_POSITION_FWD_BWD_OF_FLASH_ATTENTION = {
    r"aten::_scaled_dot_product_attention_flash_musa": {
        "type": "TFLOPS",
        "ShapeFrom": r"aten::_scaled_dot_product_attention_flash_musa",
        "bwd_func": r"aten::_scaled_dot_product_attention_flash_musa_backward",
        "formula": calculate_scaled_dot_product_attention_flash_musa_flops,
    }
}

# HTA_DISABLE_NS_ROUNDING=1 python call_graph_kernel_dur_sum.py
if __name__ == "__main__":
    import time
    base_dir = "../"
    trace_dir = str(Path(base_dir).joinpath("good_perf"))
    cfg = ParserConfig.get_default_cfg()
    # config for extracting shape info
    cfg.add_args(ParserConfig.ARGS_INPUT_SHAPE)
    ParserConfig.set_default_cfg(cfg)
    trace_files = get_trace_files(trace_dir)
    for rank, trace_file in trace_files.items():
        if rank != 32:
            continue
        # HTA starts from here
        t = Trace(trace_files={rank: trace_file}, trace_dir="")
        t.load_traces()
        # transform name and cat columns to s_name and s_cat
        # name and cat are kernel id
        t.decode_symbol_ids(use_shorten_name=False)
        set_pandas_display_options()
        t0 = time.perf_counter()
        cg = CallGraph(t)
        t1 = time.perf_counter()
        print(f"Rank {rank}, CallGraph took {t1 - t0:.2f} seconds")
        #df = cg.call_stacks[-1].full_df
        _, main_stack = get_main_stack_on_rank(cg, rank)
        df = main_stack.full_df
        #df.to_csv(f"./full_df_{rank}.csv", index=False)
        dup_func_name, need_shape_func_name = extract_dup_or_shape_func_name_from_template(output_template_to_file_debug)
        func_name = extract_func_name_from_template(output_template_to_file_debug)
        stat_info_funcs_grouped = pd.DataFrame()
        func_mapping_node_index: Dict[str, List[np.float64]] = defaultdict(list)

        for forward_func_name, func_ancestors in func_name:
            # print(f"Processing function: {forward_func_name}")
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
            # df['comm_volume'] = 0.0
            # df['tflop'] = 0.0
            df['TFLOPS'] = 0.0
            df['GB/s'] = 0.0

        tflop_bw_mapping_index = extract_shape_from_parents_to_tflops_or_bw_in_fwd(func_name, df, func_mapping_node_index, need_shape_func_name, rank, cg, SHAPE_POSITION_FWD_BWD)
        tflop_bw_mapping_index_bwd = extract_shape_from_parents_to_tflops_or_bw_in_bwd(func_name, df, func_mapping_node_index, need_shape_func_name, rank, cg, SHAPE_POSITION_FWD_BWD)
        print('tflop_bw_mapping_index_bwd:', tflop_bw_mapping_index_bwd)
        with open(f"./20260211-{rank}-repo6.txt", "w") as f:
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
                        #file_name = forward_func_name.replace('\\', '').replace('+', '').replace(':', '').replace(' ', '').replace('/', '_')
                        #df_subset.to_csv(f"./{fwd_bwd_key}_{rank}_{file_name}_tflops_bw.csv", columns=['index', 's_name', 'kernel_span', 'shape', 'TFLOPS', 'GB/s'], index=False)
                        f.write(f"{fwd_bwd_key} shape: {df_subset['shape'].iloc[0]}, ")
                        if forward_func_name.split('@')[0] in SHAPE_POSITION_FWD_BWD:
                            calculate_type = SHAPE_POSITION_FWD_BWD[forward_func_name.split('@')[0]]["type"] # TFLOPS or GB/s
                        else:
                            calculate_type = SHAPE_POSITION_FWD_BWD_OF_FLASH_ATTENTION[forward_func_name.split('@')[0]]["type"]
                        tflops_or_bw_mean = df_subset[calculate_type].mean()
                        #print(f": {forward_func_name}, tflops_mean: {tflops_mean}")
                        f.write(f" mean_time(us): {df_subset['kernel_span'].mean():.2f},")
                        f.write(f" {calculate_type if calculate_type == 'TFLOPS' else 'BW'} mean: {tflops_or_bw_mean:.2f} {calculate_type},")
                        f.write(f" q_25: {df_subset[calculate_type].quantile(.25):.2f}, q_50: {df_subset[calculate_type].quantile(.5):.2f}, q_75: {df_subset[calculate_type].quantile(.75):.2f},")
                        f.write(f" count: {len(df_subset)}\n")
                    #elif calculate_type == "GB/s":
                    #    f.write(f"  mean_time: {df_subset['kernel_span'].mean():.2f}, {calculate_type if calculate_type == 'TFLOPS' else 'BW'} mean: {tflops_or_bw_mean:.2f} {calculate_type}, q_25: {df_subset[calculate_type].quantile(.25):.2f}, q_50: {df_subset[calculate_type].quantile(.5):.2f}, q_75: {df_subset[calculate_type].quantile(.75):.2f}, count: {len(df_subset)}\n")
                    #    f.write(f"    bwd-0 shape: {df_subset_bwd_0['shape'].iloc[0] if not df_subset_bwd_0['shape'].empty else 'N/A'}, mean_time: {df_subset_bwd_0['kernel_span'].mean():.2f}, tflops_mean: {df_subset_bwd_0[calculate_type].mean():.2f} Tflops, q_25: {df_subset_bwd_0[calculate_type].quantile(.25):.2f}, q_50: {df_subset_bwd_0[calculate_type].quantile(.5):.2f}, q_75: {df_subset_bwd_0[calculate_type].quantile(.75):.2f}, count: {len(df_subset_bwd_0)}\n")
                    #    if len(df_subset_bwd_1) > 0:
                    #        f.write(f"    bwd-1 shape: {df_subset_bwd_1['shape'].iloc[0] if not df_subset_bwd_1['shape'].empty else 'N/A'}, tflops_mean: {df_subset_bwd_1[calculate_type].mean():.2f} Tflops, mean_time: {df_subset_bwd_1['kernel_span'].mean():.2f}, q_25: {df_subset_bwd_1[calculate_type].quantile(.25):.2f}, q_50: {df_subset_bwd_1[calculate_type].quantile(.5):.2f}, q_75: {df_subset_bwd_1[calculate_type].quantile(.75):.2f}, count: {len(df_subset_bwd_1)}\n")
                    #file_name = forward_func_name.replace('\\', '').replace('+', '').replace(':', '').replace(' ', '').replace('/', '_')
                else:
                    print(f'{"    " * len(func_ancestors)}{forward_func_name}')
                    f.write(f'{"    " * len(func_ancestors)}{forward_func_name}\n')
