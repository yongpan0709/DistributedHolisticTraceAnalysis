from hta.common.trace_call_graph import CallStackIdentity
from typing import Dict, List
from collections import deque
import numpy as np
import pandas as pd
from pytorch_callstack_analysis.utils.timing import time_it

@time_it("get_backward_duration")
def get_backward_duration(df, cg, rank, forward_index):
    fwdbwd_index_series = df['fwdbwd_index']
    num_kernels_series = df['num_kernels']
    s_cat_series = df['s_cat']
    first_kernel_start_series = df['first_kernel_start']
    last_kernel_end_series = df['last_kernel_end']
    """
    Todo: refact to void checking children and grand-children funcs
    """
    rank_nodes = cg.rank_to_nodes[rank]
    backward_info: Dict[np.int64, np.float64] = {}

    for forward_as_parent in forward_index:
        parents = deque([forward_as_parent])
        first_kernel_start = None
        last_kernel_end = None

        while parents:
            cur_node = parents.popleft()
            cur_node_fwdbwd_index = fwdbwd_index_series.loc[cur_node]
            if cur_node_fwdbwd_index > 0:
                cur_node_fwdbwd_num_kernels = num_kernels_series.loc[cur_node_fwdbwd_index]
                if cur_node_fwdbwd_num_kernels > 0:
                    cur_first_kernel_start = first_kernel_start_series.loc[cur_node_fwdbwd_index]
                    cur_last_kernel_end = last_kernel_end_series.loc[cur_node_fwdbwd_index]
                    if first_kernel_start is None or cur_first_kernel_start < first_kernel_start:
                        first_kernel_start = cur_first_kernel_start
                    if last_kernel_end is None or cur_last_kernel_end > last_kernel_end:
                        last_kernel_end = cur_last_kernel_end
                continue

            cur_call_stack_node = rank_nodes.get(cur_node)
            cur_node_children = cur_call_stack_node.children
            if len(cur_node_children) > 0:
                for child in cur_node_children:
                    if s_cat_series.loc[child] != 'kernel':
                        parents.append(child)

        if first_kernel_start is not None and last_kernel_end is not None:
            backward_info[forward_as_parent] = last_kernel_end - first_kernel_start

    backward_stat_info = pd.DataFrame.from_dict(backward_info, orient='index', columns=['kernel_span'])
    return backward_stat_info

@time_it("get_forward_duration_uniq")
def get_forward_duration_uniq(df, forward_func_name):
    node_index = df[df['s_name'].str.match(pat=r"^"+forward_func_name+r"$")].index
    return df[df['index'].isin(node_index)]

@time_it("get_forward_duration_dup")
def get_forward_duration_dup(df, forward_func_name, func_ancestors, cg, rank, func_mapping_node_index):
    """
    获取指定函数的 forward duration（处理重复函数名的情况）
    
    性能优化：
    1. 提前将 nearest_ancestor_index 转换为 set，避免每次循环重建
    2. 提前筛选候选索引，避免每次循环都进行 regex 匹配
    3. 使用 set.intersection() 直接判断，避免创建临时 set
    """
    node_index: List[np.int64] = []
    
    # 性能优化：提前将祖先索引转换为 set
    nearest_ancestor_index = func_mapping_node_index[func_ancestors[-1]]
    nearest_ancestor_set = set(nearest_ancestor_index)
    
    # 性能优化：提前筛选候选索引，避免每次循环都进行 regex 匹配
    candidate_indices = df[df['s_name'].str.match(pat=r"^"+forward_func_name+r"$")].index
    
    for index in candidate_indices:
        # 获取该索引对应的 pid 和 tid
        pid, tid = df.loc[index, 'pid'], df.loc[index, 'tid']
        
        # 获取从当前节点到根节点的路径
        path_to_root = cg.rank_to_stacks[rank][CallStackIdentity(rank, pid, tid)].get_path_to_root(index)
        
        # 性能优化：直接使用 set.intersection()，避免创建临时 set
        # path_to_root 是 list，nearest_ancestor_set 是 set
        # set.intersection() 可以接受任何 iterable
        if nearest_ancestor_set.intersection(path_to_root):
            node_index.append(index)
    
    return df[df['index'].isin(node_index)]