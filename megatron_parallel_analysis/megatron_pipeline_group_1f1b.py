# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Optional

import pandas as pd

from hta.common.trace_filter import NameFilter
from megatron_parallel_analysis.utils.trace_filter_utils import create_regex_for_prefix_match
from hta.configs.config import logger
from hta.configs.default_values import DEFAULT_TRACE_DIR
from megatron_parallel_analysis.megatron_pipeline_group_base import MegatronPipelineParallelGroupTraceBase


class MegatronPipelineParallel1F1BGroupTrace(MegatronPipelineParallelGroupTraceBase):
    """1F1B (One-Forward-One-Backward) 调度下的 PP group trace 分析。"""
    def __init__(self,
        trace_files: Optional[Dict[int, str]] = None,
        trace_dir: str = DEFAULT_TRACE_DIR,
        dp = -1,
        tp = -1,
        pp = -1,
        ep = -1,
        cp: int =1, 
        order: str ="tp-cp-ep-dp-pp",
        micro_bs: int =0,
        ) -> None:
        super().__init__(trace_files, trace_dir, dp, tp, pp, ep, cp, order, micro_bs)

    # Todo: func list
    @staticmethod
    def keep_comm_span_only(trace_df):
        comm_names_list = [
            'forward_step', 
            'backward_step', 
            'recv_forward', 
            'recv_backward', 
            'send_forward', 
            'send_backward', 
            'send_forward_recv_backward', 
            'send_backward_recv_forward', 
            # 'send_forward_recv_forward',
            # 'send_backward_recv_backward',
            # useing batch_isend_irecv
            # 'mccl:send', 
            # 'mccl:recv', 
            'finalize_model_grads',
            'step',
            'logical_and_across_model_parallel_group',
            'reduce_max_stat_across_model_parallel_group',
            'should_run_forward_backward',
            #'mccl:all_reduce'
        ]
        filter_comm = NameFilter(create_regex_for_prefix_match(comm_names_list))
        return filter_comm(trace_df)

    def filter_comm_only_traces(self, pp_group_id=0):
        for rank in self.all_pipeline_parallel_group_ranks[pp_group_id]:
            self.traces_comm_only[rank] = self.keep_comm_span_only(self.full_dfs[rank])

    def set_self_microbatch_id(self, trace_df: pd.DataFrame) -> None:
        trace_df.sort_values(by=['ts', 'dur'], ascending=[True, False], inplace=True)
        trace_df['micro_batch_id_forward'] = -1
        trace_df['micro_batch_id_backward'] = -1
        trace_df['recv_forward'] = trace_df['s_name'].str.contains('recv_forward').astype(int).cumsum() - 1
        trace_df['recv_backward'] = trace_df['s_name'].str.contains('recv_backward').astype(int).cumsum() - 1
        trace_df.loc[trace_df['s_name'].str.contains('forward'), 'micro_batch_id_forward'] = trace_df['recv_forward']
        trace_df.loc[trace_df['s_name'].str.contains('backward'), 'micro_batch_id_backward'] = trace_df['recv_backward']
        trace_df.drop(['recv_forward', 'recv_backward'], axis=1, inplace=True)

    def set_recv_send_microbatch_id(self, trace_df: pd.DataFrame) -> None:
        trace_df['send_prev'] = -1
        trace_df['send_next'] = -1
        trace_df['recv_prev'] = -1
        trace_df['recv_next'] = -1
        trace_df.loc[trace_df['s_name'].str.contains('recv_forward', regex=False), 'recv_prev'] = trace_df['micro_batch_id_forward']
        trace_df.loc[trace_df['s_name'].str.contains('recv_backward', regex=False), 'recv_next'] = trace_df['micro_batch_id_backward']
        trace_df.loc[trace_df['s_name'].str.contains('send_forward', regex=False), 'send_next'] = trace_df['micro_batch_id_forward']
        trace_df.loc[trace_df['s_name'].str.contains('send_backward', regex=False), 'send_prev'] = trace_df['micro_batch_id_backward']

    # Use: func annotations
    def process_pipeline_start(self, df):
        #if is_first_stage(rank, self.tensor_parallel_size, self.pipeline_parallel_size, self.data_parallel_size):
        df.loc[df['s_name'].str.contains('recv_forward'), 'recv_prev'] = -1
        df.loc[df['s_name'].str.contains('send_backward'), 'send_prev'] = -1

    def process_pipeline_end(self, df):
        #if is_last_stage(rank, self.tensor_parallel_size, self.pipeline_parallel_size, self.data_parallel_size):
        df.loc[df['s_name'].str.contains('recv_backward'), 'recv_next'] = -1
        df.loc[df['s_name'].str.contains('send_forward'), 'send_next'] = -1

    def set_micro_batch_id(self, pp_group_id: int = 0) -> None:
        """为指定 PP 组内各 rank 的 trace 设置 micro-batch id，由子类的 set_self_microbatch_id/set_recv_send_microbatch_id 实现具体算法。"""
        ranks = self.all_pipeline_parallel_group_ranks[pp_group_id]
        logger.info(f'In set micro batch id: ranks: {ranks}')
        for rank in ranks:
            self.set_self_microbatch_id(self.full_dfs[rank])
            self.set_recv_send_microbatch_id(self.full_dfs[rank])
        self.process_pipeline_start(self.full_dfs[ranks[0]])
        self.process_pipeline_end(self.full_dfs[ranks[-1]])

    def get_p2p_ranks_pairs(self, ranks):
        if len(ranks) < 2: return []
        # Todo: double check
        ranks_sorted = sorted(ranks)
        p2p_devices_pairs = []
        for i in range(len(ranks) - 1):
            p2p_devices_pairs.append([ranks[i], ranks[i+1]])
        return p2p_devices_pairs 

    def preprocess_trace_df(self, rank):
        trace_df = self.full_dfs[rank]
        sorted_trace_df = trace_df.sort_values(by=['ts', 'kernel_span'], ascending=[True, False])
        # Use regex to find the first occurrence of any 'recv_forward*' event
        recv_forward_index = sorted_trace_df[sorted_trace_df['s_name'].str.contains(r'^recv_forward.*')].index
        first_recv_forward_index = recv_forward_index[0]
        # Keep only the rows from the first 'recv_forward*' event onwards
        sorted_trace_df = sorted_trace_df.loc[first_recv_forward_index:]

        # sorted_trace_df['end'] = sorted_trace_df['ts'] + sorted_trace_df['dur']
        # sorted_trace_df.to_csv('sorted_trace_df-after-recv.csv')
        return sorted_trace_df

    def get_all_comm_df(self, sorted_trace_df, rank=None):
        return NameFilter(create_regex_for_prefix_match(['send_forward',  'recv_forward', 'send_backward',  'recv_backward']))(sorted_trace_df)

    def calculate_comm_time_total(self, all_comm_time_df):
        return all_comm_time_df['kernel_span'].sum()/1000
    
    def calculate_theoretical_bubble_time_warmup(self, all_comm_time_df, stage_id):
        if stage_id == 0:
            return 0.0
        else:
            theoretical_bubble_head_index = all_comm_time_df[all_comm_time_df['s_name'].str.contains(r'^recv_forward(?:_\d+)?$')].index[0]
            return all_comm_time_df.loc[theoretical_bubble_head_index, 'wait_time']/1000

    def calculate_bubble_time_warmup(self, all_comm_time_df, stage_id=None):
        recv_index_in_head = all_comm_time_df[all_comm_time_df['s_name'].str.contains(r'^recv_forward(?:_\d+)?$')].index
        send_index_in_head = all_comm_time_df[all_comm_time_df['s_name'].str.contains(r'^send_forward(?:_\d+)?$')].index
        return all_comm_time_df.loc[recv_index_in_head, 'wait_time'].sum()/1000 + all_comm_time_df.loc[send_index_in_head, 'wait_time'].sum()/1000

    def calculate_theoretical_bubble_time_steady(self, all_comm_time_df, stage_id):
        if stage_id == self.pipeline_parallel_size-1:
            return 0.0
        else:
            send_forward_recv_backward_index = all_comm_time_df[all_comm_time_df['s_name'].str.contains(r'^send_forward_recv_backward.*')].index
            if len(send_forward_recv_backward_index) > 0:
                return all_comm_time_df.loc[send_forward_recv_backward_index[0], 'wait_time']/1000
            else:
                return 0.0

    def calculate_bubble_time_steady(self, all_comm_time_df, stage_id=None):
        send_forward_recv_backward_index = all_comm_time_df[all_comm_time_df['s_name'].str.contains(r'^send_forward_recv_backward.*')].index
        send_backward_recv_forward_index = all_comm_time_df[all_comm_time_df['s_name'].str.contains(r'^send_backward_recv_forward.*')].index
        return all_comm_time_df.loc[send_forward_recv_backward_index, 'wait_time'].sum()/1000 + all_comm_time_df.loc[send_backward_recv_forward_index, 'wait_time'].sum()/1000
    
    def calculate_theoretical_bubble_time_cooldown(self, all_comm_time_df, stage_id):
        if stage_id == self.pipeline_parallel_size-1:
            return 0.0
        else:
            recv_backward_index = all_comm_time_df[all_comm_time_df['s_name'].str.contains(r'^recv_backward(?:_\d+)?$')].index
            return all_comm_time_df.loc[recv_backward_index, 'wait_time'].sum()/1000

    def calculate_bubble_time_cooldown(self, all_comm_time_df, stage_id=None):
        send_backward_index = all_comm_time_df[all_comm_time_df['s_name'].str.contains(r'^send_backward(?:_\d+)?$')].index
        recv_backward_index = all_comm_time_df[all_comm_time_df['s_name'].str.contains(r'^recv_backward(?:_\d+)?$')].index
        bubble_time_cooldown = all_comm_time_df.loc[send_backward_index, 'wait_time'].sum()/1000 + all_comm_time_df.loc[recv_backward_index, 'wait_time'].sum()/1000
        return bubble_time_cooldown

    def calculate_true_comm_and_overhead_wait_time(self, all_comm_time_df):
        comm_time_true = all_comm_time_df['comm_time'].sum()/1000
        overhead_wait_time_total = all_comm_time_df['wait_time'].sum()/1000
        return comm_time_true, overhead_wait_time_total

    # Todo: 'reduce_model_grads', 'step_', 'gather_model_params' 
    def calculate_optimizer_time(self, sorted_trace_df, bubble_time_final, stage_id, rank):
        optimizer_df = NameFilter(create_regex_for_prefix_match(['finalize_model_grads', 
                                                                 'step', 
                                                                 'logical_and_across_model_parallel_group',
                                                                 'reduce_max_stat_across_model_parallel_group']))(sorted_trace_df)
        #optimizer_df.to_csv(f'optimizer_df-stageid{stage_id}-rank{rank}.csv')
        optimizer_time = optimizer_df['kernel_span'].sum()/1000
        return optimizer_time - bubble_time_final
    
    def get_useful_trace_df(self, trace_df):
        return trace_df[~(trace_df[['send_prev', 'send_next', 'recv_prev', 'recv_next']] < 0).all(axis=1)]
    
    def get_p2p_trace_for_one_pair(self, rank_prev, rank_next):
        """
        Compute the point-to-point (P2P) communication times between two ranks.
        Args:
            rank_prev (int): The previous rank.
            rank_next (int): The next rank.

        Returns:
            pd.DataFrame: A DataFrame containing the bidirectional P2P communication times.
        """
        # Get the DataFrames for the given ranks
        trace_df_prev = self.full_dfs[rank_prev]
        trace_df_next = self.full_dfs[rank_next]
        useful_trace_df_prev = self.get_useful_trace_df(trace_df_prev)
        useful_trace_df_next = self.get_useful_trace_df(trace_df_next)
        # Compute forward P2P communication times
        df_p2p_forward = self._compute_p2p_forward(useful_trace_df_prev, useful_trace_df_next)
        # Link the dataframes of two adjacent pp i and pp i+1 by micro-bc id to get a forward-direction pair of send and recv
        # After linking, take min(send, recv) and save it to the comm_time column
        # Then copy the comm_time data to the new comm_time column of pp i
        # Similarly, copy the comm_time data to the new comm_time column of pp i+1
        self._update_comm_time(trace_df_prev, df_p2p_forward, 'index', 'index_on_prev')
        self._update_comm_time(trace_df_next, df_p2p_forward, 'index', 'index_on_next')
        # Compute backward P2P communication times
        df_p2p_backward = self._compute_p2p_backward(useful_trace_df_prev, useful_trace_df_next)
        self._update_comm_time(trace_df_prev, df_p2p_backward, 'index', 'index_on_prev')
        self._update_comm_time(trace_df_next, df_p2p_backward, 'index', 'index_on_next')
        # same as above. Calculate the pairwise send and recv funcs in the backward direction
        # and compute min(send, recv) into comm_time.
        # After completing the calculation of the actual comm_time for both forward and backward directions, 
        # wait_time = send/recv - comm_time 
        # Update the original DataFrames with the wait times
        self._update_wait_time(trace_df_prev)
        self._update_wait_time(trace_df_next)
        # Concatenate the forward and backward P2P DataFrames
        df_p2p_bidirection = pd.concat([df_p2p_forward, df_p2p_backward])

        return df_p2p_bidirection
    
    def establish_p2p_link_on_adjacent_ranks(self, pp_group_id=0):
        ranks = self.all_pipeline_parallel_group_ranks[pp_group_id]
        rank_pairs = self.get_p2p_ranks_pairs(ranks)
        self.traces_p2p_comm = {}
        for rank_prev, rank_next in rank_pairs:
            df_p2p_bidirection = self.get_p2p_trace_for_one_pair(rank_prev, rank_next)
            self.traces_p2p_comm[rank_prev] = df_p2p_bidirection
        
        self.trace_df_p2p_flow_events = self.combine_into_one_trace(self.traces_p2p_comm)
    
    def calculate_step_times(self, all_forward_steps_df, all_backward_steps_df):
        forward_step_avg_time = all_forward_steps_df['kernel_span'].mean()/1000
        backward_step_avg_time = all_backward_steps_df['kernel_span'].mean()/1000
        compute_time_total = all_forward_steps_df['kernel_span'].sum()/1000 + all_backward_steps_df['kernel_span'].sum()/1000
        return forward_step_avg_time, backward_step_avg_time, compute_time_total, all_forward_steps_df['kernel_span'].std()/1000, all_backward_steps_df['kernel_span'].std()/1000

    def get_bubble_time_ratio_theoretical(self, num_microbatch):
        return (self.pipeline_parallel_size - 1) / (self.pipeline_parallel_size - 1 + num_microbatch)
