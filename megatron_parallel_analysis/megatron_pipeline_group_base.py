# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import multiprocessing as mp
import os
import re
import time
from abc import ABC
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

from hta.common.trace import Trace
from hta.common.trace_filter import NameFilter
from megatron_parallel_analysis.utils.trace_filter_utils import create_regex_for_prefix_match
from hta.configs.config import logger
from hta.configs.default_values import DEFAULT_TRACE_DIR
from hta.common.trace_call_graph import CallGraph
from hta.common.trace_file import get_trace_files
from megatron_parallel_analysis.utils.parallel_state import RankGenerator
from megatron_parallel_analysis.utils.call_graph_utils import get_main_stack_on_rank
from megatron_parallel_analysis.utils.trace_parse_cache import (
    build_rank_parse_cache_metadata,
    get_rank_parse_cache_path,
    load_rank_parse_cache,
    write_rank_parse_cache,
)


def parallel_callgraph_create(rank_id, trace_file, bwd_annotation_str='backward_step'):
    logger.debug(f'rank id: {rank_id}, trace_file: {trace_file}')
    t = Trace(trace_files={rank_id: trace_file}, trace_dir="")
    t.load_traces()
    t.decode_symbol_ids(use_shorten_name=False)
    cg = CallGraph(t, bwd_annotation_str=bwd_annotation_str)
    _, main_stack = get_main_stack_on_rank(cg, rank_id)
    full_df = main_stack.full_df
    #Todo: ProfilerStep col kernel span has a wrong value
    profiler_step_mask = full_df['s_name'].str.contains(r'^ProfilerStep#.*')
    pretrain_mask = full_df['s_name'].str.match(
        pat=r"^megatron/training/training.py\(\d+\): pretrain$"
    )
    profiler_step_count = int(profiler_step_mask.sum())
    pretrain_kernel_span = full_df.loc[pretrain_mask, 'kernel_span'].to_numpy()
    pretrain_count = len(pretrain_kernel_span)
    result_df = full_df.loc[full_df["s_cat"].isin(["user_annotation", "gpu_memcpy"])].copy()
    del main_stack
    del cg
    del t
    del full_df
    if profiler_step_count > 0:
        if profiler_step_count == pretrain_count:
            result_profiler_step_mask = result_df['s_name'].str.contains(r'^ProfilerStep#.*')
            result_df.loc[result_profiler_step_mask, 'kernel_span'] = pretrain_kernel_span
        elif pretrain_count == 0:
            # Only user annotation, but set PROFILER_WITH_STACK=0
            logger.warning(
                'Skipping ProfilerStep kernel_span override because counts do not match: '
                'rank=%s, trace_file=%s, profiler_steps=%s',
                rank_id,
                trace_file,
                profiler_step_count,
            )
    else:
        raise RuntimeError('ProfilerStep annotations not found in trace, cannot override kernel_span values')
    return rank_id, result_df

# Todo: Does the pp group trace class need to know information about other tp, ep, dp groups?
class MegatronPipelineParallelGroupTraceBase(ABC):
    """
    MegatronPipelineParallelGroupTraceBase class for Megatron-LM with Pipeline Parallel group.
    Models are getting larger; load one pp group at a time to avoid OOM (out of memory) and slow parsing errors.
    """
    def __init__(
        self,
        trace_files: Optional[Dict[int, str]] = None,
        trace_dir: str = DEFAULT_TRACE_DIR,
        dp = -1,
        tp = -1,
        pp = -1,
        ep = -1,
        cp: int =1, order: str ="tp-cp-ep-dp-pp",
        micro_bs: int=0,
        bwd_annotation_str: str = 'backward_step',
        parse_cache_dir: Optional[str] = None,
        rebuild_parse_cache: bool = False,
        # pp_schedule: str = "1f1b",
        # vpp_size = -1,
    ) -> None:
        if trace_files is None:
            assert os.path.exists(trace_dir), f"Trace directory {trace_dir} does not exist!"
        self.trace_files = get_trace_files(trace_dir)
        self.trace_dir = trace_dir
        self.tensor_parallel_size = tp
        self.data_parallel_size = dp
        self.pipeline_parallel_size = pp
        self.expert_model_parallel_size = ep
        self.context_parallel_size = cp
        self.micro_bs = micro_bs
        self.bwd_annotation_str = bwd_annotation_str
        # self.pp_schedule = pp_schedule
        # self.vpp_size = vpp_size
        self.expert_decoder_rank_generator = RankGenerator(
            tp=tp,
            ep=ep,
            dp=dp,
            pp=pp,
            cp=cp,
            order=order,
            rank_offset=0,
        )
        self.all_data_parallel_group_ranks = self.expert_decoder_rank_generator.get_ranks('dp')
        self.all_tensor_parallel_group_ranks = self.expert_decoder_rank_generator.get_ranks('tp')
        self.all_pipeline_parallel_group_ranks = self.expert_decoder_rank_generator.get_ranks('pp')
        self.with_gpu_kernel = False
        self.traces_comm_only: Dict[int, pd.DataFrame] = {}
        #self.pp_group_trace: Dict[int, Trace] = {}
        self.full_dfs: Dict[int, pd.DataFrame] = {}
        self.is_parsed_per_pp_group: Dict[int, bool] = {}
        self.parse_cache_dir = parse_cache_dir
        self.rebuild_parse_cache = rebuild_parse_cache
        self.cache_only_mode = False
        if not self.trace_files and self.parse_cache_dir and not self.rebuild_parse_cache:
            cached_trace_files = {}
            for filename in os.listdir(self.parse_cache_dir):
                match = re.fullmatch(r'rank(\d+)\.parse_cache\.pkl', filename)
                if match:
                    cached_trace_files[int(match.group(1))] = os.path.join(
                        self.parse_cache_dir, filename
                    )
            if cached_trace_files:
                self.trace_files = cached_trace_files
                self.cache_only_mode = True
                logger.warning(
                    'No original trace files found under %s; using rank parse caches '
                    'without metadata validation for temporary debugging',
                    trace_dir,
                )

    def _parse_rank_tasks(self, tasks):
        num_procs = min(mp.cpu_count(), len(tasks))
        with mp.get_context("fork").Pool(num_procs) as pool:
            return pool.starmap(parallel_callgraph_create, tasks)

    def _load_cached_rank(self, rank_id: int, trace_file: str):
        if self.cache_only_mode:
            cache_path = get_rank_parse_cache_path(self.parse_cache_dir, rank_id)
            full_df = load_rank_parse_cache(cache_path, expected_metadata=None)
        else:
            metadata = build_rank_parse_cache_metadata(
                rank_id, trace_file, self.bwd_annotation_str
            )
            cache_path = get_rank_parse_cache_path(self.parse_cache_dir, rank_id)
            full_df = load_rank_parse_cache(cache_path, metadata)
        return None if full_df is None else full_df.copy(deep=True)

    def _write_cached_rank(
        self,
        rank_id: int,
        trace_file: str,
        full_df: pd.DataFrame,
    ) -> None:
        metadata = build_rank_parse_cache_metadata(
            rank_id, trace_file, self.bwd_annotation_str
        )
        cache_path = get_rank_parse_cache_path(self.parse_cache_dir, rank_id)
        write_rank_parse_cache(cache_path, metadata, full_df)

    # def get_ranks(self, pp_group_id: int = 0) -> List[int]:
    #     """返回指定 PP Group内的 rank 列表。"""
    #     return self.all_pipeline_parallel_group_ranks[pp_group_id]

    def parse_traces_per_pp_group(self, pp_group_id=0) -> None:
        if self.is_parsed_per_pp_group.get(pp_group_id, False):
            logger.warning("Traces are already parsed and loaded!")
            return

        ranks = self.all_pipeline_parallel_group_ranks[pp_group_id]
        cache_hits = []
        tasks = []
        for rank_id in ranks:
            trace_file = self.trace_files[rank_id]
            full_df = None
            if not self.rebuild_parse_cache:
                full_df = self._load_cached_rank(rank_id, trace_file)
            if full_df is None:
                if self.cache_only_mode:
                    raise RuntimeError(
                        f'Cache-only debugging mode requires a valid cache for rank {rank_id}: '
                        f'{get_rank_parse_cache_path(self.parse_cache_dir, rank_id)}'
                    )
                tasks.append((rank_id, trace_file, self.bwd_annotation_str))
            else:
                self.full_dfs[rank_id] = full_df
                cache_hits.append(rank_id)

        rebuilt_ranks = []
        if tasks:
            results = self._parse_rank_tasks(tasks)
            for rank_id, main_stack_df in results:
                logger.debug(f"rank id: {rank_id}")
                trace_file = self.trace_files[rank_id]
                self._write_cached_rank(rank_id, trace_file, main_stack_df)
                self.full_dfs[rank_id] = main_stack_df
                rebuilt_ranks.append(rank_id)

        self.is_parsed_per_pp_group[pp_group_id] = True
        logger.info(
            f'PP group {pp_group_id} trace parse cache: '
            f'hits={cache_hits}, rebuilt={rebuilt_ranks}'
        )

    def etl_traces_per_pp_group(self, redirect_trace_dir, filter_out_funcs, pp_group_id=0) -> None:
        if self.is_parsed_per_pp_group.get(pp_group_id, False):
            logger.warning("Traces are already parsed and loaded!")
            return
        tasks = []
        t0 = time.perf_counter()
        for rank in self.all_pipeline_parallel_group_ranks[pp_group_id]:
            trace_file = self.trace_files[rank]
            logger.debug(f'rank {rank} trace file:{trace_file}')
            filename = os.path.basename(trace_file)
            redirect_new_trace_path = os.path.join(redirect_trace_dir, filename)
            tasks.append((trace_file, redirect_new_trace_path))
        num_procs = min(mp.cpu_count(), len(self.all_pipeline_parallel_group_ranks[pp_group_id]))
        with mp.get_context("fork").Pool(num_procs) as pool:
            results = pool.starmap(filter_out_funcs, tasks)
            pool.close()
            pool.join()
        t1 = time.perf_counter()
        logger.debug(f"calculating critical path took {t1 - t0:2f} seconds")
        self.is_parsed_per_pp_group[pp_group_id] = True

    # @abstractmethod
    def set_self_microbatch_id(self, trace_df: pd.DataFrame) -> None:
        """根据当前 PP 算法为 trace 打上 micro-batch id(forward/backward 等）。子类必须实现。"""
        pass

    # @abstractmethod
    def set_recv_send_microbatch_id(self, trace_df: pd.DataFrame) -> None:
        """根据 send/recv 与 micro-batch 关系设置 recv_prev/recv_next/send_prev/send_next。子类必须实现。"""
        pass

    # Use: func annotations
    # @abstractmethod
    def process_pipeline_start(self, df):
        #if is_first_stage(rank, self.tensor_parallel_size, self.pipeline_parallel_size, self.data_parallel_size):
        pass

    # @abstractmethod
    def process_pipeline_end(self, df):
        #if is_last_stage(rank, self.tensor_parallel_size, self.pipeline_parallel_size, self.data_parallel_size):
        pass

    # @abstractmethod
    def set_micro_batch_id(self, pp_group_id: int = 0) -> None:
        """为指定 PP 组内各 rank 的 trace 设置 micro-batch id，由子类的 set_self_microbatch_id/set_recv_send_microbatch_id 实现具体算法。"""
        pass
    
    # @abstractmethod
    def filter_comm_only_traces(self, pp_group_id=0):
        pass
    
    # @abstractmethod
    def get_p2p_ranks_pairs(self, ranks):
        # different pp schedule algorithms, the pairs are different
        pass

    # Todo: 
    def calculate_time_per_iteration(self, rank):
        trace_df = self.full_dfs[rank]
        return trace_df[trace_df['s_name'].str.contains(r'^ProfilerStep#.*')]['kernel_span'].values[0]/1000
    


    def _compute_p2p_forward(self, df_prev, df_next):
        """
        Compute the forward P2P communication times between two DataFrames.

        Args:
            df_prev (pd.DataFrame): The previous rank DataFrame.
            df_next (pd.DataFrame): The next rank DataFrame.

        Returns:
            pd.DataFrame: A DataFrame containing the forward P2P communication times.
        """
        
        df_p2p_forward = pd.merge(df_prev, df_next, left_on='send_next', right_on='recv_prev', how='inner', suffixes=('_on_prev', '_on_next'))
        df_p2p_forward = df_p2p_forward[df_p2p_forward['send_next_on_prev'] >= 0]
        df_p2p_forward['p2p_forward'] = True
        df_p2p_forward['comm_time'] = np.minimum(df_p2p_forward['kernel_span_on_prev'], df_p2p_forward['kernel_span_on_next'])
        return df_p2p_forward

    def _compute_p2p_backward(self, df_prev, df_next):
        """
        Compute the backward P2P communication times between two DataFrames.

        Args:
            df_prev (pd.DataFrame): The previous rank DataFrame.
            df_next (pd.DataFrame): The next rank DataFrame.

        Returns:
            pd.DataFrame: A DataFrame containing the backward P2P communication times.
        """
        df_p2p_backward = pd.merge(df_prev, df_next, left_on='recv_next', right_on='send_prev', how='inner', suffixes=('_on_prev', '_on_next'))
        df_p2p_backward = df_p2p_backward[df_p2p_backward['recv_next_on_prev'] >= 0]
        df_p2p_backward['p2p_backward'] = True
        df_p2p_backward['comm_time'] = np.minimum(df_p2p_backward['kernel_span_on_prev'], df_p2p_backward['kernel_span_on_next'])
        return df_p2p_backward

    def _update_comm_time(self, original_df, p2p_df, merge_col_original, merge_col_p2p):
        """
        Update the communication times in the original DataFrame based on the P2P DataFrame.

        Args:
            original_df (pd.DataFrame): The original DataFrame to update.
            p2p_df (pd.DataFrame): The P2P DataFrame containing the communication times.
            merge_col_original (str): The column name in the original DataFrame to merge on.
            merge_col_p2p (str): The column name in the P2P DataFrame to merge on.
        """
        # Ensure 'comm_time' column exists in the original DataFrame
        if 'comm_time' not in original_df.columns:
            original_df['comm_time'] = 0.0

        original_df.loc[original_df[merge_col_original].isin(p2p_df[merge_col_p2p]), 'comm_time'] = p2p_df['comm_time'].values

    def _update_wait_time(self, df):
        """
        Update the wait times in the DataFrame based on the communication times.

        Args:
            df (pd.DataFrame): The DataFrame to update.
        """
        df['wait_time'] = df['kernel_span'] - df['comm_time']

    def get_num_microbatches(self):
        return self.micro_bs
    
    def save_traces_with_p2p_comm(self, save_path, traces=None, trace_df_p2p_flow_events=None, meta_data=None):
        if traces is None:
            traces = self.traces
        #if trace_df_p2p_flow_events is None:
        #    trace_df_p2p_flow_events = self.trace_df_p2p_flow_events
        #if meta_data is None:
        #    meta_data = self.meta_data
        trace_df_all_ranks = self.combine_into_one_trace(traces)
        self.save_trace_df_to_file(trace_df_all_ranks, save_path) # Todo: enhance flow event:, trace_df_p2p_flow_events)
    
    def save_trace_df_to_file(self, df: pd.DataFrame, output_file: str, trace_df_p2p_comm_flow: pd.DataFrame=None, meta_data: dict=None, pp_schedule: str='1f1b'):
        columns_to_keep = ['name', 'cat', 'pid', 'tid', 'ts', 'dur', 'rank']
        columns_to_drop = ['s_name', 's_cat']
        
        new_df = df[columns_to_keep].copy()
        new_df['ts'] = df['first_kernel_start'].where(df['first_kernel_start'] > 0, df['ts'])
        new_df['dur'] = df['kernel_span'].where(df['kernel_span'] > 0, df['dur'])
        new_df['name'] = df['s_name']
        new_df['cat'] = df['s_cat']
        new_df['ph'] = 'X'
        # Todo: in interleaved PP, send_fwd_recv_fwd and send_bwd_recv_bwd execute asyn and in parallel with fwd_step or bwd_step
        # so for displaying in perfetto, it muse set them with different tids.
        new_df['args'] = df.apply(lambda row: {col: row[col] for col in row.index if col not in columns_to_keep + columns_to_drop}, axis=1)
        
        trace_data = meta_data.copy() if meta_data is not None else {}
        trace_events = new_df.to_dict('records')
        #flow_events = convert_to_flow_events(trace_df_p2p_comm_flow)
        metadata_events = MegatronPipelineParallelGroupTraceBase.generate_metadata_events([tuple(x) for x in new_df[['rank', 'pid']].drop_duplicates().to_records(index=False)])
        trace_data["traceEvents"] = trace_events + metadata_events
        
        with open(output_file, 'w') as f:
            json.dump(trace_data, f, indent=4)
        
    @staticmethod
    def generate_metadata_events(rank_pid_pairs):
        metadata_events = []
        rank_pid_pairs = sorted(rank_pid_pairs)
        for i, (rank, pid) in enumerate(rank_pid_pairs):
            metadata_events.append(
                {
                    'name': 'process_sort_index',
                    'ph': 'M',
                    'pid': pid,
                    'args':{
                        'sort_index': i
                    }
                }
            )
            metadata_events.append(
                {
                    'name': 'process_name',
                    'ph': 'M',
                    'pid': pid,
                    'args':{
                        'name': f'rank {rank}'
                    }
                }
            )
        return metadata_events

    @staticmethod
    def combine_into_one_trace(traces_dict: dict):
        all_trace_dfs = []
        # Todo: debug performance issue, check if the memcpy names are correct
        memcpy_names = [
            'Memcpy1 DtoH (Device -> Pinned)',
            'Memcpy1 HtoD (Pinned -> Device)',
        ]
        for rank, trace_df in traces_dict.items():
            trace_df['rank'] = rank
            forward_step_df = trace_df.loc[trace_df['s_name'] == 'forward_step']
            if not forward_step_df.empty:
                forward_step_pid = forward_step_df['pid'].iloc[0]
                trace_df.loc[trace_df['s_name'].isin(memcpy_names), 'pid'] = forward_step_pid
            all_trace_dfs.append(trace_df)
        trace_df = pd.concat(all_trace_dfs, ignore_index=True)
        return trace_df

    @staticmethod
    def display_traces_info(traces):
        first_trace_df = next(iter(traces.values()))
        logger.info(f'total {len(traces)} traces, and each trace has {len(first_trace_df)} items')
        logger.info(first_trace_df['s_cat'].value_counts())

    def calculate_finalize_model_grads_step_time(self, sorted_trace_df):
        pattern = r'^finalize_model_grads$'
        finalize_model_grads_step_time = sorted_trace_df[sorted_trace_df['s_name'].str.contains(pattern)]['kernel_span'].values[0]
        return finalize_model_grads_step_time/1000

    def calculate_should_run_forward_backward_time(self, sorted_trace_df):
        pattern = r'^should_run_forward_backward$'
        should_run_forward_backward = sorted_trace_df[sorted_trace_df['s_name'].str.contains(pattern)]
        if should_run_forward_backward.empty:
            return 0.0
        should_run_forward_backward_time = should_run_forward_backward['kernel_span'].values[0]
        return should_run_forward_backward_time/1000

    # Todo: func name
    #       sorted_trace_df[sorted_trace_df['full_name'].str.contains(pattern, regex=True)]['dur'].values[0]   why use the first value
    def calculate_optimizer_step_time_and_bubble(self, sorted_trace_df):
        # Use regex to match 'full_name' with the pattern '?ProfilerStep?/step?'
        # Todo: func name
        # pattern = r'.*ProfilerStep.*/step.*'
        pattern = r'^step$'
        optimizer_step = sorted_trace_df[sorted_trace_df['s_name'].str.contains(pattern)].sort_values('ts').iloc[0]
        optimizer_step_time = optimizer_step['kernel_span']/1000
        optimizer_step_start_ts = optimizer_step['first_kernel_start']
        return optimizer_step_time, optimizer_step_start_ts
    
    def calculate_logical_and_across_model_parallel_group_time(self, sorted_trace_df):
        pattern = r'^logical_and_across_model_parallel_group$'
        logical_and_across_model_parallel_group_time = sorted_trace_df[sorted_trace_df['s_name'].str.contains(pattern)]['kernel_span'].values[0]
        return logical_and_across_model_parallel_group_time/1000

    def generate_report(self, pp_group_id, save_path):
        output_df = None
        detail_output_df = None
        #first_stage_optimizer_step = None
        stage_0_optimizer_step_start_ts = None
        for stage_id, rank in enumerate(sorted(self.traces_comm_only.keys())):
            sorted_trace_df = self.preprocess_trace_df(rank)
            time_per_iteration = self.calculate_time_per_iteration(rank)
            all_forward_steps_df = NameFilter(create_regex_for_prefix_match(['forward_step']))(sorted_trace_df)
            all_backward_steps_df = NameFilter(create_regex_for_prefix_match(['backward_step']))(sorted_trace_df)
            forward_step_avg_time, backward_step_avg_time, compute_time_total, fwd_std, bwd_std = self.calculate_step_times(all_forward_steps_df, all_backward_steps_df)
            all_comm_time_df = self.get_all_comm_df(sorted_trace_df, rank)
            optimizer_time, optimizer_step_start_ts = self.calculate_optimizer_step_time_and_bubble(sorted_trace_df)
            if stage_id == 0:
                stage_0_optimizer_step_start_ts = optimizer_step_start_ts
            theoretical_bubble_time_warmup = self.calculate_theoretical_bubble_time_warmup(all_comm_time_df, stage_id)
            bubble_time_warmup = self.calculate_bubble_time_warmup(all_comm_time_df, stage_id) - theoretical_bubble_time_warmup
            theoretical_bubble_time_steady = self.calculate_theoretical_bubble_time_steady(all_comm_time_df, stage_id)
            bubble_time_steady = self.calculate_bubble_time_steady(all_comm_time_df, stage_id) - theoretical_bubble_time_steady
            theoretical_bubble_time_cooldown = self.calculate_theoretical_bubble_time_cooldown(all_comm_time_df, all_backward_steps_df, stage_id, stage_0_optimizer_step_start_ts)
            bubble_time_cooldown = self.calculate_bubble_time_cooldown(all_comm_time_df, stage_id)
            finalize_model_grads_step_time = self.calculate_finalize_model_grads_step_time(sorted_trace_df)
            logical_and_across_model_parallel_group_time = self.calculate_logical_and_across_model_parallel_group_time(sorted_trace_df)
            should_run_forward_backward_time = self.calculate_should_run_forward_backward_time(sorted_trace_df)
            #bubble_time_total = bubble_time_warmup + bubble_time_steady + bubble_time_cooldown # + bubble_time_final
            comm_time_true  = self.calculate_true_comm(all_comm_time_df)
            overhead_wait_time_total = theoretical_bubble_time_warmup + bubble_time_warmup + theoretical_bubble_time_steady + bubble_time_steady + theoretical_bubble_time_cooldown + bubble_time_cooldown
            comm_time_total =  comm_time_true + overhead_wait_time_total

            num_microbatch = self.get_num_microbatches()

            info_per_rank = {
                'Global rank in a pp group, Rank_(i) + pp_size = Rank_(i+1)': rank,
                'Elapsed time per iteration': time_per_iteration,
                'Micro-Batch count': num_microbatch,
                'Sum(Micro-Batch_forward_time + Micro-Batch_backward_time)': compute_time_total,
                'PP SendRecv time': comm_time_total,
                'Finalize_model_grads_step_time': finalize_model_grads_step_time,
                'Should_run_forward_backward_time': should_run_forward_backward_time,
                'Logical_and_across_model_parallel_group_time': logical_and_across_model_parallel_group_time,
                'Optimizer_time': optimizer_time,
                'Compute time total / Elapsed time per iteration': compute_time_total / time_per_iteration,
                'PP SendRecv time / Elapsed time per iteration': comm_time_total / time_per_iteration,
                'Finalize_model_grads_step_time / Elapsed time per iteration': finalize_model_grads_step_time / time_per_iteration,
                'Should_run_forward_backward_time / Elapsed time per iteration': should_run_forward_backward_time / time_per_iteration,
                'Logical_and_across_model_parallel_group_time / Elapsed time per iteration': logical_and_across_model_parallel_group_time / time_per_iteration,
                'Optimizer_time / Elapsed time per iteration': optimizer_time / time_per_iteration,
            }
            detail_info_per_rank = {
                'Global rank in a pp group, Rank_(i) + pp_size = Rank_(i+1)': rank,
                'Elapsed time per iteration': time_per_iteration,
                'Micro-Batch count': num_microbatch,
                'SUM(micro_batch_forward_time) / Micro-Batch count': forward_step_avg_time,
                'STD(micro_batch_forward_time) ': fwd_std,
                'SUM(micro_batch_backward_time) / Micro-Batch count': backward_step_avg_time,
                'STD(micro_batch_backward_time) ': bwd_std,
                'compute_time_total': compute_time_total,
                'PP SendRecv time': comm_time_total,
                'Actual transfer time of send-recv': comm_time_true,
                'Bubble time of send-recv': overhead_wait_time_total,
                'Theoretical bubble time warmup': theoretical_bubble_time_warmup,
                'Theoretical bubble time steady': theoretical_bubble_time_steady,
                'Theoretical bubble time cooldown': theoretical_bubble_time_cooldown,
                'Non-balanced bubble time warmup': bubble_time_warmup,
                'Non-balanced bubble time steady': bubble_time_steady,
                'Non-balanced bubble time cooldown': bubble_time_cooldown,
                'Theoretical bubble time / elapsed time per iteration': (theoretical_bubble_time_warmup + theoretical_bubble_time_steady + theoretical_bubble_time_cooldown) / time_per_iteration,
                'Theoretical bubble time / SUM(micro_batch_forward_time + micro_batch_backward_time)': (theoretical_bubble_time_warmup + theoretical_bubble_time_steady + theoretical_bubble_time_cooldown) / (compute_time_total + comm_time_total),
                'bubble_ratio in paper': self.get_bubble_time_ratio_theoretical(num_microbatch),
            }
            #info_per_rank = self._generate_info_per_rank(args)

            if output_df is None:
                output_df = pd.DataFrame([info_per_rank])
                detail_output_df = pd.DataFrame([detail_info_per_rank])
            else:
                output_df.loc[len(output_df)] = info_per_rank
                detail_output_df.loc[len(detail_output_df)] = detail_info_per_rank

        if save_path is not None:
            output_df.to_csv(save_path, header=True, index=False, float_format='%.3f')
            save_root, save_ext = os.path.splitext(save_path)
            detail_output_df.to_csv(
                f'{save_root}-detail{save_ext}',
                header=True,
                index=False,
                float_format='%.3f',
            )
        #print(output_df)
        return output_df

def convert_to_flow_events(trace_df_p2p_comm_flow: pd.DataFrame):
    if trace_df_p2p_comm_flow is None: return []
    
    df_p2p_forward = trace_df_p2p_comm_flow[trace_df_p2p_comm_flow['p2p_forward'] == True]
    df_p2p_backward = trace_df_p2p_comm_flow[trace_df_p2p_comm_flow['p2p_backward'] == True]

    send_forward_pd = pd.DataFrame({
        'cat': 'p2p_forward',
        'name': 'p2p_forward',
        'ph': 's',
        'pid': df_p2p_forward['pid_on_prev'],
        'tid': df_p2p_forward['tid_on_prev'],
        'ts': np.maximum(df_p2p_forward['ts_on_prev'], df_p2p_forward['ts_on_next']),
        'id': df_p2p_forward.index,
        'args': [
            {'micro_batch_id': micro_batch_id}
            for micro_batch_id in df_p2p_forward['micro_batch_id_forward_on_prev']
        ]
    })
    
    send_backward_pd = pd.DataFrame({
        'cat': 'p2p_backward',
        'name': 'p2p_backward',
        'ph': 's',
        'pid': df_p2p_backward['pid_on_next'],
        'tid': df_p2p_backward['tid_on_next'],
        'ts': np.maximum(df_p2p_backward['ts_on_prev'], df_p2p_backward['ts_on_next']),
        'id': df_p2p_backward.index,
        'args': [
            {'micro_batch_id': micro_batch_id}
            for micro_batch_id in df_p2p_backward['micro_batch_id_backward_on_prev']
        ]
    })

    recv_forward_pd = pd.DataFrame({
        'cat': 'p2p_forward',
        'name': 'p2p_forward',
        'ph': 'f',
        'pid': df_p2p_forward['pid_on_next'],
        'tid': df_p2p_forward['tid_on_next'],
        'ts': df_p2p_forward['ts_on_next'] + df_p2p_forward['dur_on_next'],
        'id': df_p2p_forward.index,
        'bp': 'e',
        'args': [
            {'micro_batch_id': micro_batch_id}
            for micro_batch_id in df_p2p_forward['micro_batch_id_forward_on_prev']
        ]
    })

    recv_backward_pd = pd.DataFrame({
        'cat': 'p2p_backward',
        'name': 'p2p_backward',
        'ph': 'f',
        'pid': df_p2p_backward['pid_on_prev'],
        'tid': df_p2p_backward['tid_on_prev'],
        'ts': df_p2p_backward['ts_on_prev'] + df_p2p_backward['dur_on_prev'],
        'id': df_p2p_backward.index,
        'bp': 'e',
        'args': [
            {'micro_batch_id': micro_batch_id}
            for micro_batch_id in df_p2p_backward['micro_batch_id_backward_on_prev']
        ]
    })

    total_dicts = [
        item
        for pd in [send_forward_pd, send_backward_pd, recv_forward_pd, recv_backward_pd]
        for item in pd.to_dict('records')
    ]

    return total_dicts