# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import math
from typing import Dict, Optional, Tuple

import pandas as pd
from hta.common.trace_filter import NameFilter
from hta.configs.config import logger
from hta.configs.default_values import DEFAULT_TRACE_DIR
from megatron_parallel_analysis.megatron_pipeline_group_base import MegatronPipelineParallelGroupTraceBase
from megatron_parallel_analysis.utils.parallel_state import get_pp_rank_microbatches
from megatron_parallel_analysis.utils.pipeline_parallel_utils import convert_schedule_table_to_order, get_schedule_table
from megatron_parallel_analysis.utils.trace_filter_utils import create_regex_for_full_match

class MegatronPipelineParallel1F1BInterleavedGroupTrace(MegatronPipelineParallelGroupTraceBase):
    """1F1B interleaved (One-Forward-One-Backward) 调度下的 PP group trace 分析。"""
    def __init__(self,
        trace_files: Optional[Dict[int, str]] = None,
        trace_dir: str = DEFAULT_TRACE_DIR,
        dp = -1,
        tp = -1,
        pp = -1,
        ep = -1,
        cp: int =1, 
        order: str ="tp-cp-ep-dp-pp",
        #pp_schedule: str = "1f1b-interleaved",
        vpp_size = -1,
        micro_bs = 0,
        parse_cache_dir: Optional[str] = None,
        rebuild_parse_cache: bool = False,
        ) -> None:
        super().__init__(trace_files, trace_dir, dp, tp, pp, ep, cp, order, micro_bs, parse_cache_dir=parse_cache_dir, rebuild_parse_cache=rebuild_parse_cache)
        #self.pp_schedule = pp_schedule
        self.vpp_size = vpp_size

    def preprocess_trace_df(self, rank):
        #trace_df = self.full_dfs[rank]
        trace_df = self.traces_comm_only[rank]
        sorted_trace_df = trace_df.sort_values(by=['ts', 'kernel_span'], ascending=[True, False])
        # Use regex to find the first occurrence of any 'recv_forward*' event
        # Todo: communication on mooncake
        # First 'recv_forward' event is the start of the first forward pass, and we want to keep all events after that for better analysis of PP scheduling. 
        # recv_forward_index = sorted_trace_df[sorted_trace_df['s_name'].str.contains(r'^recv_forward$')].index
        #Todo: uncomment the following line after fixing the recv_forward event name in mooncake
        #first_recv_forward_index = recv_forward_index[0]
        # Keep only the rows from the first 'recv_forward*' event onwards
        #sorted_trace_df = sorted_trace_df.loc[first_recv_forward_index:]

        #sorted_trace_df.to_csv(f'preprocess_trace_df-after-{rank}.csv')
        return sorted_trace_df


    def set_vpp_stage_id(
        self,
        trace_df: pd.DataFrame,
        stage_id: int,
        rank: Optional[int] = None,
    ) -> None:
        trace_df.sort_values(by=['ts', 'dur'], ascending=[True, False], inplace=True)
        trace_df['vpp_stage_id'] = 0
        trace_df['micro_batch_id'] = -1
        num_microbatches = self.get_num_microbatches()
        num_warmup_microbatches = get_pp_rank_microbatches(num_microbatches, self.pipeline_parallel_size, stage_id, self.vpp_size, self.pipeline_parallel_size)
        schedule_table = get_schedule_table(num_microbatches, self.vpp_size, self.pipeline_parallel_size)
        micro_batch_order, _ = zip(*schedule_table)
        fwd_order, bwd_order, _ = convert_schedule_table_to_order(num_warmup_microbatches, self.vpp_size, schedule_table)
        forward_mask = trace_df['s_name'].eq('forward_step')
        backward_mask = trace_df['s_name'].eq('backward_step')
        expected_count = len(micro_batch_order)
        forward_count = int(forward_mask.sum())
        backward_count = int(backward_mask.sum())
        if forward_count != expected_count or backward_count != expected_count:
            raise ValueError(
                'Unexpected interleaved step count for '
                f'rank {rank}, PP stage {stage_id}: expected {expected_count} per '
                f'direction, got forward={forward_count}, backward={backward_count}'
            )
        trace_df.loc[forward_mask, 'micro_batch_id'] = micro_batch_order
        trace_df.loc[backward_mask, 'micro_batch_id'] = micro_batch_order
        trace_df.loc[forward_mask, 'vpp_stage_id'] = fwd_order
        trace_df.loc[backward_mask, 'vpp_stage_id'] = bwd_order

    def set_micro_batch_id(self, pp_group_id: int = 0) -> None:
        """为指定 PP 组内各 rank 的 trace 设置 micro-batch id，由子类的 set_self_microbatch_id/set_recv_send_microbatch_id 实现具体算法。"""
        ranks = self.all_pipeline_parallel_group_ranks[pp_group_id]
        logger.info(f'[1F1B interleaved] In set micro batch id: ranks: {ranks}')
        for stage_id, rank in enumerate(ranks):
            self.set_vpp_stage_id(self.traces_comm_only[rank], stage_id, rank)

    @staticmethod
    def _python_number(value):
        return value.item() if hasattr(value, 'item') else value

    def extract_rank_gpu_timeline(
        self,
        rank: int,
    ) -> Dict[int, Tuple[Tuple[float, float], ...]]:
        """Extract VPP GPU intervals per micro-batch.

        The tuple order is forward VPP stages in ascending order, followed by
        backward VPP stages in ascending order. Each interval is
        ``(first_kernel_start, kernel_span)`` in original microsecond units.
        """
        trace_df = self.traces_comm_only[rank]
        required_columns = {
            's_name',
            'micro_batch_id',
            'vpp_stage_id',
            'first_kernel_start',
            'kernel_span',
        }
        missing_columns = required_columns.difference(trace_df.columns)
        if missing_columns:
            raise ValueError(
                f'Cannot extract GPU timeline for rank {rank}; missing columns: '
                f'{sorted(missing_columns)}'
            )

        step_df = trace_df[trace_df['s_name'].isin(('forward_step', 'backward_step'))]
        intervals = {}
        duplicate_keys = []
        invalid_rows = []
        for _, row in step_df.iterrows():
            direction = 'forward' if row['s_name'] == 'forward_step' else 'backward'
            try:
                micro_batch_id = int(row['micro_batch_id'])
                signed_vpp_id = int(row['vpp_stage_id'])
                start = float(row['first_kernel_start'])
                duration = float(row['kernel_span'])
            except (TypeError, ValueError, OverflowError):
                invalid_rows.append((direction, row.get('micro_batch_id'), row.get('vpp_stage_id')))
                continue

            valid_sign = signed_vpp_id > 0 if direction == 'forward' else signed_vpp_id < 0
            logical_vpp_id = abs(signed_vpp_id) - 1
            has_kernels = 'num_kernels' not in row.index or row['num_kernels'] > 0
            if (
                micro_batch_id < 0
                or micro_batch_id >= self.micro_bs
                or not valid_sign
                or logical_vpp_id < 0
                or logical_vpp_id >= self.vpp_size
                or not math.isfinite(start)
                or not math.isfinite(duration)
                or start <= 0
                or duration < 0
                or not has_kernels
            ):
                invalid_rows.append((direction, micro_batch_id, signed_vpp_id))
                continue

            key = (direction, micro_batch_id, logical_vpp_id)
            if key in intervals:
                duplicate_keys.append(key)
                continue
            intervals[key] = (
                self._python_number(row['first_kernel_start']),
                self._python_number(row['kernel_span']),
            )

        expected_keys = {
            (direction, micro_batch_id, vpp_id)
            for direction in ('forward', 'backward')
            for micro_batch_id in range(self.micro_bs)
            for vpp_id in range(self.vpp_size)
        }
        missing_keys = sorted(expected_keys.difference(intervals))
        extra_keys = sorted(set(intervals).difference(expected_keys))
        if invalid_rows or duplicate_keys or missing_keys or extra_keys:
            raise ValueError(
                f'Invalid interleaved GPU timeline for rank {rank}: '
                f'invalid={invalid_rows}, duplicates={sorted(set(duplicate_keys))}, '
                f'missing={missing_keys}, extra={extra_keys}'
            )

        return {
            micro_batch_id: tuple(
                intervals[(direction, micro_batch_id, vpp_id)]
                for direction in ('forward', 'backward')
                for vpp_id in range(self.vpp_size)
            )
            for micro_batch_id in range(self.micro_bs)
        }

    def get_p2p_ranks_pairs(self, ranks):
        if len(ranks) < 2: return []
        # Todo: double check
        ranks_sorted = sorted(ranks)
        p2p_devices_pairs = []
        for i in range(len(ranks) - 1):
            p2p_devices_pairs.append([ranks[i], ranks[i+1]])
        return p2p_devices_pairs 
    
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
            'send_forward_recv_forward',
            'send_backward_recv_backward',
            'finalize_model_grads',
            'step',
            'logical_and_across_model_parallel_group',
            'reduce_max_stat_across_model_parallel_group',
            'should_run_forward_backward',
            # 'mccl:reduce_scatter_tensor_coalesced',
            # For debug
            'mccl:all_reduce',
            'mccl:all_to_all',
            'Memcpy1 DtoH (Device -> Pinned)',
            'Memcpy1 HtoD (Pinned -> Device)'
        ]
        filter_comm = NameFilter(create_regex_for_full_match(comm_names_list))
        return filter_comm(trace_df)

    def filter_comm_only_traces(self, pp_group_id=0):
        for rank in self.all_pipeline_parallel_group_ranks[pp_group_id]:
            self.traces_comm_only[rank] = self.keep_comm_span_only(self.full_dfs[rank])

    def establish_p2p_link_on_adjacent_ranks(self, pp_group_id=0):
        logger.info(f'[1F1B interleaved] Todo: establish p2p link on adjacent ranks for pp_group_id {pp_group_id}')
    
    def get_all_comm_df(self, sorted_trace_df, rank=None):
        all_comm_df = sorted_trace_df[
            (sorted_trace_df['num_kernels'] > 0)
            & (sorted_trace_df['s_cat'] == 'user_annotation')
        ]
        all_comm_df = all_comm_df.sort_values(by="first_kernel_start")
        """
        Calculate idle intervals for communication events.
        Using the start time and duration of communication events to calculate the end time of current event.
        Then, using the start time of the next event and the end time of current event to calculate the idle interval between two consecutive communication events.
        The interval will be set to the next event's idle_interval column. 
        For example, send_forward_recv_forward event and forward_step event,
        calculate the end ts of send_forward_recv_forward event 
        end_ts = ts(send_forward_recv_forward) + dur(send_forward_recv_forward)
        idle interval = ts(forward_step) - end_ts(send_forward_recv_forward)
        idle interval will be set to forward_step's idle_interval column.
        The idle interval of first event in all_comm_df should be equal to the first recv_forward event's cpu duration, since there is called wait for communication event before the first forward_step event, and the GPU is idle during that time.
        """
        all_comm_df["end_ts"] = all_comm_df.first_kernel_start + all_comm_df.kernel_span
        all_comm_df["prev_end_ts"] = all_comm_df.end_ts.shift(1)
        all_comm_df["idle_interval"] = all_comm_df["first_kernel_start"] - all_comm_df["prev_end_ts"]
        all_comm_df = all_comm_df[all_comm_df['s_name'].str.match(pat=r'^(forward|backward)_step$')]
        first_step_index = all_comm_df.index[0]
        first_recv_duration = self.full_dfs[rank].loc[
            self.full_dfs[rank]['s_name'].str.match(pat=r'^recv_forward$'), 'dur'
        ].iloc[0]
        all_comm_df.loc[first_step_index, 'idle_interval'] = first_recv_duration
        return all_comm_df

    def calculate_step_times(self, all_forward_steps_df, all_backward_steps_df):
        num_model_chunks = self.vpp_size
        forward_step_avg_time = []
        forward_step_std_time = []
        backward_step_avg_time = []
        backward_step_std_time = []
        
        for vpp_stage_id in range(num_model_chunks):
            fwd_vpp_stage_id = vpp_stage_id + 1
            bwd_vpp_stage_id = -fwd_vpp_stage_id
            forward_step_avg_time.append(
                round(float(all_forward_steps_df.loc[all_forward_steps_df['vpp_stage_id'] == fwd_vpp_stage_id, 'kernel_span'].mean())/1000, 2)
            )
            forward_step_std_time.append(
                round(float(all_forward_steps_df.loc[all_forward_steps_df['vpp_stage_id'] == fwd_vpp_stage_id, 'kernel_span'].std())/1000, 2)
            )
            backward_step_avg_time.append(
                round(float(all_backward_steps_df.loc[all_backward_steps_df['vpp_stage_id'] == bwd_vpp_stage_id, 'kernel_span'].mean())/1000, 2)
            )
            backward_step_std_time.append(
                round(float(all_backward_steps_df.loc[all_backward_steps_df['vpp_stage_id'] == bwd_vpp_stage_id, 'kernel_span'].std())/1000, 2)
            )

        compute_time_total = (all_forward_steps_df['kernel_span'].sum()/1000 + all_backward_steps_df['kernel_span'].sum()/1000)
        return forward_step_avg_time, backward_step_avg_time, compute_time_total, forward_step_std_time, backward_step_std_time

    def calculate_comm_time_total(self, all_comm_time_df):
        return all_comm_time_df['idle_interval'].sum()/1000

    def calculate_theoretical_bubble_time_warmup(self, all_comm_time_df, stage_id):
        if stage_id == 0:
            return 0.0
        else:
            theoretical_bubble_head_index = all_comm_time_df[all_comm_time_df['s_name'].str.match(pat=r'^forward_step$')].index[0]
            return all_comm_time_df.loc[theoretical_bubble_head_index, 'idle_interval'].sum()/1000

    def calculate_bubble_time_warmup(self, all_comm_time_df, stage_id):
        num_warmup_microbatches = get_pp_rank_microbatches(self.get_num_microbatches(), self.pipeline_parallel_size, stage_id, self.vpp_size, self.pipeline_parallel_size)
        fwd_step_in_head = all_comm_time_df[all_comm_time_df['s_name'].str.match(pat=r'^forward_step$')].index[:num_warmup_microbatches+1]
        return all_comm_time_df.loc[fwd_step_in_head, 'idle_interval'].sum()/1000
    
    def calculate_theoretical_bubble_time_steady(self, all_comm_time_df, stage_id):
        if stage_id == self.pipeline_parallel_size-1:
            return 0.0
        else:
            bwd_index = all_comm_time_df[all_comm_time_df['s_name'].str.match(pat=r'^backward_step$')].index
            if len(bwd_index) > 0:
                return all_comm_time_df.loc[bwd_index[0], 'idle_interval']/1000
            else:
                return 0.0

    def calculate_bubble_time_steady(self, all_comm_time_df, stage_id):
        num_warmup_microbatches = get_pp_rank_microbatches(self.get_num_microbatches(), self.pipeline_parallel_size, stage_id, self.vpp_size, self.pipeline_parallel_size)
        fwd_step = all_comm_time_df[all_comm_time_df['s_name'].str.match(pat=r'^forward_step$')]
        fwd_step_in_steady = fwd_step.index[num_warmup_microbatches+1:]
        bwd_step = all_comm_time_df[all_comm_time_df['s_name'].str.match(pat=r'^backward_step$')]
        bwd_step_in_steady = bwd_step.index[:-num_warmup_microbatches]
        return bwd_step.loc[bwd_step_in_steady, 'idle_interval'].sum()/1000 + fwd_step.loc[fwd_step_in_steady, 'idle_interval'].sum()/1000
    
    # Todo: update this function, since currently we cannot accurately get the idle interval because send-recv by mooncake. Also, cannot calculate the send ts of last send_backward_recv_backward
    def calculate_theoretical_bubble_time_cooldown(self, all_comm_time_df, all_backward_steps_df, stage_id, stage_0_optimizer_step_start_ts=None):
        num_warmup_microbatches = get_pp_rank_microbatches(self.get_num_microbatches(), self.pipeline_parallel_size, stage_id, self.vpp_size, self.pipeline_parallel_size)
        if stage_id == self.pipeline_parallel_size-1:
            theoretical_bubble_time = 0.0
        else:
            bwd_index = all_comm_time_df[all_comm_time_df['s_name'].str.match(pat=r'^backward_step$')].index
            if len(bwd_index) > 0:
                bwd_step_in_cooldown = bwd_index[-num_warmup_microbatches:]
                theoretical_bubble_time = all_comm_time_df.loc[
                    bwd_step_in_cooldown[:self.pipeline_parallel_size-stage_id-1], 'idle_interval'
                ].sum()
            else:
                theoretical_bubble_time = 0.0

        if stage_0_optimizer_step_start_ts is not None:
            if stage_id != 0:
                bwd_step_df = all_comm_time_df[all_comm_time_df['s_name'].str.match(pat=r'^backward_step$')]
                if len(bwd_step_df) > 0:
                    last_backward_step = bwd_step_df.iloc[-1]
                    theoretical_bubble_time += (
                        stage_0_optimizer_step_start_ts - (last_backward_step['first_kernel_start'] + last_backward_step['kernel_span'])
                    )
            else:
                if len(all_backward_steps_df) > 0:
                    last_backward_step = all_backward_steps_df.iloc[-1]
                    theoretical_bubble_time += (
                        stage_0_optimizer_step_start_ts - (last_backward_step['first_kernel_start'] + last_backward_step['kernel_span'])
                    )

        return theoretical_bubble_time/1000

    def calculate_bubble_time_cooldown(self, all_comm_time_df, stage_id):
        num_warmup_microbatches = get_pp_rank_microbatches(self.get_num_microbatches(), self.pipeline_parallel_size, stage_id, self.vpp_size, self.pipeline_parallel_size)
        bwd_index = all_comm_time_df[all_comm_time_df['s_name'].str.match(pat=r'^backward_step$')].index
        bwd_step_in_cooldown = bwd_index[-num_warmup_microbatches+(self.pipeline_parallel_size - stage_id - 1):]
        return all_comm_time_df.loc[bwd_step_in_cooldown, 'idle_interval'].sum()/1000

    # Todo: using mooncake, cannot get the accurate comm time and wait time
    def calculate_true_comm(self, all_comm_time_df):
        return 0.0
    
    def get_bubble_time_ratio_theoretical(self, num_microbatch):
        return (self.pipeline_parallel_size - 1) / num_microbatch / self.vpp_size

    @staticmethod
    def format_step_trace_name(row: pd.Series) -> str:
        if row['s_name'] not in ('forward_step', 'backward_step'):
            return row['s_name']
        micro_batch_id = row.get('micro_batch_id', -1)
        vpp_stage_id = row.get('vpp_stage_id', 0)
        if pd.isna(micro_batch_id) or int(micro_batch_id) < 0:
            return row['s_name']
        return f"{row['s_name']}_mb{int(micro_batch_id)}_vpp{abs(int(vpp_stage_id))}"

    # Todo: 增加 fwd step和bwd step batch num and flow
    def save_trace_df_to_file(
        self,
        df: pd.DataFrame,
        output_file: str,
        trace_df_p2p_comm_flow: pd.DataFrame = None,
        meta_data: dict = None,
        pp_schedule: str = '1f1b',
    ):
        columns_to_keep = ['name', 'cat', 'pid', 'tid', 'ts', 'dur', 'rank']
        columns_to_drop = ['s_name', 's_cat']
        
        new_df = df[columns_to_keep].copy()
        new_df['ts'] = df['first_kernel_start'].where(df['first_kernel_start'] > 0, df['ts'])
        new_df['dur'] = df['kernel_span'].where(df['kernel_span'] > 0, df['dur'])
        new_df['name'] = df.apply(self.format_step_trace_name, axis=1)
        new_df['cat'] = df['s_cat']
        new_df['ph'] = 'X'
        # Todo: in interleaved PP, send_fwd_recv_fwd and send_bwd_recv_bwd execute asyn and in parallel with fwd_step or bwd_step
        # so for displaying in perfetto, it muse set them with different tids.
        new_df.loc[new_df['name'].str.match(pat=r"^(send_forward_recv_forward|send_backward_recv_backward)$"), 'tid'] = 1
        new_df.loc[new_df['name'].str.match(pat=r"recv_forward$"), 'tid'] = 1
        new_df.loc[new_df['name'].str.match(pat=r"^mccl:reduce_scatter_tensor_coalesced$"), 'tid'] = 1
        new_df.loc[new_df['name'].str.match(pat=r"^mccl:all_reduce$"), 'tid'] = 1
        new_df['args'] = df.apply(lambda row: {col: row[col] for col in row.index if col not in columns_to_keep + columns_to_drop}, axis=1)

        trace_data = meta_data.copy() if meta_data is not None else {}
        trace_events = new_df.to_dict('records')
        #flow_events = convert_to_flow_events(trace_df_p2p_comm_flow)
        metadata_events = MegatronPipelineParallelGroupTraceBase.generate_metadata_events([tuple(x) for x in new_df[['rank', 'pid']].drop_duplicates().to_records(index=False)])
        trace_data["traceEvents"] = trace_events + metadata_events
        
        with open(output_file, 'w') as f:
            json.dump(trace_data, f, indent=4)
