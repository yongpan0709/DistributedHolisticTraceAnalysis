# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Optional

import pandas as pd

from hta.common.trace_filter import NameFilter
from hta.configs.config import logger
from hta.configs.default_values import DEFAULT_TRACE_DIR
from megatron_parallel_analysis.megatron_pipeline_group_base import MegatronPipelineParallelGroupTraceBase
from megatron_parallel_analysis.utils.trace_filter_utils import create_regex_for_full_match
from megatron_parallel_analysis.utils.parallel_state import get_pp_rank_microbatches

def get_pp_rank_microbatches_epoverlap(
    num_microbatches,
    num_devices,
    device_id,
    num_stages_per_device,
    microbatch_group_size_per_vp_stage,
):
    """Get the number of total, warmup, and remaining microbatches in PP scheduling.
    Default: microbatch_group_size_per_vp_stage = pipeline_model_parallel_size
    """
    return get_pp_rank_microbatches(num_microbatches,
            num_devices,
            device_id,
            num_stages_per_device,
            microbatch_group_size_per_vp_stage,) + 1


class MegatronPipelineParallel1F1BInterleavedEPOverlapGroupTrace(MegatronPipelineParallelGroupTraceBase):
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
        #pp_schedule: str = "1f1b-interleaved-epoverlap",
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

        #sorted_trace_df.to_csv(f'epoverlap_df-after-{rank}.csv')
        return sorted_trace_df


    #def set_vpp_stage_id(self, trace_df: pd.DataFrame, stage_id: int) -> None:
    #    trace_df.sort_values(by=['ts', 'dur'], ascending=[True, False], inplace=True)
    #    trace_df['vpp_stage_id'] = 0
    #    num_microbatches = self.get_num_microbatches()
    #    num_warmup_microbatches = get_pp_rank_microbatches_epoverlap(num_microbatches, self.pipeline_parallel_size, stage_id, self.vpp_size, self.pipeline_parallel_size)
    #    schedule_table = get_schedule_table(num_microbatches, self.vpp_size, self.pipeline_parallel_size)
    #    fwd_order, bwd_order, _ = convert_schedule_table_to_order(num_warmup_microbatches, self.vpp_size, schedule_table)
    #    trace_df.loc[trace_df['s_name'].str.match(pat=r'^forward_step$'), 'vpp_stage_id'] = fwd_order
    #    trace_df.loc[trace_df['s_name'].str.match(pat=r'^backward_step$'), 'vpp_stage_id'] = bwd_order

    
    def set_micro_batch_id(self, pp_group_id: int = 0) -> None:
        """为指定 PP 组内各 rank 的 trace 设置 micro-batch id，由子类的 set_self_microbatch_id/set_recv_send_microbatch_id 实现具体算法。"""
        ranks = self.all_pipeline_parallel_group_ranks[pp_group_id]
        logger.info(f'[1F1B interleaved epoverlap] In set micro batch id: ranks: {ranks}')
        #for stage_id, rank in enumerate(ranks):
        #    self.set_vpp_stage_id(self.traces_comm_only[rank], stage_id)

    #def get_p2p_ranks_pairs(self, ranks):
    #    if len(ranks) < 2: return []
    #    # Todo: double check
    #    ranks_sorted = sorted(ranks)
    #    p2p_devices_pairs = []
    #    for i in range(len(ranks) - 1):
    #        p2p_devices_pairs.append([ranks[i], ranks[i+1]])
    #    return p2p_devices_pairs 
    
    # Todo: func list
    @staticmethod
    def keep_comm_span_only(trace_df):
        comm_names_list = [
            #'forward_step', 
            #'backward_step', 
            'recv_forward', 
            'recv_backward', 
            'send_forward', 
            'send_backward', 
            #'send_forward_recv_backward', 
            #'send_backward_recv_forward', 
            'send_forward_recv_forward',
            'send_backward_recv_backward',
            'fwdbwd_epoverlap_run',
            'finalize_model_grads',
            'step',
            'logical_and_across_model_parallel_group',
            'reduce_max_stat_across_model_parallel_group',
            'should_run_forward_backward',
        ]
        filter_comm = NameFilter(create_regex_for_full_match(comm_names_list))
        return filter_comm(trace_df)

    def filter_comm_only_traces(self, pp_group_id=0):
        for rank in self.all_pipeline_parallel_group_ranks[pp_group_id]:
            self.traces_comm_only[rank] = self.keep_comm_span_only(self.full_dfs[rank])
        
    def establish_p2p_link_on_adjacent_ranks(self, pp_group_id=0):
        logger.info(f'[1F1B interleaved epoverlap] Todo: establish p2p link on adjacent ranks for pp_group_id {pp_group_id}')
    
    def get_all_comm_df(self, sorted_trace_df, rank=None):
        all_comm_df = sorted_trace_df[sorted_trace_df['num_kernels'] > 0]
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
        all_comm_df['idle_interval'].values[0] = self.full_dfs[rank].loc[self.full_dfs[rank]['s_name'].str.match(pat=r'^recv_forward$'), 'dur'].values[0]
        return all_comm_df

    def calculate_step_times(self, fwdbwd_epoverlap_run_df, stage_id=0):
        num_microbatch = self.get_num_microbatches()
        num_warmup_microbatches = get_pp_rank_microbatches_epoverlap(num_microbatch, self.pipeline_parallel_size, stage_id, self.vpp_size, self.pipeline_parallel_size)
        fwdbwd_warmup_step = fwdbwd_epoverlap_run_df[:num_warmup_microbatches]
        fwdbwd_steady_step = fwdbwd_epoverlap_run_df[num_warmup_microbatches:-num_warmup_microbatches]
        fwdbwd_cooldown_step = fwdbwd_epoverlap_run_df[-num_warmup_microbatches:]

        fwdbwd_avg = [round(float(fwdbwd_warmup_step['kernel_span'].mean())/1000, 2),
                      round(float(fwdbwd_steady_step['kernel_span'].mean())/1000, 2),
                      round(float(fwdbwd_cooldown_step['kernel_span'].mean())/1000, 2),
                      ]
        fwdbwd_std = [round(float(fwdbwd_warmup_step['kernel_span'].std())/1000, 2),
                      round(float(fwdbwd_steady_step['kernel_span'].std())/1000, 2),
                      round(float(fwdbwd_cooldown_step['kernel_span'].std())/1000, 2),
                      ]
        return fwdbwd_avg, fwdbwd_std, fwdbwd_epoverlap_run_df['kernel_span'].sum()/1000
    
    def calculate_comm_time_total(self, all_comm_time_df):
        return all_comm_time_df['idle_interval'].sum()/1000

    def calculate_theoretical_bubble_time_warmup(self, all_comm_time_df, stage_id):
        if stage_id == 0:
            return 0.0
        else:
            theoretical_bubble_head_index = all_comm_time_df[all_comm_time_df['s_name'].str.match(pat=r'^recv_forward$')].index[0]
            return all_comm_time_df.loc[theoretical_bubble_head_index, 'dur'].sum()/1000

    def calculate_bubble_time_warmup(self, all_comm_time_df, stage_id):
        num_warmup_microbatches = get_pp_rank_microbatches_epoverlap(self.get_num_microbatches(), self.pipeline_parallel_size, stage_id, self.vpp_size, self.pipeline_parallel_size)
        fwd_step_in_head = all_comm_time_df[all_comm_time_df['s_name'].str.match(pat=r'^fwdbwd_epoverlap_run$')].index[1:num_warmup_microbatches]
        return all_comm_time_df.loc[fwd_step_in_head, 'idle_interval'].sum()/1000
    
    def calculate_theoretical_bubble_time_steady(self, all_comm_time_df, stage_id):
        num_warmup_microbatches = get_pp_rank_microbatches_epoverlap(self.get_num_microbatches(), self.pipeline_parallel_size, stage_id, self.vpp_size, self.pipeline_parallel_size)
        if stage_id == self.pipeline_parallel_size-1:
            return 0.0
        else:
            fwdbwd_index = all_comm_time_df.index
            if len(all_comm_time_df) > num_warmup_microbatches:
                return all_comm_time_df.loc[fwdbwd_index[num_warmup_microbatches], 'idle_interval']/1000
            else:
                return 0.0

    def calculate_bubble_time_steady(self, all_comm_time_df, stage_id):
        num_warmup_microbatches = get_pp_rank_microbatches_epoverlap(self.get_num_microbatches(), self.pipeline_parallel_size, stage_id, self.vpp_size, self.pipeline_parallel_size)
        fwdbwd_step_in_steady = all_comm_time_df[num_warmup_microbatches:-num_warmup_microbatches]
        return fwdbwd_step_in_steady['idle_interval'].sum()/1000
    
    def calculate_theoretical_bubble_time_cooldown(self, all_comm_time_df, stage_id):
        if stage_id == self.pipeline_parallel_size-1:
            return 0.0
        else:
            num_warmup_microbatches = get_pp_rank_microbatches_epoverlap(self.get_num_microbatches(), self.pipeline_parallel_size, stage_id, self.vpp_size, self.pipeline_parallel_size)
            if len(all_comm_time_df) > num_warmup_microbatches * 2:
                fwdbwd_step_in_cooldown = all_comm_time_df[-num_warmup_microbatches:]
                return fwdbwd_step_in_cooldown['idle_interval'].sum()/1000
            else:
                return 0.0
        
    def calculate_bubble_time_cooldown(self, all_comm_time_df, stage_id):
        num_warmup_microbatches = get_pp_rank_microbatches_epoverlap(self.get_num_microbatches(), self.pipeline_parallel_size, stage_id, self.vpp_size, self.pipeline_parallel_size)
        bwd_step_in_cooldown = all_comm_time_df[-num_warmup_microbatches:]
        return bwd_step_in_cooldown['idle_interval'].sum()/1000

    # Todo: using mooncake, cannot get the accurate comm time and wait time
    def calculate_true_comm_and_overhead_wait_time(self, all_comm_time_df):
        return 0.0, 0.0
    
    def get_bubble_time_ratio_theoretical(self, num_microbatch):
        return (self.pipeline_parallel_size - 1) / num_microbatch / self.vpp_size
    
    def generate_report(self, pp_group_id, save_path):
        output_df = None

        for stage_id, rank in enumerate(sorted(self.traces_comm_only.keys())):
            sorted_trace_df = self.preprocess_trace_df(rank)
            #sorted_trace_df.to_csv(f'sorted_trace_df-{rank}.csv')
            time_per_iteration = self.calculate_time_per_iteration(rank)
            num_microbatch = self.get_num_microbatches() 

            fwdbwd_epoverlap_run_steps_df = NameFilter(create_regex_for_full_match(['fwdbwd_epoverlap_run']))(sorted_trace_df)
            #fwdbwd_epoverlap_run_steps_df.to_csv(f'fwdbwd_epoverlap_run_steps_df-{rank}-stageid-{stage_id}.csv')
            fwdbwd_epoverlap_step_avg_time, fwdbwd_epoverlap_std, compute_time_total = self.calculate_step_times(fwdbwd_epoverlap_run_steps_df, stage_id)

            # 'send_forward_recv_backward',  'send_backward_recv_forward',
            all_comm_time_df = self.get_all_comm_df(fwdbwd_epoverlap_run_steps_df, rank)
            #all_comm_time_df.to_csv(f'all_comm_time_df-{rank}-stageid-{stage_id}.csv')
            comm_time_total = self.calculate_comm_time_total(all_comm_time_df)

            theoretical_bubble_time_warmup = self.calculate_theoretical_bubble_time_warmup(sorted_trace_df, stage_id)
            bubble_time_warmup = self.calculate_bubble_time_warmup(all_comm_time_df, stage_id)
            theoretical_bubble_time_steady = self.calculate_theoretical_bubble_time_steady(all_comm_time_df, stage_id)
            bubble_time_steady = self.calculate_bubble_time_steady(all_comm_time_df, stage_id) - theoretical_bubble_time_steady
            theoretical_bubble_time_cooldown = self.calculate_theoretical_bubble_time_cooldown(all_comm_time_df, stage_id)
            bubble_time_cooldown = self.calculate_bubble_time_cooldown(all_comm_time_df, stage_id)
            finalize_model_grads_step_time = self.calculate_finalize_model_grads_step_time(sorted_trace_df)
            optimizer_time = self.calculate_optimizer_step_time_and_bubble(sorted_trace_df)
            logical_and_across_model_parallel_group_time = self.calculate_logical_and_across_model_parallel_group_time(sorted_trace_df)
            #bubble_time_total = bubble_time_warmup + bubble_time_steady + bubble_time_cooldown # + bubble_time_final
            #comm_time_total += bubble_time_final
            #comm_time_true, overhead_wait_time_total = self.calculate_true_comm_and_overhead_wait_time(all_comm_time_df)
            
            info_per_rank = {
                'rank': rank,
                'time_per_iteration': time_per_iteration,
                'num_microbatch': num_microbatch,
                'fwdbwd_epoverlap_avg_time': fwdbwd_epoverlap_step_avg_time,
                'fwdbwd_epoverlap_std_time': fwdbwd_epoverlap_std,
                'compute_time_total': compute_time_total,
                'comm_time_total': comm_time_total,
                #'comm_time_true': comm_time_true,
                #'overhead_wait_time_total': overhead_wait_time_total,
                ##'bubble_time_total': bubble_time_total,
                'bubble_time_warmup': bubble_time_warmup,
                'bubble_time_steady': bubble_time_steady,
                'bubble_time_cooldown': bubble_time_cooldown,
                'theoretical_bubble_time_warmup': theoretical_bubble_time_warmup,
                'theoretical_bubble_time_steady': theoretical_bubble_time_steady,
                'theoretical_bubble_time_cooldown': theoretical_bubble_time_cooldown,
                ##'bubble_time_final': bubble_time_final,
                #'overhead_wait_time_ratio': overhead_wait_time_total / time_per_iteration,
                #'bubble_time_ratio': overhead_wait_time_total/ (compute_time_total + comm_time_total),
                #'bubble_time_ratio_theoretical': self.get_bubble_time_ratio_theoretical(num_microbatch),
                'pipeline_parallel_size': self.pipeline_parallel_size,
                #'comm_time_true_ratio': comm_time_true / time_per_iteration,
                'comp_time_ratio': compute_time_total / time_per_iteration,
                'comm_time_ratio': comm_time_total / time_per_iteration,
                'finalize_model_grads_step_time': finalize_model_grads_step_time,
                'logical_and_across_model_parallel_group_time': logical_and_across_model_parallel_group_time,
                'optimizer_time_total': optimizer_time,
            }
            #info_per_rank = self._generate_info_per_rank(args)

            if output_df is None:
                output_df = pd.DataFrame([info_per_rank])
            else:
                output_df.loc[len(output_df)] = info_per_rank

        if save_path is not None:
            output_df.to_csv(save_path, header=True, index=False, float_format='%.3f')
        #print(output_df)
        return output_df

