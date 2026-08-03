# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import os
import logging
import math
import pickle
import shutil
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpi4py import MPI

from megatron_parallel_analysis.utils.parallel_state import RankGenerator
from megatron_parallel_analysis.utils.utils import partition_files_across_directories, prepare_directory
from hta.configs.config import logger
from megatron_parallel_analysis.megatron_pipeline_group_1f1b_interleaved_epoverlap import MegatronPipelineParallel1F1BInterleavedEPOverlapGroupTrace
from megatron_parallel_analysis.megatron_pipeline_group_1f1b import MegatronPipelineParallel1F1BGroupTrace
from megatron_parallel_analysis.megatron_pipeline_group_1f1b_interleaved import MegatronPipelineParallel1F1BInterleavedGroupTrace
from megatron_parallel_analysis.megatron_pipeline_group_base import MegatronPipelineParallelGroupTraceBase


MAX_INT = 2**31 - 1  # Maximum size for each chunk

def send_by_chunk(data, dest, tag, comm):
    """Send data in chunks to avoid MPI message size limits."""
    data = np.asarray(data, dtype='b')  # Convert to byte array
    total_size = data.nbytes
    comm.send(total_size, dest=dest, tag=tag)  # Send total size first

    num_chunks = (total_size + MAX_INT - 1) // MAX_INT  # Calculate number of chunks
    for i in range(num_chunks):
        start = i * MAX_INT
        end = min(start + MAX_INT, total_size)
        comm.Send([data[start:end], MPI.BYTE], dest=dest, tag=tag + i + 1)


def recv_by_chunk(source, tag, comm):
    """Receive data in chunks."""
    total_size = comm.recv(source=source, tag=tag)  # Receive total size first
    recv_data = np.empty(total_size, dtype='b')  # Allocate buffer

    num_chunks = (total_size + MAX_INT - 1) // MAX_INT  # Calculate number of chunks
    for i in range(num_chunks):
        start = i * MAX_INT
        end = min(start + MAX_INT, total_size)
        comm.Recv([recv_data[start:end], MPI.BYTE], source=source, tag=tag + i + 1)

    return recv_data


def gatherv_p2p(comm, sendbuf, recvbuf, root=0):
    """Gather data from all processes to root using point-to-point communication."""
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == root:
        recv_data, recvcounts, displs, recvtype = recvbuf
        recvcounts = np.asarray(recvcounts, dtype=np.int64)
        displs = np.asarray(displs, dtype=np.int64)

        for i in range(size):
            if i == root:
                recv_data[displs[i]:displs[i] + recvcounts[i]] = sendbuf
            else:
                temp_recvbuf = recv_by_chunk(source=i, tag=0, comm=comm)
                recv_data[displs[i]:displs[i] + recvcounts[i]] = temp_recvbuf
    else:
        send_by_chunk(data=sendbuf, dest=root, tag=0, comm=comm)


def gather_data(comm, send_data, use_p2p=True):
    """Gather data from all processes to root."""
    rank = comm.Get_rank()
    size = comm.Get_size()
    send_data_size = len(send_data)

    # Collect the size of data sent by each process
    send_counts = comm.gather(send_data_size, root=0)

    if rank == 0:
        recv_displs = np.zeros(size, dtype=int)
        for i in range(1, size):
            recv_displs[i] = recv_displs[i - 1] + send_counts[i - 1]
        total_recv_size = sum(send_counts)
        recv_data = np.empty(total_recv_size, dtype='b')
    else:
        recv_displs = None
        recv_data = None

    if use_p2p:
        if rank == 0:
            recvbuf_p2p = (recv_data, send_counts, recv_displs, MPI.BYTE)
        else:
            recvbuf_p2p = None
        gatherv_p2p(comm, np.frombuffer(send_data, dtype='b'), recvbuf_p2p, root=0)
    else:
        comm.Gatherv(np.frombuffer(send_data, dtype='b'), [recv_data, send_counts, recv_displs, MPI.BYTE], root=0)

    return recv_data, send_counts, recv_displs


class DistributedMegatronTraceAnalysis:
    """
    Distributed trace analysis for Megatron pipeline parallel groups.
    
    This class combines the functionality of MegatronPipelineParallelGroupTraceAnalysis
    with distributed processing capabilities to analyze pipeline parallel traces across
    multiple processes.
    """
    
    @staticmethod
    def _get_expert_data_parallel_size(dp: int, ep: int) -> int:
        if dp <= 0 or ep <= 0:
            raise ValueError(
                f'Data parallel size and expert parallel size must be positive; '
                f'got DP={dp}, EP={ep}'
            )
        if dp % ep != 0:
            raise ValueError(
                f'Total data parallel size must be divisible by expert parallel size; '
                f'got DP={dp}, EP={ep}'
            )
        return dp // ep

    def __init__(
        self, 
        trace_dir: str, 
        tp: int, 
        ep: int, 
        dp: int, 
        pp: int, 
        cp: int = 1, 
        pp_schedule: str = '1f1b', 
        vpp_size: int = 2, 
        micro_bs = 0,
        order: str = "tp-cp-ep-dp-pp",
        enable_ep_analysis: bool = False,
        rebuild_parse_cache: bool = False,
    ):
        """
        Initialize the distributed trace analyzer.
        
        Args:
            trace_dir: Directory containing trace files
            tp: Tensor parallel size
            ep: Expert model parallel size
            dp: Data parallel size
            pp: Pipeline parallel size
            cp: Context parallel size
            pp_schedule: Pipeline parallel schedule ('1f1b', '1f1b-interleaved', '1f1b-interleaved-epoverlap')
            vpp_size: Virtual pipeline parallel size
            order: Rank generation order
        """
        self.trace_dir = trace_dir.rstrip('/')
        self.tensor_parallel_size = tp
        self.data_parallel_size = dp
        self.pipeline_parallel_size = pp
        self.expert_model_parallel_size = ep
        self.context_parallel_size = cp
        self.enable_ep_analysis = enable_ep_analysis
        edp = self._get_expert_data_parallel_size(dp, ep)
        self.expert_data_parallel_size = edp
        assert pp_schedule in ['1f1b', '1f1b-interleaved', '1f1b-interleaved-epoverlap'], \
            f'Invalid pp schedule: {pp_schedule}'
        self.pp_schedule = pp_schedule
        self.order = order

        if pp_schedule in ['1f1b-interleaved', '1f1b-interleaved-epoverlap']:
            self.vpp_size = vpp_size
            assert self.vpp_size > 0, f'Invalid vpp size: {self.vpp_size}'
        else:
            self.vpp_size = None
        self.micro_bs = micro_bs
        self.rebuild_parse_cache = rebuild_parse_cache
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.world_size = self.comm.Get_size()
        self.processor_name = MPI.Get_processor_name()
        
        self.expert_decoder_rank_generator = RankGenerator(
            tp=tp, ep=ep, dp=edp, pp=pp, cp=cp, order=order, rank_offset=0
        )
        self.all_data_parallel_group_ranks = self.expert_decoder_rank_generator.get_ranks('dp')
        self.all_tensor_parallel_group_ranks = self.expert_decoder_rank_generator.get_ranks('tp')
        self.all_pipeline_parallel_group_ranks = self.expert_decoder_rank_generator.get_ranks('pp')
        self.all_expert_parallel_group_ranks = self.expert_decoder_rank_generator.get_ranks('ep')
        self.source_pp_ep_group_bundles = self._build_source_pp_ep_group_bundles() if self._should_analyze_ep() else []
        self.assigned_tasks = []
        self.analysis_list = []
        self.total_analysis_lists = None
        self.total_traces_list = None
        self.total_trace_df = None
        self.anomaly_status = None
        
        self.setup_dirs()
        self.setup_logging()
        self.init_pp_group_sub_dirs()
        self.assign_analysis_tasks()

    def _should_analyze_ep(self):
        """Return whether EP timing data should be collected and analyzed."""
        return (
            self.enable_ep_analysis
            and self.expert_model_parallel_size > 1
            and self.pp_schedule in ('1f1b', '1f1b-interleaved')
        )

    def _build_source_pp_ep_group_bundles(self):
        """Map EP groups to their source PP groups without assuming rank arithmetic."""
        rank_to_pp_location = {}
        for pp_group_id, ranks in enumerate(self.all_pipeline_parallel_group_ranks):
            for pp_stage_id, rank in enumerate(ranks):
                if rank in rank_to_pp_location:
                    raise ValueError(f'Rank {rank} occurs in multiple PP groups')
                rank_to_pp_location[rank] = (pp_group_id, pp_stage_id)

        grouped_descriptors = defaultdict(list)
        for ep_group_id, ranks in enumerate(self.all_expert_parallel_group_ranks):
            if len(ranks) != self.expert_model_parallel_size or len(set(ranks)) != len(ranks):
                raise ValueError(f'Invalid ranks for EP group {ep_group_id}: {ranks}')
            try:
                locations = [rank_to_pp_location[rank] for rank in ranks]
            except KeyError as error:
                raise ValueError(f'EP group {ep_group_id} contains unknown rank {error.args[0]}') from error
            pp_stage_ids = {location[1] for location in locations}
            source_pp_group_ids = tuple(sorted(location[0] for location in locations))
            if len(pp_stage_ids) != 1 or len(set(source_pp_group_ids)) != self.expert_model_parallel_size:
                raise ValueError(
                    f'EP group {ep_group_id} does not map to one PP stage and '
                    f'{self.expert_model_parallel_size} source PP groups: {locations}'
                )
            descriptor = {
                'ep_group_id': ep_group_id,
                'pp_stage_id': pp_stage_ids.pop(),
                'ranks': tuple(ranks),
                'source_pp_group_ids': source_pp_group_ids,
            }
            grouped_descriptors[source_pp_group_ids].append(descriptor)

        return [
            {
                'source_pp_group_ids': source_pp_group_ids,
                'ep_groups': tuple(sorted(descriptors, key=lambda item: item['pp_stage_id'])),
            }
            for source_pp_group_ids, descriptors in sorted(grouped_descriptors.items())
        ]

    @staticmethod
    def _tasks_for_worker(tasks, worker_rank, world_size):
        num_tasks_per_process, remainder = divmod(len(tasks), world_size)
        if worker_rank < remainder:
            start_index = worker_rank * (num_tasks_per_process + 1)
            end_index = start_index + num_tasks_per_process + 1
        else:
            start_index = remainder * (num_tasks_per_process + 1) + \
                         (worker_rank - remainder) * num_tasks_per_process
            end_index = start_index + num_tasks_per_process
        return tasks[start_index:end_index]

    def setup_dirs(self):
        """Setup directory structure for analysis."""
        self.workspace_dir = 'workspace'
        self.workname = os.path.basename(self.trace_dir)
        self.trace_dir_pp_group = os.path.join(self.workspace_dir, self.workname, 'trace')
        self.ep_group_stats_dir = os.path.join(self.trace_dir_pp_group, 'ep_group_stats')
        self.ep_trace_dir = os.path.join(self.trace_dir_pp_group, 'ep_trace')
        self.output_dir = os.path.join(self.workspace_dir, self.workname, 'output')
        self.stragglers_dir = os.path.join(self.output_dir, 'stragglers')
        self.log_dir = os.path.join(self.workspace_dir, self.workname, 'log')
        self.parse_cache_dir = os.path.join(self.workspace_dir, self.workname, 'cache')
        if self.rank == 0:
            prepare_directory(self.trace_dir_pp_group, force_clear=False)
            if self._should_analyze_ep():
                prepare_directory(self.ep_group_stats_dir, force_clear=True)
                prepare_directory(self.ep_trace_dir, force_clear=True)
            prepare_directory(self.log_dir, force_clear=True)
            prepare_directory(self.output_dir, force_clear=True)
            prepare_directory(self.stragglers_dir, force_clear=True)
            prepare_directory(self.parse_cache_dir, force_clear=False)
        self.comm.Barrier()
        time.sleep(3)

    def setup_logging(self):
        """Setup logging for each process."""
        log_filename = os.path.join(self.log_dir, f'log_mpirun_parallel_{self.rank}.log')
        logging.basicConfig(
            filename=log_filename,
            filemode='w',
            level=logging.DEBUG,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        logger.info(f'Process {self.rank} on {self.processor_name} started logging.')

    def init_pp_group_sub_dirs(self):
        """Initialize pipeline parallel group subdirectories."""
        logger.info('Initializing pipeline parallel group subdirectories.')
        trace_dir_for_pp_group = os.path.join(self.trace_dir_pp_group, 'pp_group')
        self.all_pp_group_sub_dirs = [
            f'{trace_dir_for_pp_group}_{i}' 
            for i in range(len(self.all_pipeline_parallel_group_ranks))
        ]

        if self.rank == 0:
            partition_files_across_directories(
                self.trace_dir,
                trace_dir_for_pp_group,
                self.all_pipeline_parallel_group_ranks,
            )
        
        self.comm.Barrier()
        time.sleep(6)
        logger.info('Initialization of pipeline parallel group subdirectories completed.')

    def assign_analysis_tasks(self):
        """Assign PP groups or complete EP bundles to MPI workers."""
        logger.info('Assigning analysis tasks.')
        if self._should_analyze_ep():
            self.assigned_tasks = self._tasks_for_worker(
                self.source_pp_ep_group_bundles, self.rank, self.world_size
            )
        else:
            tasks = list(enumerate(self.all_pp_group_sub_dirs))
            self.assigned_tasks = self._tasks_for_worker(tasks, self.rank, self.world_size)
        logger.info(f'Assigned tasks: {self.assigned_tasks}')

    def _create_pipeline_trace(self, trace_dir: str):
        """
        Create pipeline trace object based on schedule type.
        
        Args:
            trace_dir: Directory containing trace files
            
        Returns:
            Pipeline trace object
        """
        if self.pp_schedule == '1f1b':
            return MegatronPipelineParallel1F1BGroupTrace(
                trace_files=None, 
                trace_dir=trace_dir,
                dp=self.expert_data_parallel_size,
                tp=self.tensor_parallel_size,
                pp=self.pipeline_parallel_size,
                ep=self.expert_model_parallel_size,
                cp=self.context_parallel_size,
                order=self.order,
                micro_bs = self.micro_bs,
                parse_cache_dir=self.parse_cache_dir,
                rebuild_parse_cache=self.rebuild_parse_cache,
            )
        elif self.pp_schedule == '1f1b-interleaved':
            return MegatronPipelineParallel1F1BInterleavedGroupTrace(
                trace_files=None,
                trace_dir=trace_dir,
                dp=self.expert_data_parallel_size,
                tp=self.tensor_parallel_size,
                pp=self.pipeline_parallel_size,
                ep=self.expert_model_parallel_size,
                cp=self.context_parallel_size,
                order=self.order,
                vpp_size=self.vpp_size,
                micro_bs = self.micro_bs,
                parse_cache_dir=self.parse_cache_dir,
                rebuild_parse_cache=self.rebuild_parse_cache,
            )
        elif self.pp_schedule == '1f1b-interleaved-epoverlap':
            return MegatronPipelineParallel1F1BInterleavedEPOverlapGroupTrace(
                trace_files=None,
                trace_dir=trace_dir,
                dp=self.expert_data_parallel_size,
                tp=self.tensor_parallel_size,
                pp=self.pipeline_parallel_size,
                ep=self.expert_model_parallel_size,
                cp=self.context_parallel_size,
                order=self.order,
                vpp_size=self.vpp_size,
                micro_bs = self.micro_bs,
                parse_cache_dir=self.parse_cache_dir,
                rebuild_parse_cache=self.rebuild_parse_cache,
            )

    def analyze_pipeline_parallel_per_group(self, pp_group_id: int, trace_dir: str):
        """
        Analyze pipeline parallel traces for a single group.
        This method integrates the core functionality from MegatronPipelineParallelGroupTraceAnalysis.

        Args:
            pp_group_id: Pipeline parallel group ID
            trace_dir: Directory containing trace files for this group
        """
        output_dir = os.path.join(trace_dir, 'output')
        prepare_directory(output_dir, force_clear=False)

        context = (
            f'pp_group_id={pp_group_id}, mpi_rank={self.rank}, '
            f'trace_dir={trace_dir}'
        )
        logger.info(f'Analyzing pipeline parallel group {pp_group_id}')
        logger.info('PP-group analysis context: %s', context)

        try:
            pipeline_trace = self._create_pipeline_trace(trace_dir)

            logger.info('Construct CallGraph for traces')
            pipeline_trace.parse_traces_per_pp_group(pp_group_id=pp_group_id)

            logger.info('Keep comm spans only')
            pipeline_trace.filter_comm_only_traces(pp_group_id=pp_group_id)

            logger.info('Set micro batch id')
            pipeline_trace.set_micro_batch_id(pp_group_id=pp_group_id)

            logger.info('Establish P2P link on adjacent ranks')
            pipeline_trace.establish_p2p_link_on_adjacent_ranks(pp_group_id=pp_group_id)

            logger.info('Save traces with P2P comm')
            pipeline_trace.save_traces_with_p2p_comm(
                f'{output_dir}/../../pp{pp_group_id}-trace.json',
                traces=pipeline_trace.traces_comm_only
            )

            logger.info('Generate report')
            pipeline_trace.generate_report(
                pp_group_id,
                f'{output_dir}/../../report-pp{pp_group_id}.csv'
            )
        except Exception as error:
            raise RuntimeError(
                'Pipeline parallel analysis failed '
                f'({context})'
            ) from error

        return pipeline_trace

    def process_single_pp_group(self, pp_group_id: int, trace_dir: str):
        """
        Process a single pipeline parallel group.
        
        Args:
            pp_group_id: Pipeline parallel group ID
            trace_dir: Directory containing trace files for this group
            
        Returns:
            Analysis result dictionary
        """
        logger.debug(f'Processing pp group {pp_group_id} in {trace_dir}')
        
        pipeline_trace = self.analyze_pipeline_parallel_per_group(pp_group_id, trace_dir)
        
        return pipeline_trace

    def _validate_and_select_ep_bundles(self, selected_pp_group_ids, pp_group_id_range):
        """Select only complete EP bundles covered by the requested PP IDs."""
        if pp_group_id_range is None:
            return list(self.source_pp_ep_group_bundles)

        selected_bundles = []
        for bundle in self.source_pp_ep_group_bundles:
            source_ids = set(bundle['source_pp_group_ids'])
            covered_ids = source_ids.intersection(selected_pp_group_ids)
            if not covered_ids:
                continue
            if covered_ids != source_ids:
                missing_ids = sorted(source_ids.difference(covered_ids))
                raise ValueError(
                    f'PP group range {pp_group_id_range} partially overlaps EP group '
                    f'{bundle["source_pp_group_ids"]}: covered '
                    f'{sorted(covered_ids)}, missing {missing_ids}. '
                    'EP analysis requires complete bundle coverage.'
                )
            selected_bundles.append(bundle)
        if not selected_bundles:
            raise ValueError(
                f'PP group range {pp_group_id_range} does not cover any complete EP bundle.'
            )
        return selected_bundles

    def _assign_tasks_for_selection(self, selected_pp_group_ids, selected_bundles=None):
        if self._should_analyze_ep():
            tasks = self.source_pp_ep_group_bundles if selected_bundles is None else selected_bundles
        else:
            tasks = [
                (pp_group_id, self.all_pp_group_sub_dirs[pp_group_id])
                for pp_group_id in sorted(selected_pp_group_ids)
            ]
        self.assigned_tasks = self._tasks_for_worker(tasks, self.rank, self.world_size)
        logger.info('Assigned tasks: %s', self.assigned_tasks)

    def _ep_group_stats_events(self):
        """Return the stable semantic event order used in EP group statistics rows."""
        if self.pp_schedule == '1f1b':
            return ('forward', 'backward')
        if self.pp_schedule == '1f1b-interleaved':
            return tuple(
                f'{direction}_vpp{vpp_id}'
                for direction in ('forward', 'backward')
                for vpp_id in range(self.vpp_size)
            )
        raise ValueError(
            f"EP group statistics are not supported for schedule "
            f"{self.pp_schedule!r}"
        )

    @staticmethod
    def _timeline_stat_columns(event_name):
        return (
            f'{event_name}_start_min_ms',
            f'{event_name}_start_max_ms',
            f'{event_name}_duration_min_ms',
            f'{event_name}_duration_max_ms',
            f'{event_name}_duration_mean_ms',
            f'{event_name}_duration_std_ms',
        )

    def _write_ep_group_stats_csv(self, descriptor, timelines_by_rank):
        """Write cross-rank GPU timing statistics in milliseconds per micro-batch.

        Each row represents one micro-batch.  Every event is aggregated over
        all ranks in the EP descriptor.  The timeline extractors define the
        positional event order: forward/backward for 1F1B, and all forward
        VPP stages followed by all backward VPP stages for interleaved 1F1B.
        """
        event_names = self._ep_group_stats_events()
        expected_interval_count = len(event_names)
        expected_ranks = tuple(descriptor['ranks'])
        ep_group_id = descriptor['ep_group_id']
        if set(timelines_by_rank) != set(expected_ranks):
            raise ValueError(
                f'EP group {ep_group_id} rank mismatch: expected '
                f'{expected_ranks}, got {tuple(timelines_by_rank)}'
            )

        expected_micro_batches = set(range(self.micro_bs))
        validated_intervals = {}
        for rank in expected_ranks:
            timeline = timelines_by_rank[rank]
            if set(timeline) != expected_micro_batches:
                raise ValueError(
                    f'EP group {ep_group_id} rank {rank} has invalid micro-batches: '
                    f'{sorted(timeline)}'
                )
            for micro_batch_id in range(self.micro_bs):
                intervals = timeline[micro_batch_id]
                if len(intervals) != expected_interval_count:
                    raise ValueError(
                        f'EP group {ep_group_id} rank {rank}, micro-batch '
                        f'{micro_batch_id} has {len(intervals)} intervals; expected '
                        f'{expected_interval_count}'
                    )
                validated = []
                for event_name, interval in zip(event_names, intervals):
                    if len(interval) != 2:
                        raise ValueError(
                            f'EP group {ep_group_id} rank {rank}, micro-batch '
                            f'{micro_batch_id}, event {event_name} has invalid '
                            f'interval {interval!r}'
                        )
                    try:
                        start = float(interval[0])
                        duration = float(interval[1])
                    except (TypeError, ValueError, OverflowError) as error:
                        raise ValueError(
                            f'EP group {ep_group_id} rank {rank}, micro-batch '
                            f'{micro_batch_id}, event {event_name} has non-numeric '
                            f'interval {interval!r}'
                        ) from error
                    if (
                        not math.isfinite(start)
                        or not math.isfinite(duration)
                        or start <= 0
                        or duration < 0
                    ):
                        raise ValueError(
                            f'EP group {ep_group_id} rank {rank}, micro-batch '
                            f'{micro_batch_id}, event {event_name} has invalid '
                            f'start/duration ({start}, {duration})'
                        )
                    validated.append((start, duration))
                validated_intervals[(rank, micro_batch_id)] = validated

        columns = ['micro_batch_id'] + [
            column
            for event_name in event_names
            for column in self._timeline_stat_columns(event_name)
        ]
        rows = []
        for micro_batch_id in range(self.micro_bs):
            row = {'micro_batch_id': micro_batch_id}
            for event_index, event_name in enumerate(event_names):
                starts = np.asarray(
                    [
                        validated_intervals[(rank, micro_batch_id)][event_index][0]
                        for rank in expected_ranks
                    ],
                    dtype=float,
                )
                durations = np.asarray(
                    [
                        validated_intervals[(rank, micro_batch_id)][event_index][1]
                        for rank in expected_ranks
                    ],
                    dtype=float,
                )
                row.update(dict(zip(
                    self._timeline_stat_columns(event_name),
                    tuple(
                        value / 1000.0
                        for value in (
                            starts.min(),
                            starts.max(),
                            durations.min(),
                            durations.max(),
                            durations.mean(),
                            durations.std(ddof=0),
                        )
                    ),
                )))
            rows.append(row)

        output_path = os.path.join(
            self.ep_group_stats_dir,
            f"ep_group_{ep_group_id}-pp_stage_{descriptor['pp_stage_id']}.csv",
        )
        pd.DataFrame(rows, columns=columns).to_csv(
            output_path, index=False, float_format='%.3f'
        )

    def _write_ep_trace(self, descriptor, traces_by_rank, pipeline_trace):
        expected_ranks = tuple(descriptor['ranks'])
        if set(traces_by_rank) != set(expected_ranks):
            raise ValueError(
                f"EP group {descriptor['ep_group_id']} trace rank mismatch: expected "
                f'{expected_ranks}, got {tuple(traces_by_rank)}'
            )
        output_path = os.path.join(
            self.ep_trace_dir,
            f"ep_group_{descriptor['ep_group_id']}-pp_stage_"
            f"{descriptor['pp_stage_id']}-trace.json",
        )
        pipeline_trace.save_traces_with_p2p_comm(
            output_path,
            traces={rank: traces_by_rank[rank] for rank in expected_ranks},
        )

    def _process_ep_bundle(self, bundle):
        selected_source_ids = set(bundle['source_pp_group_ids'])
        timelines_by_rank = {}
        traces_by_rank = {}
        pipeline_trace = None
        for pp_group_id in sorted(selected_source_ids):
            folder = self.all_pp_group_sub_dirs[pp_group_id]
            pipeline_trace = self.process_single_pp_group(pp_group_id, folder)
            for rank in self.all_pipeline_parallel_group_ranks[pp_group_id]:
                timelines_by_rank[rank] = pipeline_trace.extract_rank_gpu_timeline(rank)
                traces_by_rank[rank] = pipeline_trace.traces_comm_only[rank]

        for descriptor in bundle['ep_groups']:
            group_timelines = {
                rank: timelines_by_rank[rank]
                for rank in descriptor['ranks']
            }
            group_traces = {
                rank: traces_by_rank[rank]
                for rank in descriptor['ranks']
            }
            self._write_ep_trace(descriptor, group_traces, pipeline_trace)
            self._write_ep_group_stats_csv(descriptor, group_timelines)

    def _validate_pp_group_id_range(self, pp_group_id_range):
        if pp_group_id_range is None:
            return set(range(len(self.all_pipeline_parallel_group_ranks)))
        start_pp_group_id, end_pp_group_id = pp_group_id_range
        if (
            start_pp_group_id < 0
            or start_pp_group_id > end_pp_group_id
            or end_pp_group_id >= len(self.all_pipeline_parallel_group_ranks)
        ):
            raise ValueError(
                f'Invalid PP group range {pp_group_id_range}; valid IDs are '
                f'0..{len(self.all_pipeline_parallel_group_ranks) - 1}'
            )
        return set(range(start_pp_group_id, end_pp_group_id + 1))

    def analyze(self, pp_group_id_range: Optional[Tuple[int, int]] = None):
        """Execute analysis on assigned tasks.

        Args:
            pp_group_id_range: Optional inclusive pp_group_id range, e.g. (0, 3).
        """
        self.analysis_list = []
        selected_pp_group_ids = self._validate_pp_group_id_range(pp_group_id_range)
        if self._should_analyze_ep():
            selected_bundles = self._validate_and_select_ep_bundles(
                selected_pp_group_ids,
                pp_group_id_range,
            )
            self._assign_tasks_for_selection(
                selected_pp_group_ids,
                selected_bundles,
            )
            for bundle in self.assigned_tasks:
                self._process_ep_bundle(bundle)
        else:
            self._assign_tasks_for_selection(selected_pp_group_ids)
            for pp_group_id, folder in self.assigned_tasks:
                logger.debug(f'Processing pp group {pp_group_id}')
                result = self.process_single_pp_group(pp_group_id, folder)
                self.analysis_list.append(result)

        # Note: Uncomment the following lines to enable full distributed analysis
        # self.gather_infos_from_all_ranks()
        # self.analyze_anomalies()
        # self.post_process()

    def gather_infos_from_all_ranks(self):
        """Gather analysis results from all ranks to root process."""
        logger.info('Gathering information from all ranks.')

        # Serialize the data to be sent
        send_data = pickle.dumps(self.analysis_list)

        # Gather data using the helper function
        recv_data, send_counts, recv_displs = gather_data(self.comm, send_data)

        if self.rank == 0:
            # The root process deserializes and merges data
            self.total_analysis_lists = []
            for i in range(self.world_size):
                start_index = recv_displs[i]
                end_index = start_index + send_counts[i]
                data_chunk = recv_data[start_index:end_index]
                analysis_list = pickle.loads(data_chunk)
                self.total_analysis_lists.extend(analysis_list)

            logger.info(f'Total analysis lists gathered: {len(self.total_analysis_lists)}')
            self.total_traces_list = [
                analysis['traces'] for analysis in self.total_analysis_lists
            ]
            self.total_trace_df = self.combine_traces()
        else:
            self.total_analysis_lists = None
            self.total_traces_list = None
            self.total_trace_df = None

    def combine_traces(self):
        """Combine traces from all pipeline groups into a single DataFrame."""
        if not self.total_traces_list:
            return None
            
        for trace in self.total_traces_list:
            for pp_stage_id, rank in enumerate(sorted(trace.keys())):
                trace_df = trace[rank]
                trace_df['rank'] = rank
                trace_df['pp_stage'] = pp_stage_id

        trace_dfs_list = [
            trace_df 
            for trace in self.total_traces_list 
            for trace_df in list(trace.values())
        ]
        return pd.concat(trace_dfs_list)

    def analyze_anomalies(self):
        """Analyze anomalies in the combined trace data."""
        if self.total_trace_df is None:
            logger.warning('No trace data available for anomaly analysis')
            return

        self.detect_anomalies_between_pp_group()
        self.detect_anomalies_between_layers()

    def detect_anomalies_between_pp_group(self):
        """Detect anomalies between pipeline parallel groups."""
        group_keys = ['full_name', 'pp_stage']
        anomaly_key = 'is_anomaly_between_pp_group'
        output_dir = os.path.join(self.stragglers_dir, 'stragglers_bewteen_pp_group')
        self.detect_anomalies_zscore_and_relative(group_keys, anomaly_key)
        self.plot_anomalies_between_pp_group(output_dir)

    def detect_anomalies_between_layers(self):
        """Detect anomalies between transformer layers."""
        self.set_full_name_without_index(
            name_index_list=['forward_step', 'backward_step', 'ParallelTransformerLayer']
        )
        group_keys = ['full_name_without_index']
        anomaly_key = 'is_anomaly_between_layers'
        output_dir = os.path.join(self.stragglers_dir, 'stragglers_bewteen_layers')
        self.detect_anomalies_zscore_and_relative(group_keys, anomaly_key)
        self.plot_anomalies_between_layers(output_dir)

    def set_full_name_without_index(self, name_index_list: List[str]):
        """
        Create a new column with full names without numeric indices.
        
        Args:
            name_index_list: List of name patterns to remove indices from
        """
        self.total_trace_df['full_name_without_index'] = self.total_trace_df['full_name']
        for name in name_index_list:
            self.total_trace_df['full_name_without_index'] = \
                self.total_trace_df['full_name_without_index'].str.replace(
                    f'{name}_\\d+', name, regex=True
                )
        self.total_trace_df['full_name_without_index'] = (
            'rank' + self.total_trace_df['rank'].astype(str) + '/' + 
            self.total_trace_df['full_name_without_index']
        )

    def detect_anomalies_zscore_and_relative(
        self, 
        group_keys: List[str] = ['full_name'], 
        anomaly_key: str = 'is_anomaly', 
        zscore_threshold: float = 2, 
        min_std: float = 0.1, 
        relative_threshold: float = 0.5
    ):
        """
        Detect anomalies based on Z-Score and relative deviation.
        
        Args:
            group_keys: Keys to group by for anomaly detection
            anomaly_key: Column name for anomaly flag
            zscore_threshold: Z-Score threshold for anomaly detection
            min_std: Minimum standard deviation to avoid extreme small values
            relative_threshold: Relative deviation threshold for anomaly detection
        """
        # Filter out groups with size 1
        group_sizes = self.total_trace_df.groupby(group_keys).size()
        valid_groups = group_sizes[group_sizes > 1].index

        # Apply the filter to the DataFrame
        filtered_df = self.total_trace_df[
            self.total_trace_df.set_index(group_keys).index.isin(valid_groups)
        ]

        # Calculate the mean and standard deviation for each group
        mean_std_df = filtered_df.groupby(group_keys)['dur'].agg(
            ['mean', 'std']
        ).reset_index()

        # Set a minimum standard deviation to avoid extremely small values
        mean_std_df['std'] = mean_std_df['std'].apply(
            lambda x: max(x, min_std)
        )

        # Merge mean and std with the original dataframe
        self.total_trace_df = self.total_trace_df.merge(
            mean_std_df, on=group_keys, suffixes=('', '_group')
        )

        # Calculate Z-Score
        self.total_trace_df['z_score'] = (
            self.total_trace_df['dur'] - self.total_trace_df['mean']
        ) / self.total_trace_df['std']

        # Calculate relative deviation
        self.total_trace_df['relative_deviation'] = (
            self.total_trace_df['dur'] - self.total_trace_df['mean']
        ).abs() / self.total_trace_df['mean']

        # Mark anomalies based on Z-Score and relative deviation
        self.total_trace_df[anomaly_key] = (
            (self.total_trace_df['z_score'].abs() > zscore_threshold) | 
            (self.total_trace_df['relative_deviation'] > relative_threshold)
        )
        self.anomaly_status = self.total_trace_df[anomaly_key]

    def plot_anomalies_between_pp_group(self, output_dir: str = ''):
        """
        Plot anomalies between pipeline parallel groups.
        
        Args:
            output_dir: Directory to save plots
        """
        if self.anomaly_status is None:
            logger.info("Anomalies have not been detected. Please run detect_anomalies first.")
            return

        group_keys = ['full_name', 'pp_stage']
        anomaly_key = 'is_anomaly_between_pp_group'

        spans_with_anomalies = self.total_trace_df[self.anomaly_status].groupby(
            group_keys
        ).any().reset_index()
        num_spans_with_anomalies = spans_with_anomalies[anomaly_key].sum()

        if num_spans_with_anomalies == 0:
            logger.info("No anomalies found.")
            return

        for _, row in spans_with_anomalies.iterrows():
            if row[anomaly_key]:
                subset_condition = True
                for key in group_keys:
                    subset_condition &= (self.total_trace_df[key] == row[key])

                subset = self.total_trace_df[subset_condition].sort_values(by='rank')

                # Create a new figure and axis object
                fig, ax = plt.subplots(figsize=(12, 6))

                # Check if all values in 'is_anomaly' are the same
                if subset[anomaly_key].all() or not subset[anomaly_key].any():
                    default_color = 'red'
                    sns.barplot(x='rank', y='dur', data=subset, color=default_color, ax=ax)
                else:
                    sns.barplot(
                        x='rank', y='dur', data=subset, hue=anomaly_key, 
                        dodge=False, palette={True: 'red', False: 'blue'}, 
                        ax=ax, legend=False
                    )

                ax.set_title(f'{row["full_name"]} - pp stage {row["pp_stage"]}')
                ax.set_xticks(range(len(subset)))
                ax.set_xticklabels(subset['rank'].astype(str), rotation=45)
                ax.set_ylabel('Duration')
                plt.tight_layout()

                filename = f"{row['full_name']}_stage{row['pp_stage']}.png"
                filename = os.path.join(output_dir, filename)
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                plt.savefig(filename)
                plt.close(fig)

    def plot_anomalies_between_layers(self, output_dir: str = ''):
        """
        Plot anomalies between transformer layers.
        
        Args:
            output_dir: Directory to save plots
        """
        if self.anomaly_status is None:
            logger.info("Anomalies have not been detected. Please run detect_anomalies first.")
            return

        group_keys = ['full_name_without_index']
        anomaly_key = 'is_anomaly_between_layers'

        spans_with_anomalies = self.total_trace_df[self.anomaly_status].groupby(
            group_keys
        ).any().reset_index()
        num_spans_with_anomalies = spans_with_anomalies[anomaly_key].sum()

        if num_spans_with_anomalies == 0:
            logger.info("No anomalies found.")
            return

        for _, row in spans_with_anomalies.iterrows():
            if row[anomaly_key]:
                subset_condition = True
                for key in group_keys:
                    subset_condition &= (self.total_trace_df[key] == row[key])

                subset = self.total_trace_df[subset_condition].copy()

                # Extract indices
                subset['layer_index'] = subset['full_name'].str.extract(
                    r'ParallelTransformerLayer_(\d+)', expand=False
                )
                subset['forward_step_index'] = subset['full_name'].str.extract(
                    r'forward_step_(\d+)', expand=False
                )
                subset['backward_step_index'] = subset['full_name'].str.extract(
                    r'backward_step_(\d+)', expand=False
                )

                subset['forward_step_index'].fillna(-1, inplace=True)
                subset['backward_step_index'].fillna(-1, inplace=True)
                subset['layer_index'].fillna(-1, inplace=True)

                def custom_sort(row):
                    if row['forward_step_index'] != -1:
                        return (int(row['forward_step_index']), int(row['layer_index']), 0)
                    elif row['backward_step_index'] != -1:
                        return (int(row['backward_step_index']), int(row['layer_index']), 1)
                    else:
                        return (float('inf'), float('inf'), float('inf'))

                subset['sorting_key'] = subset.apply(custom_sort, axis=1)
                subset = subset.sort_values(by=['sorting_key'])

                subset['x_label'] = subset.apply(
                    lambda row: (
                        f"f{row['forward_step_index']}"
                        if 'forward_step' in row['full_name']
                        else f"b{row['backward_step_index']}"
                    ) + (f"l{row['layer_index']}" if row['layer_index'] != -1 else ""),
                    axis=1
                )

                fig, ax = plt.subplots(figsize=(12, 6))

                if subset[anomaly_key].all() or not subset[anomaly_key].any():
                    default_color = 'red'
                    sns.barplot(x='x_label', y='dur', data=subset, color=default_color, ax=ax)
                else:
                    sns.barplot(
                        x='x_label', y='dur', data=subset, hue=anomaly_key, 
                        dodge=False, palette={True: 'red', False: 'blue'}, 
                        ax=ax, legend=False
                    )

                ax.set_title(f'{row["full_name_without_index"]}')
                ax.set_xticks(range(len(subset)))
                ax.set_xticklabels(subset['x_label'].astype(str), rotation=45)
                ax.set_ylabel('Duration')
                plt.tight_layout()

                filename = f"{row['full_name_without_index']}.png"
                filename = os.path.join(output_dir, filename)
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                plt.savefig(filename)
                plt.close(fig)

    def etl_single_pp_group(self, pp_group_id: int, target_trace_dir: str, filter_out_funcs):
        """
        ETL for a single pipeline parallel group.
        
        Args:
            pp_group_id: Pipeline parallel group ID
            target_trace_dir: Target trace directory
            filter_out_funcs: Functions to filter out traces
        """
        logger.info(f"pp_group_id: {pp_group_id}, in target trace dir: {target_trace_dir}")
        t = MegatronPipelineParallelGroupTraceBase(
            None, self.trace_dir, 
            dp=self.expert_data_parallel_size, 
            tp=self.tensor_parallel_size, 
            pp=self.pipeline_parallel_size, 
            ep=self.expert_model_parallel_size, 
            cp=self.context_parallel_size
        )
        t.etl_traces_per_pp_group(target_trace_dir, filter_out_funcs, pp_group_id=pp_group_id)

    def pp_etl(self, target_trace_dir: str, filter_out_funcs):
        """
        ETL for all assigned pipeline parallel groups.

        Args:
            target_trace_dir: Target trace directory
            filter_out_funcs: Functions to filter out traces
        """
        tasks = list(enumerate(self.all_pp_group_sub_dirs))
        assigned_tasks = self._tasks_for_worker(tasks, self.rank, self.world_size)
        for pp_group_id, _ in assigned_tasks:
            logger.info(f'ETL pp group {pp_group_id}')
            self.etl_single_pp_group(pp_group_id, target_trace_dir, filter_out_funcs)

    def post_process(self):
        """Post-processing after analysis completion."""
        self.comm.Barrier()
        if self.rank == 0:
            self.move_output_dir()

    def move_output_dir(self):
        """Move output directories from pp_group subdirs to main output dir."""
        src_dir = self.trace_dir_pp_group
        dest_dir = self.output_dir

        output_dirs = []
        for root, dirs, files in os.walk(src_dir):
            for dir_name in dirs:
                if dir_name == 'output':
                    output_dir_path = os.path.join(root, dir_name)
                    output_dirs.append(output_dir_path)

        for output_dir in output_dirs:
            parent_dir_name = os.path.basename(os.path.dirname(output_dir))
            dest_subpath = os.path.join(dest_dir, parent_dir_name)
            os.makedirs(dest_subpath, exist_ok=True)
            shutil.move(output_dir, dest_subpath)
            logger.info(f"Moved: {output_dir} -> {dest_subpath}")

        logger.info("All 'output' directories have been moved.")
