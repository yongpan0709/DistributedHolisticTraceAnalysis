"""
Test cases for DistributedMegatronTraceAnalysis with different PP_SCHEDULE configurations.

This module verifies that DistributedMegatronTraceAnalysis produces the
expected report CSV for supported pipeline parallel schedules.
"""

import os
import shutil
import sys
import unittest
from typing import Any, Dict

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from megatron_parallel_analysis.distribute_trace_analysis import DistributedMegatronTraceAnalysis
from megatron_parallel_analysis.utils.parallel_state import RankGenerator

from hta.utils.test_utils import get_test_data_dir
from tests.data.musa_megatron_trace.dataset_config import (
    check_dataset_exists,
    check_expected_csv_exists,
    download_and_extract_dataset,
    download_expected_csv,
    get_dataset_info,
)


def compare_csv_files(generated_path: str, expected_path: str, tolerance: float = 0.01) -> Dict[str, Any]:
    """Compare two CSV files and return comparison results."""
    if not os.path.exists(generated_path):
        return {'success': False, 'error': f'Generated file not found: {generated_path}'}

    if not os.path.exists(expected_path):
        return {'success': False, 'error': f'Expected file not found: {expected_path}'}

    generated_df = pd.read_csv(generated_path)
    expected_df = pd.read_csv(expected_path)

    if set(generated_df.columns) != set(expected_df.columns):
        return {
            'success': False,
            'error': f'Columns mismatch. Generated: {list(generated_df.columns)}, Expected: {list(expected_df.columns)}'
        }

    if len(generated_df) != len(expected_df):
        return {
            'success': False,
            'error': f'Row count mismatch. Generated: {len(generated_df)}, Expected: {len(expected_df)}'
        }

    differences = []
    for col in expected_df.columns:
        gen_values = generated_df[col].values
        exp_values = expected_df[col].values

        for i, (gen_val, exp_val) in enumerate(zip(gen_values, exp_values)):
            if isinstance(exp_val, str) or pd.isna(exp_val):
                if gen_val != exp_val and not (pd.isna(gen_val) and pd.isna(exp_val)):
                    differences.append({
                        'column': col,
                        'row': i,
                        'generated': gen_val,
                        'expected': exp_val,
                    })
            else:
                if abs(gen_val - exp_val) > tolerance * abs(exp_val) + 1e-6:
                    differences.append({
                        'column': col,
                        'row': i,
                        'generated': gen_val,
                        'expected': exp_val,
                        'diff_percent': abs(gen_val - exp_val) / abs(exp_val) * 100 if exp_val != 0 else 'inf',
                    })

    return {
        'success': len(differences) == 0,
        'differences': differences,
        'generated_rows': len(generated_df),
        'expected_rows': len(expected_df),
        'columns': list(generated_df.columns),
    }

class TestMegatronPipeline(unittest.TestCase):
    """Test Megatron pipeline analysis for supported PP schedules."""

    DATASET_NAMES = ('1f1b', '1f1b-interleaved')

    @classmethod
    def setUpClass(cls):
        cls.base_data_dir = get_test_data_dir()
        cls.megatron_trace_dir = os.path.join(
            cls.base_data_dir,
            'musa_megatron_trace',
        )
        os.makedirs(cls.megatron_trace_dir, exist_ok=True)

    @staticmethod
    def _get_expected_pp_group_ranks(dataset_info: Dict[str, Any], pp_group_id: int):
        expert_data_parallel_size = int(dataset_info['dp_size'] / dataset_info['ep_size'])
        rank_generator = RankGenerator(
            tp=dataset_info['tp_size'],
            ep=dataset_info['ep_size'],
            dp=expert_data_parallel_size,
            pp=dataset_info['pp_size'],
            cp=1,
            order="tp-cp-ep-dp-pp",
            rank_offset=0,
        )
        return rank_generator.get_ranks('pp')[pp_group_id]

    @staticmethod
    def _dataset_has_ranks(trace_dir: str, ranks) -> bool:
        if not check_dataset_exists(trace_dir):
            return False

        trace_files = os.listdir(trace_dir)
        return all(
            any(filename.startswith(f'rank{rank}.') for filename in trace_files)
            for rank in ranks
        )

    def _prepare_dataset(self, dataset_name: str, pp_group_id: int = 0):
        dataset_info = get_dataset_info(dataset_name)
        trace_dir = os.path.join(self.megatron_trace_dir, dataset_name)
        expected_csv_path = os.path.join(trace_dir, dataset_info['expected_csv_name'])
        expected_ranks = self._get_expected_pp_group_ranks(dataset_info, pp_group_id)
        if not self._dataset_has_ranks(trace_dir, expected_ranks):
            download_and_extract_dataset(
                dataset_name,
                self.megatron_trace_dir,
                force_download=True,
            )

        if not check_expected_csv_exists(expected_csv_path):
            download_expected_csv(dataset_name, self.megatron_trace_dir)

        self.assertTrue(
            self._dataset_has_ranks(trace_dir, expected_ranks),
            f"Dataset '{dataset_name}' is incomplete after preparation: {trace_dir}",
        )
        self.assertTrue(
            os.path.exists(expected_csv_path),
            f"Expected report CSV not found: {expected_csv_path}",
        )
        return dataset_info, trace_dir, expected_csv_path

    def _run_analysis_and_compare(self, dataset_name: str, pp_group_id: int = 0):
        dataset_info, trace_dir, expected_detail_csv_path = self._prepare_dataset(
            dataset_name,
            pp_group_id,
        )

        analysis_kwargs = dict(
            trace_dir=trace_dir,
            tp=dataset_info['tp_size'],
            ep=dataset_info['ep_size'],
            dp=dataset_info['dp_size'],
            pp=dataset_info['pp_size'],
            pp_schedule=dataset_info['schedule'],
            micro_bs=dataset_info['micro_batchsize'],
            microbatch_group_size_per_vp_stage=dataset_info.get(
                'microbatch_group_size_per_vp_stage'
            ),
        )
        if dataset_info['vpp_size'] is not None:
            analysis_kwargs['vpp_size'] = dataset_info['vpp_size']

        workspace_dataset_dir = os.path.join('workspace', dataset_name)
        if os.path.exists(workspace_dataset_dir):
            shutil.rmtree(workspace_dataset_dir)

        dist_megatron_analysis = DistributedMegatronTraceAnalysis(**analysis_kwargs)
        dist_megatron_analysis.analyze(pp_group_id_range=(pp_group_id, pp_group_id))

        generated_detail_csv_path = os.path.join(
            'workspace',
            dataset_name,
            'trace',
            f'report-pp{pp_group_id}-detail.csv',
        )

        self.assertTrue(
            os.path.exists(generated_detail_csv_path),
            f"Generated detail report CSV not found: {generated_detail_csv_path}",
        )
        self.assertTrue(
            os.path.exists(expected_detail_csv_path),
            f"Expected detail report CSV not found: {expected_detail_csv_path}",
        )

        comparison_result = compare_csv_files(generated_detail_csv_path, expected_detail_csv_path)
        self.assertTrue(
            comparison_result['success'],
            f"CSV comparison failed. Differences: {comparison_result.get('differences', [])}",
        )

    def test_1f1b_analysis_results(self):
        self._run_analysis_and_compare('1f1b')

    def test_1f1b_interleaved_analysis_results(self):
        self._run_analysis_and_compare('1f1b-interleaved', pp_group_id=0)


if __name__ == '__main__':
    unittest.main()
