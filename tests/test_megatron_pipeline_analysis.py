"""
Test cases for parse_megatron.py with different PP_SCHEDULE configurations.

This module verifies that DistributedMegatronTraceAnalysis produces the
expected report CSV for supported pipeline parallel schedules.
"""

import os
import sys
import unittest
from typing import Any, Dict

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from megatron_parallel_analysis.distribute_trace_analysis import DistributedMegatronTraceAnalysis

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

    def _prepare_dataset(self, dataset_name: str):
        dataset_info = get_dataset_info(dataset_name)
        trace_dir = os.path.join(self.megatron_trace_dir, dataset_name)
        expected_csv_path = os.path.join(trace_dir, dataset_info['expected_csv_name'])
        
        if not check_dataset_exists(trace_dir):
            download_and_extract_dataset(dataset_name, self.megatron_trace_dir)

        if not check_expected_csv_exists(expected_csv_path):
            download_expected_csv(dataset_name, self.megatron_trace_dir)

        self.assertTrue(
            check_dataset_exists(trace_dir),
            f"Dataset '{dataset_name}' is incomplete after preparation: {trace_dir}",
        )
        self.assertTrue(
            os.path.exists(expected_csv_path),
            f"Expected report CSV not found: {expected_csv_path}",
        )
        
        return dataset_info, trace_dir, expected_csv_path

    def _run_analysis_and_compare(self, dataset_name: str):
        dataset_info, trace_dir, expected_csv_path = self._prepare_dataset(dataset_name)

        analysis_kwargs = dict(
            trace_dir=trace_dir,
            tp=dataset_info['tp_size'],
            ep=dataset_info['ep_size'],
            dp=dataset_info['dp_size'],
            pp=dataset_info['pp_size'],
            pp_schedule=dataset_info['schedule'],
            micro_bs=dataset_info['micro_batchsize'],
        )
        if dataset_info['vpp_size'] is not None:
            analysis_kwargs['vpp_size'] = dataset_info['vpp_size']

        dist_megatron_analysis = DistributedMegatronTraceAnalysis(**analysis_kwargs)
        dist_megatron_analysis.analyze(pp_group_id_range=(0, 0))

        generated_csv_path = os.path.join(
            'workspace',
            dataset_name,
            'trace',
            'report-pp0.csv',
        )
        self.assertTrue(
            os.path.exists(generated_csv_path),
            f"Generated report CSV not found: {generated_csv_path}",
        )

        comparison_result = compare_csv_files(generated_csv_path, expected_csv_path)
        self.assertTrue(
            comparison_result['success'],
            f"CSV comparison failed. Differences: {comparison_result.get('differences', [])}",
        )

        generated_df = pd.read_csv(generated_csv_path)
        key_columns = [
            'rank',
            'time_per_iteration',
            'num_microbatch',
            'forward_step_avg_time',
            'backward_step_avg_time',
            'compute_time_total',
            'comm_time_total',
            'pipeline_parallel_size',
        ]
        for col in key_columns:
            self.assertIn(col, generated_df.columns, f"Missing key column: {col}")

        self.assertEqual(
            generated_df['pipeline_parallel_size'].iloc[0],
            dataset_info['pp_size'],
            'Pipeline parallel size mismatch',
        )
        self.assertEqual(
            generated_df['num_microbatch'].iloc[0],
            dataset_info['micro_batchsize'],
            'Microbatch size mismatch',
        )

    def test_1f1b_analysis_results(self):
        self._run_analysis_and_compare('1f1b')

    def test_1f1b_interleaved_analysis_results(self):
        self._run_analysis_and_compare('1f1b-interleaved')


if __name__ == '__main__':
    unittest.main()
