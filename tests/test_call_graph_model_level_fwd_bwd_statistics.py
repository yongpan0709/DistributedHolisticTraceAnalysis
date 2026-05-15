# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
Test cases for call_graph_model_level_fwd_bwd_statistics.py

This module tests the model-level forward/backward statistics analysis functionality.
"""
import os
import re
import sys
import unittest
from pathlib import Path

# Add musa_examples to path for imports
musa_examples_path = Path(__file__).parent
sys.path.insert(0, str(musa_examples_path))

from pytorch_callstack_analysis.call_graph_model_level_fwd_bwd_statistics import TEMPLATE_MAP, analyze_rank
from hta.common.trace_file import get_trace_files
from tests.data.musa_megatron_trace.dataset_config import (
    MODEL_MAIN_STACK_TEST_DATASET,
    prepare_model_main_stack_dataset,
)


class EndToEndTestCase(unittest.TestCase):
    """
    End-to-end test cases that run the actual script with test data.
    
    These tests require the trace data to be present in the expected location.
    """

    @classmethod
    def setUpClass(cls) -> None:
        """Set up test class with paths."""
        cls.trace_data_root = Path(__file__).parent.joinpath("data/musa_megatron_trace")
        cls.dataset_paths = prepare_model_main_stack_dataset(str(cls.trace_data_root))
        if cls.dataset_paths is None:
            cls.trace_dir = cls.trace_data_root.joinpath(MODEL_MAIN_STACK_TEST_DATASET["name"])
            cls.expected_output_path = cls.trace_dir.joinpath(
                MODEL_MAIN_STACK_TEST_DATASET["expected_txt_name"]
            )
        else:
            cls.trace_dir = Path(cls.dataset_paths["trace_dir"])
            cls.expected_output_path = Path(cls.dataset_paths["expected_txt_path"])
        cls.output_dir = cls.trace_data_root.joinpath("model_main_stack")

        expected_templates = ["default", "kimi_epoverlap"]
        for template in expected_templates:
            assert template in TEMPLATE_MAP

    def test_e2e_kimi_epoverlap_template(self) -> None:
        """
        End-to-end test for kimi_epoverlap template.

        This test runs the actual script and validates the output.
        Set environment variable RUN_E2E_TESTS=1 to enable this test.
        """
        if not self.trace_dir.exists():
            self.skipTest(f"Trace directory not found: {self.trace_dir}")

        if not self.expected_output_path.exists():
            self.skipTest(f"Expected output not found: {self.expected_output_path}")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir.joinpath(
            f"{MODEL_MAIN_STACK_TEST_DATASET['template']}-{MODEL_MAIN_STACK_TEST_DATASET['rank']}-main-stack.txt"
        )

        os.environ['HTA_DISABLE_NS_ROUNDING'] = '1'

        rank = MODEL_MAIN_STACK_TEST_DATASET["rank"]
        template_name = MODEL_MAIN_STACK_TEST_DATASET["template"]
        template = TEMPLATE_MAP[template_name]
        trace_files = get_trace_files(str(self.trace_dir))

        analyze_rank(
            rank=rank,
            trace_files=trace_files,
            template=template,
            template_name=template_name,
            output_path=str(output_path),
        )

        self.assertTrue(output_path.exists(), f"Output file was not generated: {output_path}")

        with open(output_path, 'r') as f:
            actual_content = f.read()

        with open(self.expected_output_path, 'r') as f:
            expected_content = f.read()

        self._compare_outputs(actual_content, expected_content)

    def _compare_outputs(self, actual: str, expected: str) -> None:
        """
        Compare actual and expected outputs.
        
        The comparison handles:
        1. Regex patterns in function names (e.g., \\d+ for line numbers)
        2. Floating point tolerance for statistics values
        3. NaN value handling
        """
        actual_lines = actual.strip().split('\n')
        expected_lines = expected.strip().split('\n')
        
        # Check line count
        self.assertEqual(
            len(actual_lines),
            len(expected_lines),
            f"Line count mismatch: actual={len(actual_lines)}, expected={len(expected_lines)}"
        )
        
        for i, (actual_line, expected_line) in enumerate(zip(actual_lines, expected_lines)):
            self._compare_line(actual_line, expected_line, i + 1)

    def _compare_line(self, actual: str, expected: str, line_num: int) -> None:
        """Compare a single line with tolerance for variations."""
        # Check if this is a statistics line
        if actual.strip().startswith(('fwd:', 'bwd:')):
            self._compare_stats_line(actual, expected, line_num)
        else:
            # Function name line - use regex matching
            self._compare_func_line(actual, expected, line_num)

    def _compare_stats_line(self, actual: str, expected: str, line_num: int) -> None:
        """Compare statistics line with tolerance."""
        # Pattern to extract values from stats line
        pattern = re.compile(
            r'^(\s*)(fwd|bwd):\s+'
            r'mean_percent:\s+([\d.]+|nan),\s+'
            r'mean:\s+([\d.]+|nan),\s+'
            r'q_25:\s+([\d.]+|nan),\s+'
            r'q_50:\s+([\d.]+|nan),\s+'
            r'q_75:\s+([\d.]+|nan),\s+'
            r'max:\s+([\d.]+|nan),\s+'
            r'min:\s+([\d.]+|nan),\s+'
            r'count:\s+([\d.]+|nan)\s*$'
        )
        
        actual_match = pattern.match(actual)
        expected_match = pattern.match(expected)
        
        self.assertIsNotNone(actual_match, f"Line {line_num}: Invalid actual stats format")
        self.assertIsNotNone(expected_match, f"Line {line_num}: Invalid expected stats format")
        
        # Compare indentation
        self.assertEqual(
            actual_match.group(1),
            expected_match.group(1),
            f"Line {line_num}: Indentation mismatch"
        )
        
        # Compare fwd/bwd label
        self.assertEqual(
            actual_match.group(2),
            expected_match.group(2),
            f"Line {line_num}: fwd/bwd label mismatch"
        )
        
        # Compare numeric values with tolerance
        tolerance = 0.01  # 1% tolerance
        for j, (actual_val, expected_val) in enumerate(zip(actual_match.groups()[2:], expected_match.groups()[2:])):
            if actual_val == 'nan' and expected_val == 'nan':
                continue
            elif actual_val == 'nan' or expected_val == 'nan':
                self.fail(f"Line {line_num}: NaN mismatch - actual={actual_val}, expected={expected_val}")
            else:
                actual_num = float(actual_val)
                expected_num = float(expected_val)
                if expected_num != 0:
                    relative_diff = abs(actual_num - expected_num) / expected_num
                    self.assertLessEqual(
                        relative_diff,
                        tolerance,
                        f"Line {line_num}: Value mismatch at position {j} - actual={actual_num}, expected={expected_num}"
                    )
                else:
                    self.assertAlmostEqual(
                        actual_num,
                        expected_num,
                        places=2,
                        msg=f"Line {line_num}: Value mismatch at position {j}"
                    )

    def _compare_func_line(self, actual: str, expected: str, line_num: int) -> None:
        """Compare function name line."""
        # Extract indentation
        actual_indent = len(actual) - len(actual.lstrip())
        expected_indent = len(expected) - len(expected.lstrip())
        
        self.assertEqual(
            actual_indent,
            expected_indent,
            f"Line {line_num}: Indentation mismatch"
        )
        
        # Get the function name part
        actual_func = actual.strip()
        expected_func = expected.strip()
        
        # Direct string comparison - the expected output file and actual output
        # should have identical content (both use literal \d+ notation)
        self.assertEqual(
            actual_func,
            expected_func,
            f"Line {line_num}: Function name mismatch"
        )

if __name__ == '__main__':
    unittest.main()