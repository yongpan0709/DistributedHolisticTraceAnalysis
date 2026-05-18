import os
import re
import unittest
from pathlib import Path

from hta.common.trace_file import get_trace_files
from pytorch_callstack_analysis.kernel_level_fwdbwd_statistics import analyze_rank
from tests.data.musa_megatron_trace.dataset_config import (
    KERNEL_LEVEL_TEST_DATASETS,
    get_kernel_level_dataset_info,
    prepare_kernel_level_dataset,
)


class EndToEndTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.trace_data_root = Path(__file__).parent.joinpath("data/musa_megatron_trace")
        cls.output_dir = cls.trace_data_root.joinpath("kernel_level_fwdbwd_statistics")
        cls.dataset_paths = {}
        for dataset_name, dataset in KERNEL_LEVEL_TEST_DATASETS.items():
            dataset_paths = prepare_kernel_level_dataset(dataset_name, str(cls.trace_data_root))
            if dataset_paths is None:
                dataset_dir = cls.trace_data_root.joinpath(dataset["name"])
                cls.dataset_paths[dataset_name] = {
                    "trace_dir": str(dataset_dir),
                    "trace_path": str(dataset_dir.joinpath(dataset["trace_filename"])),
                    "expected_txt_path": str(dataset_dir.joinpath(dataset["expected_txt_name"])),
                }
            else:
                cls.dataset_paths[dataset_name] = dataset_paths

    def test_e2e_bf16(self) -> None:
        self._run_dataset("bf16")

    def test_e2e_fp8(self) -> None:
        self._run_dataset("fp8")

    def _run_dataset(self, dataset_name: str) -> None:
        dataset_info = get_kernel_level_dataset_info(dataset_name)
        dataset_paths = self.dataset_paths[dataset_name]
        trace_dir = Path(dataset_paths["trace_dir"])
        expected_output_path = Path(dataset_paths["expected_txt_path"])

        if not trace_dir.exists():
            self.skipTest(f"Trace directory not found: {trace_dir}")
        if not expected_output_path.exists():
            self.skipTest(f"Expected output not found: {expected_output_path}")

        dataset_output_dir = self.output_dir.joinpath(dataset_name)
        dataset_output_dir.mkdir(parents=True, exist_ok=True)
        output_path = dataset_output_dir.joinpath(dataset_info["output_filename"])

        os.environ["HTA_DISABLE_NS_ROUNDING"] = "1"

        trace_files = get_trace_files(str(trace_dir))
        analyze_rank(
            rank=dataset_info["rank"],
            trace_files=trace_files,
            output_path=str(output_path),
        )

        self.assertTrue(output_path.exists(), f"Output file was not generated: {output_path}")

        with open(output_path, "r") as f:
            actual_content = f.read()

        with open(expected_output_path, "r") as f:
            expected_content = f.read()

        self._compare_outputs(actual_content, expected_content)

    def _compare_outputs(self, actual: str, expected: str) -> None:
        actual_lines = actual.strip().split("\n")
        expected_lines = expected.strip().split("\n")

        self.assertEqual(
            len(actual_lines),
            len(expected_lines),
            f"Line count mismatch: actual={len(actual_lines)}, expected={len(expected_lines)}",
        )

        for idx, (actual_line, expected_line) in enumerate(zip(actual_lines, expected_lines), start=1):
            self._compare_line(actual_line, expected_line, idx)

    def _compare_line(self, actual: str, expected: str, line_num: int) -> None:
        if actual.strip().startswith(("fwd-", "bwd-")):
            self._compare_stats_line(actual, expected, line_num)
        else:
            self._compare_func_line(actual, expected, line_num)

    def _compare_stats_line(self, actual: str, expected: str, line_num: int) -> None:
        pattern = re.compile(
            r"^(\s*)(fwd-\d+|bwd-\d+) shape: (.+?),\s+mean_time\(us\):\s+([\d.]+|nan),\s+"
            r"(TFLOPS|BW) mean: ([\d.]+|nan) (TFLOPS|GB/s),\s+"
            r"q_25: ([\d.]+|nan), q_50: ([\d.]+|nan), q_75: ([\d.]+|nan), count: ([\d.]+|nan)\s*$"
        )

        actual_match = pattern.match(actual)
        expected_match = pattern.match(expected)

        self.assertIsNotNone(actual_match, f"Line {line_num}: Invalid actual stats format")
        self.assertIsNotNone(expected_match, f"Line {line_num}: Invalid expected stats format")

        self.assertEqual(actual_match.group(1), expected_match.group(1), f"Line {line_num}: Indentation mismatch")
        self.assertEqual(actual_match.group(2), expected_match.group(2), f"Line {line_num}: Phase label mismatch")
        self.assertEqual(actual_match.group(3), expected_match.group(3), f"Line {line_num}: Shape mismatch")
        self.assertEqual(actual_match.group(5), expected_match.group(5), f"Line {line_num}: Metric label mismatch")
        self.assertEqual(actual_match.group(7), expected_match.group(7), f"Line {line_num}: Metric unit mismatch")

        tolerance = 0.01
        for pos, (actual_val, expected_val) in enumerate(
            zip(
                (actual_match.group(4), actual_match.group(6), actual_match.group(8), actual_match.group(9), actual_match.group(10), actual_match.group(11)),
                (expected_match.group(4), expected_match.group(6), expected_match.group(8), expected_match.group(9), expected_match.group(10), expected_match.group(11)),
            ),
            start=1,
        ):
            if actual_val == "nan" and expected_val == "nan":
                continue
            if actual_val == "nan" or expected_val == "nan":
                self.fail(f"Line {line_num}: NaN mismatch at position {pos} - actual={actual_val}, expected={expected_val}")

            actual_num = float(actual_val)
            expected_num = float(expected_val)
            if expected_num != 0:
                relative_diff = abs(actual_num - expected_num) / abs(expected_num)
                self.assertLessEqual(
                    relative_diff,
                    tolerance,
                    f"Line {line_num}: Value mismatch at position {pos} - actual={actual_num}, expected={expected_num}",
                )
            else:
                self.assertAlmostEqual(
                    actual_num,
                    expected_num,
                    places=2,
                    msg=f"Line {line_num}: Value mismatch at position {pos}",
                )

    def _compare_func_line(self, actual: str, expected: str, line_num: int) -> None:
        actual_indent = len(actual) - len(actual.lstrip())
        expected_indent = len(expected) - len(expected.lstrip())

        self.assertEqual(actual_indent, expected_indent, f"Line {line_num}: Indentation mismatch")
        self.assertEqual(actual.strip(), expected.strip(), f"Line {line_num}: Function name mismatch")


if __name__ == "__main__":
    unittest.main()
