# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Test data configuration for MUSA Megatron Pipeline tests.

This module contains metadata for test datasets used in megatron pipeline analysis tests.
The datasets are downloaded from remote URLs and extracted to the test data directory.
"""

import os
import tarfile
import urllib.request
import shutil
from typing import Dict, Optional


# Base URL for all test datasets
BASE_URL = "https://sh-repo.mthreads.com/repo/repository/mcc-ci-dependency/hta"


# Test dataset filenames (tgz files)
DATASET_FILES: Dict[str, str] = {
    "1f1b": "mccl-1f1b-tp1-pp4-dp2-ep8.tgz",
    "1f1b-interleaved": "mooncake-tp1-pp32-dp32-ep8-vpp2.tgz",
    # Future datasets will be added here:
    # "1f1b-interleaved-epoverlap": "...tgz",
}


# Expected CSV filenames for each dataset
EXPECTED_CSV_NAMES: Dict[str, str] = {
    "1f1b": "mccl-1f1b-tp1-pp4-dp2-ep8-expected_report-pp0-detail.csv",
    "1f1b-interleaved": "mooncake-tp1-pp32-dp32-ep8-vpp2-expected_report-pp1-detail.csv",
    # Future datasets will be added here:
    # "1f1b-interleaved-epoverlap": "...expected_report-pp0.csv",
}


MODEL_MAIN_STACK_TEST_DATASET: Dict[str, object] = {
    "name": "model_main_stack",
    "trace_filename": "epoverlap-drop-no-related-stream-rank16.json",
    "expected_txt_name": "kimi_epoverlap-16-expected-main-stack.txt",
    "rank": 16,
    "template": "kimi_epoverlap",
}

KERNEL_LEVEL_TEST_DATASETS: Dict[str, Dict[str, object]] = {
    "bf16": {
        "name": "kernel_level_bf16",
        "trace_filename": "kernel-level-bf16-rank8.json",
        "expected_txt_name": "kernel-level-bf16-expected-analysis-results.txt",
        "rank": 8,
        "output_filename": "kernel-level-rank8-fwdbwd.txt",
    },
    "fp8": {
        "name": "kernel_level_fp8",
        "trace_filename": "kernel-level-fp8-rank8.json",
        "expected_txt_name": "kernel-level-fp8-expected-analysis-results.txt",
        "rank": 8,
        "output_filename": "kernel-level-rank8-fwdbwd.txt",
    },
}

# Dataset metadata configuration
# Each entry contains:
# - schedule: PP_SCHEDULE type this dataset is used for
# - tp_size: Tensor parallel size
# - pp_size: Pipeline parallel size
# - dp_size: Data parallel size
# - ep_size: Expert parallel size
# - vpp_size: Virtual pipeline parallel size (optional, for interleaved schedules)
# - micro_batchsize: Micro batch size
# - description: Brief description of the dataset
MEGATRON_PIPELINE_TEST_DATASETS: Dict[str, Dict] = {
    "1f1b": {
        "schedule": "1f1b",
        "tp_size": 1,
        "pp_size": 4,
        "dp_size": 16,
        "ep_size": 8,
        "vpp_size": None,
        "micro_batchsize": 32,
        "description": "Megatron pipeline trace with 1f1b schedule (TP=1, PP=4, DP=16, EP=8)",
    },
    "1f1b-interleaved": {
        "schedule": "1f1b-interleaved",
        "tp_size": 1,
        "pp_size": 32,
        "dp_size": 32,
        "ep_size": 8,
        "vpp_size": 2,
        "micro_batchsize": 64,
        "description": "Megatron pipeline trace with 1f1b-interleaved schedule (TP=1, PP=32, DP=32, EP=8, VPP=2)",
    },
    # Future datasets will be added here:
    # "1f1b-interleaved-epoverlap": {
    #     "schedule": "1f1b-interleaved-epoverlap",
    #     "tp_size": 1,
    #     "pp_size": 4,
    #     "dp_size": 2,
    #     "ep_size": 8,
    #     "vpp_size": 2,
    #     "micro_batchsize": 32,
    #     "description": "Megatron pipeline trace with 1f1b-interleaved-epoverlap schedule",
    # },
}


def get_dataset_url(dataset_name: str) -> str:
    """
    Get the full URL for a dataset.
    
    Args:
        dataset_name: Name of the dataset (key in DATASET_FILES)
    
    Returns:
        Full URL to download the dataset
    """
    if dataset_name not in DATASET_FILES:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return f"{BASE_URL}/{DATASET_FILES[dataset_name]}"


def get_expected_csv_name(dataset_name: str) -> str:
    """
    Get the expected CSV filename for a dataset.
    
    Args:
        dataset_name: Name of the dataset (key in EXPECTED_CSV_NAMES)
    
    Returns:
        Expected CSV filename
    """
    if dataset_name not in EXPECTED_CSV_NAMES:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return EXPECTED_CSV_NAMES[dataset_name]


def get_expected_csv_url(dataset_name: str) -> str:
    """
    Get the full URL for the expected CSV of a dataset.

    Args:
        dataset_name: Name of the dataset (key in EXPECTED_CSV_NAMES)

    Returns:
        Full URL to download the expected CSV
    """
    return f"{BASE_URL}/{get_expected_csv_name(dataset_name)}"


def get_model_main_stack_trace_url() -> str:
    """Get the model main stack trace URL."""
    return f"{BASE_URL}/{MODEL_MAIN_STACK_TEST_DATASET['trace_filename']}"


def get_model_main_stack_expected_txt_url() -> str:
    """Get the model main stack expected text URL."""
    return f"{BASE_URL}/{MODEL_MAIN_STACK_TEST_DATASET['expected_txt_name']}"


def get_kernel_level_trace_url(dataset_name: str) -> str:
    """Get the kernel-level trace URL for a dataset."""
    if dataset_name not in KERNEL_LEVEL_TEST_DATASETS:
        raise ValueError(f"Unknown kernel-level dataset: {dataset_name}")

    return f"{BASE_URL}/{KERNEL_LEVEL_TEST_DATASETS[dataset_name]['trace_filename']}"


def get_kernel_level_expected_txt_url(dataset_name: str) -> str:
    """Get the kernel-level expected text URL for a dataset."""
    if dataset_name not in KERNEL_LEVEL_TEST_DATASETS:
        raise ValueError(f"Unknown kernel-level dataset: {dataset_name}")

    return f"{BASE_URL}/{KERNEL_LEVEL_TEST_DATASETS[dataset_name]['expected_txt_name']}"


def download_file(url: str, target_path: str, force_download: bool = False) -> Optional[str]:
    """Download a file if it does not exist."""
    if not force_download and os.path.exists(target_path):
        print(f"File already exists at {target_path}, skipping download")
        return target_path

    target_dir = os.path.dirname(target_path)
    if target_dir and not os.path.exists(target_dir):
        os.makedirs(target_dir)

    print(f"Downloading {url} to {target_path}...")
    try:
        urllib.request.urlretrieve(url, target_path)
        return target_path
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        if os.path.exists(target_path):
            os.remove(target_path)
        return None


def prepare_model_main_stack_dataset(
    target_dir: str,
    force_download: bool = False,
) -> Optional[Dict[str, str]]:
    """Download trace and expected text for model main stack tests."""
    dataset_dir = os.path.join(target_dir, MODEL_MAIN_STACK_TEST_DATASET["name"])
    trace_path = os.path.join(dataset_dir, MODEL_MAIN_STACK_TEST_DATASET["trace_filename"])
    expected_txt_path = os.path.join(dataset_dir, MODEL_MAIN_STACK_TEST_DATASET["expected_txt_name"])

    if not force_download and os.path.exists(trace_path) and os.path.exists(expected_txt_path):
        print(f"Model main stack dataset already exists at {dataset_dir}, skipping download")
        return {
            "trace_dir": dataset_dir,
            "trace_path": trace_path,
            "expected_txt_path": expected_txt_path,
        }

    trace_result = download_file(
        get_model_main_stack_trace_url(),
        trace_path,
        force_download=force_download,
    )
    expected_result = download_file(
        get_model_main_stack_expected_txt_url(),
        expected_txt_path,
        force_download=force_download,
    )

    if trace_result is None or expected_result is None:
        return None

    return {
        "trace_dir": dataset_dir,
        "trace_path": trace_path,
        "expected_txt_path": expected_txt_path,
    }


def prepare_kernel_level_dataset(
    dataset_name: str,
    target_dir: str,
    force_download: bool = False,
) -> Optional[Dict[str, str]]:
    """Download trace and expected text for a kernel-level statistics test dataset."""
    if dataset_name not in KERNEL_LEVEL_TEST_DATASETS:
        raise ValueError(f"Unknown kernel-level dataset: {dataset_name}")

    dataset = KERNEL_LEVEL_TEST_DATASETS[dataset_name]
    dataset_dir = os.path.join(target_dir, dataset["name"])
    trace_path = os.path.join(dataset_dir, dataset["trace_filename"])
    expected_txt_path = os.path.join(dataset_dir, dataset["expected_txt_name"])

    if not force_download and os.path.exists(trace_path) and os.path.exists(expected_txt_path):
        print(f"Kernel-level dataset '{dataset_name}' already exists at {dataset_dir}, skipping download")
        return {
            "trace_dir": dataset_dir,
            "trace_path": trace_path,
            "expected_txt_path": expected_txt_path,
        }

    trace_result = download_file(
        get_kernel_level_trace_url(dataset_name),
        trace_path,
        force_download=force_download,
    )
    expected_result = download_file(
        get_kernel_level_expected_txt_url(dataset_name),
        expected_txt_path,
        force_download=force_download,
    )

    if trace_result is None or expected_result is None:
        return None

    return {
        "trace_dir": dataset_dir,
        "trace_path": trace_path,
        "expected_txt_path": expected_txt_path,
    }


def check_dataset_exists(target_dir: str) -> bool:
    """
    Check if an extracted dataset directory exists with trace json files.

    Args:
        target_dir: Extracted dataset directory path

    Returns:
        True if the dataset directory exists and contains trace json files
    """
    if not os.path.exists(target_dir) or not os.path.isdir(target_dir):
        return False

    trace_files = [f for f in os.listdir(target_dir) if f.endswith('.json')]
    return bool(trace_files)


def check_expected_csv_exists(target_path: str) -> bool:
    """Check whether the expected CSV file exists."""
    return os.path.exists(target_path)


def download_expected_csv(dataset_name: str, target_dir: str, force_download: bool = False) -> Optional[str]:
    """
    Download the expected CSV for a dataset into its extracted directory.

    Args:
        dataset_name: Name of the dataset
        target_dir: Base directory containing extracted datasets
        force_download: If True, re-download even if the file already exists

    Returns:
        Path to the downloaded expected CSV, or None if download failed
    """
    if dataset_name not in MEGATRON_PIPELINE_TEST_DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    extracted_dir = os.path.join(target_dir, dataset_name)
    expected_csv_path = os.path.join(extracted_dir, get_expected_csv_name(dataset_name))

    if not os.path.exists(extracted_dir):
        os.makedirs(extracted_dir)

    if not force_download and os.path.exists(expected_csv_path):
        print(f"Expected CSV for '{dataset_name}' already exists at {expected_csv_path}, skipping download")
        return expected_csv_path

    url = get_expected_csv_url(dataset_name)
    print(f"Downloading expected CSV for '{dataset_name}' from {url}...")

    try:
        urllib.request.urlretrieve(url, expected_csv_path)
        print(f"Expected CSV downloaded to {expected_csv_path}")
        return expected_csv_path
    except Exception as e:
        print(f"Failed to download expected CSV for '{dataset_name}': {e}")
        if os.path.exists(expected_csv_path):
            os.remove(expected_csv_path)
        return None


def download_and_extract_dataset(
    dataset_name: str,
    target_dir: str,
    force_download: bool = False,
) -> Optional[str]:
    """
    Download and extract a test dataset.
    
    This function first checks if the dataset already exists with all required files
    (trace files and expected CSV). If all files exist, it skips the download.
    
    Args:
        dataset_name: Name of the dataset (key in MEGATRON_PIPELINE_TEST_DATASETS)
        target_dir: Directory to extract the dataset to
        force_download: If True, re-download even if dataset already exists
    
    Returns:
        Path to the extracted dataset directory, or None if download failed
    """
    if dataset_name not in MEGATRON_PIPELINE_TEST_DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    url = get_dataset_url(dataset_name)
    
    # Create target directory if it doesn't exist
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    # Check if dataset already exists with all required files
    extracted_dir = os.path.join(target_dir, dataset_name)
    if not force_download and check_dataset_exists(extracted_dir):
        print(f"Dataset '{dataset_name}' already exists at {extracted_dir}, skipping download")
        return extracted_dir
    
    # Download the dataset
    print(f"Downloading dataset '{dataset_name}' from {url}...")
    tar_path = os.path.join(target_dir, f"{dataset_name}.tgz")
    
    try:
        urllib.request.urlretrieve(url, tar_path)
        print(f"Downloaded to {tar_path}")
        
        # Extract the tarball
        print(f"Extracting {tar_path}...")
        with tarfile.open(tar_path, 'r:gz') as tar:
            # Get the top-level directory name in the tarball
            members = tar.getmembers()
            if members:
                # Extract all files
                tar.extractall(path=target_dir)
                
                # Find the extracted directory name
                top_level_dir = members[0].name.split('/')[0]
                original_extracted_path = os.path.join(target_dir, top_level_dir)
                
                # Rename to match expected dataset name if different
                if original_extracted_path != extracted_dir:
                    if os.path.exists(extracted_dir):
                        shutil.rmtree(extracted_dir)
                    shutil.move(original_extracted_path, extracted_dir)
        
        # Clean up the tar file
        os.remove(tar_path)
        print(f"Dataset extracted to {extracted_dir}")
        
        return extracted_dir
    
    except Exception as e:
        print(f"Failed to download/extract dataset '{dataset_name}': {e}")
        # Clean up partial downloads
        if os.path.exists(tar_path):
            os.remove(tar_path)
        return None


def get_dataset_info(dataset_name: str) -> Dict:
    """
    Get information about a test dataset.

    Args:
        dataset_name: Name of the dataset (key in MEGATRON_PIPELINE_TEST_DATASETS)

    Returns:
        Dictionary containing dataset metadata including URL, expected CSV name,
        and parallel configuration (TP, PP, DP, EP, VPP, micro_batchsize)
    """
    if dataset_name not in MEGATRON_PIPELINE_TEST_DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    info = MEGATRON_PIPELINE_TEST_DATASETS[dataset_name].copy()
    info["url"] = get_dataset_url(dataset_name)
    info["expected_csv_name"] = get_expected_csv_name(dataset_name)
    info["kind"] = "pipeline"

    return info


def get_kernel_level_dataset_info(dataset_name: str) -> Dict[str, object]:
    """Get metadata for a kernel-level statistics test dataset."""
    if dataset_name not in KERNEL_LEVEL_TEST_DATASETS:
        raise ValueError(f"Unknown kernel-level dataset: {dataset_name}")

    info = KERNEL_LEVEL_TEST_DATASETS[dataset_name].copy()
    info["trace_url"] = get_kernel_level_trace_url(dataset_name)
    info["expected_txt_url"] = get_kernel_level_expected_txt_url(dataset_name)
    info["kind"] = "kernel_level"
    return info


def list_available_datasets() -> list:
    """
    List all available test datasets.
    
    Returns:
        List of dataset names
    """
    return list(MEGATRON_PIPELINE_TEST_DATASETS.keys())