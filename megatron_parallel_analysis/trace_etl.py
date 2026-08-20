import argparse
import json
import os
import re
from copy import deepcopy
from functools import partial


# Todo: value missing in trace, like: '"Process Group Description": ,'
def fix_json_value_missing(file_path):
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as fr:
            json.load(fr)
    except json.JSONDecodeError:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as fr:
            s = fr.read()
        if s:
            s = re.sub(r':\s*(?=[,}])', ': ""', s)
            with open(file_path, "w", encoding="utf-8") as fw:
                fw.write(s)


FILTER_OUT_FUNCS = [
    r".*__init__.*",
    r".*__enter__.*",
    r".*__exit__.*",
    r"torch/.*__call__.*",
    r"transformer_engine/.*__call__.*",
    r"triton/.*__call__.*",
    r"torch/utils/data/_utils/pin_memory.py\(\d+\):.*",
    r"<built-in .*>",
    r"musaEventQuery",
    r"musaStreamIsCapturing",
    r"threading.py\(\d+\): .*",
    r"multiprocessing/.*\(\d+\): .*",
    r"torch/multiprocessing/.*\(\d+\): .*",
    r"torch/storage.py\(\d+\): .*",
    r"selectors.py\(\d+\): .*",
    r"socket.py\(\d+\): .*",
    r"hmac.py\(\d+\): .*",
    r"abc.py\(\d+\): .*",
]
COMBINED_PATTERN = "|".join(FILTER_OUT_FUNCS)

MOONCAKE_P2P_FUNCS = [
    "mooncake_p2p_recv_from",
    "mooncake_p2p_send_to",
]
MOONCAKE_P2P_PATTERN = "|".join(MOONCAKE_P2P_FUNCS)

FILTER_OUT_CAT_FUNCS = [
    "user_annotation",
    "gpu_user_annotation",
]

FILTER_OUT_FUNCS_FOR_EPOVERLAP = [
    r"Memcpy1 DtoH \(Device -> Pinned\)"
]

FILTER_OUT_FUNCS_FOR_EPOVERLAP_PATTERN = "|".join(FILTER_OUT_FUNCS_FOR_EPOVERLAP)

def filter_out_funcs(
    file_path,
    redirect_new_trace_path,
    combined_pattern=COMBINED_PATTERN,
    mooncake_p2p_pattern=MOONCAKE_P2P_PATTERN,
):
    print(f"redirect_new_trace_path: {redirect_new_trace_path}")
    #fix_json_value_missing(file_path)
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    dup_data = {}
    for key, value in data.items():
        if key == "traceEvents":
            dup_data.setdefault("traceEvents", [])
            for item in value:
                if "name" in item:
                    if re.match(combined_pattern, item["name"]):
                        continue
                    if re.match(mooncake_p2p_pattern, item["name"]):
                        item["cat"] = "user_annotation"
                dup_data["traceEvents"].append(deepcopy(item))
        else:
            dup_data[key] = deepcopy(value)

    with open(redirect_new_trace_path, "w", encoding="utf-8") as file:
        json.dump(dup_data, file, indent="\t")


def create_directory_if_not_exists(path):
    os.makedirs(path, exist_ok=True)
    return path


def positive_int(value):
    parsed_value = int(value)
    if parsed_value <= 0:
        raise argparse.ArgumentTypeError("value must be greater than 0")
    return parsed_value


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Filter trace events and run distributed Megatron trace ETL.",
        usage=(
            "mpirun -allow-run-as-root -np 2 --bind-to none --hostfile ./hostfile "
            "--map-by ppr:2:node --wdir /path/to/HTA "
            "python -m megatron_parallel_analysis.trace_etl "
            "--trace-dir <trace directory> --tp <tp size> --pp <pp size> "
            "--dp <dp size> --ep <ep size>"
        ),
    )
    parser.add_argument("--trace-dir", required=True, help="trace directory")
    parser.add_argument("--tp", type=int, required=True, help="tp size")
    parser.add_argument("--pp", type=int, required=True, help="pp size")
    parser.add_argument("--dp", type=int, required=True, help="dp size")
    parser.add_argument("--ep", type=int, required=True, help="ep size")
    parser.add_argument(
        "--pp-schedule",
        choices=["1f1b", "1f1b-interleaved", "1f1b-interleaved-epoverlap"],
        default="1f1b",
        help="pipeline parallel schedule",
    )
    parser.add_argument(
        "--etl-workers-per-rank",
        type=positive_int,
        default=8,
        help="maximum ETL subprocesses per MPI rank (default: 8)",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()

    from megatron_parallel_analysis.distribute_trace_analysis import (
        DistributedMegatronTraceAnalysis,
    )

    trace_dir = args.trace_dir.rstrip("/")
    redirect_path = create_directory_if_not_exists(trace_dir + "-etl")
    print(f"Origin Trace_dir: {trace_dir}, After filtering Dir: {redirect_path}")

    dist_megatron_analysis = DistributedMegatronTraceAnalysis(
        trace_dir, args.tp, args.ep, args.dp, args.pp
    )
    combined_pattern = COMBINED_PATTERN
    mooncake_p2p_pattern = MOONCAKE_P2P_PATTERN
    if args.pp_schedule == "1f1b-interleaved-epoverlap":
        combined_pattern = "|".join(
            [combined_pattern, FILTER_OUT_FUNCS_FOR_EPOVERLAP_PATTERN]
        )
    filter_out_funcs_with_pattern = partial(
        filter_out_funcs,
        combined_pattern=combined_pattern,
        mooncake_p2p_pattern=mooncake_p2p_pattern,
    )
    dist_megatron_analysis.pp_etl(
        redirect_path,
        filter_out_funcs_with_pattern,
        etl_workers_per_rank=args.etl_workers_per_rank,
    )
