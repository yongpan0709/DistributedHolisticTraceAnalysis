import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze Megatron traces with configurable parallelism and schedule parameters.",
    )
    parser.add_argument("--trace-dir", required=True, help="trace directory")
    parser.add_argument("--tp", type=int, default=1, help="tensor parallel size")
    parser.add_argument("--pp", type=int, default=2, help="pipeline parallel size")
    parser.add_argument("--dp", type=int, default=1, help="data parallel size")
    parser.add_argument("--ep", type=int, default=8, help="expert parallel size")
    parser.add_argument(
        "--pp-schedule",
        choices=["1f1b", "1f1b-interleaved", "1f1b-interleaved-epoverlap"],
        default="1f1b",
        help="pipeline parallel schedule",
    )
    parser.add_argument("--num-bs", type=int, default=16, help="number of micro batches")
    parser.add_argument("--vpp", type=int, default=2, help="virtual pipeline parallel size")
    parser.add_argument(
        "--pp-group-id-range",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        default=None,
        help="inclusive pipeline parallel group id range",
    )
    return parser.parse_args()


def main():
    from megatron_parallel_analysis.distribute_trace_analysis import DistributedMegatronTraceAnalysis

    args = parse_args()
    pp_group_id_range = None
    if args.pp_group_id_range is not None:
        pp_group_id_range = tuple(args.pp_group_id_range)
    dist_megatron_analysis = DistributedMegatronTraceAnalysis(
        args.trace_dir,
        args.tp,
        args.ep,
        args.dp,
        args.pp,
        pp_schedule=args.pp_schedule,
        vpp_size=args.vpp,
        micro_bs=args.num_bs,
    )
    dist_megatron_analysis.analyze(pp_group_id_range=pp_group_id_range)


if __name__ == '__main__':
    main()
