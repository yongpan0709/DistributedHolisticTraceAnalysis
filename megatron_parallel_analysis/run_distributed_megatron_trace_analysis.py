import argparse
import sys
import traceback


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze Megatron traces with configurable parallelism and schedule parameters.",
    )
    parser.add_argument("--trace-dir", required=True, help="trace directory")
    parser.add_argument("--tp", type=int, default=1, help="tensor parallel size")
    parser.add_argument("--pp", type=int, default=2, help="pipeline parallel size")
    parser.add_argument(
        "--dp",
        type=int,
        default=8,
        help="total data parallel size (must be divisible by expert parallel size)",
    )
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
        "--microbatch-group-size-per-virtual-pipeline-stage",
        type=int,
        default=None,
        help="contiguous micro-batches executed per virtual pipeline stage; defaults to PP size",
    )
    parser.add_argument(
        "--enable_ep_analysis",
        action="store_true",
        help="enable EP analysis for MoE models (disabled by default)",
    )
    parser.add_argument(
        "--pp-group-id-range",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        default=None,
        help="inclusive pipeline parallel group id range",
    )
    parser.add_argument(
        "--rebuild-parse-cache",
        action="store_true",
        help="rebuild caches for ranks analyzed in this run",
    )
    return parser.parse_args()


def main():
    from megatron_parallel_analysis.distribute_trace_analysis import DistributedMegatronTraceAnalysis

    args = parse_args()
    pp_group_id_range = None
    if args.pp_group_id_range is not None:
        pp_group_id_range = tuple(args.pp_group_id_range)
    try:
        dist_megatron_analysis = DistributedMegatronTraceAnalysis(
            trace_dir=args.trace_dir,
            tp=args.tp,
            ep=args.ep,
            dp=args.dp,
            pp=args.pp,
            pp_schedule=args.pp_schedule,
            vpp_size=args.vpp,
            micro_bs=args.num_bs,
            microbatch_group_size_per_vp_stage=args.microbatch_group_size_per_virtual_pipeline_stage,
            enable_ep_analysis=args.enable_ep_analysis,
            rebuild_parse_cache=args.rebuild_parse_cache,
        )
        dist_megatron_analysis.analyze(pp_group_id_range=pp_group_id_range)
    except Exception:
        print(
            'Distributed Megatron trace analysis failed: '
            f'trace_dir={args.trace_dir}, '
            f'pp_group_id_range={pp_group_id_range}, '
            f'mpi_args=(tp={args.tp}, pp={args.pp}, dp={args.dp}, ep={args.ep}, '
            f'num_bs={args.num_bs}, vpp={args.vpp}, schedule={args.pp_schedule})',
            file=sys.stderr,
        )
        traceback.print_exc()
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
