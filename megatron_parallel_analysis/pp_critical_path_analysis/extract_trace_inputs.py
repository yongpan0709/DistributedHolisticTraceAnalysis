"""Extract a compact DHTA snapshot for iterative_critical_path.py.

Run locally with --ssh pan to read remote source files, or omit --ssh when the
source files are local. --output updates only --key and preserves other datasets.
Without --output, emit JSON on stdout. Python standard library only.

This reproduces the original inline extraction: keep every complete (ph=X)
event, project fields without changing ts/dur, and normalize nonfinite args to
null. No DAG construction or hypothetical duration updates occur here.
"""
import argparse
import hashlib
import json
import math
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path


EVENT_FIELDS = ("name", "cat", "ts", "dur", "rank", "tid")
ARG_FIELDS = {
    "iteration", "kernel_span", "kernel_dur_sum", "first_kernel_start",
    "last_kernel_end", "vpp_stage_id", "micro_batch_id", "index", "end",
    "num_kernels",
}


def extract(trace_path, csv_path=None):
    path = Path(trace_path).expanduser().resolve()
    raw = path.read_bytes()
    source = json.loads(raw)
    events = []
    for event in source["traceEvents"]:
        if event.get("ph") != "X":
            continue
        projected = {key: event[key] for key in EVENT_FIELDS if key in event}
        projected["args"] = {
            key: None if isinstance(value, float) and not math.isfinite(value) else value
            for key, value in event.get("args", {}).items() if key in ARG_FIELDS
        }
        events.append(projected)
    snapshot = {
        "path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "mtime": path.stat().st_mtime,
        "events": events,
    }
    if csv_path is not None:
        snapshot["csv"] = Path(csv_path).expanduser().read_text(encoding="utf-8")
    return snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True, help="Source DHTA trace path")
    parser.add_argument("--key", required=True, help="Dataset key, e.g. 2026-08-04_pp36")
    parser.add_argument("--csv", help="Optional report CSV to retain; not needed for DAG analysis")
    parser.add_argument("--ssh", help="Read trace and CSV on this SSH host")
    parser.add_argument("--output", type=Path, help="Local output JSON; preserve other dataset keys")
    args = parser.parse_args()

    if args.ssh:
        if args.ssh.startswith("-"):
            parser.error("--ssh must name an SSH host")
        remote_args = ["python3", "-", "--trace", args.trace, "--key", args.key]
        if args.csv:
            remote_args.extend(["--csv", args.csv])
        completed = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=15", args.ssh, shlex.join(remote_args)],
            input=Path(__file__).read_text(encoding="utf-8"),
            text=True, capture_output=True,
        )
        if completed.returncode:
            raise RuntimeError(completed.stderr.strip() or "Remote extraction failed")
        payload = json.loads(completed.stdout)
    else:
        payload = {args.key: extract(args.trace, args.csv)}

    if args.output is None:
        print(json.dumps(payload, ensure_ascii=False, allow_nan=False))
        return

    output = args.output.expanduser().resolve()
    existing = json.loads(output.read_text(encoding="utf-8")) if output.exists() else {}
    if not isinstance(existing, dict) or any(
        not isinstance(value, dict) or "events" not in value
        for value in existing.values()
    ):
        raise ValueError("Existing output is not a trace_inputs dataset mapping")
    existing.update(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=output.parent,
            prefix="." + output.name + ".", suffix=".tmp", delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(existing, handle, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
        temporary.replace(output)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(f"Updated {args.key}: {len(payload[args.key]['events'])} events -> {output}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print(f"Extraction failed: {error}", file=sys.stderr)
        sys.exit(1)
