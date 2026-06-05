#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Periodic inference latency harness.

Runs a fixed-rate (default 100 Hz) inference loop and records, per cycle:

  release_jitter = wake-up time  - scheduled release   (scheduler/IRQ latency)
  compute        = inference done - wake-up time        (GPU + framework time)
  response       = inference done - scheduled release   (end-to-end; vs deadline)

A deadline miss is response > deadline (default deadline = period, i.e. an
implicit-deadline periodic task). Separating release_jitter from compute is the
whole point: under load, misses usually come from late wake-ups, not slow GPU.

Run as root for SCHED_FIFO/affinity/mlockall (see run_matrix.sh).

Example:
  sudo python3 -m harness.infer_bench \
      --backend onnxruntime --model models/mobilenetv2.onnx \
      --hz 100 --iters 100000 --warmup 2000 --cpu 5 --prio 80 \
      --label baseline --out results/baseline
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import sys
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from harness import rt_utils as rt          # noqa: E402
from harness.backends import make_backend   # noqa: E402


def _percentiles(arr_us: np.ndarray) -> dict:
    qs = [50, 90, 99, 99.9, 99.99]
    out = {f"p{q}": float(np.percentile(arr_us, q)) for q in qs}
    out["mean"] = float(arr_us.mean())
    out["std"] = float(arr_us.std())
    out["min"] = float(arr_us.min())
    out["max"] = float(arr_us.max())
    return out


def run(args) -> dict:
    backend = make_backend(args.backend, args.model)

    warnings = rt.try_apply_rt(priority=args.prio,
                               cpu=args.cpu,
                               lock_mem=not args.no_mlock)
    backend.warmup(args.warmup)

    n = args.iters
    period_ns = round(1_000_000_000 / args.hz)
    deadline_ns = period_ns if args.deadline_ms is None \
        else round(args.deadline_ms * 1_000_000)

    release_jitter = np.empty(n, dtype=np.int64)
    compute = np.empty(n, dtype=np.int64)
    response = np.empty(n, dtype=np.int64)

    # Anchor the schedule a few ms in the future so cycle 0 is not already late.
    start = rt.now_ns() + 5_000_000
    for i in range(n):
        scheduled = start + i * period_ns
        rt.sleep_until_ns(scheduled)
        wake = rt.now_ns()
        backend.infer()
        done = rt.now_ns()
        release_jitter[i] = wake - scheduled
        compute[i] = done - wake
        response[i] = done - scheduled

    misses = int(np.count_nonzero(response > deadline_ns))
    summary = {
        "meta": {
            "label": args.label,
            "backend": backend.name,
            "model": os.path.basename(args.model),
            "hz": args.hz,
            "period_us": period_ns / 1000,
            "deadline_us": deadline_ns / 1000,
            "iters": n,
            "warmup": args.warmup,
            "cpu": args.cpu,
            "prio": args.prio,
            "mlock": not args.no_mlock,
            "host": socket.gethostname(),
            "kernel": platform.release(),
            "python": platform.python_version(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "rt_warnings": warnings,
        },
        "deadline_miss_count": misses,
        "deadline_miss_rate": misses / n,
        "response_us": _percentiles(response / 1000.0),
        "compute_us": _percentiles(compute / 1000.0),
        "release_jitter_us": _percentiles(release_jitter / 1000.0),
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out + ".json", "w") as f:
        json.dump(summary, f, indent=2)
    if not args.no_raw:
        raw = np.column_stack([
            np.arange(n),
            release_jitter / 1000.0,
            compute / 1000.0,
            response / 1000.0,
            (response > deadline_ns).astype(np.int8),
        ])
        np.savetxt(args.out + ".csv", raw, delimiter=",",
                   header="cycle,release_jitter_us,compute_us,response_us,deadline_miss",
                   comments="", fmt=["%d", "%.3f", "%.3f", "%.3f", "%d"])

    r = summary["response_us"]
    print(f"[{args.label}] response p50={r['p50']:.3f} p99={r['p99']:.3f} "
          f"p99.99={r['p99.99']:.3f} max={r['max']:.3f} us | "
          f"misses={misses} ({summary['deadline_miss_rate']*100:.4f}%)")
    if warnings:
        for w in warnings:
            print(f"  ! {w}", file=sys.stderr)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="onnxruntime",
                    choices=["onnxruntime", "baseline"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--hz", type=float, default=100.0)
    ap.add_argument("--iters", type=int, default=100_000)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--deadline-ms", type=float, default=None,
                    help="default = period (implicit-deadline task)")
    ap.add_argument("--cpu", type=int, default=None,
                    help="CPU to pin to; should be an isolated core")
    ap.add_argument("--prio", type=int, default=80)
    ap.add_argument("--no-mlock", action="store_true")
    ap.add_argument("--no-raw", action="store_true",
                    help="skip per-cycle CSV (keep only summary JSON)")
    ap.add_argument("--label", default="run")
    ap.add_argument("--out", default="results/run")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
