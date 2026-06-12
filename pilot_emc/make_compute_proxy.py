#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Generate a compute-bound ONNX model — the opposite roofline anchor to
make_decode_proxy.py.

L chained square GEMMs ([N,N] x [N,N] fp16): arithmetic intensity ~ N/3
FLOP/byte (N=2048 -> ~680), far above Orin's compute/bandwidth ratio, so
latency tracks GPU clock and should be nearly EMC-invariant. Together with
the GEMV decode proxy (AI ~ 1) this brackets every real workload on the
roofline; the EMC pilot showed batch-1 real models (mobilenetv2 +11%,
vit_small +18% over the 2133->3199 MHz range) all lean memory-bound, so a
synthetic anchor is the only clean compute-bound control available.

Weights are scaled by 1/sqrt(N) to keep activations finite in fp16.

Run with a python that has `onnx` (vit_export venv):
  ~/mobile/rt-infer-bench/vit_export/bin/python3 pilot_emc/make_compute_proxy.py
"""

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def build(n: int, layers: int, shared_weight: bool) -> onnx.ModelProto:
    rng = np.random.default_rng(seed=42)
    scale = 1.0 / np.sqrt(n)

    inits, nodes = [], []
    if shared_weight:
        # One L2-resident weight reused every layer: DRAM traffic ~0, so the
        # workload is compute-bound in REALIZED terms, not just nominal AI.
        # (v1 lesson: a 2048^3 GEMM has nominal AI ~683 but cuBLAS panel
        # re-reads against a few-MB L2 drop realized AI to ~65 — bandwidth-
        # bound again. 6.2 TFLOPS measured = 81GB/s x 65 exactly.)
        inits.append(numpy_helper.from_array(
            (rng.standard_normal((n, n)) * scale).astype(np.float16), "W"))
    prev = "x"
    for i in range(layers):
        w_name = "W" if shared_weight else f"W{i}"
        if not shared_weight:
            w = (rng.standard_normal((n, n)) * scale).astype(np.float16)
            inits.append(numpy_helper.from_array(w, w_name))
        out = f"h{i}" if i < layers - 1 else "y"
        nodes.append(helper.make_node("MatMul", [prev, w_name], [out]))
        prev = out

    graph = helper.make_graph(
        nodes, "compute_proxy",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT16, [n, n])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT16, [n, n])],
        initializer=inits,
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17)],
        producer_name="jetson-latency-lab/pilot_emc",
    )
    model.ir_version = 10
    onnx.checker.check_model(model)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--layers", type=int, default=64)
    ap.add_argument("--no-shared-weight", action="store_true",
                    help="v1 behavior: per-layer weights (streams from DRAM)")
    ap.add_argument("--out", default=str(Path(__file__).parent /
                                         "models/compute_proxy_fp16.onnx"))
    args = ap.parse_args()

    shared = not args.no_shared_weight
    gflop = args.layers * 2 * args.n ** 3 / 1e9
    nw = 1 if shared else args.layers
    mb = nw * args.n * args.n * 2 / 2 ** 20
    print(f"building {args.layers}x GEMM {args.n}^3 fp16, "
          f"{'shared' if shared else 'per-layer'} weights: "
          f"{gflop:.1f} GFLOP/inference, {mb:.0f} MiB weights")
    model = build(args.n, args.layers, shared)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(out))
    print(f"wrote {out} ({out.stat().st_size / 2**20:.0f} MiB)")


if __name__ == "__main__":
    main()
