#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Generate a memory-bandwidth-bound ONNX model that mimics LLM decode.

Batch-1 decode is a chain of GEMVs: every step streams the full weight
matrix from DRAM to produce one token, so latency tracks memory bandwidth,
not compute. This proxy is exactly that — L layers of x(1xH) @ W(HxH) in
fp16 — with arithmetic intensity ~1 FLOP/byte, far below Orin's
compute/bandwidth ratio, so EMC frequency is the binding resource.

Why a proxy instead of a real SLM: no LLM runtime is installed on this
board, and the pilot's kill-criterion only needs a workload whose latency
is dominated by DRAM streaming. The real SLM goes in the main campaign.

Weights are scaled by 1/sqrt(H) so activations stay finite in fp16 across
the chain (NaN/inf wouldn't change timing on GPU, but clean numerics keep
the run comparable to real models).

Run with a python that has `onnx` (vit_export venv):
  ~/mobile/rt-infer-bench/vit_export/bin/python3 make_decode_proxy.py \
      --hidden 4096 --layers 12 --out models/decode_proxy_fp16.onnx
"""

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def build(hidden: int, layers: int) -> onnx.ModelProto:
    rng = np.random.default_rng(seed=42)
    scale = 1.0 / np.sqrt(hidden)

    inits, nodes = [], []
    prev = "x"
    for i in range(layers):
        w_name = f"W{i}"
        w = (rng.standard_normal((hidden, hidden)) * scale).astype(np.float16)
        inits.append(numpy_helper.from_array(w, w_name))
        out = f"h{i}" if i < layers - 1 else "y"
        nodes.append(helper.make_node("MatMul", [prev, w_name], [out]))
        prev = out

    graph = helper.make_graph(
        nodes, "decode_proxy",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT16, [1, hidden])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT16, [1, hidden])],
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
    ap.add_argument("--hidden", type=int, default=4096)
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--out", default=str(Path(__file__).parent /
                                         "models/decode_proxy_fp16.onnx"))
    args = ap.parse_args()

    nbytes = args.layers * args.hidden * args.hidden * 2
    print(f"building {args.layers}x{args.hidden}x{args.hidden} fp16 "
          f"({nbytes / 2**20:.0f} MiB of weights streamed per inference)")
    model = build(args.hidden, args.layers)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(out))
    print(f"wrote {out} ({out.stat().st_size / 2**20:.0f} MiB)")


if __name__ == "__main__":
    main()
