# SPDX-License-Identifier: Apache-2.0
"""
Inference backends.

The default OnnxRuntimeBackend is a faithful copy of the inference path in your
baseline bench.py, so the stress runs are apples-to-apples with your published
baseline (mean 3.882 ms, p99.99 3.978 ms, 0 misses):

  * SessionOptions: ORT_ENABLE_ALL, intra=1, inter=1  (single ORT thread)
  * CUDAExecutionProvider
  * dummy input: first input, symbolic dims -> 1, seed=42 standard_normal f32
  * hot call: sess.run(None, {name: x})   <-- NOT io_binding (matches baseline)

Because this mirrors bench.py exactly, you do NOT need to touch
YourBaselineBackend; just use `--backend onnxruntime`. The baseline slot is kept
only if you later change your baseline inference and want to wire it directly.

A backend is anything with:
    .name : str
    .warmup(n: int) -> None
    .infer() -> None        # one inference, no Python-side allocation
"""

from __future__ import annotations

import sys

import numpy as np

PROVIDER_MAP = {
    "cuda": "CUDAExecutionProvider",
    "trt": "TensorrtExecutionProvider",
    "cpu": "CPUExecutionProvider",
}


def _make_dummy_input(sess):
    """Identical logic to bench.py make_dummy_input (fixed seed=42)."""
    inp = sess.get_inputs()[0]
    shape = [1 if (d is None or isinstance(d, str)) else d for d in inp.shape]
    dtype_map = {
        "tensor(float)": np.float32,
        "tensor(float16)": np.float16,
        "tensor(int64)": np.int64,
        "tensor(int32)": np.int32,
    }
    dtype = dtype_map.get(inp.type, np.float32)
    rng = np.random.default_rng(seed=42)
    if np.issubdtype(dtype, np.floating):
        x = rng.standard_normal(shape).astype(dtype)
    else:
        x = rng.integers(0, 1000, size=shape, dtype=dtype)
    return inp.name, x


class OnnxRuntimeBackend:
    """Faithful copy of bench.py's inference path."""

    def __init__(self, model_path: str, provider_key: str = "cuda"):
        import onnxruntime as ort

        provider = PROVIDER_MAP[provider_key]
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1

        self.sess = ort.InferenceSession(str(model_path), sess_options=so,
                                         providers=[provider])
        actual = self.sess.get_providers()[0]
        if actual != provider:
            sys.stderr.write(f"WARNING: requested {provider} got {actual}\n")
        self.name = f"onnxruntime-{provider_key}"

        self.input_name, self.x = _make_dummy_input(self.sess)
        self._feed = {self.input_name: self.x}      # built once, reused
        self._run = self.sess.run                    # bind out of the loop

    def warmup(self, n: int) -> None:
        for _ in range(n):
            self._run(None, self._feed)

    def infer(self) -> None:
        self._run(None, self._feed)


class YourBaselineBackend:
    """Optional: only needed if your baseline inference differs from the above."""

    def __init__(self, model_path: str):
        self.name = "baseline"
        raise NotImplementedError(
            "Use --backend onnxruntime (it already mirrors bench.py). "
            "Only wire this if you change your baseline inference."
        )

    def warmup(self, n: int) -> None:  # pragma: no cover
        ...

    def infer(self) -> None:  # pragma: no cover
        ...


def make_backend(kind: str, model_path: str):
    if kind == "onnxruntime":
        return OnnxRuntimeBackend(model_path)
    if kind == "baseline":
        return YourBaselineBackend(model_path)
    raise ValueError(f"unknown backend: {kind}")
