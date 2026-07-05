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
    # symbolic dims -> 1, except the leading (batch) dim, which honors
    # BENCH_BATCH (default 1) so batch sweeps need no per-batch model files
    import os as _os
    _b = int(_os.environ.get("BENCH_BATCH", "1"))
    shape = []
    for _j, d in enumerate(inp.shape):
        if d is None or isinstance(d, str):
            shape.append(_b if _j == 0 else 1)
        else:
            shape.append(d)
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

        if provider_key == "trt":
            # TensorRT EP: build an fp16 engine (the deployment-realistic,
            # fused/optimized path) and cache it so repeat cells reuse it.
            # Falls back to CUDA EP for ops TRT cannot place.
            import os
            cache = os.environ.get("TRT_CACHE", "/tmp/trt_cache")
            os.makedirs(cache, exist_ok=True)
            providers = [
                ("TensorrtExecutionProvider", {
                    "trt_fp16_enable": True,
                    "trt_engine_cache_enable": True,
                    "trt_engine_cache_path": cache,
                }),
                ("CUDAExecutionProvider", {}),
            ]
        else:
            providers = [provider]

        self.sess = ort.InferenceSession(str(model_path), sess_options=so,
                                         providers=providers)
        actual = self.sess.get_providers()[0]
        if actual != provider:
            # Hard error, not a warning: a silent CPU-EP fallback overnight
            # writes complete, plausible-looking JSONs measured on the wrong
            # device, and campaign resume logic then keeps them forever.
            raise RuntimeError(f"requested {provider} but session uses {actual}")
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
    if kind == "onnxruntime-trt":
        return OnnxRuntimeBackend(model_path, "trt")
    if kind == "baseline":
        return YourBaselineBackend(model_path)
    raise ValueError(f"unknown backend: {kind}")
