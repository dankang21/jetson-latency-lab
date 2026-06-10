#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
YOLOv8 full-pipeline cycle: preprocess -> inference -> postprocess(NMS).

This module defines ONE detection cycle with each stage individually timed, plus
a knob for scene complexity (number of valid boxes fed to NMS). It does not do
the RT loop -- that lives in run_yolo_pipeline.py. Keeping the cycle separate
lets us validate stage timing before wrapping it in SCHED_FIFO.

Why synthetic box injection (the `n_boxes` knob):
A fixed dummy image produces a fixed (often zero) detection count, so it cannot
exercise NMS across scene complexity. Real inference still runs (real GPU time),
but the postprocess stage receives a controlled N valid boxes. This isolates
"scene complexity -> NMS cost -> end-to-end" as a clean curve. Real images are
fed through the same path later as anchors.
"""
import time
import numpy as np
import onnxruntime as ort

try:
    import torch
    from torchvision.ops import nms as tv_nms
    _HAVE_TV = True
except Exception:
    _HAVE_TV = False


def make_session(model_path, backend="cuda", fp16=False):
    """backend: 'cuda' | 'tensorrt'. fp16 only affects tensorrt."""
    if backend == "tensorrt":
        provs = [("TensorrtExecutionProvider", {"trt_fp16_enable": bool(fp16)}),
                 "CUDAExecutionProvider"]
    else:
        provs = ["CUDAExecutionProvider"]
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    s = ort.InferenceSession(model_path, sess_options=so, providers=provs)
    return s


def preprocess(img_hwc_u8, size=640):
    """Standard YOLO preprocess: (already-resized HxWx3 uint8) -> 1x3xHxW f32 [0,1].
    We hand it an already-sized buffer so we measure the per-cycle cast/normalize/
    transpose cost, not disk decode. Returns contiguous float32 NCHW."""
    x = img_hwc_u8.astype(np.float32) / 255.0      # normalize
    x = np.transpose(x, (2, 0, 1))                 # HWC -> CHW
    x = np.ascontiguousarray(x[None])              # add batch, contiguous
    return x


def synth_boxes(n, dev="cpu"):
    """N valid boxes (xyxy) + scores, to feed NMS at a controlled scene complexity."""
    if n <= 0:
        if _HAVE_TV:
            return (torch.zeros((0, 4)), torch.zeros((0,)))
        return (np.zeros((0, 4), np.float32), np.zeros((0,), np.float32))
    xy = np.random.rand(n, 4).astype(np.float32) * 500
    xy[:, 2:] = xy[:, :2] + np.random.rand(n, 2).astype(np.float32) * 140 + 10
    sc = np.random.rand(n).astype(np.float32) * 0.5 + 0.3
    if _HAVE_TV:
        t = torch.from_numpy(xy); s = torch.from_numpy(sc)
        if dev == "cuda":
            t = t.cuda(); s = s.cuda()
        return t, s
    return xy, sc


def postprocess(raw_out, n_boxes, nms_dev="cpu", iou_th=0.45):
    """Decode + NMS. raw_out is the real model output (kept to measure the decode
    over the true 84x8400 tensor); n_boxes controls how many survive to NMS."""
    raw = raw_out[0]                       # [84, 8400]
    boxes = raw[:4].T                      # real decode work over full tensor
    scores = raw[4:].T
    _ = scores.max(1)                      # conf reduction (real cost)
    xy, sc = synth_boxes(n_boxes, dev=nms_dev)   # controlled scene complexity
    if n_boxes <= 0:
        return 0
    if _HAVE_TV:
        keep = tv_nms(xy, sc, iou_th)
        if nms_dev == "cuda":
            torch.cuda.synchronize()
        return int(keep.numel())
    return n_boxes


def run_cycle(sess, inp_name, img_u8, x_pre, n_boxes, nms_dev="cpu"):
    """One full cycle, returning per-stage microseconds.
    x_pre: caller may pass a prebuilt input to skip preprocess timing isolation;
    here we always time preprocess fresh from img_u8."""
    t0 = time.perf_counter()
    x = preprocess(img_u8)
    t1 = time.perf_counter()
    out = sess.run(None, {inp_name: x})
    t2 = time.perf_counter()
    postprocess(out, n_boxes, nms_dev=nms_dev)
    t3 = time.perf_counter()
    return ((t1 - t0) * 1e6, (t2 - t1) * 1e6, (t3 - t2) * 1e6, (t3 - t0) * 1e6)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--backend", default="cuda", choices=["cuda", "tensorrt"])
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--nms-dev", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--boxes", type=int, default=100)
    ap.add_argument("--iters", type=int, default=200)
    a = ap.parse_args()

    print(f"torchvision NMS: {_HAVE_TV}")
    s = make_session(a.model, a.backend, a.fp16)
    print("providers:", s.get_providers())
    inp = s.get_inputs()[0].name
    img = (np.random.rand(640, 640, 3) * 255).astype(np.uint8)

    if a.backend == "tensorrt":
        print("building TensorRT engine (minutes)...")
    for _ in range(30):
        run_cycle(s, inp, img, None, a.boxes, a.nms_dev)

    pre, inf, post, e2e = [], [], [], []
    for _ in range(a.iters):
        p, i, q, e = run_cycle(s, inp, img, None, a.boxes, a.nms_dev)
        pre.append(p); inf.append(i); post.append(q); e2e.append(e)
    for name, arr in [("preprocess", pre), ("inference", inf),
                      ("postproc/NMS", post), ("end-to-end", e2e)]:
        arr.sort()
        print(f"  {name:14s} p50={arr[len(arr)//2]/1000:.3f}ms  "
              f"p99={arr[int(len(arr)*0.99)]/1000:.3f}ms")
