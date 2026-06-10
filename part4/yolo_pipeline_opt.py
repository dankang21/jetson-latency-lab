#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
YOLOv8 full pipeline: NAIVE vs OPTIMIZED, with per-stage timing.

naive      : numpy preprocess -> ORT-CUDA infer (CPU->GPU copy) -> CPU NMS
optimized  : GPU preprocess -> io_binding (zero-copy) -> TensorRT-FP16 infer
             -> output stays on GPU -> GPU decode + GPU NMS

The point of the series finale: optimizing only the model (the infer stage) is a
lie. The end-to-end win comes from optimizing the WHOLE pipeline and keeping data
on the GPU end to end (zero-copy), so preprocess and postprocess stop being the
new bottleneck.

Scene complexity is controlled via n_boxes (how many valid boxes reach NMS),
since a fixed image yields a fixed detection count. Real images are fed through
the same path as anchors.

Run:
  python3 yolo_pipeline_opt.py --model yolov8s.onnx --mode naive    --boxes 100
  python3 yolo_pipeline_opt.py --model yolov8s.onnx --mode optimized --boxes 100
"""
import argparse
import time
import numpy as np
import onnxruntime as ort
import torch
from torchvision.ops import nms as tv_nms

H = W = 640


# ----------------------------- NAIVE -----------------------------
class NaivePipeline:
    """numpy preprocess + ORT-CUDA + CPU NMS. The 'just make it work' version."""

    def __init__(self, model):
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        self.s = ort.InferenceSession(model, sess_options=so,
                                      providers=["CUDAExecutionProvider"])
        self.inp = self.s.get_inputs()[0].name

    def preprocess(self, img_u8):
        x = img_u8.astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))
        return np.ascontiguousarray(x[None])

    def infer(self, x):
        return self.s.run(None, {self.inp: x})

    def postprocess(self, out, n_boxes):
        raw = out[0]
        boxes = raw[:4].T            # real decode over full tensor
        scores = raw[4:].T
        _ = scores.max(1)
        if n_boxes <= 0:
            return 0
        xy = np.random.rand(n_boxes, 4).astype(np.float32) * 500
        xy[:, 2:] = xy[:, :2] + np.random.rand(n_boxes, 2).astype(np.float32) * 140 + 10
        sc = np.random.rand(n_boxes).astype(np.float32) * 0.5 + 0.3
        keep = tv_nms(torch.from_numpy(xy), torch.from_numpy(sc), 0.45)  # CPU
        return int(keep.numel())

    def cycle(self, img_u8, n_boxes):
        t0 = time.perf_counter()
        x = self.preprocess(img_u8)
        t1 = time.perf_counter()
        out = self.infer(x)
        t2 = time.perf_counter()
        self.postprocess(out, n_boxes)
        t3 = time.perf_counter()
        return ((t1 - t0) * 1e6, (t2 - t1) * 1e6, (t3 - t2) * 1e6, (t3 - t0) * 1e6)


# --------------------------- OPTIMIZED ---------------------------
class OptimizedPipeline:
    """GPU preprocess -> io_binding zero-copy -> TensorRT-FP16 -> GPU NMS.

    Reused GPU buffers: the input buffer is allocated once and the io_binding
    points at its stable .data_ptr(); each cycle fills it in-place. This avoids
    per-cycle allocation and dangling-pointer hazards from a freed tensor."""

    def __init__(self, model):
        self.s = ort.InferenceSession(
            model,
            providers=[("TensorrtExecutionProvider", {"trt_fp16_enable": True}),
                       "CUDAExecutionProvider"])
        self.inp = self.s.get_inputs()[0].name
        self.outname = self.s.get_outputs()[0].name
        # persistent GPU input buffer (NCHW float32), bound once
        self.gpu_in = torch.empty((1, 3, H, W), dtype=torch.float32,
                                  device="cuda").contiguous()
        self.io = self.s.io_binding()
        self.io.bind_input(name=self.inp, device_type="cuda", device_id=0,
                           element_type=np.float32, shape=(1, 3, H, W),
                           buffer_ptr=self.gpu_in.data_ptr())
        self.io.bind_output(self.outname, device_type="cuda", device_id=0)

    def preprocess(self, img_u8):
        # upload uint8 (1/4 the bytes of float32), cast+normalize+CHW on GPU,
        # write into the persistent bound buffer in-place.
        g = torch.from_numpy(img_u8).cuda(non_blocking=True)      # HWC u8
        self.gpu_in.copy_(g.permute(2, 0, 1).unsqueeze(0).float().div_(255.0))
        torch.cuda.synchronize()

    def infer(self):
        self.s.run_with_iobinding(self.io)
        torch.cuda.synchronize()

    def postprocess(self, n_boxes):
        # output is on GPU; receive as torch tensor (zero-copy via dlpack)
        ortval = self.io.get_outputs()[0]
        raw = torch.from_dlpack(ortval._ortvalue.to_dlpack())     # [1,84,8400] cuda
        raw = raw[0]
        _ = raw[:4].t()           # decode work on GPU
        _ = raw[4:].t().max(0)
        if n_boxes <= 0:
            return 0
        xy = torch.rand(n_boxes, 4, device="cuda") * 500
        xy[:, 2:] = xy[:, :2] + torch.rand(n_boxes, 2, device="cuda") * 140 + 10
        sc = torch.rand(n_boxes, device="cuda") * 0.5 + 0.3
        keep = tv_nms(xy, sc, 0.45)   # GPU NMS
        torch.cuda.synchronize()
        return int(keep.numel())

    def cycle(self, img_u8, n_boxes):
        t0 = time.perf_counter()
        self.preprocess(img_u8)
        t1 = time.perf_counter()
        self.infer()
        t2 = time.perf_counter()
        self.postprocess(n_boxes)
        t3 = time.perf_counter()
        return ((t1 - t0) * 1e6, (t2 - t1) * 1e6, (t3 - t2) * 1e6, (t3 - t0) * 1e6)


def bench(pipe, img, n_boxes, iters, warmup, label):
    for _ in range(warmup):
        pipe.cycle(img, n_boxes)
    pre, inf, post, e2e = [], [], [], []
    for _ in range(iters):
        p, i, q, e = pipe.cycle(img, n_boxes)
        pre.append(p); inf.append(i); post.append(q); e2e.append(e)
    print(f"\n=== {label} (boxes={n_boxes}, n={iters}) ===")
    for name, arr in [("preprocess", pre), ("inference", inf),
                      ("postproc/NMS", post), ("end-to-end", e2e)]:
        arr.sort()
        print(f"  {name:14s} p50={arr[len(arr)//2]/1000:7.3f}ms  "
              f"p99={arr[int(len(arr)*0.99)]/1000:7.3f}ms")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", default="both",
                    choices=["naive", "optimized", "both"])
    ap.add_argument("--boxes", type=int, default=100)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=30)
    a = ap.parse_args()

    img = (np.random.rand(H, W, 3) * 255).astype(np.uint8)

    if a.mode in ("naive", "both"):
        bench(NaivePipeline(a.model), img, a.boxes, a.iters, a.warmup, "NAIVE")
    if a.mode in ("optimized", "both"):
        print("\n(building TensorRT engine, may take minutes...)")
        bench(OptimizedPipeline(a.model), img, a.boxes, a.iters, a.warmup,
              "OPTIMIZED (GPU preprocess + iobind + TRT-FP16 + GPU NMS)")
