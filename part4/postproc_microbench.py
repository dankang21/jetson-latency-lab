#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Postprocess micro-benchmark: where does YOLOv8 postprocess time actually go?

This isolates each postprocess step on a real model output so the cost of each
can be read separately. It is the measurement behind the "NMS is not the
bottleneck" table in the writeup:

    decode (max over 80 classes + transpose) : ~0.19 ms
    filter (confidence threshold, gather)    : ~0.46 ms
    coordinate transform (xywh -> xyxy)      : ~0.70 ms   <-- the real cost
    NMS (the algorithm)                      : ~0.20 ms

The headline: the coordinate transform (four small strided tensor ops) costs
several times the NMS algorithm, because for small tensors the per-op GPU
kernel-launch overhead dominates the actual work -- not the O(n^2) of NMS.

------------------------------------------------------------------------------
NOTE ON THE 'nms' COLUMN IN real_image_anchor.py
------------------------------------------------------------------------------
real_image_anchor.py reports an end-to-end pipeline, and its 'nms' column times
ONE fused region: coordinate transform + (optional .cpu() copy) + NMS + the
trailing cuda synchronize. That region is ~1.0 ms (CPU NMS) / ~1.2 ms (GPU NMS)
and is dominated by the coordinate transform and synchronization, NOT by NMS.

This micro-benchmark SPLITS that region. So the 'nms' here (~0.2 ms, the
algorithm alone) and the 'nms' column in real_image_anchor.py (~1.0 ms, the
whole fused region) measure DIFFERENT spans on purpose. They are consistent:
  real_anchor 'nms' (~1.0) ~= coord (0.70) + .cpu() copy (0.17) + NMS (0.20)
This script exists to make that decomposition reproducible.
------------------------------------------------------------------------------

Usage:
  python3 postproc_microbench.py --model yolov8s.onnx --image imgs/some.jpg
  python3 postproc_microbench.py --model yolov8s.onnx          # random input
"""
import argparse
import statistics as st
import time
import numpy as np
import onnxruntime as ort
import torch
from torchvision.ops import nms as tv_nms

H = W = 640


def make_session(model):
    return ort.InferenceSession(
        model,
        providers=[("TensorrtExecutionProvider", {"trt_fp16_enable": True}),
                   "CUDAExecutionProvider"])


def load_input(path):
    if path is None:
        return torch.rand(1, 3, H, W, device="cuda").contiguous()
    from PIL import Image
    im = np.ascontiguousarray(
        np.asarray(Image.open(path).convert("RGB").resize((W, H)), dtype=np.uint8)).copy()
    g = torch.from_numpy(im).cuda()
    return g.permute(2, 0, 1).unsqueeze(0).float().div_(255.0).contiguous()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--image", default=None,
                    help="real image; omit for random input (decode/filter/coord "
                         "costs are input-independent, NMS depends on box count)")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--nms-dev", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=30)
    a = ap.parse_args()

    s = make_session(a.model)
    inp = s.get_inputs()[0].name
    outn = s.get_outputs()[0].name
    gin = load_input(a.image)

    io = s.io_binding()
    io.bind_input(name=inp, device_type="cuda", device_id=0,
                  element_type=np.float32, shape=(1, 3, H, W),
                  buffer_ptr=gin.data_ptr())
    io.bind_output(outn, device_type="cuda", device_id=0)

    print(f"(TensorRT engine build on first run, may take minutes)  "
          f"input={'random' if a.image is None else a.image}  NMS on {a.nms_dev}\n")
    for _ in range(a.warmup):
        s.run_with_iobinding(io); torch.cuda.synchronize()

    D, F, C, P, N, filt = [], [], [], [], [], []
    for _ in range(a.iters):
        s.run_with_iobinding(io); torch.cuda.synchronize()
        ov = io.get_outputs()[0]
        raw = torch.from_dlpack(ov._ortvalue.to_dlpack())[0]   # [84,8400] cuda

        # decode
        torch.cuda.synchronize(); t0 = time.perf_counter()
        boxes = raw[:4].t(); scores = raw[4:].t(); conf, _ = scores.max(1)
        torch.cuda.synchronize(); t1 = time.perf_counter()
        # filter
        keep = conf > a.conf
        bb = boxes[keep]; cc = conf[keep]
        n = int(bb.shape[0])
        torch.cuda.synchronize(); t2 = time.perf_counter()
        # coordinate transform (xywh -> xyxy): four small strided ops
        if n > 0:
            xy = torch.empty_like(bb)
            xy[:, 0] = bb[:, 0] - bb[:, 2] / 2
            xy[:, 1] = bb[:, 1] - bb[:, 3] / 2
            xy[:, 2] = bb[:, 0] + bb[:, 2] / 2
            xy[:, 3] = bb[:, 1] + bb[:, 3] / 2
        torch.cuda.synchronize(); t3 = time.perf_counter()
        # optional copy to CPU for CPU NMS
        if n > 0 and a.nms_dev == "cpu":
            xy_n = xy.cpu(); cc_n = cc.cpu()
        else:
            xy_n, cc_n = (xy, cc) if n > 0 else (None, None)
        if a.nms_dev == "cuda":
            torch.cuda.synchronize()
        t4 = time.perf_counter()
        # NMS algorithm alone
        if n > 0:
            tv_nms(xy_n, cc_n, a.iou)
        if a.nms_dev == "cuda":
            torch.cuda.synchronize()
        t5 = time.perf_counter()

        D.append((t1 - t0) * 1e3); F.append((t2 - t1) * 1e3)
        C.append((t3 - t2) * 1e3); P.append((t4 - t3) * 1e3)
        N.append((t5 - t4) * 1e3); filt.append(n)

    def p50(x): return st.median(x)
    print(f"boxes into NMS (filter survivors): {filt[0]}")
    print(f"  decode (max+transpose)      p50={p50(D):.3f} ms")
    print(f"  filter (conf>th gather)     p50={p50(F):.3f} ms")
    print(f"  coord transform (xywh->xyxy) p50={p50(C):.3f} ms   <-- dominant")
    print(f"  .cpu() copy ({a.nms_dev})         p50={p50(P):.3f} ms")
    print(f"  NMS algorithm only          p50={p50(N):.3f} ms")
    print(f"  ---")
    print(f"  coord+copy+NMS (= real_anchor 'nms' column) "
          f"p50={p50(C)+p50(P)+p50(N):.3f} ms")


if __name__ == "__main__":
    main()
