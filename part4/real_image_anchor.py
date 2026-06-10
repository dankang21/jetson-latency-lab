#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Real-image anchors for the YOLO pipeline finale, with postprocess BROKEN DOWN.

Runs the OPTIMIZED pipeline (GPU preprocess -> iobind -> TRT-FP16 -> GPU NMS) on
real images of differing scene complexity, with REAL detections, and splits the
postprocess into decode / filter / NMS so we can see which one actually costs.

The finding this validates: "NMS" is not the postprocess bottleneck. NMS is
~free; the cost is the decode of the 8400 raw rows (fixed) plus the confidence
FILTER (boolean gather), which grows with how many detections survive -- i.e.
with scene complexity. A synthetic fixed input hides this because nothing passes
the filter.

Usage:
  python3 real_image_anchor.py --model yolov8s.onnx --images imgs/*.jpg
"""
import argparse
import time
import statistics as st
import numpy as np
import onnxruntime as ort
import torch
from torchvision.ops import nms as tv_nms

H = W = 640


def load_resize(path):
    from PIL import Image
    im = Image.open(path).convert("RGB").resize((W, H))
    return np.ascontiguousarray(np.asarray(im, dtype=np.uint8)).copy()


class OptPipelineReal:
    def __init__(self, model, conf_th=0.25, iou_th=0.45, nms_dev="cuda"):
        self.nms_dev = nms_dev
        self.s = ort.InferenceSession(
            model,
            providers=[("TensorrtExecutionProvider", {"trt_fp16_enable": True}),
                       "CUDAExecutionProvider"])
        self.inp = self.s.get_inputs()[0].name
        self.outname = self.s.get_outputs()[0].name
        self.gpu_in = torch.empty((1, 3, H, W), dtype=torch.float32,
                                  device="cuda").contiguous()
        self.io = self.s.io_binding()
        self.io.bind_input(name=self.inp, device_type="cuda", device_id=0,
                           element_type=np.float32, shape=(1, 3, H, W),
                           buffer_ptr=self.gpu_in.data_ptr())
        self.io.bind_output(self.outname, device_type="cuda", device_id=0)
        self.conf_th = conf_th
        self.iou_th = iou_th

    def cycle(self, img_u8):
        # preprocess
        t0 = time.perf_counter()
        g = torch.from_numpy(img_u8).cuda(non_blocking=True)
        self.gpu_in.copy_(g.permute(2, 0, 1).unsqueeze(0).float().div_(255.0))
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        # infer
        self.s.run_with_iobinding(self.io)
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        # postprocess, split: decode / filter / nms
        ov = self.io.get_outputs()[0]
        raw = torch.from_dlpack(ov._ortvalue.to_dlpack())[0]   # [84,8400] cuda
        ta = time.perf_counter()
        boxes = raw[:4].t(); scores = raw[4:].t(); conf, _ = scores.max(1)
        torch.cuda.synchronize(); tb = time.perf_counter()
        keep = conf > self.conf_th
        bb = boxes[keep]; cc = conf[keep]
        n_filt = int(bb.shape[0])
        torch.cuda.synchronize(); tc = time.perf_counter()
        n_final = 0
        if n_filt > 0:
            xy = torch.empty_like(bb)
            xy[:, 0] = bb[:, 0] - bb[:, 2] / 2; xy[:, 1] = bb[:, 1] - bb[:, 3] / 2
            xy[:, 2] = bb[:, 0] + bb[:, 2] / 2; xy[:, 3] = bb[:, 1] + bb[:, 3] / 2
            if self.nms_dev == "cpu":
                xy = xy.cpu(); cc = cc.cpu()
            n_final = int(tv_nms(xy, cc, self.iou_th).numel())
        torch.cuda.synchronize(); td = time.perf_counter()
        return dict(
            filt=n_filt, final=n_final,
            pre=(t1 - t0) * 1e3, inf=(t2 - t1) * 1e3,
            decode=(tb - ta) * 1e3, filter=(tc - tb) * 1e3, nms=(td - tc) * 1e3,
            post=(td - ta) * 1e3, e2e=(t1 - t0 + t2 - t1 + td - ta) * 1e3,
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--images", nargs="+", required=True)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--nms-dev", default="cuda", choices=["cuda", "cpu"])
    a = ap.parse_args()

    pipe = OptPipelineReal(a.model, nms_dev=a.nms_dev)
    print(f"(TensorRT engine build on first run, may take minutes)  NMS on {a.nms_dev}\n")
    hdr = (f"{'image':22s} {'filt':>5s} {'fin':>4s} | "
           f"{'pre':>6s} {'inf':>6s} {'dec':>6s} {'filt':>6s} {'nms':>6s} "
           f"{'post':>6s} {'e2e':>6s}  (ms p50)")
    print(hdr); print("-" * len(hdr))
    rows_csv = []
    for path in sorted(a.images):
        img = load_resize(path)
        for _ in range(a.warmup):
            pipe.cycle(img)
        samples = [pipe.cycle(img) for _ in range(a.iters)]
        m = lambda k: st.median(s[k] for s in samples)
        filt, fin = samples[0]["filt"], samples[0]["final"]
        name = path.split("/")[-1][:22]
        print(f"{name:22s} {filt:5d} {fin:4d} | "
              f"{m('pre'):6.3f} {m('inf'):6.3f} {m('decode'):6.3f} "
              f"{m('filter'):6.3f} {m('nms'):6.3f} {m('post'):6.3f} {m('e2e'):6.3f}")
        rows_csv.append((name, filt, fin, m('pre'), m('inf'),
                         m('decode'), m('filter'), m('nms'), m('post'), m('e2e')))

    # also dump CSV for charting
    import csv
    with open(f"results/p4_real_anchor_{a.nms_dev}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "filt", "final", "pre", "inf",
                    "decode", "filter", "nms", "post", "e2e"])
        w.writerows(rows_csv)
    print(f"\n-> results/p4_real_anchor_{a.nms_dev}.csv")


if __name__ == "__main__":
    main()
