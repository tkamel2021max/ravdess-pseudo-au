"""Measure student-network inference latency (excludes face detection / cropping).
Writes results/latency.json. Run on the machine you want to quote in the paper."""
import argparse
import json
import time
from pathlib import Path

import torch

from train_eval import CROP, T, Net


def timeit(fn, dev, n=50, warm=10):
    for _ in range(warm):
        fn()
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    if dev.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/latency.json")
    a = ap.parse_args()
    res = {}
    devs = [torch.device("cpu")] + ([torch.device("cuda")] if torch.cuda.is_available() else [])
    for dev in devs:
        for bb, tmp in [("resnet18", "transformer"), ("resnet18", "tcn"), ("mobilenet_v3_small", "tcn"), ("mobilenet_v3_small", "transformer"),
                        ("vit_b_16", "transformer")]:
            m = Net(bb, tmp, "ordinal", pretrained=False).to(dev).eval()
            clip = torch.randn(1, T, 3, CROP, CROP, device=dev)
            with torch.no_grad():
                clip_ms = timeit(lambda: m(clip), dev)
                frame_ms = timeit(lambda: m.bb(clip[:, 0]), dev)
            res[f"{dev.type}/{bb}+{tmp}"] = dict(ms_per_clip16=clip_ms, ms_per_frame_amortised=clip_ms / T,
                                                  ms_backbone_single_frame=frame_ms,
                                                  params_M=sum(p.numel() for p in m.parameters()) / 1e6)
            print(dev.type, bb, tmp, res[f"{dev.type}/{bb}+{tmp}"])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
