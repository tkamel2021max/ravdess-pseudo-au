"""Run MediaPipe FaceLandmarker on RAVDESS videos.

Per video -> features/<stem>.npz with
  frames  (N,S,S,3) uint8   square face crop (box enlarged by 25 %), RGB
  au      (N,16) float32    RAW pseudo-AU = mean of mapped blendshape scores in [0,1]
  bs      (N,52) float32    all MediaPipe blendshape scores (for the baseline)
Calibration to [0,5] is done later, on training folds only (see au_common.compute_q99).
Filename convention: MM-VV-EE-II-SS-RR-AA (modality, vocal channel, emotion, intensity,
statement, repetition, actor). Only modality 01 (audio-visual) is kept by default.
"""
import argparse
import urllib.request
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import mediapipe as mp
from mediapipe.tasks import python as mpp
from mediapipe.tasks.python import vision
from au_common import AU_MAP

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
             "face_landmarker/float16/1/face_landmarker.task")
_lm = None
cv2.setNumThreads(1)


def init(model_path):
    global _lm
    opts = vision.FaceLandmarkerOptions(
        base_options=mpp.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.IMAGE, output_face_blendshapes=True, num_faces=1)
    _lm = vision.FaceLandmarker.create_from_options(opts)


def process(job):
    path, out_dir, stride, size = job
    p = Path(path)
    out = Path(out_dir) / f"{p.stem}.npz"
    if out.exists():
        return p.stem
    mod, voc, emo, inten, _, _, actor = [int(t) for t in p.stem.split("-")]
    cap = cv2.VideoCapture(str(p))
    frames, aus, bss, total, i, names = [], [], [], 0, 0, None
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if i % stride == 0:
            total += 1
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            res = _lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
            if res.face_landmarks and res.face_blendshapes:
                h, w = rgb.shape[:2]
                lm = res.face_landmarks[0]
                xs = np.array([q.x for q in lm]) * w
                ys = np.array([q.y for q in lm]) * h
                cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
                side = max(xs.max() - xs.min(), ys.max() - ys.min()) * 1.25
                x0, x1 = int(max(cx - side / 2, 0)), int(min(cx + side / 2, w))
                y0, y1 = int(max(cy - side / 2, 0), ), int(min(cy + side / 2, h))
                crop = cv2.resize(rgb[y0:y1, x0:x1], (size, size), interpolation=cv2.INTER_AREA)
                d = {c.category_name: c.score for c in res.face_blendshapes[0]}
                names = sorted(d)
                bss.append([d[k] for k in names])
                aus.append([np.mean([d[b] for b in bl]) for _, bl in AU_MAP.values()])
                frames.append(crop)
        i += 1
    cap.release()
    if not frames:
        return None
    np.savez_compressed(out, frames=np.stack(frames), au=np.array(aus, np.float32),
                        bs=np.array(bss, np.float32), bs_names=np.array(names),
                        total=total, actor=actor, emotion=emo - 1, vocal=voc, intensity=inten, modality=mod)
    return p.stem


def build_index(out_dir):
    rows = []
    for f in sorted(Path(out_dir).glob("*.npz")):
        z = np.load(f)
        n = int(z["au"].shape[0])
        rows.append(dict(file=f.name, actor=int(z["actor"]), emotion=int(z["emotion"]), vocal=int(z["vocal"]),
                         intensity=int(z["intensity"]), n_det=n, n_total=int(z["total"]),
                         det_rate=n / max(int(z["total"]), 1)))
    df = pd.DataFrame(rows)
    df.to_csv(Path(out_dir) / "index.csv", index=False)
    print(df.groupby("emotion").size().rename("videos"))
    print(f"{len(df)} videos, mean detection rate {df.det_rate.mean():.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/features")
    ap.add_argument("--stride", type=int, default=2, help="keep 1 frame out of `stride` (30 fps -> 15 fps)")
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--modality", type=int, default=1, help="1 = audio-visual, 2 = video-only")
    ap.add_argument("--model", default="face_landmarker.task")
    a = ap.parse_args()
    if not Path(a.model).exists():
        print("downloading FaceLandmarker model ...")
        urllib.request.urlretrieve(MODEL_URL, a.model)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    vids = [p for p in sorted(Path(a.raw).rglob("*.mp4")) if int(p.stem.split("-")[0]) == a.modality]
    print(f"{len(vids)} videos to process")
    jobs = [(str(v), a.out, a.stride, a.size) for v in vids]
    with Pool(a.workers, initializer=init, initargs=(a.model,)) as pool:
        for i, r in enumerate(pool.imap_unordered(process, jobs), 1):
            if i % 50 == 0:
                print(f"{i}/{len(jobs)}")
    build_index(a.out)


if __name__ == "__main__":
    main()
