"""Shared utilities (numpy/pandas only): AU mapping, folds, calibration, metrics.

All AU metrics are computed from additive per-video sufficient statistics, which
makes video-level bootstrap confidence intervals cheap and exact.
"""
from pathlib import Path
import numpy as np
import pandas as pd

# AU -> (description, MediaPipe blendshapes averaged). Heuristic mapping, see paper Sec. 3.
AU_MAP = {
    1: ("Inner brow raiser", ["browInnerUp"]),
    2: ("Outer brow raiser", ["browOuterUpLeft", "browOuterUpRight"]),
    4: ("Brow lowerer", ["browDownLeft", "browDownRight"]),
    5: ("Upper lid raiser", ["eyeWideLeft", "eyeWideRight"]),
    6: ("Cheek raiser", ["cheekSquintLeft", "cheekSquintRight"]),
    7: ("Lid tightener", ["eyeSquintLeft", "eyeSquintRight"]),
    9: ("Nose wrinkler", ["noseSneerLeft", "noseSneerRight"]),
    10: ("Upper lip raiser", ["mouthUpperUpLeft", "mouthUpperUpRight"]),
    12: ("Lip corner puller", ["mouthSmileLeft", "mouthSmileRight"]),
    14: ("Dimpler", ["mouthDimpleLeft", "mouthDimpleRight"]),
    15: ("Lip corner depressor", ["mouthFrownLeft", "mouthFrownRight"]),
    16: ("Lower lip depressor", ["mouthLowerDownLeft", "mouthLowerDownRight"]),
    17: ("Chin raiser", ["mouthShrugLower"]),
    20: ("Lip stretcher", ["mouthStretchLeft", "mouthStretchRight"]),
    23: ("Lip pressor", ["mouthPressLeft", "mouthPressRight"]),
    26: ("Jaw drop", ["jawOpen"]),
}
AU_IDS = list(AU_MAP)
N_AU = len(AU_IDS)
EMOTIONS = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]
K_LEVELS = 5      # ordinal thresholds 0..4 on the [0,5] scale
THR = 2.0         # FACS-style "active" threshold (intensity >= 2 on the 0-5 scale)
MIN_FRAMES = 8    # videos with fewer detected frames are dropped


# ----------------------------------------------------------------- data
def load_index(feat_dir, actors=None):
    idx = pd.read_csv(Path(feat_dir) / "index.csv")
    idx = idx[idx.n_det >= MIN_FRAMES]
    if actors:
        idx = idx[idx.actor.isin(actors)]
    return idx.sort_values("file").reset_index(drop=True)


def load_arrays(feat_dir, idx, with_frames=True):
    frames, aus, bss = [], [], []
    for f in idx.file:
        z = np.load(Path(feat_dir) / f)
        aus.append(z["au"].astype(np.float32))
        bss.append(z["bs"].astype(np.float32))
        if with_frames:
            frames.append(z["frames"])
    return frames, aus, bss


def make_folds(actors, k):
    """Actor-independent folds, balanced by gender (odd actor id = male, even = female)."""
    actors = sorted(actors)
    m = [a for a in actors if a % 2 == 1]
    f = [a for a in actors if a % 2 == 0]
    return [m[i::k] + f[i::k] for i in range(k)]


def split_train_val(train_actors, seed, frac=0.15):
    rng = np.random.default_rng(seed)
    tr = sorted(train_actors)
    m = [a for a in tr if a % 2 == 1]
    f = [a for a in tr if a % 2 == 0]
    n_val = max(2, round(frac * len(tr)))
    nm, nf = min(n_val // 2, len(m)), min(n_val - n_val // 2, len(f))
    val = list(rng.choice(m, nm, replace=False)) + list(rng.choice(f, nf, replace=False))
    return [int(a) for a in val]


def compute_q99(aus, vids):
    """Per-AU 99th percentile of the raw pseudo-labels, computed on TRAINING videos only."""
    a = np.concatenate([aus[v] for v in vids])
    return np.maximum(np.quantile(a, 0.99, axis=0), 0.05).astype(np.float32)


def scale_au(raw, q99):
    """raw blendshape mean in [0,1] -> pseudo-AU intensity in [0,5]."""
    return 5.0 * np.clip(raw / q99, 0.0, 1.0)


def save_oof(path, au_true, au_pred, emo_true, emo_prob):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    d = dict(emo_true=np.asarray(emo_true), emo_prob=np.asarray(emo_prob, dtype=np.float32))
    if au_true is not None:
        d["frame_vid"] = np.concatenate([np.full(len(a), i) for i, a in enumerate(au_true)])
        d["au_true"] = np.concatenate(au_true).astype(np.float32)
        d["au_pred"] = np.concatenate(au_pred).astype(np.float32)
    np.savez_compressed(path, **d)


# ----------------------------------------------------------------- AU metrics
def frame_stats(vid, y, p, thr=THR):
    """Per-video additive sufficient statistics -> array (V, A, 11).
    columns: n, sum y, sum p, sum y^2, sum p^2, sum yp, sum |y-p|, tp, fp, fn, tn."""
    starts = np.r_[0, np.flatnonzero(np.diff(vid)) + 1]
    assert len(starts) == vid.max() + 1, "frames must be ordered by video id"
    yb, pb = y >= thr, p >= thr
    cols = [np.ones_like(y), y, p, y * y, p * p, y * p, np.abs(y - p),
            yb & pb, (~yb) & pb, yb & (~pb), (~yb) & (~pb)]
    return np.stack([np.add.reduceat(c.astype(np.float64), starts, axis=0) for c in cols], -1)


def metrics_from_sums(S):
    """S (..., A, 11) summed statistics -> dict of (..., A) arrays.
    ICC(3,1) with k=2 'raters' (label, prediction), consistency; NaN if the label has zero variance."""
    n, sy, sp, syy, spp, syp, sae, tp, fp, fn, tn = [S[..., i] for i in range(11)]
    with np.errstate(all="ignore"):
        my, mp = sy / n, sp / n
        vy, vp = syy / n - my ** 2, spp / n - mp ** 2
        cov = syp / n - my * mp
        pcc = cov / np.sqrt(vy * vp)
        ccc = 2 * cov / (vy + vp + (my - mp) ** 2)
        m = (my + mp) / 2
        ssr = 2 * ((syy + 2 * syp + spp) / 4 - n * m ** 2)
        ssc = n * (my - mp) ** 2 / 2
        sst = syy + spp - 2 * n * m ** 2
        sse = sst - ssr - ssc
        msr, mse = ssr / (n - 1), sse / (n - 1)
        icc = np.where(vy > 1e-8, (msr - mse) / (msr + mse), np.nan)
        prev = (tp + fn) / n
        f1 = np.where(tp + fn > 0, 2 * tp / (2 * tp + fp + fn), np.nan)
        bacc = (tp / (tp + fn) + tn / (tn + fp)) / 2
    return dict(icc=icc, pcc=pcc, ccc=ccc, mae=sae / n, f1=f1, bacc=bacc, prev=prev)


def au_boot(S, n_boot=1000, seed=0):
    """S (n_seeds, V, A, 11). Video-level bootstrap; metrics averaged over seeds."""
    rng = np.random.default_rng(seed)
    V = S.shape[1]
    with np.errstate(all="ignore"):
        pt = {k: np.nanmean(v, 0) for k, v in metrics_from_sums(S.sum(1)).items()}
        boots = {k: [] for k in pt}
        for _ in range(n_boot):
            w = rng.multinomial(V, np.full(V, 1.0 / V)).astype(np.float64)
            m = metrics_from_sums(np.einsum("v,svak->sak", w, S))
            for k in m:
                boots[k].append(np.nanmean(m[k], 0))
    return pt, {k: np.array(v) for k, v in boots.items()}


def agg_ci(point, boots, valid, k):
    p = np.nanmean(point[k][valid])
    b = np.nanmean(boots[k][:, valid], 1)
    return p, np.nanpercentile(b, 2.5), np.nanpercentile(b, 97.5)


def per_au_ci(point, boots, k):
    return point[k], np.nanpercentile(boots[k], 2.5, 0), np.nanpercentile(boots[k], 97.5, 0)


# ----------------------------------------------------------------- emotion metrics
def emo_metrics_one(y, prob):
    pred = prob.argmax(-1)
    conf = np.bincount(y * 8 + pred, minlength=64).reshape(8, 8).astype(float)
    tp = np.diag(conf)
    with np.errstate(all="ignore"):
        rec = tp / conf.sum(1)
        prec = tp / conf.sum(0)
        f1 = 2 * prec * rec / (prec + rec)
    return dict(acc=tp.sum() / conf.sum(), macro_f1=np.nanmean(np.nan_to_num(f1, nan=0.0)[conf.sum(1) > 0]),
                uar=np.nanmean(rec))


def emo_boot(y, prob, n_boot=1000, seed=0):
    """prob (n_seeds, V, 8). Returns {metric: (mean, lo, hi)}."""
    rng = np.random.default_rng(seed)
    V = len(y)

    def avg(idx):
        ms = [emo_metrics_one(y[idx], p[idx]) for p in prob]
        return {k: np.mean([m[k] for m in ms]) for k in ms[0]}

    pt = avg(np.arange(V))
    bs = [avg(rng.integers(0, V, V)) for _ in range(n_boot)]
    return {k: (pt[k], np.percentile([b[k] for b in bs], 2.5), np.percentile([b[k] for b in bs], 97.5)) for k in pt}
