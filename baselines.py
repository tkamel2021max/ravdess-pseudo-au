"""Baselines evaluated with exactly the same actor-independent folds as the deep models.

  const      : AU = training-fold mean, emotion = training prior         (floor)
  lr_gt_au   : logistic regression on video statistics of the 16 pseudo-AUs (MediaPipe, no student)
  lr_blend   : logistic regression on video statistics of the 52 raw blendshapes
lr_* need MediaPipe at test time: they bound what the RGB student can hope to match.
"""
import argparse
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from au_common import compute_q99, load_arrays, load_index, make_folds, save_oof, scale_au


def video_stats(arrs):
    return np.stack([np.concatenate([a.mean(0), a.std(0), a.max(0)]) for a in arrs])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", default="data/features")
    ap.add_argument("--out", default="results")
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--actors", type=int, nargs="*")
    a = ap.parse_args()

    idx = load_index(a.feat, a.actors)
    _, aus, bss = load_arrays(a.feat, idx, with_frames=False)
    emo = idx.emotion.values
    V = len(idx)
    folds = make_folds(sorted(idx.actor.unique()), a.folds)

    X_au, X_bs = video_stats(aus), video_stats(bss)
    au_true, au_pred = [None] * V, [None] * V
    prior_prob = np.zeros((V, 8))
    prob = {"lr_gt_au": np.zeros((V, 8)), "lr_blend": np.zeros((V, 8))}

    for fold in folds:
        te = np.flatnonzero(idx.actor.isin(fold).values)
        tr = np.flatnonzero(~idx.actor.isin(fold).values)
        q99 = compute_q99(aus, tr)
        mean = np.concatenate([scale_au(aus[v], q99) for v in tr]).mean(0)
        prior = np.bincount(emo[tr], minlength=8) / len(tr)
        for v in te:
            au_true[v] = scale_au(aus[v], q99)
            au_pred[v] = np.tile(mean, (len(aus[v]), 1))
            prior_prob[v] = prior
        for name, X in (("lr_gt_au", X_au), ("lr_blend", X_bs)):
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0))
            clf.fit(X[tr], emo[tr])
            p = clf.predict_proba(X[te])
            full = np.zeros((len(te), 8))
            full[:, clf.classes_] = p
            prob[name][te] = full

    out = Path(a.out)
    save_oof(out / "const" / "seed0.npz", au_true, au_pred, emo, prior_prob)
    for name in prob:
        save_oof(out / name / "seed0.npz", None, None, emo, prob[name])
    print("baselines saved in", out)


if __name__ == "__main__":
    main()
