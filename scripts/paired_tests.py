"""Paired bootstrap tests of emotion accuracy against the base model (paper, Table 3).

Resamples either videos or actors (all videos of an actor together); per-video correctness is
averaged over seeds; Holm correction over the compared configurations.

    python scripts/paired_tests.py --results results --feat data/features --out results/paired_tests.csv
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

EXPS = ['bb_vit', 'bb_mobilenet', 'loss_l1', 'tmp_none', 'direct_emotion',
        'tmp_tcn', 'loss_ccc', 'lr_gt_au', 'lr_blend']


def correct(res, e):
    out = []
    for f in sorted((res / e).glob('seed?.npz')):
        z = np.load(f)
        out.append(z['emo_prob'].argmax(1) == z['emo_true'])
    return np.mean(out, 0)


def holm(p, alpha=0.05):
    keep = np.zeros(len(p), bool)
    for k, i in enumerate(np.argsort(p)):
        if p[i] <= alpha / (len(p) - k):
            keep[i] = True
        else:
            break
    return keep


def two_sided(b):
    return min(1.0, 2 * min((b <= 0).mean(), (b >= 0).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default='results')
    ap.add_argument('--feat', default='data/features')
    ap.add_argument('--out', default='results/paired_tests.csv')
    ap.add_argument('--boot', type=int, default=5000)
    a = ap.parse_args()
    res = Path(a.results)
    idx = pd.read_csv(Path(a.feat) / 'index.csv')
    idx = idx[idx.n_det >= 8].sort_values('file').reset_index(drop=True)   # same order as the results
    actors = idx.actor.values
    base = correct(res, 'base')
    assert len(base) == len(actors), 'video order mismatch'
    rng = np.random.default_rng(0)
    vid = rng.integers(0, len(base), (a.boot, len(base)))
    groups = [np.flatnonzero(actors == x) for x in np.unique(actors)]
    n = np.array([len(g) for g in groups])
    act = rng.integers(0, len(groups), (a.boot, len(groups)))
    rows = []
    for e in EXPS:
        d = correct(res, e) - base
        bv = d[vid].mean(1)
        s = np.array([d[g].sum() for g in groups])
        ba = s[act].sum(1) / n[act].sum(1)
        rows.append(dict(experiment=e, delta_acc=d.mean(),
                         video_ci_low=np.percentile(bv, 2.5), video_ci_high=np.percentile(bv, 97.5), video_p=two_sided(bv),
                         actor_ci_low=np.percentile(ba, 2.5), actor_ci_high=np.percentile(ba, 97.5), actor_p=two_sided(ba)))
    df = pd.DataFrame(rows)
    df['video_holm'] = holm(df.video_p.values)
    df['actor_holm'] = holm(df.actor_p.values)
    df.round(4).to_csv(a.out, index=False)
    print(df.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
