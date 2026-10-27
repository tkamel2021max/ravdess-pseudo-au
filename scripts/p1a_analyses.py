# ======================================================================
#  P1a — Additional analyses on CPU (no GPU, no retraining)
#  Reads the predictions already saved on Drive and writes tables + figures
#  to My Drive/ravdess_au_project/results/p1a/  (and downloads a zip).
# ======================================================================
import os, sys, json, time, shutil, tarfile, warnings, subprocess
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
os.environ['MPLBACKEND'] = 'Agg'
T0 = time.time()
def step(t): print(f"\n=== {t}   [{(time.time()-T0)/60:.1f} min]", flush=True)

try:
    from google.colab import drive, files
    drive.mount('/content/drive')
    P = Path('/content/drive/MyDrive/ravdess_au_project'); LOCAL = Path('/content'); IN_COLAB = True
except ImportError:                                   # local test
    P = Path(os.environ['P1A_PROJ']); LOCAL = Path(os.environ['P1A_LOCAL']); IN_COLAB = False
R = P / 'results'; OUT = R / 'p1a'; OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(P))
from au_common import (AU_IDS, AU_MAP, EMOTIONS, load_index, load_arrays, make_folds, compute_q99,
                       scale_au, frame_stats, metrics_from_sums)

# ---------------------------------------------------------------- data
step("1. Features (raw MediaPipe pseudo-AUs and blendshapes)")
FEAT = LOCAL / 'features_p1a'
if not (FEAT / 'index.csv').exists():
    FEAT.mkdir(exist_ok=True)
    tar = P / 'data' / 'features.tar'
    if tar.exists():
        shutil.copy2(tar, LOCAL / 'f.tar')
        with tarfile.open(LOCAL / 'f.tar') as tf: tf.extractall(FEAT)
        (LOCAL / 'f.tar').unlink()
    else:
        FEAT = P / 'data' / 'features'                # slower: read from Drive
idx = load_index(FEAT)
_, aus_raw, bss = load_arrays(FEAT, idx, with_frames=False)
actors, emo = idx.actor.values, idx.emotion.values
V, A = len(idx), len(AU_IDS)
uact = sorted(np.unique(actors))
folds = make_folds(uact, 4)
fold_of = {a: fi for fi, f in enumerate(folds) for a in f}
vfold = np.array([fold_of[a] for a in actors])
agroups = [np.flatnonzero(actors == a) for a in uact]
print(V, "videos,", len(uact), "actors")

def load_exp(e):
    out = []
    for f in sorted((R / e).glob('seed?.npz')):
        z = np.load(f); out.append({k: z[k] for k in z.files})
    return out
EXPS = ['base', 'direct_emotion', 'tmp_none', 'tmp_tcn', 'loss_l1', 'loss_ccc', 'bb_mobilenet', 'bb_vit']
data = {e: load_exp(e) for e in EXPS + ['const', 'lr_gt_au', 'lr_blend'] if (R / e).exists()}
assert len(data['base'][0]['emo_true']) == V, "video order mismatch"
AU_EXPS = [e for e in EXPS if e != 'direct_emotion' and e in data]

# valid AUs (prevalence 1-99 % on pooled labels, as in the paper)
S0 = frame_stats(data['base'][0]['frame_vid'], data['base'][0]['au_true'], data['base'][0]['au_pred'])
prev0 = metrics_from_sums(S0.sum(0))['prev']
valid = (prev0 >= 0.01) & (prev0 <= 0.99)
print("valid AUs:", [f"AU{a}" for a, v in zip(AU_IDS, valid) if v])

# ================================================================= RQ1
step("2. RQ1 — raw / calibrated signal audit (16 AUs)")
raw = np.concatenate(aus_raw)
q99_fold = [compute_q99(aus_raw, np.flatnonzero(vfold != fi)) for fi in range(4)]   # training actors of each fold
cal = np.concatenate([scale_au(aus_raw[v], q99_fold[vfold[v]]) for v in range(V)])
rows = []
for j, au in enumerate(AU_IDS):
    x, c = raw[:, j], cal[:, j]
    q = [q99_fold[fi][j] for fi in range(4)]
    rows.append({'AU': au, 'description': AU_MAP[au][0], 'blendshapes': '+'.join(AU_MAP[au][1]),
                 'raw_mean': x.mean(), 'raw_std': x.std(), 'raw_min': x.min(), 'raw_median': np.median(x),
                 'raw_p99': np.quantile(x, 0.99), 'raw_max': x.max(),
                 'frac_raw_below_0.001': (x < 1e-3).mean(), 'frac_raw_ge_0.02': (x >= 0.02).mean(),
                 'calib_q99_min': min(q), 'calib_q99_max': max(q), 'q99_floor_hit': bool(min(q) <= 0.05 + 1e-9),
                 'calibrated_prevalence': (c >= 2).mean(), 'calibrated_label_var': c.var()})
audit = pd.DataFrame(rows)
audit['status'] = np.where(audit.calibrated_label_var < 1e-8, 'zero variance',
                   np.where(audit.calibrated_prevalence == 0, 'zero prevalence, non-zero variance', 'ok'))
audit.round(5).to_csv(OUT / 'rq1_signal_audit.csv', index=False)
print(audit[['AU', 'raw_median', 'raw_p99', 'raw_max', 'q99_floor_hit', 'calibrated_prevalence', 'status']].round(4).to_string(index=False))

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 7})
fig, ax = plt.subplots(figsize=(7, 2.8))
ax.boxplot([np.log10(raw[:, j] + 1e-6) for j in range(A)], showfliers=False, widths=0.6,
           medianprops=dict(color='#eb6834'), boxprops=dict(color='#2a78d6'), whiskerprops=dict(color='#2a78d6'),
           capprops=dict(color='#2a78d6'))
ax.axhline(np.log10(0.02), color='#6b6b66', lw=0.6, ls='--'); ax.text(A + 0.4, np.log10(0.02), 'activation\n(raw 0.02)', fontsize=5.5, va='center')
ax.set_xticks(range(1, A + 1), [f'AU{a}' for a in AU_IDS], rotation=60)
ax.set_ylabel('log10 raw pseudo-AU score'); ax.set_xlabel('Action Unit')
for s in ('top', 'right'): ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig(OUT / 'fig_raw_distributions.pdf'); fig.savefig(OUT / 'fig_raw_distributions.png', dpi=150); plt.close(fig)

step("3. RQ1 — fidelity per actor (mean ICC over valid AUs), dispersion across seeds")
def actor_icc(z):
    S = frame_stats(z['frame_vid'], z['au_true'], z['au_pred'])
    return np.stack([metrics_from_sums(S[g].sum(0))['icc'] for g in agroups])     # (actors, A)
per_actor, per_actor_au = {}, {}
for e in AU_EXPS:
    M = np.stack([actor_icc(z) for z in data[e]])                                 # (seeds, actors, A)
    per_actor_au[e] = M
    per_actor[e] = np.nanmean(M[:, :, valid], 2)                                  # (seeds, actors)
rng = np.random.default_rng(0); draws = rng.integers(0, len(uact), (5000, len(uact)))
tab = pd.DataFrame({'actor': uact, 'gender': ['M' if a % 2 else 'F' for a in uact], 'fold': [fold_of[a] + 1 for a in uact]})
summ = []
for e in AU_EXPS:
    m = per_actor[e].mean(0)
    tab[f'{e}_icc_mean'] = m; tab[f'{e}_icc_sd_seeds'] = per_actor[e].std(0)
    b = m[draws].mean(1)
    summ.append(dict(experiment=e, mean_of_actor_icc=m.mean(), actor_ci_low=np.percentile(b, 2.5),
                     actor_ci_high=np.percentile(b, 97.5), min_actor=m.min(), max_actor=m.max(),
                     sd_between_actors=m.std(), mean_sd_between_seeds=per_actor[e].std(0).mean()))
tab.round(4).to_csv(OUT / 'rq1_per_actor_icc.csv', index=False)
summ = pd.DataFrame(summ); summ.round(4).to_csv(OUT / 'rq1_per_actor_summary.csv', index=False)
print(summ.round(3).to_string(index=False))

M = per_actor_au['base'].mean(0)                                                   # (actors, A)
disp = pd.DataFrame(dict(AU=AU_IDS, valid=valid, prevalence=prev0,
                         icc_mean_over_actors=np.nanmean(M, 0), icc_sd_over_actors=np.nanstd(M, 0),
                         icc_min_actor=np.nanmin(M, 0), icc_max_actor=np.nanmax(M, 0),
                         n_actors_defined=np.sum(~np.isnan(M), 0)))
disp.round(4).to_csv(OUT / 'rq1_per_au_actor_dispersion.csv', index=False)

fig, ax = plt.subplots(figsize=(7, 2.6))
o = np.argsort(per_actor['base'].mean(0))
ax.bar(range(len(uact)), per_actor['base'].mean(0)[o], 0.7, color='#2a78d6',
       yerr=per_actor['base'].std(0)[o], ecolor='#3d3d3a', error_kw=dict(elinewidth=0.6, capsize=1.2))
ax.set_xticks(range(len(uact)), [f"A{uact[i]}" for i in o], rotation=60)
ax.set_ylabel('Mean ICC(3,1) over valid AUs'); ax.set_xlabel('Test actor (sorted)'); ax.set_ylim(0, 1)
for s in ('top', 'right'): ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig(OUT / 'fig_per_actor_icc.pdf'); fig.savefig(OUT / 'fig_per_actor_icc.png', dpi=150); plt.close(fig)

step("4. RQ1 — within-video agreement (temporal) vs global ICC, base model")
def ccc(y, p):
    my, mp = y.mean(), p.mean(); vy, vp = y.var(), p.var()
    return 2 * ((y - my) * (p - mp)).mean() / (vy + vp + (my - mp) ** 2)
wv = np.full((len(data['base']), V, A), np.nan); wr = wv.copy()
for s, z in enumerate(data['base']):
    vid, y, p = z['frame_vid'], z['au_true'], z['au_pred']
    starts = np.r_[0, np.flatnonzero(np.diff(vid)) + 1, len(vid)]
    for v in range(V):
        yy, pp = y[starts[v]:starts[v + 1]], p[starts[v]:starts[v + 1]]
        if len(yy) < 8: continue
        for j in range(A):
            if yy[:, j].std() > 0.05 and pp[:, j].std() > 1e-6:
                wv[s, v, j] = ccc(yy[:, j], pp[:, j]); wr[s, v, j] = np.corrcoef(yy[:, j], pp[:, j])[0, 1]
Sg = np.stack([frame_stats(z['frame_vid'], z['au_true'], z['au_pred']) for z in data['base']]).sum(1)
icc_g = np.nanmean(metrics_from_sums(Sg)['icc'], 0)
intra = pd.DataFrame(dict(AU=AU_IDS, valid=valid, global_icc=icc_g,
                          within_video_ccc=np.nanmean(np.nanmean(wv, 1), 0),
                          within_video_pearson=np.nanmean(np.nanmean(wr, 1), 0),
                          frac_videos_with_label_motion=np.mean(~np.isnan(wv[0]), 0)))
intra.round(4).to_csv(OUT / 'rq1_within_video.csv', index=False)
print(intra.round(3).to_string(index=False))

step("5. RQ1 — sensitivity to the activation threshold (1, 2, 3), base model")
rows = []
for thr in (1.0, 2.0, 3.0):
    Ms = [metrics_from_sums(frame_stats(z['frame_vid'], z['au_true'], z['au_pred'], thr).sum(0)) for z in data['base']]
    for j, au in enumerate(AU_IDS):
        pr = np.mean([m['prev'][j] for m in Ms])
        rows.append(dict(threshold=thr, AU=au, prevalence=pr, f1=np.nanmean([m['f1'][j] for m in Ms]),
                         f1_always_active=2 * pr / (1 + pr) if pr > 0 else np.nan,
                         balanced_acc=np.nanmean([m['bacc'][j] for m in Ms])))
thr_tab = pd.DataFrame(rows); thr_tab.round(4).to_csv(OUT / 'rq1_threshold_sensitivity.csv', index=False)
print(thr_tab.pivot(index='AU', columns='threshold', values='prevalence').round(3).to_string())

# ================================================================= RQ2
step("6. RQ2 — accuracy, macro-F1 and NLL per fold and seed")
def emo_metrics(y, pr):
    pred = pr.argmax(1)
    cm = np.bincount(y * 8 + pred, minlength=64).reshape(8, 8).astype(float)
    with np.errstate(all='ignore'):
        rec, prc = np.diag(cm) / cm.sum(1), np.diag(cm) / cm.sum(0)
        f1 = np.nan_to_num(2 * prc * rec / (prc + rec))
    return dict(acc=np.mean(pred == y), macro_f1=f1[cm.sum(1) > 0].mean(),
                nll=-np.log(np.clip(pr[np.arange(len(y)), y], 1e-8, 1)).mean())
rows = []
for e, zs in data.items():
    for s, z in enumerate(zs):
        for fi in range(4):
            m = vfold == fi
            rows.append(dict(experiment=e, seed=s, fold=fi + 1, **emo_metrics(emo[m], z['emo_prob'][m])))
pfs = pd.DataFrame(rows); pfs.round(4).to_csv(OUT / 'rq2_per_fold_seed.csv', index=False)
print(pfs[pfs.experiment.isin(['base', 'direct_emotion'])].pivot_table(index=['experiment', 'seed'], columns='fold', values='acc').round(3).to_string())

step("7. RQ2 — AU bottleneck and non-linear MediaPipe baselines (nested model selection)")
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold, GridSearchCV
def stats(arrs): return np.stack([np.concatenate([a.mean(0), a.std(0), a.max(0)]) for a in arrs])
def split_frames(z):
    vid = z['frame_vid']; st = np.r_[0, np.flatnonzero(np.diff(vid)) + 1, len(vid)]
    return [z['au_pred'][st[v]:st[v + 1]] for v in range(V)], [z['au_true'][st[v]:st[v + 1]] for v in range(V)]
MODELS = {
    'LR': (make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
           {'logisticregression__C': [0.01, 0.1, 1, 10]}),
    'GBT': (HistGradientBoostingClassifier(random_state=0, max_iter=200),
            {'learning_rate': [0.05, 0.1], 'max_depth': [3, None]}),
    'MLP': (make_pipeline(StandardScaler(), MLPClassifier(max_iter=2000, random_state=0)),
            {'mlpclassifier__hidden_layer_sizes': [(64,), (128,)], 'mlpclassifier__alpha': [1e-3, 1e-2]}),
}
def nested(Xtr_fn, Xte_fn, model):
    """Outer: the 4 actor-independent folds; inner: 3-fold GroupKFold over training actors (dev data only)."""
    est, grid = MODELS[model]; prob = np.zeros((V, 8)); chosen = []
    for fi in range(4):
        tr, te = np.flatnonzero(vfold != fi), np.flatnonzero(vfold == fi)
        gs = GridSearchCV(est, grid, cv=GroupKFold(3), scoring='neg_log_loss', n_jobs=-1)
        gs.fit(Xtr_fn(fi)[tr], emo[tr], groups=actors[tr])
        pr = gs.predict_proba(Xte_fn(fi)[te]); full = np.zeros((len(te), 8)); full[:, gs.best_estimator_.classes_] = pr
        prob[te] = full; chosen.append(gs.best_params_)
    return prob, chosen
extra, chosen_log = {}, {}
X_au, X_bs = stats(aus_raw), stats(bss)
for m in MODELS:                                                      # teacher at test time (MediaPipe)
    extra[f'mediapipe_au_{m}'], chosen_log[f'mediapipe_au_{m}'] = nested(lambda fi: X_au, lambda fi: X_au, m)
    extra[f'mediapipe_bs_{m}'], chosen_log[f'mediapipe_bs_{m}'] = nested(lambda fi: X_bs, lambda fi: X_bs, m)
    print("  done", m, flush=True)
# AU bottleneck: emotion from the 16 AUs PREDICTED by the base student (no MediaPipe at test time)
bott_a, bott_b = [], []
for s, z in enumerate(data['base']):
    pred_frames, _ = split_frames(z); X_pred = stats(pred_frames)
    # (a) trained on teacher pseudo-AUs (calibrated on the fold's training actors), applied to student predictions
    def X_teacher(fi):
        q = compute_q99(aus_raw, np.flatnonzero(vfold != fi))
        return stats([scale_au(a, q) for a in aus_raw])
    pa, _ = nested(X_teacher, lambda fi: X_pred, 'LR'); bott_a.append(pa)
    # (b) stacked: trained and tested on out-of-fold student predictions
    pb, _ = nested(lambda fi: X_pred, lambda fi: X_pred, 'LR'); bott_b.append(pb)
extra['au_bottleneck_teacher_trained'] = bott_a
extra['au_bottleneck_stacked_oof'] = bott_b
json.dump({k: [{kk: str(vv) for kk, vv in d.items()} for d in v] for k, v in chosen_log.items()},
          open(OUT / 'rq2_selected_hyperparameters.json', 'w'), indent=1)

step("8. RQ2 — overall metrics and paired actor-level tests vs base")
def as_list(p): return p if isinstance(p, list) else [p]
allp = {e: [z['emo_prob'] for z in zs] for e, zs in data.items()}
allp.update({k: as_list(v) for k, v in extra.items()})
rows = []
for e, ps in allp.items():
    ms = [emo_metrics(emo, p) for p in ps]
    rows.append(dict(model=e, n_runs=len(ps), **{k: np.mean([m[k] for m in ms]) for k in ms[0]}))
overall = pd.DataFrame(rows); overall.round(4).to_csv(OUT / 'rq2_overall.csv', index=False)
print(overall.round(3).to_string(index=False))

D = 5000; dr = rng.integers(0, len(uact), (D, len(uact)))
def actor_suff(p):
    """Per-actor sufficient statistics: confusion matrix (8x8) and NLL sum -> exact, fast actor bootstrap."""
    pred = p.argmax(1); nll = -np.log(np.clip(p[np.arange(V), emo], 1e-8, 1))
    cm = np.stack([np.bincount(emo[g] * 8 + pred[g], minlength=64).reshape(8, 8) for g in agroups]).astype(float)
    return cm, np.array([nll[g].sum() for g in agroups]), np.array([len(g) for g in agroups])
def from_cm(cm, nll, n):                       # cm (..., 8, 8), nll (...), n (...)
    tp = np.diagonal(cm, axis1=-2, axis2=-1); rows_, cols_ = cm.sum(-1), cm.sum(-2)
    with np.errstate(all='ignore'):
        rec, prc = tp / rows_, tp / cols_
        f1 = np.nan_to_num(2 * prc * rec / (prc + rec))
    mf1 = np.where(rows_ > 0, f1, np.nan); mf1 = np.nanmean(mf1, -1)
    return dict(acc=tp.sum(-1) / n, macro_f1=mf1, nll=nll / n)
def boot_all(ps):
    out = {k: np.zeros(D) for k in ('acc', 'macro_f1', 'nll')}
    for p in ps:
        cm, nl, n = actor_suff(p)
        m = from_cm(cm[dr].sum(1), nl[dr].sum(1), n[dr].sum(1))
        for k in out: out[k] += m[k] / len(ps)
    return out
cmp_rows = []
base_b = boot_all(allp['base'])
base_pt = {k: np.mean([emo_metrics(emo, p)[k] for p in allp['base']]) for k in ('acc', 'macro_f1', 'nll')}
for e in [k for k in allp if k != 'base']:
    row = dict(model=e); be = boot_all(allp[e])
    for k in ('acc', 'macro_f1', 'nll'):
        b = be[k] - base_b[k]
        row[f'delta_{k}'] = np.mean([emo_metrics(emo, p)[k] for p in allp[e]]) - base_pt[k]
        row[f'{k}_ci_low'], row[f'{k}_ci_high'] = np.percentile(b, 2.5), np.percentile(b, 97.5)
        row[f'{k}_p'] = min(1.0, 2 * min((b <= 0).mean(), (b >= 0).mean()))
    cmp_rows.append(row)
cmp = pd.DataFrame(cmp_rows); cmp.round(4).to_csv(OUT / 'rq2_paired_actor_vs_base.csv', index=False)
print(cmp[['model', 'delta_acc', 'acc_ci_low', 'acc_ci_high', 'acc_p', 'delta_nll', 'nll_p']].round(3).to_string(index=False))

# ---------------------------------------------------------------- export
shutil.make_archive(str(LOCAL / 'p1a_results'), 'zip', OUT)
print(f"\nDONE in {(time.time()-T0)/60:.1f} min. Results in My Drive/ravdess_au_project/results/p1a/")
if IN_COLAB:
    files.download(str(LOCAL / 'p1a_results.zip'))
