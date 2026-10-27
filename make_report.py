"""Aggregate results/*/seed*.npz -> paper/generated/{tab_*.tex, numbers.tex, fig_*.pdf}.
All CIs are 95 % video-level bootstrap intervals (metrics averaged over seeds)."""
import argparse
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from au_common import (AU_IDS, AU_MAP, EMOTIONS, agg_ci, au_boot, emo_boot, frame_stats, per_au_ci)

warnings.filterwarnings("ignore")
NAMES = {
    "base": "ResNet-18 + causal Transf. + ordinal (base)",
    "bb_mobilenet": "backbone: MobileNetV3-S",
    "bb_vit": "backbone: ViT-B/16",
    "base_sel_emo": "base, epoch selected on emotion NLL",
    "vit_ccc": "ViT-B/16 + CCC+L1 (best combination)",
    "tmp_none": "temporal: none (frame-wise)",
    "tmp_tcn": "temporal: causal TCN",
    "loss_l1": "loss: L1",
    "loss_ccc": "loss: CCC+L1",
    "direct_emotion": "no AU supervision (direct emotion CNN)",
    "const": "Baseline: constant / class prior",
    "lr_gt_au": "Baseline: LR on MediaPipe pseudo-AU stats$^\\dagger$",
    "lr_blend": "Baseline: LR on 52 blendshape stats$^\\dagger$",
}
ORDER = list(NAMES)


def fmt(t, d=3):
    return f"{t[0]:.{d}f}\\,[{t[1]:.{d}f},{t[2]:.{d}f}]"


def load(res, name):
    files = sorted((res / name).glob("seed*.npz"))
    if not files:
        return None
    S, P, y = [], [], None
    for f in files:
        z = np.load(f)
        if "au_true" in z:
            S.append(frame_stats(z["frame_vid"], z["au_true"], z["au_pred"]))
        P.append(z["emo_prob"])
        y = z["emo_true"]
    return (np.stack(S) if S else None), np.stack(P), y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="paper/generated")
    ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args()
    res, out = Path(a.results), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    data = {n: load(res, n) for n in ORDER}
    data = {n: d for n, d in data.items() if d is not None}
    if "base" not in data:
        raise SystemExit("run train_eval.py --exps base first")

    # valid AUs: prevalence (>= FACS threshold) in [1 %, 99 %] on the pooled labels
    pt_ref = au_boot(data["base"][0], 10)[0]
    valid = (pt_ref["prev"] >= 0.01) & (pt_ref["prev"] <= 0.99)

    rows, cache = [], {}
    for n, (S, P, y) in data.items():
        e = emo_boot(y, P, a.boot)
        r = dict(name=n, emo=e)
        if S is not None and n != 'direct_emotion':   # direct model: untrained AU head
            pt, bs = au_boot(S, a.boot)
            cache[n] = (pt, bs)
            for k in ("icc", "ccc", "mae", "f1"):
                r[k] = agg_ci(pt, bs, valid, k)
        rows.append(r)

    # ---------- ablation table
    L = [r"\begin{table*}[t]\centering\small",
         r"\caption{Actor-independent results (out-of-fold, mean over seeds; 95\,\% video-level bootstrap CI). "
         r"AU metrics are averaged over the " + str(int(valid.sum())) + r" AUs with 1--99\,\% prevalence at the FACS threshold (intensity $\ge 2$); "
         r"MAE on the 0--5 scale. $^\dagger$ needs MediaPipe at test time.}\label{tab:ablation}",
         r"\resizebox{\textwidth}{!}{\begin{tabular}{lccccc}\toprule",
         r"Configuration & Emotion acc. & Emotion macro-F1 & AU ICC(3,1) & AU CCC & AU MAE \\\midrule"]
    for r in rows:
        au = ("--", "--", "--")
        if "icc" in r:
            au = (fmt(r["icc"]), fmt(r["ccc"]), fmt(r["mae"]))
        L.append(f"{NAMES[r['name']]} & {fmt(r['emo']['acc'])} & {fmt(r['emo']['macro_f1'])} & {au[0]} & {au[1]} & {au[2]} \\\\")
    L += [r"\bottomrule\end{tabular}}\end{table*}"]
    (out / "tab_ablation.tex").write_text("\n".join(L))

    # ---------- per-AU table (base) + trivial "always active" F1
    pt, bs = cache["base"]
    icc, ccc, mae, f1 = (per_au_ci(pt, bs, k) for k in ("icc", "ccc", "mae", "f1"))
    L = [r"\begin{table*}[t]\centering\small",
         r"\caption{Per-AU results of the base model (out-of-fold). Prev.: fraction of frames with pseudo-AU $\ge 2$. "
         r"F1$_{\rm all}$: F1 of the trivial ``always active'' classifier, the reference any detector must beat. "
         r"ICC is undefined (--) when the label has no variance.}\label{tab:perau}",
         r"\resizebox{\textwidth}{!}{\begin{tabular}{rlccccccc}\toprule",
         r"AU & Description & Prev. & ICC(3,1) & CCC & MAE & F1 & F1$_{\rm all}$ & Bal. acc. \\\midrule"]
    for i, au in enumerate(AU_IDS):
        pr = pt["prev"][i]
        f_all = 2 * pr / (1 + pr) if pr > 0 else float("nan")

        def c(t):
            return "--" if np.isnan(t[0][i]) else fmt((t[0][i], t[1][i], t[2][i]), 2)

        L.append(f"{au} & {AU_MAP[au][0]} & {pr:.3f} & {c(icc)} & {c(ccc)} & {c(mae)} & {c(f1)} & "
                 f"{'--' if np.isnan(f_all) else f'{f_all:.2f}'} & {pt['bacc'][i]:.2f} \\\\")
    L += [r"\bottomrule\end{tabular}}\end{table*}"]
    (out / "tab_per_au.tex").write_text("\n".join(L))

    # ---------- figures
    x = np.arange(len(AU_IDS))
    fig, ax = plt.subplots(figsize=(7, 2.8))
    for off, (k, lab) in zip((-0.2, 0.2), (("icc", "ICC(3,1)"), ("ccc", "CCC"))):
        m, lo, hi = per_au_ci(pt, bs, k)
        ax.bar(x + off, np.nan_to_num(m), 0.4, yerr=[np.nan_to_num(m - lo), np.nan_to_num(hi - m)], label=lab, capsize=1.5)
    ax.set_xticks(x, [f"AU{a_}" for a_ in AU_IDS], rotation=60, fontsize=7)
    ax.set_ylim(-0.1, 1.05)
    ax.set_xlabel("Action Unit")
    ax.set_ylabel("Agreement with MediaPipe pseudo-AU")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "fig_per_au.pdf")
    plt.close(fig)

    S_, P_, y_ = data["base"]
    conf = np.zeros((8, 8))
    for p in P_:
        conf += np.bincount(y_ * 8 + p.argmax(-1), minlength=64).reshape(8, 8)
    conf = conf / conf.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.imshow(conf, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(8), [e.capitalize() for e in EMOTIONS], rotation=60, fontsize=7)
    ax.set_yticks(range(8), [e.capitalize() for e in EMOTIONS], fontsize=7)
    for i in range(8):
        for j in range(8):
            ax.text(j, i, f"{conf[i, j]:.2f}", ha="center", va="center", fontsize=6)
    ax.set_xlabel("Predicted emotion")
    ax.set_ylabel("True emotion")
    fig.tight_layout()
    fig.savefig(out / "fig_emo_cm.pdf")
    plt.close(fig)

    # ---------- macros for the abstract / text
    b = next(r for r in rows if r["name"] == "base")
    M = {"nVideos": len(y_), "nSeeds": len(P_), "emoAcc": fmt(b["emo"]["acc"]), "emoMF": fmt(b["emo"]["macro_f1"]),
         "iccMean": fmt(b["icc"]), "cccMean": fmt(b["ccc"]), "maeMean": fmt(b["mae"]), "nValidAU": int(valid.sum())}
    for key, n in (("emoAccPrior", "const"), ("emoAccLR", "lr_gt_au"), ("emoAccDirect", "direct_emotion")):
        r = next((r for r in rows if r["name"] == n), None)
        if r:
            M[key] = fmt(r["emo"]["acc"])
    lat = res / "latency.json"
    if lat.exists():
        j = json.load(open(lat))
        for k, v in j.items():
            if k.endswith("resnet18+tcn"):
                M["lat" + k.split("/")[0].capitalize()] = f"{v['ms_per_frame_amortised']:.1f}"
    (out / "numbers.tex").write_text("\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in M.items()))
    print("written to", out)


if __name__ == "__main__":
    main()
