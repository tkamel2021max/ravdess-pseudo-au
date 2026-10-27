"""Actor-independent training / evaluation of the RGB student (pseudo-AU + emotion, multi-task).

Design (one factor changed at a time w.r.t. `base`):
  backbone : resnet18 | mobilenet_v3_small | vit_b_16
  temporal : none | transformer (causal) | tcn (causal, dilations 1,2,4)
  loss     : ordinal (soft cumulative targets) | l1 | ccc (CCC + L1)
  direct_emotion : lam_au = 0  -> no AU supervision (direct CNN baseline)
Protocol: K actor-independent, gender-balanced folds; inside each training set a few actors are
held out for model selection (best epoch on validation); the test fold is evaluated ONCE.
Results are resumable: an experiment/seed with an existing .npz is skipped.
"""
import argparse
import copy
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from au_common import (K_LEVELS, N_AU, compute_q99, load_arrays, load_index, make_folds, save_oof,
                       scale_au, split_train_val)

T, CROP, D = 16, 112, 256
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)

BASE = dict(backbone="resnet18", temporal="transformer", loss="ordinal", lam_au=1.0, lam_emo=0.5)
EXPS = {
    "base": {},
    "bb_mobilenet": {"backbone": "mobilenet_v3_small"},
    "bb_vit": {"backbone": "vit_b_16"},
    "base_sel_emo": {"select": "emo"},          # base, epoch selected on emotion NLL only (RQ2 control)
    "vit_ccc": {"backbone": "vit_b_16", "loss": "ccc"},  # best-factor combination (not run in the paper)
    "tmp_none": {"temporal": "none"},
    "tmp_tcn": {"temporal": "tcn"},
    "loss_l1": {"loss": "l1"},
    "loss_ccc": {"loss": "ccc"},
    "direct_emotion": {"lam_au": 0.0},
}


# ----------------------------------------------------------------- model
class Backbone(nn.Module):
    def __init__(self, name, pretrained=True):
        super().__init__()
        w = "DEFAULT" if pretrained else None
        self.resize = None
        if name == "resnet18":
            m = torchvision.models.resnet18(weights=w)
            self.dim, m.fc = 512, nn.Identity()
        elif name == "mobilenet_v3_small":
            m = torchvision.models.mobilenet_v3_small(weights=w)
            self.dim, m.classifier = 576, nn.Identity()
        elif name == "vit_b_16":
            m = torchvision.models.vit_b_16(weights=w)
            self.dim, m.heads, self.resize = 768, nn.Identity(), 224
        else:
            raise ValueError(name)
        self.m = m

    def forward(self, x):
        if self.resize:
            x = F.interpolate(x, size=self.resize, mode="bilinear", align_corners=False)
        return self.m(x)


class CausalBlock(nn.Module):
    def __init__(self, d, dil):
        super().__init__()
        self.pad, self.conv, self.norm = 2 * dil, nn.Conv1d(d, d, 3, dilation=dil), nn.LayerNorm(d)

    def forward(self, x):                     # (B,T,d)
        h = self.conv(F.pad(x.transpose(1, 2), (self.pad, 0))).transpose(1, 2)
        return x + F.gelu(self.norm(h))


class Net(nn.Module):
    def __init__(self, backbone, temporal, loss, pretrained=True, **_):
        super().__init__()
        self.loss_type, self.temporal = loss, temporal
        self.bb = Backbone(backbone, pretrained)
        self.proj = nn.Sequential(nn.Linear(self.bb.dim, D), nn.GELU(), nn.Dropout(0.1))
        if temporal == "transformer":
            self.pos = nn.Parameter(torch.randn(1, T, D) * 0.02)
            layer = nn.TransformerEncoderLayer(D, 4, 2 * D, 0.1, activation="gelu", batch_first=True, norm_first=True)
            self.enc = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
            self.register_buffer("mask", torch.triu(torch.ones(T, T, dtype=torch.bool), 1))  # causal
        elif temporal == "tcn":
            self.tcn = nn.Sequential(*[CausalBlock(D, d) for d in (1, 2, 4)])
        self.au_head = nn.Linear(D, N_AU * (K_LEVELS if loss == "ordinal" else 1))
        self.emo_head = nn.Linear(D, 8)

    def forward(self, x):                     # x (B,T,3,H,W)
        B, Tn = x.shape[:2]
        h = self.proj(self.bb(x.flatten(0, 1))).view(B, Tn, D)
        if self.temporal == "transformer":
            h = self.enc(h + self.pos[:, :Tn], mask=self.mask[:Tn, :Tn])
        elif self.temporal == "tcn":
            h = self.tcn(h)
        out = self.au_head(h)
        if self.loss_type == "ordinal":
            out = out.view(B, Tn, N_AU, K_LEVELS)
        return out, self.emo_head(h.mean(1))

    def au_hat(self, out):
        return torch.sigmoid(out).sum(-1) if self.loss_type == "ordinal" else 5 * torch.sigmoid(out)


def ccc_t(p, y):
    p, y = p.flatten(0, 1), y.flatten(0, 1)
    mp, my = p.mean(0), y.mean(0)
    cov = ((p - mp) * (y - my)).mean(0)
    return 2 * cov / (p.var(0, unbiased=False) + y.var(0, unbiased=False) + (mp - my) ** 2 + 1e-3)


def au_loss(out, y, kind):
    if kind == "ordinal":
        # y in [0,5]; soft cumulative targets tau_k = clip(y-k,0,1); mean over B,T,A,K (1/(BTAK))
        k = torch.arange(K_LEVELS, device=y.device).view(1, 1, 1, K_LEVELS)
        return F.binary_cross_entropy_with_logits(out, torch.clamp(y.unsqueeze(-1) - k, 0, 1))
    yh = 5 * torch.sigmoid(out)
    l1 = (yh - y).abs().mean() / 5
    return l1 if kind == "l1" else l1 + (1 - ccc_t(yh, y)).mean()


# ----------------------------------------------------------------- batching
def make_batch(frames, aus, q99, vids, starts, train, dev):
    xs, ys = [], []
    for v, s in zip(vids, starts):
        n = len(frames[v])
        ii = np.minimum(np.arange(s, s + T), n - 1)
        H = frames[v].shape[1]
        oy, ox = (np.random.randint(0, H - CROP + 1, 2) if train else ((H - CROP) // 2,) * 2)
        xs.append(frames[v][ii][:, oy:oy + CROP, ox:ox + CROP])
        ys.append(scale_au(aus[v][ii], q99))
    x = torch.from_numpy(np.stack(xs)).to(dev).permute(0, 1, 4, 2, 3).float() / 255
    if train:
        B = x.shape[0]
        flip = torch.rand(B, device=dev) < 0.5           # AUs are bilateral averages -> flip-invariant
        x = torch.where(flip.view(B, 1, 1, 1, 1), x.flip(-1), x)
        c = (0.8 + 0.4 * torch.rand(B, 1, 1, 1, 1, device=dev))
        b = (torch.rand(B, 1, 1, 1, 1, device=dev) - 0.5) * 0.2
        x = (x * c + b).clamp(0, 1)
    x = (x - MEAN.to(dev)) / STD.to(dev)
    return x, torch.from_numpy(np.stack(ys)).to(dev)


@torch.no_grad()
def predict(model, frames, aus, q99, vids, dev, bs=32):
    """Per-frame AU predictions (non-overlapping windows, last one aligned to the end) and
    video-level emotion probabilities (mean of clip softmax)."""
    model.eval()
    au_p = [np.zeros((len(frames[v]), N_AU), np.float32) for v in vids]
    esum, ecnt = np.zeros((len(vids), 8)), np.zeros(len(vids))
    jobs = []
    for j, v in enumerate(vids):
        n = len(frames[v])
        st = list(range(0, n - T + 1, T)) if n >= T else [0]
        if n > T and st[-1] + T < n:
            st.append(n - T)
        jobs += [(j, s) for s in st]
    for i in range(0, len(jobs), bs):
        chunk = jobs[i:i + bs]
        x, _ = make_batch(frames, aus, q99, [vids[j] for j, _ in chunk], [s for _, s in chunk], False, dev)
        with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
            out, el = model(x)
        yh = model.au_hat(out.float()).cpu().numpy()
        pr = F.softmax(el.float(), -1).cpu().numpy()
        for k, (j, s) in enumerate(chunk):
            n = len(frames[vids[j]])
            ii = np.minimum(np.arange(s, s + T), n - 1)
            au_p[j][ii] = yh[k]
            esum[j] += pr[k]
            ecnt[j] += 1
    return au_p, esum / ecnt[:, None]


def train_fold(cfg, frames, aus, emo, tr, va, q99, seed, args, dev):
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    model = Net(**cfg).to(dev)
    bbp = list(model.bb.parameters())
    ids = {id(p) for p in bbp}
    other = [p for p in model.parameters() if id(p) not in ids]
    lr_bb = args.lr * (0.15 if cfg["backbone"] == "vit_b_16" else 0.3)
    opt = torch.optim.AdamW([{"params": bbp, "lr": lr_bb}, {"params": other, "lr": args.lr}], weight_decay=1e-2)
    steps = max(1, len(tr) * args.clips_per_video // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[lr_bb, args.lr], total_steps=steps * args.epochs, pct_start=0.1)
    scaler = torch.cuda.amp.GradScaler(enabled=dev.type == "cuda")
    best = (1e9, None, -1)
    for ep in range(args.epochs):
        model.train()
        for _ in range(steps):
            vids = rng.choice(tr, args.batch)
            starts = [rng.integers(0, max(len(frames[v]) - T, 0) + 1) for v in vids]
            x, y = make_batch(frames, aus, q99, vids, starts, True, dev)
            with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
                out, el = model(x)
            loss = 0.0
            if cfg["lam_au"] > 0:
                loss = loss + cfg["lam_au"] * au_loss(out.float(), y, cfg["loss"])
            if cfg["lam_emo"] > 0:
                loss = loss + cfg["lam_emo"] * F.cross_entropy(el.float(), torch.as_tensor(emo[vids], device=dev))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
        p_au, p_emo = predict(model, frames, aus, q99, va, dev)
        l1 = np.mean([np.abs(p - scale_au(aus[v], q99)).mean() for p, v in zip(p_au, va)]) / 5
        nll = -np.log(p_emo[np.arange(len(va)), emo[va]] + 1e-8).mean()
        crit = nll if cfg.get("select") == "emo" else (cfg["lam_au"] > 0) * l1 + (cfg["lam_emo"] > 0) * nll
        print(f"    ep{ep + 1:02d} loss={float(loss):.3f} val_L1n={l1:.3f} val_nll={nll:.3f}", flush=True)
        if crit < best[0]:
            best = (crit, copy.deepcopy(model.state_dict()), ep + 1)
    model.load_state_dict(best[1])
    return model, best[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", default="data/features")
    ap.add_argument("--out", default="results")
    ap.add_argument("--exps", nargs="+", default=["base"], help="names from EXPS or 'all'")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--clips_per_video", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--actors", type=int, nargs="*")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev = torch.device(args.device)
    exps = list(EXPS) if args.exps == ["all"] else args.exps

    idx = load_index(args.feat, args.actors)
    frames, aus, _ = load_arrays(args.feat, idx)
    emo = idx.emotion.values
    actors = sorted(idx.actor.unique())
    folds = make_folds(actors, args.folds)
    V = len(idx)
    print(f"{V} videos, {len(actors)} actors, {args.folds} folds, device={dev}")

    for name in exps:
        cfg = {**BASE, **EXPS[name]}
        for seed in args.seeds:
            path = Path(args.out) / name / f"seed{seed}.npz"
            if path.exists():
                print(f"[skip] {name} seed{seed}")
                continue
            t0 = time.time()
            au_true, au_pred, prob = [None] * V, [None] * V, np.zeros((V, 8))
            for fi, fold in enumerate(folds):
                te = np.flatnonzero(idx.actor.isin(fold).values)
                tr_act = [a for a in actors if a not in fold]
                va_act = split_train_val(tr_act, seed * 100 + fi)
                va = np.flatnonzero(idx.actor.isin(va_act).values)
                tr = np.flatnonzero(idx.actor.isin(set(tr_act) - set(va_act)).values)
                q99 = compute_q99(aus, tr)
                print(f"[{name} seed{seed} fold{fi + 1}/{len(folds)}] train={len(tr)} val={len(va)} test={len(te)}", flush=True)
                model, best_ep = train_fold(cfg, frames, aus, emo, tr, va, q99, seed, args, dev)
                p_au, p_emo = predict(model, frames, aus, q99, te, dev)
                for j, v in enumerate(te):
                    au_true[v], au_pred[v], prob[v] = scale_au(aus[v], q99), p_au[j], p_emo[j]
                print(f"    best epoch {best_ep}; test acc {np.mean(p_emo.argmax(1) == emo[te]):.3f}", flush=True)
            save_oof(path, au_true, au_pred, emo, prob)
            print(f"saved {path} ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
