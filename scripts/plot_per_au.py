"""Per-AU ICC/CCC bar chart of the paper (Fig. 1), drawn from the per-AU table written by make_report.py.

    python scripts/plot_per_au.py paper/generated/tab_per_au.tex paper/generated/fig_per_au.pdf
"""
import re, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

src, out = sys.argv[1], sys.argv[2]
rows = [l for l in open(src).read().splitlines() if '&' in l][1:]
def ci(c):
    m = re.match(r'\s*(-?[\d.]+)\\,\[(-?[\d.]+),(-?[\d.]+)\]', c)
    return tuple(map(float, m.groups())) if m else (np.nan,) * 3
aus, icc, ccc = [], [], []
for r in rows:
    c = [x.replace('\\\\', '').strip() for x in r.split('&')]
    aus.append(int(c[0])); icc.append(ci(c[3])); ccc.append(ci(c[4]))
icc, ccc = np.array(icc), np.array(ccc)

plt.rcParams.update({'font.size': 7, 'axes.edgecolor': '#8a8a85', 'axes.linewidth': 0.6,
                     'xtick.color': '#3d3d3a', 'ytick.color': '#3d3d3a', 'axes.labelcolor': '#3d3d3a'})
fig, ax = plt.subplots(figsize=(7, 2.8))
x = np.arange(len(aus)); w = 0.36
for off, (v, lab, col) in zip((-w / 2 - 0.01, w / 2 + 0.01),
                              ((icc, 'ICC(3,1)', '#2a78d6'), (ccc, 'CCC', '#eb6834'))):
    ok = ~np.isnan(v[:, 0])
    ax.bar(x[ok] + off, v[ok, 0], w, color=col, label=lab, zorder=2, linewidth=0)
    ax.errorbar(x[ok] + off, v[ok, 0], yerr=[v[ok, 0] - v[ok, 1], v[ok, 2] - v[ok, 0]],
                fmt='none', ecolor='#3d3d3a', elinewidth=0.6, capsize=1.2, zorder=3)
for i, a in enumerate(aus):
    if a in (6, 9):
        ax.text(i, 0.04, 'prev. 0\n(excluded)', ha='center', va='bottom', fontsize=5.5, color='#6b6b66')
ax.set_xticks(x, [f'AU{a}' for a in aus], rotation=60)
ax.set_ylim(0, 1.0); ax.set_xlim(-0.6, len(aus) - 0.4)
ax.set_xlabel('Action Unit'); ax.set_ylabel('Agreement with MediaPipe pseudo-AU')
ax.yaxis.grid(True, color='#e4e3dd', linewidth=0.5, zorder=0); ax.set_axisbelow(True)
for s in ('top', 'right'): ax.spines[s].set_visible(False)
ax.legend(frameon=False, fontsize=7, loc='upper right', ncol=2)
fig.tight_layout()
fig.savefig(out); fig.savefig(out.replace('.pdf', '.png'), dpi=160)
