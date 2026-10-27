"""Write the actor identifiers of every split (fold x seed), exactly as used by train_eval.py.

    python scripts/export_splits.py --out results/splits.csv
"""
import argparse, csv, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from au_common import make_folds, split_train_val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='results/splits.csv')
    ap.add_argument('--folds', type=int, default=4)
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    a = ap.parse_args()
    actors = list(range(1, 25))
    rows = []
    for fi, fold in enumerate(make_folds(actors, a.folds)):
        tr_act = [x for x in actors if x not in fold]
        for seed in a.seeds:
            va = sorted(split_train_val(tr_act, seed * 100 + fi))   # same call as train_eval.py
            tr = sorted(set(tr_act) - set(va))
            rows.append(dict(fold=fi + 1, seed=seed, test=' '.join(map(str, sorted(fold))),
                             validation=' '.join(map(str, va)), train=' '.join(map(str, tr))))
    with open(a.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    for r in rows:
        print(r['fold'], r['seed'], 'test:', r['test'], '| val:', r['validation'])


if __name__ == '__main__':
    main()
