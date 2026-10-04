"""shift_v3.py -- E7: size shift 15x15 -> 19x19 with a label budget on the target distribution.
Recalibrate with n_t labelled 19x19 mazes (n_t = 50 ... 1000); risk measured on 1000 held-out 19x19 mazes."""
import json, numpy as np
from chart import *
from v3lib import load, families
S, _ = families(load('pool_maze_main.npz'), ablation=False); Tg, _ = families(load('pool_maze_ood19.npz'), ablation=False)
ns, nt = S['CHART'][0].shape[1], Tg['CHART'][0].shape[1]; R = 200; out = {}
for a in (0.02, 0.05):
    out[str(a)] = {}
    for name in ('CHART', 'CHART-bonf', 'TRM+Q-LTT'):
        out[str(a)][name] = {}
        for nt_lab in (0, 50, 100, 250, 500, 1000):
            rng = np.random.default_rng(17); rows = []
            for r in range(R):
                pt = rng.permutation(nt); test = pt[nt//2:]
                if nt_lab == 0: cal = rng.permutation(ns)[:ns//2]; src = S
                else: cal = pt[:nt_lab]; src = Tg
                m, _ = ltt_select_v3(*src[name][:3], cal, a, 0.1, 0.25, 256, method=src[name][4], rng=rng)
                ev = evaluate(*Tg[name][:3], m, test); rows.append([ev['risk'], ev['cov'], ev['cost'], m < 0])
            X = np.array(rows, float)
            out[str(a)][name][nt_lab] = dict(risk=X[:, 0].mean(), viol=float((X[:, 0] > a).mean()), cov=X[:, 1].mean(), nfe=X[:, 2].mean(), empty=X[:, 3].mean())
            print(a, name, nt_lab, out[str(a)][name][nt_lab], flush=True)
json.dump(out, open('results_shift_v3.json', 'w'), default=float)
