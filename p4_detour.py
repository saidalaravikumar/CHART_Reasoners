"""p4_detour.py -- compute and acceptance versus the detour forced by buildings on the unseen cities for the
most frequently selected configuration at alpha = 0.02 (at alpha = 0.05 CHART selects one halted pass for
every instance, so compute is flat by design).  Same draws as exp_real.py P3."""
import json, numpy as np
from chart import *
from v3lib import load, families
Pm, Pf = load('pool_street_main.npz'), load('pool_street_fresh.npz'); Im, If = np.load('street_main.npz'), np.load('street_fresh.npz')
F1, _ = families(Pm, ablation=False); F2, _ = families(Pf, ablation=False)
F = {k: tuple(np.concatenate([F1[k][i], F2[k][i]], 1) for i in range(3)) + F1[k][3:] for k in ('CHART', 'CHART-static', 'TRM+Q-LTT')}
N = F['CHART'][0].shape[1]
def detour(I, n):
    s = np.argwhere(I['X'][:n, 1] > 0)[:, 1:]; g = np.argwhere(I['X'][:n, 2] > 0)[:, 1:]
    return I['D'][:n] - np.abs(s - g).sum(1)
det = np.r_[detour(Im, len(Pm['D'])), detour(If, len(Pf['D']))]
e0 = np.r_[~Pm['cor'][0.0][:, 0, 5], ~Pf['cor'][0.0][:, 0, 5]]
out = {}
for a in (0.02, 0.05):
    ch = {}
    for name in ('CHART', 'CHART-static', 'TRM+Q-LTT'):
        rng = np.random.default_rng(99); c = []
        for r in range(300):
            perm = rng.permutation(N); m, _ = ltt_select_v3(*F[name][:3], perm[:2000], a, 0.1, 0.25, 256, method=F[name][4], rng=rng); c.append(m)
        v, k = np.unique(c, return_counts=True); ch[name] = int(v[np.argmax(k)])
    bins = [(0, 0), (1, 2), (3, 4), (5, 8), (9, 100)]; rows = []
    for lo, hi in bins:
        s = (det >= lo) & (det <= hi); row = dict(lo=lo, hi=hi, n=int(s.sum()), err_det=float(e0[s].mean()))
        for name in ch:
            ACC, ERR, COST = F[name][:3]; m = ch[name]
            row[name] = dict(nfe=float(COST[m][s].mean()), cov=float(ACC[m][s].mean()), risk=float((ERR[m] & ACC[m])[s].sum()/max(ACC[m][s].sum(), 1)))
        rows.append(row)
    out[str(a)] = dict(config={k: str(F[k][3][v]) for k, v in ch.items()}, bins=rows)
    print(a, out[str(a)]['config']); [print(r) for r in rows]
json.dump(out, open('results_real_detour.json', 'w'), default=float)
