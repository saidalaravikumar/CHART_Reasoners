"""seeds.py -- robustness of empirical conclusions across training seeds.
Seeds 0, 1, 2 of the maze reasoner (same recipe, same number of optimizer steps) are recorded on the same 6000 mazes
(main pool 4000 + first 2000 of the fresh pool) and evaluated with the population-risk protocol (300 random
calibration splits, n = 2000, held-out risk on the remaining 4000).  usage: python seeds.py [R]"""
import sys, json, numpy as np
from chart import *
from v3lib import load, families, SIG, KMAX, TMAX, TB, QGRID
R = int(sys.argv[1]) if len(sys.argv) > 1 else 300
def sub(P, n):
    return dict(D=P['D'][:n], ans={s: v[:n] for s, v in P['ans'].items()}, cor={s: v[:n] for s, v in P['cor'].items()}, q={s: v[:n] for s, v in P['q'].items()})
def halt_q(P):
    n = len(P['D']); g, A_, E_, C_ = [], [], [], []
    for m in (1, 2, 3):
        idx = stability_halt(P['ans'][0.0][:, :1], m, TMAX)[:, 0]
        sc = P['q'][0.0][np.arange(n), 0, idx]; er = ~P['cor'][0.0][np.arange(n), 0, idx]
        for q in QGRID: g.append((m, q)); A_.append(sc >= q); E_.append(er); C_.append((idx+1).astype(float))
    o = sorted(range(len(g)), key=lambda i: (-g[i][1], g[i][0]))
    return (np.array(A_)[o], np.array(E_)[o], np.array(C_)[o], [g[i] for i in o], 'given_d1skip')
POOLS = {0: ('pool_maze_main.npz', 'pool_maze_fresh.npz'), 1: ('pool_maze_s1_main.npz', 'pool_maze_s1_fresh.npz'), 2: ('pool_maze_s2_main.npz', 'pool_maze_s2_fresh.npz')}
METH = [('CHART', 'CHART'), ('CHART, Bonferroni', 'CHART-bonf'), ('PTRM + halt + majority + LTT', 'Stop: fixed-K'), ('PTRM + majority + LTT', 'CHART-static'),
        ('TRM + halt + LTT', 'TRM + halt + LTT'), ('TRM + Q-head + LTT', 'TRM+Q-LTT'), ('PTRM + Q-head + LTT', 'PTRM+Q-LTT'), ('CHART without LTT (plug-in)', 'CHART-naive')]
out = {}
for seed, (pm, pf) in POOLS.items():
    Ps = [sub(load(pm), 4000), sub(load(pf), 2000)]; Fs = []
    for P in Ps:
        F, _ = families(P, ablation=True); F['TRM + halt + LTT'] = halt_q(P); Fs.append(F)
    F = {k: tuple(np.concatenate([Fs[0][k][i], Fs[1][k][i]], 1) for i in range(3)) + Fs[0][k][3:] for k in Fs[0]}
    Pc = {'cor': {s: np.concatenate([Ps[0]['cor'][s], Ps[1]['cor'][s]]) for s in [0.0]+SIG}, 'ans': {s: np.concatenate([Ps[0]['ans'][s], Ps[1]['ans'][s]]) for s in [0.0]+SIG}}
    n = len(Pc['cor'][0.0]); ih = stability_halt(Pc['ans'][0.0], 1, TMAX)[:, 0]
    acc = dict(det6=float(Pc['cor'][0.0][:, 0, TB-1].mean()), det16=float(Pc['cor'][0.0][:, 0, TMAX-1].mean()),
               peak=float(Pc['cor'][0.0][:, 0].mean(0).max()), peakT=int(Pc['cor'][0.0][:, 0].mean(0).argmax()+1),
               halt_err=float((~Pc['cor'][0.0][np.arange(n), 0, ih]).mean()), halt_nfe=float((ih+1).mean()))
    out[seed] = dict(acc=acc, res={})
    print('seed', seed, acc, flush=True)
    N = F['CHART'][0].shape[1]
    for a in (0.02, 0.05):
        out[seed]['res'][str(a)] = {}
        for lab, key in METH:
            ACC, ERR, COST, grid, meth = F[key]; rng = np.random.default_rng(99); rows = []
            for r in range(R):
                perm = rng.permutation(N); cal, rest = perm[:2000], perm[2000:]
                m = ltt_select_v3(ACC, ERR, COST, cal, a, 0.1, 0.25, 256, method=meth, rng=rng)[0]
                ev = evaluate(ACC, ERR, COST, m, rest); rows.append([ev['risk'], ev['cov'], ev['cost'], m < 0])
            X = np.array(rows, float); k = int((X[:, 0] > a).sum())
            out[seed]['res'][str(a)][lab] = dict(risk=X[:, 0].mean(), risk_sd=X[:, 0].std(), viol=k/R, viol_ci=list(binom_ci(k, R)), cov=X[:, 1].mean(), cov_sd=X[:, 1].std(), nfe=X[:, 2].mean())
            print(seed, a, f'{lab:30s} risk {100*X[:,0].mean():.2f} viol {100*k/R:.1f} cov {100*X[:,1].mean():.1f} nfe {X[:,2].mean():.1f}', flush=True)
    json.dump(out, open('results_seeds.json', 'w'), default=float)
print('saved')
