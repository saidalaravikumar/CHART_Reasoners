"""e1_population.py -- E1 done with independent calibration draws.
The main and fresh pools are merged (maze: 14000, Sudoku: 6000 instances). In each of R repetitions a
calibration set of n_cal instances is drawn at random, LTT selects lambda_hat, and R(lambda_hat) is
estimated on ALL remaining instances (12000 / 4500), a proxy for the population risk that is disjoint
from the calibration set.  This removes the dependence created by re-splitting a single 4000-instance pool."""
import sys, json, numpy as np
from chart import *
from v3lib import load, families
task = sys.argv[1]; R = int(sys.argv[2]) if len(sys.argv) > 2 else 300
ncal = 2000 if task == 'maze' else 1500
F1, _ = families(load(f'pool_{task}_main.npz'), ablation=False); F2, _ = families(load(f'pool_{task}_fresh.npz'), ablation=False)
F = {k: (np.concatenate([F1[k][0], F2[k][0]], 1), np.concatenate([F1[k][1], F2[k][1]], 1), np.concatenate([F1[k][2], F2[k][2]], 1), F1[k][3], F1[k][4]) for k in F1}
N = F['CHART'][0].shape[1]; out = dict(task=task, N=N, ncal=ncal, R=R, res={})
for a in (0.01, 0.02, 0.05, 0.1):
    out['res'][str(a)] = {}
    for name, (ACC, ERR, COST, grid, meth) in F.items():
        rng = np.random.default_rng(99); rows = []
        for r in range(R):
            perm = rng.permutation(N); cal, rest = perm[:ncal], perm[ncal:]
            m, _ = ltt_select_v3(ACC, ERR, COST, cal, a, 0.1, 0.25, 256, method=meth, rng=rng)
            ev = evaluate(ACC, ERR, COST, m, rest); rows.append([ev['risk'], ev['cov'], ev['cost'], m < 0])
        X = np.array(rows, float); v = X[:, 0] > a; k = int(v.sum())
        out['res'][str(a)][name] = dict(risk=X[:, 0].mean(), risk_sd=X[:, 0].std(), viol=k/R, viol_ci=list(binom_ci(k, R)),
                                        cov=X[:, 1].mean(), cov_sd=X[:, 1].std(), nfe=X[:, 2].mean(), nfe_sd=X[:, 2].std(), empty=X[:, 3].mean(),
                                        risks=X[:, 0].tolist())
        print(f"{a} {name:14s} risk {X[:,0].mean():.4f} viol {k/R:.3f} ci {binom_ci(k,R)[1]:.3f} cov {X[:,1].mean():.3f} nfe {X[:,2].mean():.1f} empty {X[:,3].mean():.2f}", flush=True)
json.dump(out, open(f'results_e1_{task}.json', 'w'), default=float)
