"""experiments_v3.py -- evaluation and validation experiments for CHART:
  E1  population-risk check: calibrate on the main pool, evaluate the selected configuration on a
      large fresh pool (10000 mazes / 4500 Sudoku-Min puzzles) -> P(R(lambda_hat) <= alpha)
  E2  stopping-rule ablation under the SAME LTT: e-process (plug-in and mixture bet), r-agree,
      Adaptive-Consistency Beta stop, SPRT, fixed-K plurality with halting
  E3  calibration-size curve
  E5  a-priori ordered LTT: plain / D2-skip (v2 Remark 1) / D1-skip (valid) FWER on the fresh pool
  E7  size shift with a small labelled target set (label-budget curve)
  E10 error attribution (Corollary 1 terms), fallback rate and end-to-end cost
usage: python experiments_v3.py maze|sudoku [R]"""
import sys, json, time, numpy as np
from chart import *
from chart import _anchor

task = sys.argv[1]; R = int(sys.argv[2]) if len(sys.argv) > 2 else 200
SIG = [0.25, 0.5, 1.0]; KMAX = 16; TMAX = 16; TB = 6; DELTA = 0.1; MU = 0.25; CREF = KMAX*TMAX
ALPHAS = [0.01, 0.02, 0.05, 0.1]
t0 = time.time()

from v3lib import load, families, pop_risk, GRID_SEQ, QGRID

P = load(f'pool_{task}_main.npz'); PF = load(f'pool_{task}_fresh.npz')
F, H = families(P); FF, HF = families(PF)
n = len(P['D']); nF = len(PF['D'])
print('families', time.time()-t0, 'n', n, 'fresh', nF, flush=True)
res = dict(task=task, R=R, n=n, n_fresh=nF, delta=DELTA, alphas=ALPHAS)

# ---------------- E1 + E2 + E10 : split study with population risk on the fresh pool
E = {}
for a in ALPHAS:
    E[a] = {}
    for name, (ACC, ERR, COST, grid, meth) in F.items():
        rng = np.random.default_rng(2026); rows = []; ch = []
        for r in range(R):
            perm = rng.permutation(n); cal, test = perm[:n//2], perm[n//2:]
            m, _ = ltt_select_v3(ACC, ERR, COST, cal, a, DELTA, MU, CREF, method=meth, rng=rng)
            ev = evaluate(ACC, ERR, COST, m, test); rp, cp = pop_risk(FF, name, m)
            costp = float(FF[name][2][m].mean()) if m >= 0 else 0.0
            rows.append([ev['risk'], ev['cov'], ev['cost'], rp, cp, costp, m < 0]); ch.append(m)
        X = np.array(rows, float); vpop = (X[:, 3] > a)
        E[a][name] = dict(risk=X[:, 0].mean(), risk_se=X[:, 0].std()/np.sqrt(R), risk_sd=X[:, 0].std(),
                          viol_test=float((X[:, 0] > a).mean()), viol_pop=float(vpop.mean()),
                          viol_pop_ci=[float(x) for x in binom_ci(int(vpop.sum()), R)],
                          cov=X[:, 1].mean(), cov_sd=X[:, 1].std(), cost=X[:, 2].mean(), cost_sd=X[:, 2].std(),
                          risk_pop=X[:, 3].mean(), cov_pop=X[:, 4].mean(), cost_pop=X[:, 5].mean(), empty=X[:, 6].mean(),
                          risk_pop_all=X[:, 3].tolist(), chosen=[int(c) for c in ch])
        print(f"a={a} {name:16s} risk {X[:,0].mean():.4f} Rpop {X[:,3].mean():.4f} viol_test {(X[:,0]>a).mean():.3f} viol_pop {vpop.mean():.3f} cov {X[:,1].mean():.3f} nfe {X[:,2].mean():.1f} empty {X[:,6].mean():.2f}", flush=True)
res['E1'] = {str(a): {k: {kk: (float(vv) if isinstance(vv, (np.floating, float, np.bool_)) else vv) for kk, vv in v.items()} for k, v in d.items()} for a, d in E.items()}
print('E1/E2 done', time.time()-t0, flush=True)

# ---------------- E5 : skip policies for a-priori ordered LTT (Q-head family), FWER on the fresh pool
QA, QE, QC = F['TRM+Q-LTT'][:3]; trueR = np.array([(FF['TRM+Q-LTT'][1][m] & FF['TRM+Q-LTT'][0][m]).sum()/max(FF['TRM+Q-LTT'][0][m].sum(), 1) for m in range(len(QGRID))])
E5 = {}
for a in ALPHAS:
    E5[a] = {}
    for meth in ('given_plain', 'given_d2skip', 'given_d1skip'):
        rng = np.random.default_rng(5); bad = 0; cov = []
        for r in range(R):
            perm = rng.permutation(n); cal, test = perm[:n//2], perm[n//2:]
            m, cert = ltt_select_v3(QA, QE, QC, cal, a, DELTA, MU, CREF, method=meth, rng=rng)
            bad += any(trueR[c] > a for c in cert); cov.append(QA[m, test].mean() if m >= 0 else 0.0)
        E5[a][meth] = dict(fwer=bad/R, cov=float(np.mean(cov)))
res['E5'] = {str(a): v for a, v in E5.items()}
print('E5', res['E5'], flush=True)

# ---------------- E3 : calibration-size curve (CHART fixed sequence and Bonferroni), population risk
E3 = {}
a3 = 0.05 if task == 'maze' else 0.02
for ncal in (250, 500, 1000, 2000):
    E3[ncal] = {}
    for name in ('CHART', 'CHART-bonf', 'TRM+Q-LTT'):
        ACC, ERR, COST, grid, meth = F[name]; rng = np.random.default_rng(3); rows = []
        for r in range(R):
            cal = rng.permutation(n)[:ncal]
            m, _ = ltt_select_v3(ACC, ERR, COST, cal, a3, DELTA, MU, CREF, method=meth, rng=rng)
            rp, cp = pop_risk(FF, name, m); rows.append([rp, cp, m < 0])
        X = np.array(rows, float)
        E3[ncal][name] = dict(risk_pop=X[:, 0].mean(), cov_pop=X[:, 1].mean(), viol_pop=float((X[:, 0] > a3).mean()), empty=X[:, 2].mean())
res['E3'] = dict(alpha=a3, data={str(k): v for k, v in E3.items()})
print('E3', time.time()-t0, flush=True)

# ---------------- E10 : error attribution (Corollary 1) on the fresh pool for the most chosen CHART config
E10 = {}
for a in ALPHAS:
    ch = [c for c in E[a]['CHART']['chosen'] if c >= 0]
    if not ch: continue
    vals, cnts = np.unique(ch, return_counts=True); m = int(vals[np.argmax(cnts)]); s, mm, k0, th, be = GRID_SEQ[m]
    A, C, cst = HF[(s, mm)]; acc, cc, used, comp = consensus_eprocess(A, C, cst, k0, th, be, KMAX)
    anchor, anc_cor, _ = _anchor(A, C, k0)
    pi_anchor = (A == anchor[:, None]).mean(1)          # basin mass of the anchor from 16 trajectories
    err = acc & ~cc
    E10[a] = dict(config=[float(x) for x in GRID_SEQ[m]], cov=float(acc.mean()), joint_err=float(err.mean()),
                  cond_risk=float(err.sum()/max(acc.sum(), 1)),
                  err_low_mass=float((err & (pi_anchor <= th)).mean()), err_dominant_wrong=float((err & (pi_anchor > th)).mean()),
                  bound_tight=float(be*np.mean((pi_anchor <= th) & ~anc_cor) + np.mean((pi_anchor > th) & ~anc_cor)),
                  fallback=float(1 - acc.mean()), nfe=float(comp.mean()))
res['E10'] = {str(a): v for a, v in E10.items()}
print('E10', res['E10'], flush=True)
json.dump(res, open(f'results_v3_{task}.json', 'w'))
print('saved', time.time()-t0)
