"""exp_real.py -- practical study on real maps (MovingAI city street maps and Dragon Age: Origins indoor maps).
Uses the same configuration families (v3lib.families), the same LTT layer and the same population-risk
protocol as the synthetic study, so every number is directly comparable with Section VIII.
  P1 depth curves on unseen cities (overthinking, value of width)
  P2 accept-all references (no abstention): TRM, TRM+ACT halt, PTRM best-of-16, plurality-of-16, pass@16
  P3 population risk with independent calibration draws on the 4 unseen cities (main+fresh pools merged)
  P4 compute versus difficulty (detour length) for the selected CHART configuration
  P5 city -> indoor shift (Dragon Age maps): no recalibration, then recalibration with n_t indoor labels
usage: python exp_real.py [R]"""
import sys, json, time, numpy as np
from chart import *
from v3lib import load, families, SIG, KMAX, TMAX, TB

R = int(sys.argv[1]) if len(sys.argv) > 1 else 300
DELTA, MU, CREF, ALPHAS = 0.1, 0.25, KMAX*TMAX, [0.01, 0.02, 0.05, 0.1]
t0 = time.time(); out = dict(R=R, delta=DELTA, alphas=ALPHAS)
Pm, Pf, Pd = load('pool_street_main.npz'), load('pool_street_fresh.npz'), load('pool_street_dao.npz')
Im, If, Idao = np.load('street_main.npz'), np.load('street_fresh.npz'), np.load('dao_main.npz')

def detour(I, n):
    s = np.argwhere(I['X'][:n, 1] > 0)[:, 1:]; g = np.argwhere(I['X'][:n, 2] > 0)[:, 1:]
    return I['D'][:n] - np.abs(s - g).sum(1)                     # extra steps forced by obstacles

# ---------------- P1 depth curves (unseen cities, main pool) and on the indoor pool
def depth(P):
    d = dict(T=list(range(1, TMAX+1)), det=P['cor'][0.0][:, 0].mean(0).tolist())
    for s in SIG:
        d[f'single_{s}'] = P['cor'][s].mean((0, 1)).tolist(); d[f'pass16_{s}'] = P['cor'][s].any(1).mean(0).tolist()
        d[f'maj16_{s}'] = [float(majority_static(P['ans'][s], P['cor'][s], 16, t)[1].mean()) for t in range(TMAX)]
    return d
out['P1'] = dict(street=depth(Pm), dao=depth(Pd))
print('P1 det street', np.round(out['P1']['street']['det'], 3), flush=True)

# ---------------- P2 accept-all references (street main + dao)
def refs(P):
    cor, ans, qv = P['cor'], P['ans'], P['q']; n = len(P['D']); r = {}
    r['TRM (det.) T=6'] = dict(err=float((~cor[0.0][:, 0, TB-1]).mean()), nfe=float(TB))
    r['TRM (det.) T=16'] = dict(err=float((~cor[0.0][:, 0, TMAX-1]).mean()), nfe=float(TMAX))
    tq = np.where((qv[0.0][:, 0] > 0).any(1), (qv[0.0][:, 0] > 0).argmax(1), TMAX-1)
    r['TRM+ACT halt'] = dict(err=float((~cor[0.0][np.arange(n), 0, tq]).mean()), nfe=float((tq+1).mean()))
    ih = stability_halt(ans[0.0], 1, TMAX)[:, 0]
    r['TRM+stability halt'] = dict(err=float((~cor[0.0][np.arange(n), 0, ih]).mean()), nfe=float((ih+1).mean()))
    for s in SIG:
        ks = qv[s][:, :, TB-1].argmax(1)
        r[f'PTRM K=16 s={s}'] = dict(err=float((~cor[s][np.arange(n), ks, TB-1]).mean()), nfe=float(KMAX*TB))
        r[f'Plurality K=16 s={s}'] = dict(err=float((~majority_static(ans[s], cor[s], 16, TB-1)[1]).mean()), nfe=float(KMAX*TB))
        r[f'pass@16 s={s}'] = dict(err=float((~cor[s][:, :, TB-1].any(1)).mean()), nfe=float(KMAX*TB))
    return r
out['P2'] = dict(street=refs(Pm), dao=refs(Pd))
for k, v in out['P2']['street'].items(): print(f'  {k:24s} err {v["err"]:.4f} nfe {v["nfe"]:.1f}', flush=True)

# ---------------- P3 population risk with independent calibration draws (street main + fresh)
F1, _ = families(Pm, ablation=True); F2, _ = families(Pf, ablation=True)
F = {k: (np.concatenate([F1[k][0], F2[k][0]], 1), np.concatenate([F1[k][1], F2[k][1]], 1),
         np.concatenate([F1[k][2], F2[k][2]], 1), F1[k][3], F1[k][4]) for k in F1}
N = F['CHART'][0].shape[1]; NCAL = 2000; out['P3'] = dict(N=N, ncal=NCAL, res={}); chosen = {}
print('families', time.time()-t0, 'N', N, flush=True)
for a in ALPHAS:
    out['P3']['res'][str(a)] = {}; chosen[a] = {}
    for name, (ACC, ERR, COST, grid, meth) in F.items():
        rng = np.random.default_rng(99); rows = []; ch = []
        for r in range(R):
            perm = rng.permutation(N); cal, rest = perm[:NCAL], perm[NCAL:]
            m, _ = ltt_select_v3(ACC, ERR, COST, cal, a, DELTA, MU, CREF, method=meth, rng=rng)
            ev = evaluate(ACC, ERR, COST, m, rest); rows.append([ev['risk'], ev['cov'], ev['cost'], m < 0]); ch.append(m)
        X = np.array(rows, float); v = X[:, 0] > a; k = int(v.sum())
        vals, cnts = np.unique(ch, return_counts=True); chosen[a][name] = int(vals[np.argmax(cnts)])
        out['P3']['res'][str(a)][name] = dict(risk=X[:, 0].mean(), risk_sd=X[:, 0].std(), viol=k/R, viol_ci=list(binom_ci(k, R)),
                                             cov=X[:, 1].mean(), cov_sd=X[:, 1].std(), nfe=X[:, 2].mean(), nfe_sd=X[:, 2].std(),
                                             empty=X[:, 3].mean(), risks=X[:, 0].tolist(),
                                             mode_cfg=str(grid[chosen[a][name]]) if chosen[a][name] >= 0 else 'none')
        print(f"{a} {name:14s} risk {X[:,0].mean():.4f} viol {k/R:.3f} cov {X[:,1].mean():.3f} nfe {X[:,2].mean():.1f} empty {X[:,3].mean():.2f}", flush=True)

# ---------------- P4 compute and acceptance versus detour (selected CHART configuration at alpha=0.05)
det = np.r_[detour(Im, len(Pm['D'])), detour(If, len(Pf['D']))]
m = chosen[0.05]['CHART']; ACC, ERR, COST = F['CHART'][:3]; ACCq = F['TRM+Q-LTT'][0][chosen[0.05]['TRM+Q-LTT']] if chosen[0.05]['TRM+Q-LTT'] >= 0 else np.zeros(N, bool)
bins = [(0, 0), (1, 2), (3, 4), (5, 8), (9, 100)]; P4 = []
for lo, hi in bins:
    s = (det >= lo) & (det <= hi)
    P4.append(dict(lo=lo, hi=hi, n=int(s.sum()), nfe=float(COST[m][s].mean()), cov=float(ACC[m][s].mean()),
                   risk=float((ERR[m] & ACC[m])[s].sum()/max(ACC[m][s].sum(), 1)), cov_q=float(ACCq[s].mean()),
                   err_det=float(np.r_[~Pm['cor'][0.0][:, 0, TB-1], ~Pf['cor'][0.0][:, 0, TB-1]][s].mean())))
out['P4'] = dict(config=str(F['CHART'][3][m]), bins=P4); print('P4', P4, flush=True)

# ---------------- P5 city -> indoor shift with a label budget on the indoor maps
Fd, _ = families(Pd, ablation=False); nd = Fd['CHART'][0].shape[1]; out['P5'] = {}
for a in (0.02, 0.05, 0.1):
    out['P5'][str(a)] = {}
    for name in ('CHART', 'CHART-bonf', 'TRM+Q-LTT', 'PTRM+Q-LTT'):
        out['P5'][str(a)][name] = {}
        for nt in (0, 50, 100, 250, 500, 1000):
            rng = np.random.default_rng(17); rows = []
            for r in range(R):
                pt = rng.permutation(nd); test = pt[nd//2:]
                if nt == 0: cal = rng.permutation(N)[:NCAL]; src = F
                else: cal = pt[:nt]; src = Fd
                mm, _ = ltt_select_v3(*src[name][:3], cal, a, DELTA, MU, CREF, method=src[name][4], rng=rng)
                ev = evaluate(*Fd[name][:3], mm, test); rows.append([ev['risk'], ev['cov'], ev['cost'], mm < 0])
            X = np.array(rows, float)
            out['P5'][str(a)][name][nt] = dict(risk=X[:, 0].mean(), viol=float((X[:, 0] > a).mean()), cov=X[:, 1].mean(), nfe=X[:, 2].mean(), empty=X[:, 3].mean())
            print('P5', a, name, nt, {k: round(float(v), 4) for k, v in out['P5'][str(a)][name][nt].items()}, flush=True)
json.dump(out, open('results_real.json', 'w'), default=float)
print('saved', time.time()-t0)
