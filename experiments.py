"""experiments.py -- all CHART experiments on recorded pools; writes results_<task>.json
usage: python experiments.py maze|sudoku  [R splits]"""
import sys, json, time, numpy as np
from scipy.optimize import curve_fit
from chart import *

task = sys.argv[1]; R = int(sys.argv[2]) if len(sys.argv) > 2 else 200
SIG = [0.25, 0.5, 1.0]; KMAX = 16; TMAX = 16; TB = 6;   # TB: baseline depth = training depth
DELTA = 0.1; MU = 0.25; COST_REF = KMAX*TMAX
ALPHAS = [0.01, 0.02, 0.05, 0.1]
d = np.load(f'pool_{task}_main.npz'); diff = d['difficulty']; n = len(diff)
ans = {s: d[f'ans_{s}'] for s in [0.0]+SIG}; cor = {s: d[f'cor_{s}'] for s in [0.0]+SIG}; qv = {s: d[f'q_{s}'] for s in [0.0]+SIG}
res = dict(task=task, n=n, R=R, delta=DELTA, alphas=ALPHAS, mu=MU)
t0 = time.time()

# ------------------------------------------------------------------ E1 depth curves
E1 = dict(T=list(range(1, TMAX+1)), det=cor[0.0][:, 0].mean(0).tolist())
for s in SIG:
    E1[f'single_{s}'] = cor[s].mean((0, 1)).tolist()
    E1[f'pass16_{s}'] = cor[s].any(1).mean(0).tolist()
    E1[f'maj16_{s}'] = [float(majority_static(ans[s], cor[s], 16, t)[1].mean()) for t in range(TMAX)]
res['E1'] = E1; print('E1 done', time.time()-t0, flush=True)

# ------------------------------------------------------------------ configuration families
fam = {}
# (a) CHART-seq : sigma, m, k0, theta0, beta
grid_seq = [(s, m, k0, th, be) for s in SIG for m in (1, 2, 3) for k0 in (1, 2, 3)
            for th in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9) for be in (1.0, 0.5, 0.3, 0.2, 0.1, 0.05, 0.02)
            if not (be == 1.0 and th != 0.5)]
halt = {}
for s in SIG:
    for m in (1, 2, 3):
        halt[(s, m)] = halted_outputs(ans[s], cor[s], stability_halt(ans[s], m, TMAX))
A_, E_, C_ = [], [], []
for (s, m, k0, th, be) in grid_seq:
    A, C, cst = halt[(s, m)]
    acc, cc, used, comp = consensus_eprocess(A, C, cst, k0, th, be, KMAX)
    A_.append(acc); E_.append(~cc); C_.append(comp)
fam['CHART-seq'] = (np.array(A_), np.array(E_), np.array(C_), grid_seq, 'fst')
print('seq family', time.time()-t0, flush=True)
# (b) CHART-static : sigma, K, T, tau  (ablation: no e-process, no depth halting)
grid_st = []; A_, E_, C_ = [], [], []
maj = {}
for s in SIG:
    for K in (2, 4, 8, 16):
        for T in (2, 4, 6, 8, 12, 16):
            agr, cc, cst = majority_static(ans[s], cor[s], K, T-1); maj[(s, K, T)] = (agr, cc)
            for tau in (0.5, 0.625, 0.75, 0.875, 1.0):
                grid_st.append((s, K, T, tau)); A_.append(agr >= tau - 1e-9); E_.append(~cc); C_.append(cst)
fam['CHART-static'] = (np.array(A_), np.array(E_), np.array(C_), grid_st, 'fst')
# (c) TRM deterministic + Q-threshold certified by LTT (guaranteed, single pass) -- SGR/CRC-style
qgrid = np.linspace(10, -4, 57)                              # strict -> loose (a-priori order)
sc = qv[0.0][:, 0, TB-1]; e0 = ~cor[0.0][:, 0, TB-1]
fam['TRM+Q-LTT'] = (np.array([sc >= g for g in qgrid]), np.tile(e0, (len(qgrid), 1)), np.full((len(qgrid), n), float(TB)), list(qgrid), 'fst_given')
fam['TRM+Q-naive'] = (fam['TRM+Q-LTT'][0], fam['TRM+Q-LTT'][1], fam['TRM+Q-LTT'][2], list(qgrid), 'naive')
# (d) PTRM best-of-K by Q-head (sigma chosen per family member), certified by LTT on max-Q
grid_pt = []; A_, E_, C_ = [], [], []
for s in SIG:
    kstar = qv[s][:, :, TB-1].argmax(1); smax = qv[s][:, :, TB-1].max(1); ep = ~cor[s][np.arange(n), kstar, TB-1]
    for g in qgrid: grid_pt.append((s, g)); A_.append(smax >= g); E_.append(ep); C_.append(np.full(n, float(KMAX*TB)))
# order: for each threshold (strict->loose) all sigmas -> a-priori order
order = sorted(range(len(grid_pt)), key=lambda i: (-grid_pt[i][1], grid_pt[i][0]))
fam['PTRM+Q-LTT'] = (np.array(A_)[order], np.array(E_)[order], np.array(C_)[order], [grid_pt[i] for i in order], 'fst_given')
# (e) naive CHART (empirical selection, no finite-sample test)
fam['CHART-naive'] = fam['CHART-seq'][:4] + ('naive',)
# ablation: no stability halting (every trajectory runs Tmax steps)
A_, E_, C_ = [], [], []
for (s, m, k0, th, be) in grid_seq:
    if m != 1: continue
    idxT = np.full(ans[s].shape[:2], TMAX-1); A, C, cst = halted_outputs(ans[s], cor[s], idxT)
    acc, cc, used, comp = consensus_eprocess(A, C, cst, k0, th, be, KMAX); A_.append(acc); E_.append(~cc); C_.append(comp)
fam['Abl: no halting'] = (np.array(A_), np.array(E_), np.array(C_), [g for g in grid_seq if g[1] == 1], 'fst')
sel = [i for i, g in enumerate(grid_seq) if g[2] == 1]
fam['Abl: k0=1 only'] = (fam['CHART-seq'][0][sel], fam['CHART-seq'][1][sel], fam['CHART-seq'][2][sel], [grid_seq[i] for i in sel], 'fst')
sel = [i for i, g in enumerate(grid_seq) if g[3] >= 0.5]
fam['Abl: theta0>=0.5'] = (fam['CHART-seq'][0][sel], fam['CHART-seq'][1][sel], fam['CHART-seq'][2][sel], [grid_seq[i] for i in sel], 'fst')
for s0 in SIG:
    sel = [i for i, g in enumerate(grid_seq) if g[0] == s0]
    fam[f'Abl: sigma={s0}'] = (fam['CHART-seq'][0][sel], fam['CHART-seq'][1][sel], fam['CHART-seq'][2][sel], [grid_seq[i] for i in sel], 'fst')
fam['CHART-bonf'] = fam['CHART-seq'][:4] + ('bonf',)
print('families built', time.time()-t0, flush=True)

# accept-all reference methods (no abstention)
ref = {}
ref['TRM (det.)'] = dict(err=float(e0.mean()), cost=float(TB))
ref['TRM (det.) T=16'] = dict(err=float((~cor[0.0][:, 0, TMAX-1]).mean()), cost=float(TMAX))
tq = np.where((qv[0.0][:, 0] > 0).any(1), (qv[0.0][:, 0] > 0).argmax(1), TMAX-1)
ih = stability_halt(ans[0.0], 1, TMAX)[:, 0]
ref['TRM+stability halt'] = dict(err=float((~cor[0.0][np.arange(n), 0, ih]).mean()), cost=float((ih+1).mean()))
for s in SIG:
    A, C, cst = halt[(s, 1)]; ref[f'Single noisy traj + halt s={s}'] = dict(err=float((~C[:, 0]).mean()), cost=float(cst[:, 0].mean()))
ref['TRM+ACT halt'] = dict(err=float((~cor[0.0][np.arange(n), 0, tq]).mean()), cost=float((tq+1).mean()))
for s in SIG:
    for TT in (TB, TMAX):
        kstar = qv[s][:, :, TT-1].argmax(1)
        ref[f'PTRM K=16 s={s} T={TT}'] = dict(err=float((~cor[s][np.arange(n), kstar, TT-1]).mean()), cost=float(KMAX*TT))
        ref[f'Majority K=16 s={s} T={TT}'] = dict(err=float((~majority_static(ans[s], cor[s], 16, TT-1)[1]).mean()), cost=float(KMAX*TT))
        ref[f'Oracle pass@16 s={s} T={TT}'] = dict(err=float((~cor[s][:, :, TT-1].any(1)).mean()), cost=float(KMAX*TT))
res['ref'] = ref

# ------------------------------------------------------------------ E3 main split experiment
E3 = {}; chosen = {}
for a in ALPHAS:
    E3[a] = {}; chosen[a] = {}
    for name, (ACC, ERR, COST, grid, meth) in fam.items():
        if name.startswith('Abl') and a not in (0.02, 0.05): continue
        rr = []; ch = []
        rng = np.random.default_rng(2026)
        for r in range(R):
            perm = rng.permutation(n); cal, test = perm[:n//2], perm[n//2:]
            m, _ = ltt_select(ACC, ERR, COST, cal, a, DELTA, MU, COST_REF, method=meth, rng=rng)
            ev = evaluate(ACC, ERR, COST, m, test); rr.append([ev['risk'], ev['cov'], ev['cost']]); ch.append(m)
        rr = np.array(rr)
        E3[a][name] = dict(risk=float(rr[:, 0].mean()), risk_sd=float(rr[:, 0].std()), viol=float((rr[:, 0] > a).mean()),
                           cov=float(rr[:, 1].mean()), cov_sd=float(rr[:, 1].std()), cost=float(rr[:, 2].mean()),
                           risks=rr[:, 0].tolist(), empty=float(np.mean(np.array(ch) < 0)))
        vals, cnts = np.unique(ch, return_counts=True); chosen[a][name] = int(vals[np.argmax(cnts)])
        print(f'alpha {a} {name:14s} risk {rr[:,0].mean():.4f} viol {(rr[:,0]>a).mean():.3f} cov {rr[:,1].mean():.3f} cost {rr[:,2].mean():.1f}', flush=True)
res['E3'] = {str(a): v for a, v in E3.items()}
res['chosen'] = {str(a): {k: (str(fam[k][3][v]) if v >= 0 else 'none') for k, v in c.items()} for a, c in chosen.items()}
print('E3 done', time.time()-t0, flush=True)

# ------------------------------------------------------------------ E2 agreement reliability  (sigma*, K=16, T=16)
s_star = 0.5
agr, cc = maj[(s_star, 16, 16)]
bins = np.linspace(0, 1, 11); E2 = dict(center=[], err=[], frac=[])
for lo, hi in zip(bins[:-1], bins[1:]):
    sel = (agr > lo) & (agr <= hi + 1e-9)
    if sel.sum() >= 10: E2['center'].append(float((lo+hi)/2)); E2['err'].append(float((~cc[sel]).mean())); E2['frac'].append(float(sel.mean()))
res['E2'] = E2

# ------------------------------------------------------------------ E5 adaptive compute vs difficulty (main alpha)
a_main = 0.05 if task == 'maze' else 0.05
ACC, ERR, COST, grid, _ = fam['CHART-seq']; mstar = chosen[a_main]['CHART-seq']
if mstar >= 0:
    qs = np.quantile(diff, np.linspace(0, 1, 7)); qs[-1] += 1
    E5 = dict(center=[], cost=[], cov=[], risk=[], static_cost=[])
    ms = chosen[a_main]['CHART-static']
    for lo, hi in zip(qs[:-1], qs[1:]):
        sel = (diff >= lo) & (diff < hi)
        E5['center'].append(float(diff[sel].mean())); E5['cost'].append(float(COST[mstar, sel].mean()))
        E5['cov'].append(float(ACC[mstar, sel].mean())); E5['risk'].append(float((ERR[mstar, sel] & ACC[mstar, sel]).sum()/max(ACC[mstar, sel].sum(), 1)))
        E5['static_cost'].append(float(fam['CHART-static'][2][ms, sel].mean()) if ms >= 0 else None)
    E5['config'] = str(grid[mstar]); res['E5'] = E5

# ------------------------------------------------------------------ E6 width-depth compute law (static majority, accept all)
E6 = dict(K=[1, 2, 4, 8, 16], budgets=[16, 32, 64, 128], err={})
s = s_star
for B in E6['budgets']:
    row = []
    for K in E6['K']:
        T = B // K
        if 1 <= T <= TMAX:
            if K == 1: row.append(float((~cor[s][:, 0, T-1]).mean()))
            else: row.append(float((~majority_static(ans[s], cor[s], K, T-1)[1]).mean()))
        else: row.append(None)
    E6['err'][B] = row
pT = np.array(cor[s].mean((0, 1)))                          # mean single-trajectory accuracy vs T
f = lambda T, pinf, c, rho: pinf - c*rho**T
try:
    (pinf, c, rho), _ = curve_fit(f, np.arange(1, TMAX+1), pT, p0=[pT[-1], 1.0, 0.7], bounds=([0, 0, 0.01], [1, 10, 0.999]))
    Dl = pinf - 0.5; Ts = np.linspace(0.1, 60, 60000); h = c*rho**Ts*(1 + 2*Ts*np.log(1/rho))
    ok = (Dl - c*rho**Ts) > 0
    Tstar = float(Ts[ok][np.argmin(np.abs(h[ok] - Dl))]) if ok.any() and Dl > 0 else None
    E6['fit'] = dict(pinf=float(pinf), c=float(c), rho=float(rho), Tstar=Tstar)
except Exception as ex:
    E6['fit'] = dict(error=str(ex))
# per-instance Hoeffding prediction  eps(K,T) <= P(p_x<=1/2) + E[exp(-2K(p_x-1/2)^2) 1{p_x>1/2}]
px = cor[s].mean(1)                                          # [n,T] per-instance trajectory accuracy estimate
E6['bound'] = {K: [float(np.mean(np.where(px[:, t] > 0.5, np.exp(-2*K*(px[:, t]-0.5)**2), 1.0))) for t in range(TMAX)] for K in (1, 4, 16)}
E6['emp'] = {K: [float((~majority_static(ans[s], cor[s], K, t)[1]).mean()) if K > 1 else float((~cor[s][:, 0, t]).mean()) for t in range(TMAX)] for K in (1, 4, 16)}
res['E6'] = E6
print('E6 done', time.time()-t0, flush=True)

# ------------------------------------------------------------------ E7 e-process sample size vs consensus strength
A, C, cst = halt[(s_star, 1)]
th, be = 0.5, 0.1
acc, cc, used, comp = consensus_eprocess(A, C, cst, 1, th, be, KMAX)
p_hat = (A == A[:, :1]).mean(1)                              # agreement with anchor over all 16 draws
E7 = dict(center=[], used=[], accrate=[], wald=[])
for lo, hi in [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 0.97), (0.97, 1.01)]:
    sel = (p_hat > lo) & (p_hat <= hi)
    if sel.sum() < 10: continue
    p = min(p_hat[sel].mean(), 0.999)
    bs = (p - th)/(th*(1-th)); bs = min(bs, 0.95/th)
    g = p*np.log1p(bs*(1-th)) + (1-p)*np.log1p(-bs*th)
    E7['center'].append(float(p_hat[sel].mean())); E7['used'].append(float(used[sel].mean() - 1))
    E7['accrate'].append(float(acc[sel].mean())); E7['wald'].append(float((np.log(1/be) + np.log1p(bs*(1-th)))/g))
res['E7'] = E7

# ------------------------------------------------------------------ E8 sigma ablation: dominant-wrong-attractor mass
E8 = dict(sigma=SIG, eps={}, acc_single=[], maj=[], pass16=[])
for s in SIG:
    A, C, _ = halt[(s, 1)]
    E8['acc_single'].append(float(C.mean())); E8['pass16'].append(float(C.any(1).mean()))
    E8['maj'].append(float(majority_static(ans[s], cor[s], 16, TMAX-1)[1].mean()))
    for th in (0.5, 0.7, 0.9):
        eps = []
        for i in range(n):
            vals, inv, cnt = np.unique(A[i], return_inverse=True, return_counts=True)
            pi = cnt/len(A[i]); wrong = np.array([not C[i, np.flatnonzero(inv == j)[0]] for j in range(len(vals))])
            eps.append(pi[(pi > th) & wrong].sum())
        E8['eps'].setdefault(str(th), []).append(float(np.mean(eps)))
res['E8'] = E8
print('E8 done', time.time()-t0, flush=True)

json.dump(res, open(f'results_{task}.json', 'w'))
np.savez_compressed(f'families_{task}.npz', **{k.replace('+','p').replace(' ','_'): v[0] for k, v in fam.items() if k in ('CHART-seq',)})
print('saved results', time.time()-t0)
