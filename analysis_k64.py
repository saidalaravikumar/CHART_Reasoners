"""analysis_k64.py -- E4 (width-depth law on the full grid, K up to 64) and K_max sensitivity.
Uses pool_<task>_k64.npz: 1500 instances, 64 noisy trajectories (sigma=0.5), 16 outer steps."""
import sys, json, numpy as np
from chart import *
task = sys.argv[1]; R = int(sys.argv[2]) if len(sys.argv) > 2 else 200
d = np.load(f'pool_{task}_k64.npz'); ans, cor = d['ans_0.5'], d['cor_0.5']; n, K64, T = ans.shape
Ks = [1, 2, 4, 8, 16, 32, 64]; out = dict(task=task, n=int(n), Ks=Ks, T=list(range(1, T+1)))
# plurality error on the full (T, K) grid
err = {}
for K in Ks:
    err[K] = [float((~cor[:, 0, t]).mean()) if K == 1 else float((~majority_static(ans, cor, K, t)[1]).mean()) for t in range(T)]
out['err'] = {str(k): v for k, v in err.items()}
# iso-budget optimum and the population-bound optimum of Eq. (13)
px = cor.mean(1)                                             # [n, T] per-instance single-trajectory accuracy (64 draws)
def bound(K, t):
    p = px[:, t]; return float(np.mean(np.where(p > 0.5, np.exp(-2*K*(p-0.5)**2), 1.0)))
iso = {}
for B in (16, 32, 64, 128, 256, 512, 1024):
    pts = [(K, B//K) for K in Ks if B % K == 0 and 1 <= B//K <= T]
    if not pts: continue
    e = [err[K][t-1] for K, t in pts]; b = [bound(K, t-1) for K, t in pts]
    iso[B] = dict(K=[p[0] for p in pts], T=[p[1] for p in pts], err=e, bound=b,
                  T_emp=pts[int(np.argmin(e))][1], T_bound=pts[int(np.argmin(b))][1])
out['iso'] = {str(k): v for k, v in iso.items()}
# fitted homogeneous model p(T) = p_inf - c0 rho^T with bootstrap confidence intervals, and T* from Eq. (14)
from scipy.optimize import curve_fit
f = lambda t, pinf, c, rho: pinf - c*rho**t
def fit(pT):
    (pinf, c, rho), _ = curve_fit(f, np.arange(1, T+1), pT, p0=[pT[-1], 1.0, 0.5], bounds=([0, 0, 0.01], [1, 20, 0.999]))
    Dl = pinf - 0.5; Ts = np.linspace(0.05, 40, 40000); ok = (Dl - c*rho**Ts) > 0
    h = c*rho**Ts*(1 + 2*Ts*np.log(1/rho)); Tst = float(Ts[ok][np.argmin(np.abs(h[ok]-Dl))]) if ok.any() and Dl > 0 else np.nan
    return pinf, c, rho, Tst
base = fit(px.mean(0)); rng = np.random.default_rng(0); boots = []
for b in range(300):
    idx = rng.integers(0, n, n)
    try: boots.append(fit(px[idx].mean(0)))
    except Exception: pass
boots = np.array(boots)
out['fit'] = dict(pinf=base[0], c0=base[1], rho=base[2], Tstar=base[3],
                  ci={k: [float(np.nanpercentile(boots[:, i], 2.5)), float(np.nanpercentile(boots[:, i], 97.5))] for i, k in enumerate(['pinf', 'c0', 'rho', 'Tstar'])})
# K_max sensitivity for CHART (sigma = 0.5, same grid otherwise), population risk on the held-out half
grid = [(m, k0, th, be) for m in (1, 2, 3) for k0 in (1, 2, 3) for th in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
        for be in (1.0, 0.5, 0.3, 0.2, 0.1, 0.05, 0.02) if not (be == 1.0 and th != 0.5)]
out['kmax'] = {}
alphas = (0.02, 0.05) if task == 'maze' else (0.01, 0.02)
for Kmax in (16, 32, 64):
    A_, E_, C_ = [], [], []
    for (m, k0, th, be) in grid:
        A, C, cst = halted_outputs(ans[:, :Kmax], cor[:, :Kmax], stability_halt(ans[:, :Kmax], m, T))
        acc, cc, used, comp = consensus_eprocess(A, C, cst, k0, th, be, Kmax); A_.append(acc); E_.append(~cc); C_.append(comp)
    ACC, ERR, COST = np.array(A_), np.array(E_), np.array(C_); out['kmax'][Kmax] = {}
    for a in alphas:
        rng = np.random.default_rng(11); rows = []
        for r in range(R):
            perm = rng.permutation(n); cal, test = perm[:n//2], perm[n//2:]
            mm, _ = ltt_select(ACC, ERR, COST, cal, a, 0.1, 0.25, Kmax*T, method='fst', rng=rng)
            ev = evaluate(ACC, ERR, COST, mm, test); rows.append([ev['risk'], ev['cov'], ev['cost'], mm < 0])
        X = np.array(rows, float)
        out['kmax'][Kmax][str(a)] = dict(risk=X[:, 0].mean(), viol=float((X[:, 0] > a).mean()), cov=X[:, 1].mean(), nfe=X[:, 2].mean(), empty=X[:, 3].mean())
        print(task, 'Kmax', Kmax, a, out['kmax'][Kmax][str(a)], flush=True)
out['kmax'] = {str(k): v for k, v in out['kmax'].items()}
json.dump(out, open(f'results_k64_{task}.json', 'w'), default=float)
print('iso', {B: (v['T_emp'], v['T_bound']) for B, v in iso.items()}, 'fit', out['fit'])
