"""decomp.py -- component decomposition under the population-risk protocol.
Same reasoner, same recorded trajectories, same calibration draws (seed 99) and the same LTT for every variant, so that
differences in coverage / risk / compute are caused by the component that is switched on or off:
stochastic trajectories, stability halting, the width rule (Q-head selection, fixed-K majority, e-process) and LTT.
usage: python decomp.py maze|sudoku|street [R]   ->  results_decomp_<task>.json"""
import sys, json, time, numpy as np
from chart import *
from v3lib import load, families, SIG, KMAX, TMAX, TB, GRID_SEQ, QGRID

task = sys.argv[1]; R = int(sys.argv[2]) if len(sys.argv) > 2 else 300
POOLS = dict(maze=('pool_maze_main.npz', 'pool_maze_fresh.npz', 2000, (0.02, 0.05)),
             sudoku=('pool_sudoku_main.npz', 'pool_sudoku_fresh.npz', 1500, (0.01, 0.02)),
             street=('pool_street_main.npz', 'pool_street_fresh.npz', 2000, (0.02, 0.05)))[task]
t0 = time.time()

def extra(P):
    """Additional configuration families evaluated in component decomposition."""
    n = len(P['D']); F = {}
    # PTRM with fixed K = 16 at the training depth, best trajectory by Q-head, no abstention, no LTT
    s = 0.5; ks = P['q'][s][:, :, TB-1].argmax(1)
    F['PTRM fixed K'] = (np.ones((1, n), bool), ~P['cor'][s][np.arange(n), ks, TB-1][None], np.full((1, n), float(KMAX*TB)), [s], 'none')
    # single stability-halted trajectory, threshold on its Q-logit, a-priori ordered, certified by LTT (valid D1 skip)
    for name, sigs in (('TRM + halt + LTT', [0.0]), ('PTRM + halt + LTT', SIG)):
        g, A_, E_, C_ = [], [], [], []
        for sg in sigs:
            for m in (1, 2, 3):
                idx = stability_halt(P['ans'][sg][:, :1], m, TMAX)[:, 0]
                sc = P['q'][sg][np.arange(n), 0, idx]; er = ~P['cor'][sg][np.arange(n), 0, idx]; cs = (idx + 1).astype(float)
                for q in QGRID: g.append((sg, m, q)); A_.append(sc >= q); E_.append(er); C_.append(cs)
        order = sorted(range(len(g)), key=lambda i: (-g[i][2], g[i][0], g[i][1]))
        F[name] = (np.array(A_)[order], np.array(E_)[order], np.array(C_)[order], [g[i] for i in order], 'given_d1skip')
    # e-process + LTT with the depth fixed instead of stability halting
    for name, T in (('e-process, no halting (T=16)', TMAX), ('e-process, fixed depth T=4', 4)):
        g, A_, E_, C_ = [], [], [], []
        for (sg, m, k0, th, be) in GRID_SEQ:
            if m != 1: continue
            idx = np.full(P['ans'][sg].shape[:2], T-1); A, C, cst = halted_outputs(P['ans'][sg], P['cor'][sg], idx)
            acc, cc, used, comp = consensus_eprocess(A, C, cst, k0, th, be, KMAX)
            g.append((sg, k0, th, be)); A_.append(acc); E_.append(~cc); C_.append(comp)
        F[name] = (np.array(A_), np.array(E_), np.array(C_), g, 'fst')
    return F

Fs = []
for p in POOLS[:2]:
    P = load(p); F, _ = families(P, ablation=True); F.update(extra(P)); Fs.append(F); print('families', p, time.time()-t0, flush=True)
F = {k: tuple(np.concatenate([Fs[0][k][i], Fs[1][k][i]], 1) for i in range(3)) + Fs[0][k][3:] for k in Fs[0]}
VAR = [('PTRM fixed K', 'PTRM fixed K'), ('PTRM + Q-head + LTT', 'PTRM+Q-LTT'), ('PTRM + majority + LTT', 'CHART-static'),
       ('TRM + halt + LTT', 'TRM + halt + LTT'), ('PTRM + halt + LTT', 'PTRM + halt + LTT'), ('PTRM + halt + majority + LTT', 'Stop: fixed-K'),
       ('e-process, no halting (T=16)', 'e-process, no halting (T=16)'), ('e-process, fixed depth T=4', 'e-process, fixed depth T=4'),
       ('CHART without LTT (plug-in)', 'CHART-naive'), ('CHART', 'CHART'), ('CHART, Bonferroni', 'CHART-bonf')]
N = F['CHART'][0].shape[1]; NCAL = POOLS[2]; out = dict(task=task, N=N, ncal=NCAL, R=R, res={})
for a in POOLS[3]:
    out['res'][str(a)] = {}
    for lab, key in VAR:
        ACC, ERR, COST, grid, meth = F[key]; rng = np.random.default_rng(99); rows = []
        for r in range(R):
            perm = rng.permutation(N); cal, rest = perm[:NCAL], perm[NCAL:]
            m = 0 if meth == 'none' else ltt_select_v3(ACC, ERR, COST, cal, a, 0.1, 0.25, 256, method=meth, rng=rng)[0]
            ev = evaluate(ACC, ERR, COST, m, rest); rows.append([ev['risk'], ev['cov'], ev['cost'], m < 0])
        X = np.array(rows, float); k = int((X[:, 0] > a).sum())
        out['res'][str(a)][lab] = dict(risk=X[:, 0].mean(), risk_sd=X[:, 0].std(), viol=k/R, viol_ci=list(binom_ci(k, R)),
                                       cov=X[:, 1].mean(), cov_sd=X[:, 1].std(), nfe=X[:, 2].mean(), empty=X[:, 3].mean())
        print(f"{a} {lab:32s} risk {100*X[:,0].mean():.2f} viol {100*k/R:.1f} cov {100*X[:,1].mean():.1f} nfe {X[:,2].mean():.1f} empty {X[:,3].mean():.2f}", flush=True)
json.dump(out, open(f'results_decomp_{task}.json', 'w'), default=float)
print('saved', time.time()-t0)
