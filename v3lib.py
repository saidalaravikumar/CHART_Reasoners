"""v3lib.py -- shared configuration families for evaluation experiments (see experiments_v3.py)."""
import numpy as np
from chart import *
from chart import _anchor
SIG = [0.25, 0.5, 1.0]; KMAX = 16; TMAX = 16; TB = 6

def load(path):
    d = np.load(path)
    return dict(D=d['difficulty'], ans={s: d[f'ans_{s}'] for s in [0.0]+SIG}, cor={s: d[f'cor_{s}'] for s in [0.0]+SIG},
                q={s: d[f'q_{s}'] for s in [0.0]+SIG})

GRID_SEQ = [(s, m, k0, th, be) for s in SIG for m in (1, 2, 3) for k0 in (1, 2, 3)
            for th in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9) for be in (1.0, 0.5, 0.3, 0.2, 0.1, 0.05, 0.02)
            if not (be == 1.0 and th != 0.5)]
QGRID = np.linspace(10, -4, 57)

def families(P, ablation=True):
    n = len(P['D']); F = {}; H = {}
    for s in SIG:
        for m in (1, 2, 3):
            H[(s, m)] = halted_outputs(P['ans'][s], P['cor'][s], stability_halt(P['ans'][s], m, TMAX))
    for name, fn in (('CHART', consensus_eprocess), ('CHART-mix', consensus_mixture)):
        A_, E_, C_ = [], [], []
        for (s, m, k0, th, be) in GRID_SEQ:
            A, C, cst = H[(s, m)]; acc, cc, used, comp = fn(A, C, cst, k0, th, be, KMAX)
            A_.append(acc); E_.append(~cc); C_.append(comp)
        F[name] = (np.array(A_), np.array(E_), np.array(C_), GRID_SEQ, 'fst')
    F['CHART-bonf'] = F['CHART'][:4] + ('bonf',); F['CHART-naive'] = F['CHART'][:4] + ('naive',)
    # CHART static (fixed K, T, agreement threshold)
    g, A_, E_, C_ = [], [], [], []
    for s in SIG:
        for K in (2, 4, 8, 16):
            for T in (2, 4, 6, 8, 12, 16):
                agr, cc, cst = majority_static(P['ans'][s], P['cor'][s], K, T-1)
                for tau in (0.5, 0.625, 0.75, 0.875, 1.0):
                    g.append((s, K, T, tau)); A_.append(agr >= tau - 1e-9); E_.append(~cc); C_.append(cst)
    F['CHART-static'] = (np.array(A_), np.array(E_), np.array(C_), g, 'fst')
    # Q-head baselines at the training depth, a-priori ordered, valid D1-skip
    sc = P['q'][0.0][:, 0, TB-1]; e0 = ~P['cor'][0.0][:, 0, TB-1]
    QA = np.array([sc >= q for q in QGRID]); QE = np.tile(e0, (len(QGRID), 1)); QC = np.full((len(QGRID), n), float(TB))
    F['TRM+Q-LTT'] = (QA, QE, QC, list(QGRID), 'given_d1skip')
    F['TRM+Q-naive'] = (QA, QE, QC, list(QGRID), 'naive')
    g, A_, E_, C_ = [], [], [], []
    for s in SIG:
        ks = P['q'][s][:, :, TB-1].argmax(1); smax = P['q'][s][:, :, TB-1].max(1); ep = ~P['cor'][s][np.arange(n), ks, TB-1]
        for q in QGRID: g.append((s, q)); A_.append(smax >= q); E_.append(ep); C_.append(np.full(n, float(KMAX*TB)))
    order = sorted(range(len(g)), key=lambda i: (-g[i][1], g[i][0]))
    F['PTRM+Q-LTT'] = (np.array(A_)[order], np.array(E_)[order], np.array(C_)[order], [g[i] for i in order], 'given_d1skip')
    if not ablation: return F, H
    # ---- E2 stopping-rule ablation (all on halted trajectories, same sigma/m/k0 grid, same LTT)
    g, A_, E_, C_ = [], [], [], []
    for s in SIG:
        for m in (1, 2, 3):
            A, C, cst = H[(s, m)]
            for k0 in (1, 2, 3):
                for r in (1, 2, 3, 4, 6, 8):
                    for sd in (1, 2, 3):
                        acc, cc, comp = stop_ragree(A, C, cst, k0, r, sd, KMAX); g.append((s, m, k0, r, sd)); A_.append(acc); E_.append(~cc); C_.append(comp)
    F['Stop: r-agree'] = (np.array(A_), np.array(E_), np.array(C_), g, 'fst')
    g, A_, E_, C_ = [], [], [], []
    for s in SIG:
        for m in (1, 2, 3):
            A, C, cst = H[(s, m)]
            for k0 in (1, 2, 3):
                for th1 in (0.7, 0.8, 0.9, 0.95):
                    for th in (0.3, 0.5, 0.6):
                        if th1 <= th: continue
                        for Aup in (np.log(3), np.log(10), np.log(30)):
                            acc, cc, comp = stop_sprt(A, C, cst, k0, th, th1, Aup, -np.log(10), KMAX)
                            g.append((s, m, k0, th, th1, Aup)); A_.append(acc); E_.append(~cc); C_.append(comp)
    F['Stop: SPRT'] = (np.array(A_), np.array(E_), np.array(C_), g, 'fst')
    g, A_, E_, C_ = [], [], [], []
    for s in SIG:
        for m in (1, 2, 3):
            A, C, cst = H[(s, m)]
            for conf in (0.7, 0.8, 0.9, 0.95, 0.99):
                acc, cc, comp = stop_beta(A, C, cst, conf, KMAX); g.append((s, m, conf)); A_.append(acc); E_.append(~cc); C_.append(comp)
    F['Stop: AC-Beta'] = (np.array(A_), np.array(E_), np.array(C_), g, 'fst')
    g, A_, E_, C_ = [], [], [], []
    for s in SIG:
        for m in (1, 2, 3):
            A, C, cst = H[(s, m)]
            for K in (1, 2, 4, 8, 16):
                agr, cc, comp = majority_halted(A, C, cst, K)
                for tau in ((1.0,) if K == 1 else (0.5, 0.625, 0.75, 0.875, 1.0)):
                    g.append((s, m, K, tau)); A_.append(agr >= tau - 1e-9); E_.append(~cc); C_.append(comp)
    F['Stop: fixed-K'] = (np.array(A_), np.array(E_), np.array(C_), g, 'fst')
    return F, H

def pop_risk(F, name, m):
    ACC, ERR = F[name][0], F[name][1]
    if m < 0: return 0.0, 0.0
    a = ACC[m]; return float((ERR[m] & a).sum()/max(a.sum(), 1)), float(a.mean())

