"""shift.py -- E9: certificates under distribution shift (calibrate on 15x15 mazes, deploy on 19x19)
and the remedy (recalibrate on a small labelled sample of the deployment distribution)."""
import json, numpy as np
from chart import *
SIG = [0.25, 0.5, 1.0]; KMAX = 16; TMAX = 16; DELTA = 0.1; MU = 0.25; R = 200
grid = [(s, m, k0, th, be) for s in SIG for m in (1, 2, 3) for k0 in (1, 2, 3)
        for th in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9) for be in (1.0, 0.5, 0.3, 0.2, 0.1, 0.05, 0.02)
        if not (be == 1.0 and th != 0.5)]
def family(path):
    d = np.load(path); A_, E_, C_ = [], [], []; H = {}
    for (s, m, k0, th, be) in grid:
        if (s, m) not in H: H[(s, m)] = halted_outputs(d[f'ans_{s}'], d[f'cor_{s}'], stability_halt(d[f'ans_{s}'], m, TMAX))
        A, C, cst = H[(s, m)]; acc, cc, used, comp = consensus_eprocess(A, C, cst, k0, th, be, KMAX)
        A_.append(acc); E_.append(~cc); C_.append(comp)
    q = d['q_0.0'][:, 0, 5]; e0 = ~d['cor_0.0'][:, 0, 5]; qg = np.linspace(10, -4, 57)
    return (np.array(A_), np.array(E_), np.array(C_)), (np.array([q >= g for g in qg]), np.tile(e0, (57, 1)), np.full((57, len(q)), 6.0)), float(e0.mean())
S, Sq, e_src = family('pool_maze_main.npz'); Tg, Tq, e_tgt = family('pool_maze_ood19.npz')
ns, nt = S[0].shape[1], Tg[0].shape[1]
out = dict(err_det_src=e_src, err_det_tgt=e_tgt)
for a in (0.02, 0.05):
    rng = np.random.default_rng(7); res = {'no-recal': [], 'recal': [], 'Q no-recal': [], 'Q recal': []}
    for r in range(R):
        cal_s = rng.permutation(ns)[:ns//2]; pt = rng.permutation(nt); cal_t, test_t = pt[:nt//2], pt[nt//2:]
        # CHART calibrated on source, applied to target
        m, _ = ltt_select(*S, cal_s, a, DELTA, MU, 256, method='fst', rng=rng); res['no-recal'].append(evaluate(*Tg, m, test_t))
        m, _ = ltt_select(*Tg, cal_t, a, DELTA, MU, 256, method='fst', rng=rng); res['recal'].append(evaluate(*Tg, m, test_t))
        m, _ = ltt_select(*Sq, cal_s, a, DELTA, MU, 256, method='fst_given', rng=rng); res['Q no-recal'].append(evaluate(*Tq, m, test_t))
        m, _ = ltt_select(*Tq, cal_t, a, DELTA, MU, 256, method='fst_given', rng=rng); res['Q recal'].append(evaluate(*Tq, m, test_t))
    out[str(a)] = {k: dict(risk=float(np.mean([x['risk'] for x in v])), viol=float(np.mean([x['risk'] > a for x in v])),
                          cov=float(np.mean([x['cov'] for x in v])), cost=float(np.mean([x['cost'] for x in v]))) for k, v in res.items()}
    print(a, json.dumps(out[str(a)], indent=0))
json.dump(out, open('results_shift.json', 'w')); print('det error src/tgt', e_src, e_tgt)
