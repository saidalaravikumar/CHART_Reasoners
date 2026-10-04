"""export_figs.py -- converts results_*.json (and pools) into one JSON per panel for matlab_fig_1.m.
No smoothing or hand adjustment: every value is read from the simulation outputs."""
import json, numpy as np, os
from chart import *
OUT = '../latex_6/matlab/data'; os.makedirs(OUT, exist_ok=True)
PAL = dict(blue=[0, .447, .698], verm=[.835, .369, 0], green=[0, .62, .451], purple=[.482, .196, .58],
           pink=[.8, .475, .655], amber=[.902, .624, 0], sky=[.337, .706, .914], grey=[.35, .35, .35])
STY = {'CHART-seq': dict(color=PAL['blue'], marker='o', line='-', proposed=True, label='CHART (proposed)'),
       'CHART-static': dict(color=PAL['verm'], marker='s', line='--', label='CHART static'),
       'TRM+Q-LTT': dict(color=PAL['green'], marker='d', line='-.', label='TRM+Q, LTT'),
       'PTRM+Q-LTT': dict(color=PAL['purple'], marker='^', line=':', label='PTRM+Q, LTT'),
       'TRM+Q-naive': dict(color=PAL['pink'], marker='v', line='-', label='TRM+Q, plug in'),
       'CHART-naive': dict(color=PAL['amber'], marker='p', line='--', label='CHART, plug in')}
json.dump(STY, open('../latex_6/matlab/palette.json', 'w'), indent=1)
def S(name, x, y, **kw):
    d = dict(STY.get(name, {})); d.update(kw); d['x'] = list(map(float, x)); d['y'] = list(map(float, y)); return d
def save(p): json.dump(p, open(f"{OUT}/{p['name']}.json", 'w'), indent=1)
R = {t: json.load(open(f'results_{t}.json')) for t in ('maze', 'sudoku')}
A = [0.01, 0.02, 0.05, 0.1]
# ---------------- Fig. depth curves (pair)
for t, lab in (('maze', 'a'), ('sudoku', 'b')):
    E1 = R[t]['E1']; T = E1['T']
    save(dict(name=f'fig_depth_{t}', layout='pair', xlabel='Outer recursion step {\\itT}', ylabel='Exact accuracy',
              xlim=[0.5, 16.5], xticks=[1, 4, 8, 12, 16], ylim=[0.2, 1.02],
              series=[S('', T, E1['det'], label='TRM, deterministic', color=PAL['grey'], marker='s', line='--'),
                      S('', T, E1['single_0.5'], label='One noisy trajectory', color=PAL['sky'], marker='v', line=':'),
                      S('', T, E1['maj16_0.5'], label='Majority of 16', color=PAL['verm'], marker='d', line='-.'),
                      S('', T, E1['pass16_0.5'], label='Oracle pass@16', color=PAL['green'], marker='^', line='-'),
                      dict(kind='vline', x=6, label='Training depth')]))
# ---------------- reliability + dominant wrong attractor mass (pair)
save(dict(name='fig_reliability', layout='pair', xlabel='Trajectory agreement {\\itA}_{16}', ylabel='Error rate of plurality answer',
          xlim=[0, 1.02], ylim=[0, 1],
          series=[S('', R['maze']['E2']['center'], R['maze']['E2']['err'], label='Maze', color=PAL['blue'], marker='o', line='-'),
                  S('', R['sudoku']['E2']['center'], R['sudoku']['E2']['err'], label='Sudoku', color=PAL['verm'], marker='s', line='--')]))
ser = []
for t, c, mk in (('maze', PAL['blue'], 'o'), ('sudoku', PAL['verm'], 's')):
    for th, ls in (('0.5', '-'), ('0.9', '--')):
        ser.append(S('', R[t]['E8']['sigma'], np.array(R[t]['E8']['eps'][th])*100, label=f'{t.capitalize()}, {{\\it\\theta}}_0 = {th}', color=c, marker=mk, line=ls, open=(th == '0.9')))
save(dict(name='fig_wrong_attractor', layout='pair', xlabel='Noise level {\\it\\sigma}', ylabel='Dominant wrong basin mass (%)',
          xlim=[0.2, 1.05], xticks=[0.25, 0.5, 1.0], series=ser))
# ---------------- risk / coverage / violation vs alpha (trio rows; one shared legend in the Sudoku coverage panel)
meths = ['CHART-seq', 'CHART-static', 'TRM+Q-LTT', 'PTRM+Q-LTT', 'TRM+Q-naive', 'CHART-naive']
XT = dict(xscale='log', xlim=[0.008, 0.125], xticks=A, xticklabels=['0.01', '0.02', '0.05', '0.1'])
for t in ('maze', 'sudoku'):
    E3 = R[t]['E3']
    ser = [S(m, A, [100*E3[str(a)][m]['risk'] for a in A], label='') for m in meths] + [dict(kind='ref', x=[0.009, 0.11], y=[0.9, 11], line=':')]
    save(dict(name=f'fig_risk_{t}', layout='trio', xlabel='Target risk {\\it\\alpha}', ylabel='Test selective risk (%)',
              yticks=[0, 2, 4, 6, 8, 10, 12], ylim=[0, 12], lock_ylim=True, series=ser,
              annotations=[dict(x=0.0145, y=5.0, text='risk = \\alpha')], **XT))
    lab = (t == 'sudoku')
    ser = [S(m, A, [E3[str(a)][m]['cov'] for a in A], label=(STY[m]['label'] if lab else '')) for m in meths]
    save(dict(name=f'fig_cov_{t}', layout='trio', xlabel='Target risk {\\it\\alpha}', ylabel='Coverage', ylim=[0, 1.05],
              lock_ylim=True, legend='southeast', series=ser, **XT))
    ser = [S(m, A, [100*E3[str(a)][m]['viol'] for a in A], label='') for m in meths] + [dict(kind='hline', y=10, line='--')]
    save(dict(name=f'fig_viol_{t}', layout='trio', xlabel='Target risk {\\it\\alpha}', ylabel='Violation frequency (%)',
              ylim=[0, 75], lock_ylim=True, series=ser, **XT))
# ---------------- adaptive compute vs difficulty (pair) -- recomputed from pools for the chosen configurations
cfgs = {'maze': ((1.0, 1, 3, 0.3, 0.5), 0.05, (1.0, 8, 4)), 'sudoku': ((0.25, 1, 1, 0.3, 0.5), 0.02, (0.5, 4, 6))}
E5x = {}
for t in ('maze', 'sudoku'):
    d = np.load(f'pool_{t}_main.npz'); D = d['difficulty']
    (s, m, k0, th, be), a, (ss, KK, TT) = cfgs[t]
    A_, C_, c_ = halted_outputs(d[f'ans_{s}'], d[f'cor_{s}'], stability_halt(d[f'ans_{s}'], m, 16))
    acc, cc, used, comp = consensus_eprocess(A_, C_, c_, k0, th, be, 16)
    if t == 'maze': edges = np.quantile(D, np.linspace(0, 1, 7)); edges[-1] += 1
    else: edges = np.array([0, 55.5, 56.5, 57.5, 58.5, 100])
    xc, nfe, cov, rk = [], [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (D >= lo) & (D < hi)
        if sel.sum() < 30: continue
        xc.append(D[sel].mean()); nfe.append(comp[sel].mean()); cov.append(acc[sel].mean()); rk.append((acc & ~cc)[sel].sum()/max(acc[sel].sum(), 1))
    E5x[t] = dict(x=xc, nfe=nfe, cov=cov, risk=rk, static=KK*TT, cfg=cfgs[t][0])
    xl = 'Shortest path length' if t == 'maze' else 'Number of blank cells'
    save(dict(name=f'fig_adaptive_{t}', layout='pair', xlabel=xl, ylabel='Recursion steps per instance (NFE)',
              ylim=[0, 40] if t == 'maze' else [0, 30], lock_ylim=True, legend='southeast',
              series=[S('CHART-seq', xc, nfe), S('', xc, [KK*TT]*len(xc), label=f'CHART static ({KK} x {TT})', color=PAL['verm'], marker='s', line='--')]))
json.dump(E5x, open('adaptive_bins.json', 'w'), indent=1)
# ---------------- width-depth law (pair)
for t in ('maze', 'sudoku'):
    E6 = R[t]['E6']; ser = []
    for B, c, mk, ls in zip([16, 32, 64, 128], [PAL['blue'], PAL['verm'], PAL['green'], PAL['purple']], 'osd^', ['-', '--', '-.', ':']):
        K = [k for k, e in zip(E6['K'], E6['err'][str(B)]) if e is not None]; e = [100*x for x in E6['err'][str(B)] if x is not None]
        ser.append(S('', K, e, label=f'Budget {{\\itB}} = {B}', color=c, marker=mk, line=ls))
    save(dict(name=f'fig_widthdepth_{t}', layout='pair', xlabel='Width {\\itK} (depth {\\itT} = {\\itB}/{\\itK})', ylabel='Plurality error (%)',
              xscale='log', xticks=[1, 2, 4, 8, 16], xticklabels=['1', '2', '4', '8', '16'], xlim=[0.85, 19],
              yscale='log', ylim=[1, 100] if t == 'sudoku' else [5, 100], yticks=[1, 2, 5, 10, 20, 50, 100] if t == 'sudoku' else [5, 10, 20, 50, 100],
              yticklabels=['1', '2', '5', '10', '20', '50', '100'] if t == 'sudoku' else ['5', '10', '20', '50', '100'], series=ser))
# ---------------- e-process sample size (col)
ser = []
for t, c, mk in (('maze', PAL['blue'], 'o'), ('sudoku', PAL['verm'], 's')):
    E7 = R[t]['E7']; ser.append(S('', E7['center'], E7['used'], label=f'{t.capitalize()}, measured', color=c, marker=mk, line='-'))
w = R['maze']['E7']; keep = [i for i, v in enumerate(w['wald']) if v < 40]
ser.append(S('', [w['center'][i] for i in keep], [w['wald'][i] for i in keep], label='Oracle Kelly bound (Prop. 2)', color=PAL['grey'], marker='none', line='--'))
ser.append(dict(kind='hline', y=15, line=':', label='Budget {\\itK}_{max} - {\\itk}_0'))
save(dict(name='fig_eprocess', layout='col', xlabel='Consensus strength {\\itp} of the anchor basin', ylabel='Test trajectories drawn',
          xlim=[0.5, 1.02], ylim=[0, 30], series=ser))
print('exported', len(os.listdir(OUT)))
