"""export_figs_v3.py -- JSON panels (matlab_fig_1 format) for the revision.  Same palette and markers as
version 2 (../matlab/palette.json); every value is read from the result files or recomputed from pools."""
import json, os, numpy as np
from chart import *
from chart import _anchor
OUT = '../latex_6/matlab/data'; os.makedirs(OUT, exist_ok=True)
PAL = dict(blue=[0, .447, .698], verm=[.835, .369, 0], green=[0, .62, .451], purple=[.482, .196, .58],
           pink=[.8, .475, .655], amber=[.902, .624, 0], sky=[.337, .706, .914], grey=[.35, .35, .35])
STY = json.load(open('../latex_6/matlab/palette.json'))
STY.update({'Stop: SPRT': dict(color=PAL['green'], marker='d', line='-.', label='SPRT stop'),
            'Stop: r-agree': dict(color=PAL['purple'], marker='^', line=':', label='r-agree stop'),
            'Stop: fixed-K': dict(color=PAL['verm'], marker='s', line='--', label='Fixed K, halted'),
            'Stop: AC-Beta': dict(color=PAL['pink'], marker='v', line='-', label='Beta stop (AC)'),
            'CHART-mix': dict(color=PAL['sky'], marker='h', line='--', label='CHART, mixture bet'),
            'CHART-bonf': dict(color=PAL['grey'], marker='>', line='-.', label='CHART, Bonferroni')})
def S(name, x, y, **kw):
    d = dict(STY.get(name, {})); d.update(kw); d['x'] = [float(v) for v in x]; d['y'] = [float(v) for v in y]; return d
def save(p): json.dump(p, open(f"{OUT}/{p['name']}.json", 'w'), indent=1)
A = [0.01, 0.02, 0.05, 0.1]; XT = dict(xscale='log', xlim=[0.008, 0.125], xticks=A, xticklabels=['0.01', '0.02', '0.05', '0.1'])
E1 = {t: json.load(open(f'results_e1_{t}.json'))['res'] for t in ('maze', 'sudoku')}
V3 = {t: json.load(open(f'results_v3_{t}.json')) for t in ('maze', 'sudoku')}
K64 = {t: json.load(open(f'results_k64_{t}.json')) for t in ('maze', 'sudoku')}
meths = ['CHART-seq', 'CHART-static', 'TRM+Q-LTT', 'PTRM+Q-LTT', 'TRM+Q-naive', 'CHART-naive']
key = lambda m: 'CHART' if m == 'CHART-seq' else m

# ---- (1) risk / coverage / population violation vs alpha, independent calibration draws (replaces v2 panels)
for t in ('maze', 'sudoku'):
    E = E1[t]
    save(dict(name=f'fig_risk_{t}', layout='trio', xlabel='Target risk {\\it\\alpha}', ylabel='Held-out selective risk (%)',
              yticks=[0, 2, 4, 6, 8, 10, 12], ylim=[0, 12], lock_ylim=True,
              series=[S(m, A, [100*E[str(a)][key(m)]['risk'] for a in A], label='') for m in meths] + [dict(kind='ref', x=[0.009, 0.11], y=[0.9, 11], line=':')],
              annotations=[dict(x=0.0135, y=5.2, text='risk = \\alpha')], **XT))
    save(dict(name=f'fig_cov_{t}', layout='trio', xlabel='Target risk {\\it\\alpha}', ylabel='Coverage', ylim=[0, 1.05], lock_ylim=True,
              legend='southeast', series=[S(m, A, [E[str(a)][key(m)]['cov'] for a in A], label=(STY[m]['label'] if t == 'sudoku' else '')) for m in meths], **XT))
    save(dict(name=f'fig_viol_{t}', layout='trio', xlabel='Target risk {\\it\\alpha}', ylabel='Violation frequency (%)', ylim=[0, 75], lock_ylim=True,
              series=[S(m, A, [100*E[str(a)][key(m)]['viol'] for a in A], label='') for m in meths] + [dict(kind='hline', y=10, line='--')], **XT))

# ---- (2) ECDF of the held-out risk of the selected configuration (the guarantee as a picture)
for t, a in (('maze', 0.05), ('sudoku', 0.02)):
    ser = []
    for m in ('CHART-seq', 'TRM+Q-LTT', 'TRM+Q-naive', 'CHART-naive'):
        r = np.sort(np.array(E1[t][str(a)][key(m)]['risks']))*100; y = np.arange(1, len(r)+1)/len(r)
        ser.append(S(m, r, y, marker='none'))
    ser += [dict(kind='vline', x=100*a, line='--', color=PAL['grey']), dict(kind='hline', y=0.9, line=':', label='1 - {\\it\\delta}')]
    xl = [2.0, 8.5] if t == 'maze' else [0.3, 4.0]
    save(dict(name=f'fig_ecdf_{t}', layout='pair', xlabel='Held-out selective risk of the selected configuration (%)', ylabel='Empirical CDF over 300 draws',
              xlim=xl, ylim=[0, 1.02], lock_ylim=True, legend='southeast', series=ser))

# ---- (3) stopping-rule ablation under the same LTT: coverage versus NFE (points = alpha values)
for t in ('maze', 'sudoku'):
    E = V3[t]['E1']; ser = []
    for m in ('CHART', 'CHART-mix', 'Stop: SPRT', 'Stop: r-agree', 'Stop: fixed-K', 'Stop: AC-Beta'):
        nm = 'CHART-seq' if m == 'CHART' else m
        al = (0.02, 0.05) if t == 'maze' else (0.01, 0.02)
        ser.append(S(nm, [E[str(a)][m]['cost'] for a in al], [E[str(a)][m]['cov'] for a in al]))
    save(dict(name=f'fig_stoprules_{t}', layout='pair', xlabel='Recursion steps per instance (NFE)', ylabel='Coverage at the certified risk',
              ylim=[0, 1.1] if t == 'maze' else [0.6, 1.02], legend='southeast' if t == 'maze' else 'southwest', series=ser))

# ---- (4) width-depth law on the full grid (K up to 64)
for t in ('maze', 'sudoku'):
    iso = K64[t]['iso']; ser = []
    cols = [PAL['blue'], PAL['verm'], PAL['green'], PAL['purple'], PAL['pink'], PAL['amber'], PAL['grey']]
    for (B, v), c, mk, ls in zip(sorted(iso.items(), key=lambda kv: int(kv[0])), cols, 'osd^vph', ['-', '--', '-.', ':', '-', '--', '-.']):
        if int(B) > 512: continue
        ser.append(S('', v['K'], [100*e for e in v['err']], label=f'{{\\itB}} = {B}', color=c, marker=mk, line=ls))
    save(dict(name=f'fig_wdgrid_{t}', layout='pair', xlabel='Width {\\itK} (depth {\\itT} = {\\itB}/{\\itK})', ylabel='Plurality error (%)',
              xscale='log', xticks=[1, 2, 4, 8, 16, 32, 64], xticklabels=['1', '2', '4', '8', '16', '32', '64'], xlim=[0.85, 75],
              yscale='log', ylim=[1, 100] if t == 'sudoku' else [4, 100],
              yticks=[1, 2, 5, 10, 20, 50, 100] if t == 'sudoku' else [5, 10, 20, 50, 100],
              yticklabels=['1', '2', '5', '10', '20', '50', '100'] if t == 'sudoku' else ['5', '10', '20', '50', '100'], legend_cols=2, series=ser))

# ---- (5) wealth fan (maze) and compute Lorenz curves
d = np.load('pool_maze_main.npz'); A_, C_, c_ = halted_outputs(d['ans_0.5'], d['cor_0.5'], stability_halt(d['ans_0.5'], 1, 16))
P, anc_cor, acc, status = kelly_paths(A_, C_, c_, 1, 0.5, 0.1, 16)
rng = np.random.default_rng(4); ph = (A_[:, 1:] == A_[:, :1]).mean(1)
ci = np.r_[rng.choice(np.flatnonzero(anc_cor & (ph == 1)), 3, replace=False), rng.choice(np.flatnonzero(anc_cor & (ph < 1) & (ph >= 0.6)), 22, replace=False)]
wi = rng.choice(np.flatnonzero(~anc_cor), 25, replace=False)
ser = []
for k, i in enumerate(np.r_[wi, ci]):
    y = P[i]; j = np.flatnonzero(~np.isnan(y)); good = i in ci
    first = (k == 0) or (k == len(wi))
    ser.append(dict(x=[0] + (j+1).tolist(), y=[0.0] + y[j].tolist(), color=PAL['blue'] if good else PAL['verm'], marker='none',
                    line='-' if good else '--', label=('correct anchor' if good else 'wrong anchor') if first else ''))
ser.append(dict(kind='hline', y=float(np.log(10)), line=':', label='log(1/{\\it\\beta})'))
save(dict(name='fig_wealth', layout='pair', xlabel='Test trajectory {\\itj}', ylabel='Log wealth log {\\itE_j}', xlim=[0, 15.5], ylim=[-3, 3.5], lock_ylim=True,
          legend='southwest', series=ser))
Lz = []
for t, (s, m, k0, th, be), c, mk in (('maze', (1.0, 1, 3, 0.3, 0.5), PAL['blue'], 'o'), ('sudoku', (0.25, 1, 1, 0.3, 0.5), PAL['verm'], 's')):
    d = np.load(f'pool_{t}_main.npz'); A_, C_, c_ = halted_outputs(d[f'ans_{s}'], d[f'cor_{s}'], stability_halt(d[f'ans_{s}'], m, 16))
    _, _, _, comp = consensus_eprocess(A_, C_, c_, k0, th, be, 16)
    x = np.sort(comp); L = np.r_[0, np.cumsum(x)/x.sum()]; q = np.linspace(0, 1, len(L)); G = 1 - 2*np.trapezoid(L, q)
    idx = np.unique(np.r_[np.linspace(0, len(L)-1, 21).astype(int)])
    Lz.append(S('', q[idx], L[idx], label=f'CHART, {t} (Gini {G:.2f})', color=c, marker=mk, line='-' if t == 'maze' else '--'))
    json.dump(dict(gini=G), open(f'gini_{t}.json', 'w'))
Lz.append(S('', [0, 1], [0, 1], label='Fixed compute (Gini 0)', color=PAL['grey'], marker='none', line=':'))
save(dict(name='fig_lorenz', layout='pair', xlabel='Cumulative share of instances', ylabel='Cumulative share of recursion steps',
          xlim=[0, 1], ylim=[0, 1.02], legend='northwest', series=Lz))

# ---- (6) calibration size and target-label budget under shift
ser = []
for t, a, c, mk in (('maze', 0.05, PAL['blue'], 'o'), ('sudoku', 0.02, PAL['verm'], 's')):
    D = V3[t]['E3']['data']; ns = sorted(int(k) for k in D)
    ser.append(S('', ns, [D[str(k)]['CHART']['cov_pop'] for k in ns], label=f'CHART, {t}', color=c, marker=mk, line='-'))
    ser.append(S('', ns, [D[str(k)]['CHART-bonf']['cov_pop'] for k in ns], label=f'Bonferroni, {t}', color=c, marker=mk, line='--', open=True))
save(dict(name='fig_calsize', layout='pair', xlabel='Calibration instances {\\itn}', ylabel='Coverage at the certified risk',
          xscale='log', xticks=[250, 500, 1000, 2000], xticklabels=['250', '500', '1000', '2000'], xlim=[220, 2300], ylim=[0, 1.05], legend='southeast', series=ser))
SH = json.load(open('results_shift_v3.json'))['0.05']; ser = []
for m, nm in (('CHART', 'CHART-seq'), ('CHART-bonf', 'CHART-bonf'), ('TRM+Q-LTT', 'TRM+Q-LTT')):
    ks = [k for k in sorted(SH[m], key=int) if int(k) > 0]
    ser.append(S(nm, [int(k) for k in ks], [SH[m][k]['cov'] for k in ks]))
save(dict(name='fig_shiftlabels', layout='pair', xlabel='Labelled 19 x 19 mazes used for recalibration', ylabel='Coverage at {\\it\\alpha} = 0.05',
          xscale='log', xticks=[50, 100, 250, 500, 1000], xticklabels=['50', '100', '250', '500', '1000'], xlim=[45, 1100], ylim=[0, 1.0], legend='northwest', series=ser))
print('exported', sorted(os.listdir(OUT)))
