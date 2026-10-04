"""export_figs_v7.py -- figure panels of the revision (matlab_fig_1 JSON):
 * Fig. 5 and Fig. 7(a,b): error bars (risk and coverage: +-1 SD over the 300 calibration splits; violation frequency:
   95% Clopper-Pearson interval), in-panel legends removed (one common legend strip per figure, fig_legend_*.tex)
 * Fig. 6(c,d): plurality error versus depth T = B/K for every budget B, with the model optimum T* (dashed) and the
   empirical optimum (dotted)
 * Fig. 7(c): number of queries per detour bin printed in the panel"""
import json, copy, numpy as np
D = '../latex_8/matlab/data'
def ld(n): return json.load(open(f'{D}/{n}.json'))
def sv(p): json.dump(p, open(f"{D}/{p['name']}.json", 'w'), indent=1)
A = [0.01, 0.02, 0.05, 0.1]
KEY = {'CHART (proposed)': 'CHART', 'CHART static': 'CHART-static', 'TRM+Q, LTT': 'TRM+Q-LTT', 'PTRM+Q, LTT': 'PTRM+Q-LTT',
       'TRM+Q, plug in': 'TRM+Q-naive', 'CHART, plug in': 'CHART-naive', 'CHART, Bonferroni': 'CHART-bonf'}
STY = json.load(open('../latex_8/matlab/palette.json'))
order = ['CHART-seq', 'CHART-static', 'TRM+Q-LTT', 'PTRM+Q-LTT', 'TRM+Q-naive', 'CHART-naive']
keys = ['CHART', 'CHART-static', 'TRM+Q-LTT', 'PTRM+Q-LTT', 'TRM+Q-naive', 'CHART-naive']

def with_err(p, E, metric):
    ser = []; j = 0
    for s in p['series']:
        if s.get('kind') in ('ref', 'hline', 'vline'): ser.append(s); continue
        k = keys[j]; j += 1; s = dict(s); s['label'] = ''
        if metric == 'risk':
            m = [100*E[str(a)][k]['risk'] for a in A]; sd = [100*E[str(a)][k]['risk_sd'] for a in A]; lo = hi = sd
        elif metric == 'cov':
            m = [E[str(a)][k]['cov'] for a in A]; sd = [E[str(a)][k]['cov_sd'] for a in A]
            lo = [min(sd_, v) for sd_, v in zip(sd, m)]; hi = [min(sd_, 1.0-v) for sd_, v in zip(sd, m)]
        else:
            m = [100*E[str(a)][k]['viol'] for a in A]; ci = [E[str(a)][k]['viol_ci'] for a in A]
            lo = [v-100*c[0] for v, c in zip(m, ci)]; hi = [100*c[1]-v for v, c in zip(m, ci)]
        s.update(kind='errorbar', x=A, y=m, err_lo=lo, err_hi=hi); ser.append(s)
    p['series'] = ser; p.pop('legend', None); return p

for t in ('maze', 'sudoku'):
    E = json.load(open(f'results_e1_{t}.json'))['res']
    for metric, nm in (('risk', f'fig_risk_{t}'), ('cov', f'fig_cov_{t}'), ('viol', f'fig_viol_{t}')):
        p = with_err(ld(nm), E, metric); p['name'] = nm + '_e'
        if metric == 'viol': p['ylim'] = [0, 80]
        sv(p)
# real maps (Fig. 7 a, b)
E = json.load(open('results_real.json'))['P3']['res']
p = with_err(ld('fig_cov_street_q'), E, 'cov'); p['name'] = 'fig_cov_street_e'; p['ylim'] = [0, 1.05]; p['yticks'] = [0, 0.2, 0.4, 0.6, 0.8, 1.0]; sv(p)
p = with_err(ld('fig_viol_street_q'), E, 'viol'); p['name'] = 'fig_viol_street_e'; p['ylim'] = [0, 80]; p['yticks'] = [0, 20, 40, 60, 80]; sv(p)
# Fig. 7 c: counts per bin, no legend
p = ld('fig_detour_street_q'); DT = json.load(open('results_real_detour.json'))['0.02']['bins']
for s in p['series']: s['label'] = ''
fmt = lambda n: f"{n/1000:.1f}k" if n >= 1000 else str(n)
p['annotations'] = [dict(x=x, y=yy, text=fmt(b['n']), italic=False) for x, yy, b in zip([0, 1.5, 3.5, 6.5, 10], [0.625, 0.67, 0.625, 0.625, 0.625], DT)]
p['name'] = 'fig_detour_street_e'; p.pop('legend', None); sv(p)
p = ld('fig_shift_dao_q'); [s.__setitem__('label', '') for s in p['series']]; p['name'] = 'fig_shift_dao_e'; p['ylim'] = [0, 1.02]; p['lock_ylim'] = True; p.pop('legend', None); sv(p)
# Fig. 6 a, b without legends
for t in ('maze', 'sudoku'):
    p = ld(f'fig_adaptive_{t}_q'); [s.__setitem__('label', '') for s in p['series']]; p['name'] = f'fig_adaptive_{t}_e'
    p['ylim'] = [0, 40] if t == 'maze' else [0, 30]; p['yticks'] = [0, 10, 20, 30, 40] if t == 'maze' else [0, 10, 20, 30]; p.pop('legend', None); sv(p)
# Fig. 6 c, d: error versus depth with the model optimum
PAL = [[0, .447, .698], [.835, .369, 0], [0, .62, .451], [.482, .196, .58], [.8, .475, .655], [.902, .624, 0]]
for t in ('maze', 'sudoku'):
    K = json.load(open(f'results_k64_{t}.json')); ser = []
    for (B, v), c, mk, ls in zip(sorted(K['iso'].items(), key=lambda kv: int(kv[0])), PAL, 'osd^vp', ['-', '--', '-.', ':', '-', '--']):
        if int(B) > 512: continue
        T = np.array(v['T']); e = np.array(v['err']); o = np.argsort(T)
        ser.append(dict(x=T[o].tolist(), y=(100*e[o]).tolist(), color=c, marker=mk, line=ls, label=''))
    ts = K['fit']['Tstar']
    ser.append(dict(kind='vline', x=ts, line='--', color=[.2, .2, .2], label=''))
    ser.append(dict(kind='vline', x=4, line=':', color=[.45, .45, .45], label=''))
    sv(dict(name=f'fig_wdT_{t}', layout='quad', xlabel='Depth {\\itT} = {\\itB}/{\\itK}', ylabel='Plurality error (%)', xscale='log',
            xticks=[1, 2, 4, 8, 16], xticklabels=['1', '2', '4', '8', '16'], xlim=[0.85, 19], yscale='log',
            ylim=[4, 100] if t == 'maze' else [1, 100], lock_ylim=True,
            yticks=[5, 10, 20, 50, 100] if t == 'maze' else [1, 2, 5, 10, 20, 50, 100],
            yticklabels=['5', '10', '20', '50', '100'] if t == 'maze' else ['1', '2', '5', '10', '20', '50', '100'],
            annotations=[dict(x=ts*0.93, y=(70 if t == 'maze' else 60), text='{\\itT}^{\\ast}', halign='right', italic=False)], series=ser))
    print(t, 'T*', ts)
print('ok')
