"""export_figs_real.py -- JSON panels (matlab_fig_1 format) for the practical study on real maps.
Every panel copies the style of its synthetic counterpart (same palette, markers, axes conventions), so the
real-map figure reads exactly like Figs. 5 and 6.  Values come from results_real.json."""
import json, os, copy, numpy as np
OUT = '../latex_6/matlab/data'; os.makedirs(OUT, exist_ok=True)
STY = json.load(open('../latex_6/matlab/palette.json'))
PAL = dict(blue=[0, .447, .698], verm=[.835, .369, 0], green=[0, .62, .451], purple=[.482, .196, .58],
           pink=[.8, .475, .655], amber=[.902, .624, 0], sky=[.337, .706, .914], grey=[.35, .35, .35])
STY.update({'CHART-bonf': dict(color=PAL['grey'], marker='>', line='-.', label='CHART, Bonferroni')})
def S(name, x, y, **kw):
    d = dict(STY.get(name, {})); d.update(kw); d['x'] = [float(v) for v in x]; d['y'] = [float(v) for v in y]; return d
def save(p): json.dump(p, open(f"{OUT}/{p['name']}.json", 'w'), indent=1)
R = json.load(open('results_real.json'))
A = [0.01, 0.02, 0.05, 0.1]; XT = dict(xscale='log', xlim=[0.008, 0.125], xticks=A, xticklabels=['0.01', '0.02', '0.05', '0.1'])
meths = ['CHART-seq', 'CHART-static', 'TRM+Q-LTT', 'PTRM+Q-LTT', 'TRM+Q-naive', 'CHART-naive']
key = lambda m: 'CHART' if m == 'CHART-seq' else m
E = R['P3']['res']

# (a), (e) depth curves: reuse the series layout of the synthetic maze panel, replace the data
tmpl = json.load(open(f'{OUT}/fig_depth_maze.json'))
for dom, nm in (('street', 'fig_depth_street'), ('dao', 'fig_depth_dao')):
    P = R['P1'][dom]; p = copy.deepcopy(tmpl); p['name'] = nm; p['layout'] = 'trio'; p['ylim'] = [0.4, 1.0]; p['lock_ylim'] = True
    p['yticks'] = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]; p['legend'] = 'southeast'
    src = {'TRM, deterministic': P['det'], 'One noisy trajectory': P['single_0.5'], 'Majority of 16': P['maj16_0.5'],
           'Oracle pass@16': P['pass16_0.5']}
    ser = []
    for s in p['series']:
        lab = s.get('label', '')
        if s.get('kind') == 'vline': ser.append(s); continue
        for k, v in src.items():
            if lab.startswith(k.split(',')[0]) or lab == k:
                s['y'] = [float(u) for u in v]; ser.append(s); break
    for s in ser:
        if dom == 'dao': s['label'] = ''
    p['series'] = ser; save(p)

# (b), (c) coverage and violation frequency versus alpha on unseen cities
save(dict(name='fig_cov_street', layout='trio', xlabel='Target risk {\\it\\alpha}', ylabel='Coverage', ylim=[0, 1.05], lock_ylim=True,
          legend='southeast', series=[S(m, A, [E[str(a)][key(m)]['cov'] for a in A], label=(STY[m]['label'] if i < 3 else '')) for i, m in enumerate(meths)], **XT))
save(dict(name='fig_viol_street', layout='trio', xlabel='Target risk {\\it\\alpha}', ylabel='Violation frequency (%)', ylim=[0, 100], lock_ylim=True, legend='northeast',
          series=[S(m, A, [100*E[str(a)][key(m)]['viol'] for a in A], label=(STY[m]['label'] if i >= 3 else '')) for i, m in enumerate(meths)] + [dict(kind='hline', y=10, line='--')], **XT))

# (d) coverage versus detour length at alpha = 0.02 (most frequently selected configurations, p4_detour.py)
DT = json.load(open('results_real_detour.json'))['0.02']['bins']; xs = [0, 1.5, 3.5, 6.5, 10]; xl = ['0', '1-2', '3-4', '5-8', '>8']
save(dict(name='fig_detour_street', layout='trio', xlabel='Detour forced by buildings (steps)', ylabel='Coverage at {\\it\\alpha} = 0.02',
          xlim=[-0.8, 10.8], xticks=xs, xticklabels=xl, ylim=[0.6, 1.0], lock_ylim=True, legend='southwest',
          series=[S(m, xs, [b[k]['cov'] for b in DT]) for m, k in (('CHART-seq', 'CHART'), ('CHART-static', 'CHART-static'), ('TRM+Q-LTT', 'TRM+Q-LTT'))]))

# (f) city -> indoor shift: coverage after recalibration with n_t indoor labels (alpha = 0.05)
SH = R['P5']['0.05']; ser = []
for m, nm in (('CHART', 'CHART-seq'), ('CHART-bonf', 'CHART-bonf'), ('TRM+Q-LTT', 'TRM+Q-LTT'), ('PTRM+Q-LTT', 'PTRM+Q-LTT')):
    ks = [k for k in sorted(SH[m], key=int) if int(k) > 0]
    ser.append(S(nm, [int(k) for k in ks], [SH[m][k]['cov'] for k in ks], label={'CHART': 'CHART', 'CHART-bonf': 'CHART, Bonf.', 'TRM+Q-LTT': 'TRM+Q', 'PTRM+Q-LTT': 'PTRM+Q'}[m]))
save(dict(name='fig_shift_dao', layout='trio', xlabel='Labelled indoor windows for recalibration', ylabel='Coverage at {\\it\\alpha} = 0.05',
          xscale='log', xticks=[50, 100, 250, 500, 1000], xticklabels=['50', '100', '250', '500', '1000'], xlim=[45, 1100], ylim=[0, 1.0], yticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
          legend='northwest', series=ser))
print('exported real-map panels')
