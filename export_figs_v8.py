"""export_figs_v8.py -- unified plotting style for figures:
  line width 1.1 pt, markers 3.5 pt, error bars 0.6 pt drawn under the curves, guides 0.8 pt, horizontal grid only,
  line style + marker shape distinguish the series (readable in greyscale), local legends.
Plotting includes selective risk per detour bin with marginal target alpha = 0.02 as reference line."""
import json, copy, numpy as np
D = '../latex_8/matlab/data'
def ld(n): return json.load(open(f'{D}/{n}.json'))
def sv(p): json.dump(p, open(f"{D}/{p['name']}.json", 'w'), indent=1)
THIN = dict(lw_pt=1.1, ms_pt=3.5, err_lw=0.6, guide_lw=0.8, ygrid_only=True)
PAL = dict(blue=[0, .447, .698], verm=[.835, .369, 0], green=[0, .62, .451], purple=[.482, .196, .58], pink=[.8, .475, .655],
           amber=[.902, .624, 0], sky=[.337, .706, .914], grey=[.35, .35, .35])
def style(p, **kw): p = dict(p); p.update(THIN); p.update(kw); return p

# ---------------------------------------------------------------- Fig. 4 (2 x 2, pair layout)
DS = [('TRM, deterministic', 'TRM (deterministic)', PAL['grey'], 'o', '-'), ('One noisy trajectory', 'One noisy trajectory', PAL['sky'], '^', '--'),
      ('Majority of 16', 'Majority of 16', PAL['verm'], 's', '-.'), ('Oracle pass@16', 'Oracle pass@16', PAL['green'], 'd', ':')]
for t in ('maze', 'sudoku'):
    p = ld(f'fig_depth_{t}'); src = {s.get('label'): s for s in p['series']}; ser = []
    for old, lab, c, mk, ls in DS:
        s = src[old]; ser.append(dict(x=s['x'], y=s['y'], label=lab, color=c, marker=mk, line=ls))
    v = [s for s in p['series'] if s.get('kind') == 'vline'][0]
    ser.append(dict(kind='vline', x=v['x'], line='--', color=[.3, .3, .3], label='Training depth'))
    sv(style(dict(name=f'fig4_depth_{t}', layout='pair', xlabel=p['xlabel'], ylabel=p['ylabel'], xlim=p['xlim'], xticks=[1, 4, 8, 12, 16],
                  ylim=p['ylim'], lock_ylim=True, legend='southeast', series=ser), ms_pt=3.0))
p = ld('fig_reliability'); ser = []
for s, c, mk, ls in zip(p['series'], (PAL['blue'], PAL['verm']), 'os', ('-', '--')):
    ser.append(dict(x=s['x'], y=s['y'], label=s['label'], color=c, marker=mk, line=ls))
ser.append(dict(kind='vline', x=0.9, line=':', color=[.3, .3, .3], label=''))
sv(style(dict(name='fig4_reliability', layout='pair', xlabel='Agreement of 16 trajectories {\\itA}_{16}', ylabel='Error rate of the plurality answer',
              xlim=[0, 1.02], ylim=[0, 1], lock_ylim=True, legend='southwest', series=ser,
              annotations=[dict(x=0.885, y=0.62, text='High agreement:', halign='right', italic=False), dict(x=0.885, y=0.55, text='low plurality error', halign='right', italic=False)]),
         lw_pt=1.4, ms_pt=4.5))
p = ld('fig_wrong_attractor'); sty = [(PAL['blue'], 'o', '-'), (PAL['blue'], 's', '--'), (PAL['verm'], '^', '-'), (PAL['verm'], 'd', '--')]
ser = [dict(x=s['x'], y=s['y'], label=s['label'], color=c, marker=mk, line=ls) for s, (c, mk, ls) in zip(p['series'], sty)]
sv(style(dict(name='fig4_wrong', layout='pair', xlabel='Noise level {\\it\\sigma}', ylabel='Dominant wrong basin mass, estimate (%)', xlim=[0.2, 1.05],
              xticks=[0.25, 0.5, 1.0], ylim=[0, 4.2], lock_ylim=True, legend='northeast', series=ser), ms_pt=4.0))

# ---------------------------------------------------------------- Fig. 5 (trio, error bars) and Fig. 7(a, b)
for n in ('fig_risk_maze_e', 'fig_cov_maze_e', 'fig_viol_maze_e', 'fig_risk_sudoku_e', 'fig_cov_sudoku_e', 'fig_viol_sudoku_e'):
    p = style(ld(n)); p['name'] = n.replace('fig_', 'fig5_').replace('_e', ''); sv(p)
for n in ('fig_cov_street_e', 'fig_viol_street_e'):
    p = style(ld(n)); p['name'] = n.replace('fig_', 'fig7_').replace('_e', ''); sv(p)

# ---------------------------------------------------------------- Fig. 6
for t, cfg in (('maze', '8 x 4'), ('sudoku', '4 x 6')):
    p = style(ld(f'fig_adaptive_{t}_e')); p['name'] = f'fig6_adaptive_{t}'
    p['series'][0]['label'] = 'CHART'; p['series'][1]['label'] = f'CHART static ({cfg})'
    p['ylim'] = [0, 52] if t == 'maze' else [0, 40]; p['yticks'] = [0, 10, 20, 30, 40, 50] if t == 'maze' else [0, 10, 20, 30, 40]
    p['legend'] = 'northwest'; p['lock_ylim'] = True; sv(p)
for t in ('maze', 'sudoku'):
    p = style(ld(f'fig_wdT_{t}')); p['name'] = f'fig6_wdT_{t}'; Bs = [16, 32, 64, 128, 256, 512]; j = 0
    for s in p['series']: s['label'] = ''          # legend strip above panels (c), (d): make_legends.py
    p['legend'] = 'none'
    p.pop('annotations', None); sv(p)

# ---------------------------------------------------------------- Fig. 7(c): selective risk per detour bin, (d): recalibration
DT = json.load(open('results_real_detour.json'))['0.02']['bins']; xs = [0, 1.5, 3.5, 6.5, 10]
fmt = lambda n: f"{n/1000:.1f}k" if n >= 1000 else str(n)
ser = [dict(x=xs, y=[100*b[k]['risk'] for b in DT], label=lab, color=c, marker=mk, line=ls)
       for k, lab, c, mk, ls in (('CHART', 'CHART', PAL['blue'], 'o', '-'), ('CHART-static', 'CHART static', PAL['verm'], 's', '--'), ('TRM+Q-LTT', 'TRM+Q, LTT', PAL['green'], 'd', '-.'))]
ser.append(dict(kind='hline', y=2.0, line='--', color=[.3, .3, .3], label=''))
sv(style(dict(name='fig7_detour_risk', layout='quad', xlabel='Detour around buildings (steps)', ylabel='Selective risk at {\\it\\alpha} = 0.02 (%)',
              xlim=[-0.8, 10.8], xticks=xs, xticklabels=['0', '1-2', '3-4', '5-8', '9+'], ylim=[0, 6.6], yticks=[0, 1, 2, 3, 4], lock_ylim=True,
              legend='northwest', legend_cols=1, series=ser,
              annotations=[dict(x=10.6, y=1.45, text='target {\\it\\alpha}', halign='right', italic=False)])))
p = style(ld('fig_shift_dao_e')); p['name'] = 'fig7_shift_dao'
for s, lab in zip(p['series'], ('CHART', 'CHART, Bonf.', 'TRM+Q', 'PTRM+Q')): s['label'] = lab
p['ylim'] = [0, 1.9]; p['yticks'] = [0, 0.2, 0.4, 0.6, 0.8, 1.0]; p['legend'] = 'northwest'; p['legend_cols'] = 1; p['lock_ylim'] = True; sv(p)
print('ok')
