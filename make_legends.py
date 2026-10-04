"""make_legends.py -- common legend strips (TikZ, standalone) for Figs. 5-7, in the matlab_fig_1 palette."""
import json
P = json.load(open('../latex_8/matlab/palette.json'))
P['CHART-bonf'] = dict(color=[.35, .35, .35], marker='>', line='-.', label='CHART, Bonferroni')
MK = {'o': 'mark=*', 's': 'mark=square*', 'd': 'mark=diamond*', '^': 'mark=triangle*', 'v': 'mark=triangle*, mark options={rotate=180}',
      'p': 'mark=star, mark options={line width=0.9pt}', '>': 'mark=triangle*, mark options={rotate=-90}'}
LS = {'-': 'solid', '--': 'dashed', '-.': 'dash dot', ':': 'dotted'}
def entry(c, mk, ls, lab, i):
    col = '{rgb,1:red,%.3f;green,%.3f;blue,%.3f}' % tuple(c)
    m = MK.get(mk, '')
    extra = (', ' + m.split('mark options={')[1][:-1]) if 'mark options' in m else ''
    mk_ = m.split(', mark options')[0] if m else ''
    prev = f'n{i-1}.east' if i > 0 else '0,0'
    s = f'\\path ({prev}) ++({0.35 if i > 0 else 0},0) coordinate (s{i});\n'
    s += f'\\draw[color={col}, line width=1.1pt, {LS[ls]}] (s{i}) -- ++(0.65,0);\n'
    if m: s += f'\\draw[color={col}, {mk_}, mark options={{fill={col}, draw={col}{extra}}}, mark size=2.1pt] plot coordinates {{($(s{i})+(0.325,0)$)}};\n'
    s += f'\\node[anchor=west, font=\\footnotesize, inner sep=1pt] (n{i}) at ($(s{i})+(0.7,0)$) {{{lab}}};\n'
    return s
def strip(name, items, widths=None):
    body = ''.join(entry(*it, i) for i, it in enumerate(items))
    tex = ('\\documentclass[border=2pt]{standalone}\n\\usepackage{amsmath}\n\\usepackage{tikz}\n\\usetikzlibrary{plotmarks,calc}\n\\begin{document}\n'
           '\\begin{tikzpicture}\n' + body + '\\end{tikzpicture}\n\\end{document}\n')
    open(f'../latex_8/{name}.tex', 'w').write(tex)
def m(k, lab=None): d = P[k]; return (d['color'], d['marker'], d['line'], lab or d['label'])
grey = [.4, .4, .4]
strip('fig_legend_methods', [m('CHART-seq'), m('CHART-static'), m('TRM+Q-LTT'), m('PTRM+Q-LTT'), m('TRM+Q-naive', 'TRM+Q, plug-in'), m('CHART-naive', 'CHART, plug-in'),
      (grey, '', ':', 'risk $=\\alpha$'), (grey, '', '--', '$\\delta=10\\%$')], [3.0, 2.55, 2.3, 2.4, 2.55, 2.6, 1.9, 1.6])
strip('fig_legend_real_a', [m('CHART-seq', 'CHART'), m('CHART-bonf', 'CHART, Bonf.'), m('CHART-static', 'CHART static'), m('TRM+Q-LTT', 'TRM+Q, LTT')])
strip('fig_legend_real_b', [m('PTRM+Q-LTT', 'PTRM+Q, LTT'), m('TRM+Q-naive', 'TRM+Q, plug-in'), m('CHART-naive', 'CHART, plug-in'), (grey, '', '--', '$\\delta$')])
PAL = [[0, .447, .698], [.835, .369, 0], [0, .62, .451], [.482, .196, .58], [.8, .475, .655], [.902, .624, 0]]
WD = [(c, mk, ls, f'$B={b}$') for c, mk, ls, b in zip(PAL, 'osd^vp', ['-', '--', '-.', ':', '-', '--'], [16, 32, 64, 128, 256, 512])]
strip('fig_legend_wd_a', WD[:4])
strip('fig_legend_wd_b', WD[4:] + [([.2, .2, .2], '', '--', 'model $T^\\ast$'), ([.45, .45, .45], '', ':', 'empirical optimum')])
print('ok')
