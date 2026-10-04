"""make_tables.py -- LaTeX tables generated from the result files:
tab_main_{maze,sudoku}.tex : population-risk protocol, mean +- SD over the 300 splits, violation with 95% CP interval
tab_decomp.tex              : component decomposition (decomp.py)
tab_seeds.tex               : seed robustness (seeds.py)"""
import json, os
OUT = '../latex_8'
def f1(x): return f'{100*x:.1f}'
def f2(x): return f'{100*x:.2f}'
ROWS = [('\\textbf{CHART (proposed)}', 'CHART'), ('CHART, Bonferroni', 'CHART-bonf'), ('PTRM + majority + LTT (CHART static)', 'CHART-static'),
        ('TRM + Q-head + LTT', 'TRM+Q-LTT'), ('PTRM + Q-head + LTT', 'PTRM+Q-LTT'), None, ('TRM + Q-head, plug-in', 'TRM+Q-naive'), ('CHART, plug-in', 'CHART-naive')]
def main_rows(task, alphas):
    E = json.load(open(f'results_e1_{task}.json'))['res']
    cert = [k for _, k in [r for r in ROWS if r] if 'naive' not in k]
    best = {a: max(E[str(a)][k]['cov'] for k in cert) for a in alphas}; L = []
    for r in ROWS:
        if r is None: L.append('\\cmidrule{2-9}'); continue
        lab, k = r; cells = []
        for a in alphas:
            v = E[str(a)][k]; lo, hi = v['viol_ci']
            cov = f"{f1(v['cov'])}$\\pm${f1(v['cov_sd'])}"
            if k in cert and f1(v['cov']) == f1(best[a]): cov = f"\\textbf{{{f1(v['cov'])}}}$\\pm${f1(v['cov_sd'])}"
            cells += [f"{f2(v['risk'])}$\\pm${f2(v['risk_sd'])}", f"{f1(v['viol'])} [{f1(lo)}, {f1(hi)}]", cov, f"{v['nfe']:.1f}"]
        L.append(' & ' + lab + ' & ' + ' & '.join(cells) + ' \\\\')
    return L
L = ['\\begin{table*}[!t]', '\\centering', '\\caption{Population-Risk Protocol on Mazes and Sudoku-Min}', '\\label{tab:main}', '\\setlength{\\tabcolsep}{2.6pt}',
     '\\begin{tabular}{@{}cl|cccc|cccc@{}}', '\\toprule',
     ' & & \\multicolumn{4}{c|}{target $\\alpha_1$} & \\multicolumn{4}{c}{target $\\alpha_2$} \\\\',
     ' & Method & Risk (\\%) & Viol. (\\%) [95\\% CI] & Cov. (\\%) & NFE & Risk (\\%) & Viol. (\\%) [95\\% CI] & Cov. (\\%) & NFE \\\\', '\\midrule']
rm = main_rows('maze', (0.02, 0.05)); rs = main_rows('sudoku', (0.01, 0.02))
L += ['\\multirow{%d}{*}{\\rotatebox{90}{Maze, $\\alpha_1=0.02$, $\\alpha_2=0.05$}}' % len(rm)] ; L[-1] += rm[0][1:] if False else ''
L[-1] = '\\multirow{%d}{*}{\\rotatebox{90}{\\footnotesize Maze}}' % (sum(1 for x in rm if 'cmidrule' not in x)) + rm[0]
L += rm[1:] + ['\\midrule']
L += ['\\multirow{%d}{*}{\\rotatebox{90}{\\footnotesize Sudoku-Min}}' % (sum(1 for x in rs if 'cmidrule' not in x)) + rs[0]] + rs[1:]
L += ['\\bottomrule', '\\end{tabular}',
      '\\par\\vspace{2pt}\\parbox{\\textwidth}{\\footnotesize Mean $\\pm$ standard deviation over the 300 random calibration splits; violation frequency with a 95\\% Clopper--Pearson interval computed as if the splits were independent (see Section~\\ref{sec:setup}). NFE: recursion steps per instance. Bold: highest coverage among risk-certified methods. The two targets are $\\alpha_1=0.02$, $\\alpha_2=0.05$ for mazes and $\\alpha_1=0.01$, $\\alpha_2=0.02$ for Sudoku-Min.}',
      '\\end{table*}']
open(f'{OUT}/tab_main.tex', 'w').write('\n'.join(L) + '\n')

# ---------------- decomposition
VAR = [('PTRM, fixed $K=16$, $T=6$', 'PTRM fixed K', '\\cmark', '--', 'Q-head', '--'),
       ('PTRM + Q-head + LTT', 'PTRM + Q-head + LTT', '\\cmark', '--', 'Q-head', '\\cmark'),
       ('PTRM + majority + LTT', 'PTRM + majority + LTT', '\\cmark', '--', 'fixed $K$', '\\cmark'),
       ('TRM + halt + LTT', 'TRM + halt + LTT', '--', '\\cmark', 'Q-head', '\\cmark'),
       ('PTRM + halt + LTT', 'PTRM + halt + LTT', '\\cmark', '\\cmark', 'Q-head', '\\cmark'),
       ('PTRM + halt + majority + LTT', 'PTRM + halt + majority + LTT', '\\cmark', '\\cmark', 'fixed $K$', '\\cmark'),
       ('e-process, no halting ($T=16$)', 'e-process, no halting (T=16)', '\\cmark', '--', 'e-process', '\\cmark'),
       ('e-process, fixed depth $T=4$', 'e-process, fixed depth T=4', '\\cmark', '$T=4$', 'e-process', '\\cmark'),
       ('CHART without LTT (plug-in)', 'CHART without LTT (plug-in)', '\\cmark', '\\cmark', 'e-process', '--'),
       ('\\textbf{CHART}', 'CHART', '\\cmark', '\\cmark', 'e-process', '\\cmark'),
       ('CHART, Bonferroni', 'CHART, Bonferroni', '\\cmark', '\\cmark', 'e-process', '\\cmark')]
BL = [('maze', '0.02'), ('maze', '0.05'), ('sudoku', '0.02'), ('street', '0.02')]
D = {t: json.load(open(f'results_decomp_{t}.json'))['res'] for t in ('maze', 'sudoku', 'street')}
L = ['\\begin{table*}[!t]', '\\centering', '\\caption{Component Decomposition on the Same Recorded Trajectories}', '\\label{tab:decomp}',
     '\\setlength{\\tabcolsep}{2.6pt}', '\\begin{tabular}{@{}lcccc|ccc|ccc|ccc|ccc@{}}', '\\toprule',
     ' & & & & & \\multicolumn{3}{c|}{Maze, $\\alpha=0.02$} & \\multicolumn{3}{c|}{Maze, $\\alpha=0.05$} & \\multicolumn{3}{c|}{Sudoku-Min, $\\alpha=0.02$} & \\multicolumn{3}{c}{Cities, $\\alpha=0.02$} \\\\',
     'Variant & Noise & Halting & Width rule & LTT & Cov. & Viol. & NFE & Cov. & Viol. & NFE & Cov. & Viol. & NFE & Cov. & Viol. & NFE \\\\', '\\midrule']
for lab, key, noise, halt, width, ltt in VAR:
    cells = []
    for t, a in BL:
        v = D[t][a][key]; cells += [f1(v['cov']), f1(v['viol']), f"{v['nfe']:.1f}"]
    L.append(f'{lab} & {noise} & {halt} & {width} & {ltt} & ' + ' & '.join(cells) + ' \\\\')
    if key in ('PTRM fixed K', 'PTRM + majority + LTT', 'PTRM + halt + majority + LTT', 'e-process, fixed depth T=4'): L.append('\\midrule')
L += ['\\bottomrule', '\\end{tabular}', '\\par\\vspace{2pt}\\parbox{\\textwidth}{\\footnotesize Coverage (Cov.) and violation frequency (Viol.) in \\%, mean over the 300 random calibration splits of Section~\\ref{sec:setup}; NFE: recursion steps per instance. ``Halting'' is stability halting; ``fixed $K$'' is the plurality of $K$ trajectories with an agreement threshold, $K$ and the threshold chosen by LTT. The accept-all PTRM row has no LTT and releases every answer, so its violation frequency is 100\\% whenever its error exceeds $\\alpha$.}',
      '\\end{table*}']
open(f'{OUT}/tab_decomp.tex', 'w').write('\n'.join(L) + '\n')

# ---------------- seeds
if os.path.exists('results_seeds.json'):
    S = json.load(open('results_seeds.json')); seeds = sorted(S, key=int)
    M = ['CHART', 'PTRM + halt + majority + LTT', 'PTRM + majority + LTT', 'TRM + halt + LTT', 'TRM + Q-head + LTT', 'PTRM + Q-head + LTT', 'CHART without LTT (plug-in)']
    L = ['\\begin{table}[!t]', '\\centering', '\\caption{Three Training Seeds of the Maze Reasoner}', '\\label{tab:seeds}', '\\setlength{\\tabcolsep}{2.4pt}',
         '\\resizebox{\\columnwidth}{!}{%', '\\begin{tabular}{@{}l' + 'c'*len(seeds) + '|' + 'c'*len(seeds) + '@{}}', '\\toprule',
         f' & \\multicolumn{{{len(seeds)}}}{{c|}}{{$\\alpha=0.02$}} & \\multicolumn{{{len(seeds)}}}{{c}}{{$\\alpha=0.05$}} \\\\',
         'Seed & ' + ' & '.join(seeds) + ' & ' + ' & '.join(seeds) + ' \\\\', '\\midrule',
         'Peak accuracy (\\%) & ' + ' & '.join(f1(S[s]['acc']['peak']) for s in seeds) + ' & ' + ' & '.join(f1(S[s]['acc']['peak']) for s in seeds) + ' \\\\',
         'Halted error (\\%) & ' + ' & '.join(f1(S[s]['acc']['halt_err']) for s in seeds) + ' & ' + ' & '.join(f1(S[s]['acc']['halt_err']) for s in seeds) + ' \\\\', '\\midrule',
         '\\multicolumn{' + str(1+2*len(seeds)) + '}{@{}l}{\\emph{Coverage (\\%) / violation frequency (\\%)}} \\\\']
    for k in M:
        lab = '\\textbf{CHART}' if k == 'CHART' else k
        cells = [f"{f1(S[s]['res'][a][k]['cov'])}/{f1(S[s]['res'][a][k]['viol'])}" for a in ('0.02', '0.05') for s in seeds]
        L.append(lab + ' & ' + ' & '.join(cells) + ' \\\\')
    L += ['\\bottomrule', '\\end{tabular}}', '\\par\\vspace{2pt}\\parbox{\\columnwidth}{\\footnotesize Same 6000 mazes for every seed, 300 random calibration splits with $n=2000$ and held-out risk on the other 4000.}', '\\end{table}']
    open(f'{OUT}/tab_seeds.tex', 'w').write('\n'.join(L) + '\n')
print('tables written')
