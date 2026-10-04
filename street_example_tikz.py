"""street_example_tikz.py -- the practical counterpart of maze_example_tikz.py: one easy and one hard
15x15 window from the UNSEEN cities (Boston, Milan, New York, Shanghai), K=8 noisy trajectories of the
fine-tuned reasoner (sigma=1.0, stability halting m=1), basins drawn as in Fig. 3.
Output: ../latex_6/fig_street_examples.tex"""
import json, numpy as np, torch
from common import *
from chart import stability_halt
torch.set_num_threads(2)
cfg = json.load(open('trainlog_street.json'))['cfg']
m = TinyRecursiveReasoner('maze', cfg['C'], cfg['n'], cfg['T_cyc']); m.load_state_dict(torch.load('ckpt_street.pt')); m.eval()
d = np.load('street_main.npz'); X, Y, L = torch.tensor(d['X'][:400]).float(), torch.tensor(d['Y'][:400]), d['D'][:400]
K, T, sg = 8, 16, 1.0; gen = torch.Generator().manual_seed(3)
with torch.no_grad():
    x = X.repeat_interleave(K, 0); xe = m.embed(x); yy, zz = m.init_state(xe, sg, gen)
    P = []
    for t in range(T):
        yy, zz = m.outer_step(xe, yy, zz, grad=False, sigma=sg, gen=gen); P.append(predict('maze', m.decode(yy), x).numpy())
P = np.stack(P, 1).reshape(400, K, T, 15, 15)
H = np.array([[[hash(P[i, k, t].tobytes()) for t in range(T)] for k in range(K)] for i in range(400)])
idx = stability_halt(H, 1, T)                               # [400,K]
Fin = np.array([[P[i, k, idx[i, k]] for k in range(K)] for i in range(400)])
cor = np.array([[(Fin[i, k] == Y[i].numpy()).all() for k in range(K)] for i in range(400)])
agree = np.array([len(set(map(bytes, [f.tobytes() for f in Fin[i]]))) for i in range(400)])
man = np.array([np.abs(np.argwhere(X[i, 1].numpy() > 0)[0] - np.argwhere(X[i, 2].numpy() > 0)[0]).sum() for i in range(400)])
easy = [i for i in range(400) if agree[i] == 1 and cor[i].all() and L[i] - man[i] >= 4]
hard = sorted([i for i in range(400) if 2 <= agree[i] <= 3 and cor[i].any() and not cor[i].all()], key=lambda i: (-agree[i], -(K - cor[i].sum())))
print('easy', len(easy), 'hard', len(hard))
def draw(i, ox, title, wall='black!75'):
    """basins drawn with colour AND pattern (readable in greyscale); a legend lists the trajectories T1..TK of each basin"""
    s = []; W = X[i, 0].numpy(); g = 0.16
    s.append(f'\\node[font=\\scriptsize] at ({ox+15*g/2:.2f},{15*g+0.25:.2f}) {{{title}}};')
    for r in range(15):
        for c in range(15):
            if W[r, c] > 0.5: s.append(f'\\fill[{wall}] ({ox+c*g:.2f},{(14-r)*g:.2f}) rectangle ++({g},{g});')
    keys = [Fin[i, k].tobytes() for k in range(K)]; uniq = sorted(set(keys), key=lambda u: -keys.count(u))
    sty = ['fill=blue!65', 'pattern=north east lines, pattern color=orange!90!black', 'pattern=crosshatch dots, pattern color=green!45!black']
    edge = ['blue!65', 'orange!90!black', 'green!45!black']
    for j, u in enumerate(uniq[:3]):
        k = keys.index(u); M = Fin[i, k]; off = [0.0, 0.045, -0.045][j]
        for r in range(15):
            for c in range(15):
                if M[r, c]: s.append(f'\\filldraw[{sty[j]}, draw={edge[j]}, line width=0.2pt] ({ox+c*g+0.03+off:.2f},{(14-r)*g+0.03+off:.2f}) rectangle ++({g-0.06},{g-0.06});')
    sr, sc_ = np.argwhere(X[i, 1].numpy() > 0)[0]; gr, gc = np.argwhere(X[i, 2].numpy() > 0)[0]
    s.append(f'\\node[circle, fill=red, draw=white, inner sep=1.3pt] at ({ox+sc_*g+g/2:.2f},{(14-sr)*g+g/2:.2f}) {{}};')
    s.append(f'\\node[star, fill=red, draw=white, inner sep=1.1pt] at ({ox+gc*g+g/2:.2f},{(14-gr)*g+g/2:.2f}) {{}};')
    s.append(f'\\draw[black!60] ({ox},0) rectangle ({ox+15*g:.2f},{15*g:.2f});')
    cnt = [keys.count(u) for u in uniq]; okl = ['correct' if cor[i, keys.index(u)] else 'wrong' for u in uniq]
    for j, u in enumerate(uniq[:3]):
        ids = [k+1 for k in range(K) if keys[k] == u]; runs = []
        for v in ids:
            if runs and v == runs[-1][1] + 1: runs[-1][1] = v
            else: runs.append([v, v])
        members = ','.join(f'{a}' if a == b else (f'{a},{b}' if b == a+1 else f'{a}\\text{{--}}{b}') for a, b in runs)
        y = -0.28 - 0.3*j
        s.append(f'\\filldraw[{sty[j]}, draw={edge[j]}] ({ox:.2f},{y-0.07:.2f}) rectangle ++(0.14,0.14);')
        s.append(f'\\node[font=\\scriptsize, anchor=west] at ({ox+0.17:.2f},{y:.2f}) {{$B_{j+1}$: $T_{{{members}}}$, {okl[j]}}};')
    return s, cnt, okl
from realmaps import TEST_CITIES
city = lambda i: {'NewYork': 'New York'}.get(TEST_CITIES[int(d['M'][i])//3], TEST_CITIES[int(d['M'][i])//3])
se, ce, oe = draw(easy[0], 0.0, f'{city(easy[0])}: one basin', wall='black!45')
sh, ch, oh = draw(hard[0], 3.7, f'{city(hard[0])}: three basins', wall='black!45')
tex = ['\\begin{tikzpicture}'] + se + sh + ['\\end{tikzpicture}']
open('../latex_8/fig_street_examples.tex', 'w').write('\n'.join(tex))
json.dump(dict(easy=int(easy[0]), hard=int(hard[0]), easy_counts=ce, hard_counts=ch, easy_ok=oe, hard_ok=oh, L_easy=int(L[easy[0]]), L_hard=int(L[hard[0]]), city_easy=city(easy[0]), city_hard=city(hard[0])), open('street_example_info.json', 'w'))
print(ce, oe, ch, oh)
