"""rollout.py -- record trajectory pools of the stochastic recursive reasoner (resumable in chunks).
For every instance and noise level sigma, run K i.i.d. noisy trajectories for T outer steps and store the
answer hash, exact correctness and Q-logit at every step.  Work is split into (sigma, block) units saved
as part files, so the job can be resumed across short sessions; the final call merges the parts.
usage: python rollout.py TASK TAG   (TASK in maze|sudoku|street) N K T sig1,sig2,.. [G_or_dataset] [time_budget_s] [seed] [offset]
(seed: maze-generator seed, default 12345+G; offset: first instance of a Sudoku file)"""
import os, sys, time, json, glob, numpy as np, torch
from common import *
torch.set_num_threads(int(os.environ.get('THREADS', 2)))
task, tag, n, K, T = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
sigmas = [float(s) for s in sys.argv[6].split(',')]
src = sys.argv[7] if len(sys.argv) > 7 else '15'
budget = float(sys.argv[8]) if len(sys.argv) > 8 else 1e9
SEED = int(sys.argv[9]) if len(sys.argv) > 9 else None
OFF = int(sys.argv[10]) if len(sys.argv) > 10 else 0
t_begin = time.time()
# practical maps (v5): task 'street' uses the fine-tuned reasoner ckpt_street.pt; src is an instance file
MTASK = 'maze' if (task in ('maze', 'street') or task.startswith('maze_s')) else task   # maze_s<k>: seed-k maze reasoner
cfg = json.load(open(f'trainlog_{task}.json'))['cfg']
model = TinyRecursiveReasoner(MTASK, cfg['C'], cfg['n'], cfg['T_cyc']); model.load_state_dict(torch.load(f'ckpt_{task}.pt')); model.eval()
if task == 'street':
    d = np.load(src); X, Y, D = d['X'][OFF:OFF+n].astype(np.float32), d['Y'][OFF:OFF+n], d['D'][OFF:OFF+n]
elif MTASK == 'maze':
    G = int(src); X, Y, D = maze_dataset(n, G, seed=(12345 + G) if SEED is None else SEED)   # seeds disjoint from training (1) and validation (999)
else:
    d = np.load(src); X, Y, D = d['P'][OFF:OFF+n].astype(np.int64), d['S'][OFF:OFF+n].astype(np.int64), d['H'][OFF:OFF+n]
X, Y = torch.tensor(X), torch.tensor(Y)
BLK = 250; os.makedirs(f'parts_{task}_{tag}', exist_ok=True)
for sg in sigmas:
    Kr = 1 if sg == 0 else K
    for b in range(0, n, BLK):
        fn = f'parts_{task}_{tag}/s{sg}_b{b}.npz'
        if os.path.exists(fn): continue
        if time.time() - t_begin > budget: print('budget reached'); sys.exit(0)
        nb = min(BLK, n - b); gen = torch.Generator().manual_seed(int(1000*sg) + 7 + 100003*b + (0 if SEED is None else 7919*SEED))
        ans = np.zeros((nb, Kr, T), np.int64); cor = np.zeros((nb, Kr, T), bool); qv = np.zeros((nb, Kr, T), np.float32)
        mb = max(1, 256 // Kr)
        with torch.no_grad():
            for c in range(0, nb, mb):
                x = X[b+c:b+min(c+mb, nb)].repeat_interleave(Kr, 0); y = Y[b+c:b+min(c+mb, nb)].repeat_interleave(Kr, 0); m = x.shape[0]//Kr
                xe = model.embed(x); yy, zz = model.init_state(xe, sg if sg > 0 else None, gen)
                for t in range(T):
                    yy, zz = model.outer_step(xe, yy, zz, grad=False, sigma=sg if sg > 0 else None, gen=gen)
                    pr = predict(MTASK, model.decode(yy), x)
                    ans[c:c+m, :, t] = answer_hash(pr).reshape(m, Kr)
                    cor[c:c+m, :, t] = exact_correct(MTASK, pr, y).numpy().reshape(m, Kr)
                    qv[c:c+m, :, t] = model.qvalue(yy).numpy().reshape(m, Kr)
        np.savez_compressed(fn + '.tmp.npz', ans=ans, cor=cor, q=qv); os.replace(fn + '.tmp.npz', fn)   # atomic
        print(f'sigma {sg} block {b}: acc(traj0,T) {cor[:,0,-1].mean():.3f}  elapsed {time.time()-t_begin:.0f}s', flush=True)
out = dict(difficulty=D)
for sg in sigmas:
    parts = [np.load(f'parts_{task}_{tag}/s{sg}_b{b}.npz') for b in range(0, n, BLK)]
    for k in ('ans', 'cor', 'q'): out[f'{k}_{sg}'] = np.concatenate([p[k] for p in parts])
np.savez_compressed(f'pool_{task}_{tag}.npz', **out); print('saved', f'pool_{task}_{tag}.npz')
