"""timing.py -- wall-clock cost of one outer recursion step (batch of 256, CPU, 2 threads) and of classical
exact solvers (BFS for mazes, bitmask DFS for Sudoku), used to convert NFE into latency."""
import time, json, numpy as np, torch
from common import *
torch.set_num_threads(2); out = {}
for task in ('maze', 'sudoku'):
    cfg = json.load(open(f'trainlog_{task}.json'))['cfg']
    m = TinyRecursiveReasoner(task, cfg['C'], cfg['n'], cfg['T_cyc']); m.load_state_dict(torch.load(f'ckpt_{task}.pt')); m.eval()
    out[f'{task}_params'] = sum(p.numel() for p in m.parameters())
    if task == 'maze': X, Y, _ = maze_dataset(256, 15, 4242); X = torch.tensor(X)
    else: d = np.load('sudoku_hard.npz'); X = torch.tensor(d['P'][:256].astype(np.int64))
    with torch.no_grad():
        xe = m.embed(X); y, z = m.init_state(xe, 0.5)
        for _ in range(2): y, z = m.outer_step(xe, y, z, grad=False, sigma=0.5)
        t = time.time(); R = 10
        for _ in range(R): y, z = m.outer_step(xe, y, z, grad=False, sigma=0.5)
    out[f'{task}_ms_per_step_per_instance'] = 1000*(time.time()-t)/R/256
from common import _bfs, _count_solutions
Xm, Ym, _ = maze_dataset(500, 15, 4343); t = time.time()
for x in Xm:
    W = x[0].astype(np.uint8); s = tuple(np.argwhere(x[1] > 0)[0]); _bfs(W, s)
out['bfs_ms'] = 1000*(time.time()-t)/500
d = np.load('sudoku_hard.npz'); t = time.time()
for p in d['P'][:200]: _count_solutions(p.reshape(9, 9).astype(np.int64), cap=1)
out['dfs_sudoku_ms'] = 1000*(time.time()-t)/200
print(out); json.dump(out, open('timing.json', 'w'))
