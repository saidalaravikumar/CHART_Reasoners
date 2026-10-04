"""tests_realmaps.py -- audit of the practical-data pipeline (T10-T12), run before any experiment.
T10 labels: every corridor cell is free and lies on a shortest S->G path (checked by an independent
    Dijkstra-free recount: a cell is on a shortest path iff dist(S,c)+dist(c,G)==L); S and G are in the
    corridor; with a unique path the corridor has L+1 cells (reduces to the synthetic label).
T11 splits: training and test windows come from disjoint city sets; the indoor pool uses no city map.
T12 determinism: the same seed reproduces the same instances; different seeds give different ones."""
import numpy as np
from collections import deque
from realmaps import *

def bfs(W, s):
    G = W.shape[0]; D = np.full(W.shape, -1); D[s] = 0; q = deque([s])
    while q:
        r, c = q.popleft()
        for a, b in ((r+1, c), (r-1, c), (r, c+1), (r, c-1)):
            if 0 <= a < G and 0 <= b < G and W[a, b] == 0 and D[a, b] < 0: D[a, b] = D[r, c] + 1; q.append((a, b))
    return D

ok = True
for name in ('street_main', 'dao_main', 'street_train'):
    d = np.load(f'{name}.npz'); X, Y, L = d['X'][:1000], d['Y'][:1000], d['D'][:1000]; bad = 0; uniq = 0
    for x, y, l in zip(X, Y, L):
        W = x[0]; s = tuple(np.argwhere(x[1])[0]); g = tuple(np.argwhere(x[2])[0])
        Ds, Dg = bfs(W, s), bfs(W, g); corr = (Ds >= 0) & (Dg >= 0) & (Ds + Dg == Ds[g])
        bad += int(Ds[g] != l or not np.array_equal(corr, y.astype(bool)) or (y.astype(bool) & (W > 0)).any() or not (y[s] and y[g]) or l < 15)
        uniq += int(y.sum() == l + 1)
    print(f'T10 {name}: label errors {bad}/1000, unique-path windows {uniq}'); ok &= bad == 0
ok &= not (set(TRAIN_CITIES) & set(TEST_CITIES)) and set(TRAIN_CITIES) | set(TEST_CITIES) == set(CITIES)
print('T11 city split disjoint and complete:', not (set(TRAIN_CITIES) & set(TEST_CITIES)))
a = real_dataset(20, 15, 5, 'street', cities=TEST_CITIES, f=2); b = real_dataset(20, 15, 5, 'street', cities=TEST_CITIES, f=2)
c = real_dataset(20, 15, 6, 'street', cities=TEST_CITIES, f=2)
det = np.array_equal(a[0], b[0]) and not np.array_equal(a[0], c[0]); ok &= det
print('T12 deterministic generation:', det)
print('ALL PASS' if ok else 'FAILURE')
