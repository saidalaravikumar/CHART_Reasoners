"""realmaps.py -- practical navigation instances from the MovingAI grid benchmarks
(N. Sturtevant, IEEE Trans. Comput. Intell. AI Games 4(2):144-148, 2012; https://movingai.com/benchmarks).
 * street : city maps of 10 cities discretised from real building/road layouts (buildings = obstacles)
 * dao    : Dragon Age: Origins level maps (human-designed indoor rooms and corridors)
An instance is a GxG window cut from a map, a start S and a goal Gl in the same free component with
4-connected shortest distance >= min_len.  Open city blocks admit many equally short routes, so the label
is the shortest-route corridor: every free cell lying on at least one shortest S->Gl path
(D_S + D_G == L).  For a maze with a unique shortest path this is exactly the path used for the synthetic
mazes, so the reasoner, loss, correctness test and CHART layer are unchanged."""
import os, glob, numpy as np
from common import _bfs

DATA = os.environ.get('MAPDATA', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mapdata'))
CITIES = ['Berlin', 'Boston', 'Denver', 'London', 'Milan', 'Moscow', 'NewYork', 'Paris', 'Shanghai', 'Sydney']
TRAIN_CITIES = ['Berlin', 'Denver', 'London', 'Moscow', 'Paris', 'Sydney']     # 6 cities for training
TEST_CITIES = ['Boston', 'Milan', 'NewYork', 'Shanghai']                       # 4 unseen cities for calibration/test

def load_map(fn):
    L = open(fn).read().split('\n'); h = int(L[1].split()[1]); w = int(L[2].split()[1])
    return np.array([[0 if ch in '.GS' else 1 for ch in r[:w]] for r in L[4:4+h]], np.uint8)   # 1 = obstacle

def downsample(M, f):
    """Majority pooling by f (a coarse cell is blocked if at least half of its pixels are blocked)."""
    if f == 1: return M
    h, w = (M.shape[0]//f)*f, (M.shape[1]//f)*f
    return (M[:h, :w].reshape(h//f, f, w//f, f).mean((1, 3)) >= 0.5).astype(np.uint8)

def city_maps(cities, res=256, f=2):
    return [downsample(load_map(os.path.join(DATA, 'street', f'{c}_{i}_{res}.map')), f) for c in cities for i in range(3)]

def dao_maps(f=1, min_side=40):
    out = []
    for fn in sorted(glob.glob(os.path.join(DATA, 'dao', '*.map'))):
        M = downsample(load_map(fn), f)
        if min(M.shape) >= min_side and 0.05 < M.mean() < 0.85: out.append(M)
    return out

def make_real(maps, G, rng, min_len=None, max_corr=None):
    """Returns (x[3,G,G] wall/start/goal, y[G,G] corridor mask, L, map index)."""
    if min_len is None: min_len = G
    if max_corr is None: max_corr = 3*G
    while True:
        k = int(rng.integers(len(maps))); M = maps[k]
        r0 = int(rng.integers(M.shape[0]-G+1)); c0 = int(rng.integers(M.shape[1]-G+1))
        W = M[r0:r0+G, c0:c0+G].copy()
        free = np.argwhere(W == 0)
        if len(free) < 2*G: continue
        s = tuple(free[rng.integers(len(free))]); D, _ = _bfs(W, s)
        cand = np.argwhere(D >= min_len)
        if len(cand) == 0: continue
        g = tuple(cand[rng.integers(len(cand))]); L = int(D[g])
        Dg, _ = _bfs(W, g); corr = (D >= 0) & (Dg >= 0) & (D + Dg == L)
        if corr.sum() > max_corr: continue                  # skip trivially open windows
        x = np.zeros((3, G, G), np.float32); x[0] = W; x[1][s] = 1; x[2][g] = 1
        return x, corr.astype(np.uint8), L, k

def real_dataset(n, G, seed, domain='street', cities=None, min_len=None, **kw):
    rng = np.random.default_rng(seed)
    maps = city_maps(cities or TRAIN_CITIES, **kw) if domain == 'street' else dao_maps(**kw)
    X = np.zeros((n, 3, G, G), np.float32); Y = np.zeros((n, G, G), np.uint8); Lp = np.zeros(n, int); Mi = np.zeros(n, int)
    for i in range(n): X[i], Y[i], Lp[i], Mi[i] = make_real(maps, G, rng, min_len)
    return X, Y, Lp, Mi
