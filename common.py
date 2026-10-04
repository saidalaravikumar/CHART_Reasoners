"""
common.py -- data generators, tiny recursive reasoner (TRM-style) and helpers.
Project: CHART (Certified Halting And Risk-controlled Trajectories) for tiny recursive reasoners.
All randomness is seeded; every function is deterministic given its seed.
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, hashlib
from collections import deque

# ----------------------------------------------------------------------------------
# 1.  MAZE generation  (grid G x G, G odd; walls=1, free=0; unique shortest path S->G)
# ----------------------------------------------------------------------------------
def _perfect_maze(G, rng):
    """Iterative recursive-backtracker on the (G//2)x(G//2) cell lattice."""
    W = np.ones((G, G), np.uint8)
    nc = G // 2
    vis = np.zeros((nc, nc), bool)
    st = [(rng.integers(nc), rng.integers(nc))]
    vis[st[0]] = True; W[2*st[0][0]+1, 2*st[0][1]+1] = 0
    while st:
        r, c = st[-1]
        nb = [(r+dr, c+dc) for dr, dc in ((1,0),(-1,0),(0,1),(0,-1))
              if 0 <= r+dr < nc and 0 <= c+dc < nc and not vis[r+dr, c+dc]]
        if not nb: st.pop(); continue
        nr, nc2 = nb[rng.integers(len(nb))]
        W[2*r+1+(nr-r), 2*c+1+(nc2-c)] = 0; W[2*nr+1, 2*nc2+1] = 0
        vis[nr, nc2] = True; st.append((nr, nc2))
    return W

def _bfs(W, s):
    """BFS distances and number of shortest paths (capped at 2) from s."""
    G = W.shape[0]; D = -np.ones((G, G), int); Nsp = np.zeros((G, G), int)
    D[s] = 0; Nsp[s] = 1; q = deque([s])
    while q:
        r, c = q.popleft()
        for dr, dc in ((1,0),(-1,0),(0,1),(0,-1)):
            a, b = r+dr, c+dc
            if 0 <= a < G and 0 <= b < G and W[a, b] == 0:
                if D[a, b] < 0:
                    D[a, b] = D[r, c] + 1; Nsp[a, b] = Nsp[r, c]; q.append((a, b))
                elif D[a, b] == D[r, c] + 1:
                    Nsp[a, b] = min(2, Nsp[a, b] + Nsp[r, c])
    return D, Nsp

def make_maze(G, rng, p_loop=0.08, min_len=None):
    """Returns (x[3,G,G] float32 : wall,start,goal ; y[G,G] uint8 path mask ; path length).
    Loops are added by knocking out interior walls with prob p_loop; instances whose
    shortest S->G path is not unique are rejected so that the label is well defined."""
    if min_len is None: min_len = G
    while True:
        W = _perfect_maze(G, rng)
        # braid: remove removable interior walls (between two free cells)
        for r in range(1, G-1):
            for c in range(1, G-1):
                if W[r, c] == 1 and rng.random() < p_loop:
                    if (W[r-1, c] == 0 and W[r+1, c] == 0 and W[r, c-1] == 1 and W[r, c+1] == 1) or \
                       (W[r, c-1] == 0 and W[r, c+1] == 0 and W[r-1, c] == 1 and W[r+1, c] == 1):
                        W[r, c] = 0
        free = np.argwhere(W == 0)
        s = tuple(free[rng.integers(len(free))])
        D, Nsp = _bfs(W, s)
        cand = np.argwhere(D >= min_len)
        if len(cand) == 0: continue
        g = tuple(cand[rng.integers(len(cand))])
        if Nsp[g] != 1: continue
        # back-track the unique path
        Dg, _ = _bfs(W, g)
        L = D[g]
        path = (D >= 0) & (Dg >= 0) & (D + Dg == L)
        # uniqueness => cells with D+Dg==L form exactly one path of L+1 cells
        if path.sum() != L + 1: continue
        x = np.zeros((3, G, G), np.float32); x[0] = W; x[1][s] = 1; x[2][g] = 1
        return x, path.astype(np.uint8), int(L)

def maze_dataset(n, G, seed, p_loop=0.08, min_len=None):
    rng = np.random.default_rng(seed)
    X = np.zeros((n, 3, G, G), np.float32); Y = np.zeros((n, G, G), np.uint8); Lp = np.zeros(n, int)
    for i in range(n):
        X[i], Y[i], Lp[i] = make_maze(G, rng, p_loop, min_len)
    return X, Y, Lp

# ----------------------------------------------------------------------------------
# 2.  SUDOKU generation (9x9, unique solution verified with a bitmask solver)
# ----------------------------------------------------------------------------------
def _sud_full(rng):
    base = 3; side = 9
    def pattern(r, c): return (base*(r % base) + r//base + c) % side
    rb = rng.permutation(3); rows = [g*3 + r for g in rng.permutation(3) for r in rng.permutation(3)]
    cols = [g*3 + c for g in rng.permutation(3) for c in rng.permutation(3)]
    nums = rng.permutation(9) + 1
    B = np.array([[nums[pattern(r, c)] for c in cols] for r in rows], np.int8)
    if rng.random() < 0.5: B = B.T.copy()
    return B

def _count_solutions(P, cap=2):
    """Bitmask DFS with MRV; returns min(#solutions, cap)."""
    rows = [0]*9; cols = [0]*9; boxes = [0]*9; empt = []
    for r in range(9):
        for c in range(9):
            v = int(P[r, c])
            if v:
                b = 1 << v; rows[r] |= b; cols[c] |= b; boxes[(r//3)*3 + c//3] |= b
            else: empt.append((r, c))
    FULL = 0b1111111110
    cnt = [0]
    def rec(empt):
        if cnt[0] >= cap: return
        if not empt: cnt[0] += 1; return
        best = None; bm = None; bc = 10
        for i, (r, c) in enumerate(empt):
            m = FULL & ~(rows[r] | cols[c] | boxes[(r//3)*3 + c//3])
            k = bin(m).count('1')
            if k < bc: bc = k; best = i; bm = m
            if k == 0: return
        r, c = empt[best]; rest = empt[:best] + empt[best+1:]; bx = (r//3)*3 + c//3
        m = bm
        while m:
            b = m & -m; m ^= b
            rows[r] |= b; cols[c] |= b; boxes[bx] |= b
            rec(rest)
            rows[r] ^= b; cols[c] ^= b; boxes[bx] ^= b
            if cnt[0] >= cap: return
    rec(empt)
    return cnt[0]

def make_sudoku(rng, hmin, hmax):
    S = _sud_full(rng)
    h = int(rng.integers(hmin, hmax + 1))
    P = S.copy()
    order = rng.permutation(81); removed = 0
    for idx in order:
        if removed >= h: break
        r, c = divmod(int(idx), 9); v = P[r, c]; P[r, c] = 0
        if _count_solutions(P) != 1: P[r, c] = v
        else: removed += 1
    return P, S, removed

def sudoku_dataset(n, seed, hmin=30, hmax=58):
    rng = np.random.default_rng(seed)
    P = np.zeros((n, 81), np.int64); S = np.zeros((n, 81), np.int64); H = np.zeros(n, int)
    for i in range(n):
        p, s, h = make_sudoku(rng, hmin, hmax); P[i] = p.ravel(); S[i] = s.ravel(); H[i] = h
    return P, S, H

# ----------------------------------------------------------------------------------
# 3.  Tiny Recursive Reasoner (TRM-style): z <- f(x+y+z) (n times), y <- f(y+z)
# ----------------------------------------------------------------------------------
class RMSNormC(nn.Module):
    def __init__(self, C, dim):
        super().__init__(); self.g = nn.Parameter(torch.ones(C)); self.dim = dim
    def forward(self, h):
        r = h.pow(2).mean(self.dim, keepdim=True).add(1e-6).rsqrt()
        shape = [1]*h.dim(); shape[self.dim] = -1
        return h * r * self.g.view(shape)

class ConvCore(nn.Module):
    """Weight-tied core f for grids: two residual conv blocks + RMS norm (channel dim)."""
    def __init__(self, C):
        super().__init__()
        self.c = nn.ModuleList([nn.Conv2d(C, C, 3, padding=1) for _ in range(4)])
        self.n1 = RMSNormC(C, 1); self.n2 = RMSNormC(C, 1)
    def forward(self, h):
        h = self.n1(h + self.c[1](F.gelu(self.c[0](h))))
        h = self.n2(h + self.c[3](F.gelu(self.c[2](h))))
        return h

class MixerCore(nn.Module):
    """Weight-tied core f for Sudoku (81 tokens): two MLP-mixer blocks (token + channel mixing)."""
    def __init__(self, D, L=81, E=4):
        super().__init__()
        self.tok = nn.ModuleList([nn.Sequential(nn.Linear(L, 2*L), nn.GELU(), nn.Linear(2*L, L)) for _ in range(2)])
        self.ch = nn.ModuleList([nn.Sequential(nn.Linear(D, E*D), nn.GELU(), nn.Linear(E*D, D)) for _ in range(2)])
        self.n = nn.ModuleList([RMSNormC(D, -1) for _ in range(4)])
    def forward(self, h):              # h: [B, 81, D]
        for i in range(2):
            h = self.n[2*i](h + self.tok[i](h.transpose(1, 2)).transpose(1, 2))
            h = self.n[2*i+1](h + self.ch[i](h))
        return h

class TinyRecursiveReasoner(nn.Module):
    """task in {'maze','sudoku'}. One outer (supervision) step = T_cyc cycles of
    [n latent updates z<-f(x+y+z); one answer update y<-f(y+z)]; gradients only through
    the last cycle (TRM recipe). Noise sigma is injected into the initial latent state (PTRM)."""
    def __init__(self, task, C=48, n=2, T_cyc=2):
        super().__init__()
        self.task, self.C, self.n, self.T_cyc = task, C, n, T_cyc
        if task == 'maze':
            self.emb = nn.Conv2d(3, C, 3, padding=1); self.core = ConvCore(C)
            self.head = nn.Conv2d(C, 1, 1)
        else:
            self.emb_tok = nn.Embedding(10, C); self.pos = nn.Parameter(0.02*torch.randn(1, 81, C))
            self.core = MixerCore(C); self.head = nn.Linear(C, 9)
        self.qhead = nn.Linear(C, 1)
        self.y0 = nn.Parameter(torch.zeros(C)); self.z0 = nn.Parameter(torch.zeros(C))

    def embed(self, x):
        return self.emb(x) if self.task == 'maze' else self.emb_tok(x) + self.pos

    def init_state(self, xe, sigma, gen=None):
        if self.task == 'maze':
            shp = xe.shape; y = self.y0.view(1, -1, 1, 1).expand(shp).clone(); z = self.z0.view(1, -1, 1, 1).expand(shp).clone()
        else:
            shp = xe.shape; y = self.y0.view(1, 1, -1).expand(shp).clone(); z = self.z0.view(1, 1, -1).expand(shp).clone()
        if sigma is not None:
            s = sigma if torch.is_tensor(sigma) else torch.tensor(float(sigma))
            s = s.view(-1, *([1]*(len(shp)-1))) if s.dim() > 0 else s
            y = y + s*torch.randn(shp, generator=gen); z = z + s*torch.randn(shp, generator=gen)
        return y, z

    def cycle(self, xe, y, z):
        for _ in range(self.n): z = self.core(xe + y + z)
        y = self.core(y + z)
        return y, z

    def outer_step(self, xe, y, z, grad=True, sigma=None, gen=None):
        """PTRM: fresh Gaussian noise is injected into the latent z at every outer (deep-recursion) step."""
        if sigma is not None:
            s = sigma if torch.is_tensor(sigma) else torch.tensor(float(sigma))
            s = s.view(-1, *([1]*(z.dim()-1))) if s.dim() > 0 else s
            z = z + s*torch.randn(z.shape, generator=gen)
        with torch.no_grad():
            for _ in range(self.T_cyc - 1): y, z = self.cycle(xe, y, z)
        if grad: y, z = self.cycle(xe, y, z)
        else:
            with torch.no_grad(): y, z = self.cycle(xe, y, z)
        return y, z

    def decode(self, y):
        if self.task == 'maze': return self.head(y).squeeze(1)          # [B,G,G] logits
        return self.head(y)                                              # [B,81,9]

    def qvalue(self, y):
        pooled = y.mean((2, 3)) if self.task == 'maze' else y.mean(1)
        return self.qhead(pooled).squeeze(-1)

def predict(task, logits, x):
    """Discrete answer from logits. maze: path mask on free cells; sudoku: digits, clues kept."""
    if task == 'maze':
        return ((logits > 0) & (x[:, 0] < 0.5)).to(torch.uint8)
    pred = logits.argmax(-1) + 1
    return torch.where(x > 0, x, pred)

def exact_correct(task, pred, y):
    if task == 'maze': return (pred == y).flatten(1).all(1)
    return (pred == y).all(1)

def answer_hash(pred):
    """64-bit hash of each discrete answer (identity of the attractor/basin reached)."""
    a = pred.cpu().numpy().reshape(pred.shape[0], -1).astype(np.uint8)
    return np.array([int.from_bytes(hashlib.blake2b(r.tobytes(), digest_size=8).digest(), 'little', signed=True) for r in a], np.int64)
