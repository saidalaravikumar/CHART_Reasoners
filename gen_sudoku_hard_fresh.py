"""Fresh Sudoku-Min pool (4500 puzzles, seeds disjoint from the main pool) for the population-risk check."""
import numpy as np
from multiprocessing import Pool
from common import make_sudoku
def shard(seed):
    rng = np.random.default_rng(seed); P, S, H = [], [], []
    for i in range(375):
        p, s, h = make_sudoku(rng, 64, 64); P.append(p.ravel()); S.append(s.ravel()); H.append(h)
    return np.array(P, np.int8), np.array(S, np.int8), np.array(H)
if __name__ == '__main__':
    with Pool(4) as pl: out = pl.map(shard, [888000 + k for k in range(12)])
    P = np.concatenate([o[0] for o in out]); S = np.concatenate([o[1] for o in out]); H = np.concatenate([o[2] for o in out])
    np.savez_compressed('sudoku_hard_fresh.npz', P=P, S=S, H=H); print(len(P), H.mean(), H.min(), H.max())
