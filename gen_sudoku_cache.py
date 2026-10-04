"""Pre-generate Sudoku datasets (unique-solution puzzles, 30-58 blanks) in parallel shards -> .npz cache."""
import numpy as np, sys
from multiprocessing import Pool
from common import sudoku_dataset
def shard(args):
    n, seed = args; P, S, H = sudoku_dataset(n, seed=seed); return P.astype(np.int8), S.astype(np.int8), H
if __name__ == '__main__':
    for name, n, base in [('val', 500, 999), ('pool', 3000, 12345), ('train', 40000, 1)]:
        ns = 8 if n >= 3000 else 1; per = n // ns
        with Pool(4) as p: out = p.map(shard, [(per, base*100 + k) for k in range(ns)])
        P = np.concatenate([o[0] for o in out]); S = np.concatenate([o[1] for o in out]); H = np.concatenate([o[2] for o in out])
        np.savez_compressed(f'sudoku_{name}.npz', P=P, S=S, H=H); print(name, len(P), H.mean(), flush=True)
