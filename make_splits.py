"""make_splits.py -- exports the index sets of the 300 random calibration splits used by e1_population.py, exp_real.py,
decomp.py and seeds.py (generator numpy default_rng(99), one fresh generator per method and target).
Methods that split the calibration set into D1/D2 ('fst', 'given_d1skip') draw that split from the same stream right
after each calibration draw; methods that do not ('bonf', 'naive', accept-all) do not.  Both streams are exported:
  cal_<task>_split[r]   : calibration indices of split r for the D1/D2 methods, d1_<task>[r] their D1 part
  cal_<task>_plain[r]   : calibration indices of split r for the Bonferroni / plug-in / accept-all methods
The held-out set of split r is the complement of its calibration indices."""
import numpy as np
SIZES = dict(maze=(14000, 2000), sudoku=(6000, 1500), street=(9000, 2000), maze_seeds=(6000, 2000))
out = {}
for t, (N, n) in SIZES.items():
    g = np.random.default_rng(99); cal, d1 = [], []
    for r in range(300):
        perm = g.permutation(N); c = perm[:n]; cal.append(c); p = g.permutation(c); d1.append(p[:int(0.3*n)])
    g = np.random.default_rng(99); plain = [g.permutation(N)[:n] for r in range(300)]
    out[f'cal_{t}_split'] = np.array(cal, np.uint16); out[f'd1_{t}'] = np.array(d1, np.uint16); out[f'cal_{t}_plain'] = np.array(plain, np.uint16)
np.savez_compressed('calibration_splits.npz', **out); print({k: v.shape for k, v in out.items()})
