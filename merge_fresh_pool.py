"""Rebuild pool_maze_fresh.npz (10000 mazes) from its two halves (split only to respect file-size limits)."""
import numpy as np
a, b = np.load('pool_maze_fresh_part1.npz'), np.load('pool_maze_fresh_part2.npz')
np.savez_compressed('pool_maze_fresh.npz', **{k: np.concatenate([a[k], b[k]]) for k in a.files}); print('ok')
