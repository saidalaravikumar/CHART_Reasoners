#!/bin/bash
# seed-2 maze reasoner: training, then main (4000, same instances as seed 0) and fresh (2000, generator seed 777001) pools
export SEED=2 THREADS=2
python3 train_maze_seed.py 99999 > log_seed2_train.txt 2>&1
cp trainlog_maze_s2.json trainlog_maze_s2.json.bak 2>/dev/null
python3 rollout.py maze_s2 main 4000 16 16 0,0.25,0.5,1.0 15 > log_seed2_main.txt 2>&1
python3 rollout.py maze_s2 fresh 2000 16 16 0,0.25,0.5,1.0 15 1e9 777001 > log_seed2_fresh.txt 2>&1
