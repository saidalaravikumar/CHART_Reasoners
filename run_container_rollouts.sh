#!/bin/bash
# container share of the practical recordings (unseen-city main pool and indoor pool)
export THREADS=2
python3 rollout.py street main 3000 16 16 0,0.25,0.5,1.0 street_main.npz > log_main.txt 2>&1
python3 rollout.py street dao 2000 16 16 0,0.25,0.5,1.0 dao_main.npz > log_dao.txt 2>&1
