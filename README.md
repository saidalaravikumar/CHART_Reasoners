# Recursive Reasoning and Risk-Controlled Adaptive Testing

Implementation of recursive reasoning models with confidence-halted anytime recursive testing (CHART). The framework couples depth recursion via stability halting, width stochastic exploration, an anytime-valid consensus e-process, and Learn-Then-Test (LTT) risk calibration across Mazes, Sudoku-Min, and real-world grid map routing (MovingAI benchmarks).

## Requirements

- Python 3.8+
- PyTorch (`torch`)
- NumPy
- SciPy

Install dependencies via:
```bash
pip install torch numpy scipy
```

## Execution Workflow

### 1. Audit and Mathematical Tests
Run the test suite to verify stability halting, Ville e-process type-I error control, Wald bounds, Hoeffding width bounds, and dataset splits:
```bash
python tests_audit.py
python tests_realmaps.py
```

### 2. Dataset Preparation
Generate the real-world street and game map benchmarks from the map archives:
```bash
python gen_real_data.py
```
This produces `street_{train,val,main,fresh}.npz` and `dao_main.npz`.

To generate Sudoku datasets:
```bash
python gen_sudoku_hard.py
python gen_sudoku_hard_fresh.py
python gen_sudoku_cache.py
```

### 3. Training Reasoners

- **Train Base Maze Reasoner:**
  ```bash
  python train.py maze
  ```
- **Train Across Additional Random Seeds (resumable):**
  ```bash
  SEED=1 python train_maze_seed.py 160
  SEED=2 python train_maze_seed.py 160
  ```
- **Train Sudoku Reasoner:**
  ```bash
  python train.py sudoku
  ```
- **Fine-tune on Real Maps (Sim-to-Real):**
  ```bash
  THREADS=4 python train_real.py 50 160
  ```

### 4. Trajectory Rollouts

Record stochastic trajectory pools across depth and width:

- **Maze:**
  ```bash
  python rollout.py maze main 4000 16 16 0,0.25,0.5,1.0 15
  python rollout.py maze fresh 2000 16 16 0,0.25,0.5,1.0 15 1e9 777001
  ```
- **Sudoku:**
  ```bash
  python rollout.py sudoku main 1500 16 16 0,0.25,0.5,1.0 sudoku_hard.npz
  python rollout.py sudoku fresh 3000 16 16 0,0.25,0.5,1.0 sudoku_hard_fresh.npz
  ```
- **Real Maps:**
  ```bash
  python rollout.py street main 3000 16 16 0,0.25,0.5,1.0 street_main.npz
  python rollout.py street fresh 6000 16 16 0,0.25,0.5,1.0 street_fresh.npz
  python rollout.py street dao 2000 16 16 0,0.25,0.5,1.0 dao_main.npz
  ```
  *(Or run `bash run_container_rollouts.sh`)*

### 5. Experiments & Evaluation

- **Population Selective Risk Evaluation:**
  ```bash
  python experiments_v3.py maze
  python experiments_v3.py sudoku
  ```
- **Component Decomposition:**
  ```bash
  python decomp.py maze 300
  python decomp.py sudoku 300
  python decomp.py street 300
  ```
  *(Or run `bash run_decomp.sh`)*
- **Seed Robustness Study:**
  ```bash
  python seeds.py 300
  ```
- **Real Maps Evaluation & Detour Analysis:**
  ```bash
  python exp_real.py 300
  python p4_detour.py
  ```

### 6. Results & Visualizations

- Generate calibration split indices:
  ```bash
  python make_splits.py
  ```
- Generate summary tables:
  ```bash
  python make_tables.py
  ```
- Export figure panels and legend strips:
  ```bash
  python export_figs_v8.py
  python make_legends.py
  ```
