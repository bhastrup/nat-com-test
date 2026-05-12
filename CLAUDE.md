# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the codebase for "Rediscovering Chemical Space from First Principles with Reinforcement Learning" (Nature Communications). It implements a reinforcement learning system that trains agents to construct molecules atom-by-atom, guided by quantum chemistry rewards (via xTB).

## Environment Setup

```bash
# Create and activate the conda environment
mamba env create -f env.yaml
mamba activate atomcomposer

# Set PYTHONPATH (one-time, from project root)
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo "export PYTHONPATH=$(pwd)" > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
mkdir -p $CONDA_PREFIX/etc/conda/deactivate.d
echo "unset PYTHONPATH" > $CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh
```

Python 3.11, PyTorch 2.1.1, CUDA 12.1. Key dependencies: `ase`, `xtb-python`, `rdkit`, `gymnasium`, `wandb`, `submitit`, `streamlit`.

## Key Commands

```bash
# Dataset preparation (run once, saves to data/)
python scripts/prep/preprocess_data.py
python scripts/prep/split_train_test.py

# Launch RL training (Agent A from paper)
python scripts/train/experiments/nat-com-version/a.py

# Launch web app for interactive inference
streamlit run app_store/About.py

# Verify install
python -c 'import torch, ase, rdkit, streamlit, pandas; print("All core packages working")'
```

Training uses `wandb` for logging — run `wandb login` before training.

## Architecture

### Core Training Loop (`src/rl/trainer.py`)
The `Trainer` class orchestrates all training. It supports:
- **Online RL** (PPO via environment rollouts)
- **Offline pretraining** (BC or MARWIL from dataset)
- Both simultaneously (offline BC/MARWIL + online PPO fine-tuning)

Entry point: `scripts/train/run_bc.py::pretrain()`, which is submitted via `scripts/train/experiments/nat-com-version/a.py` using `submitit` (supports both local and SLURM execution).

### Environments (`src/rl/envs/`)
- `AbstractMolecularEnvironment` — base Gymnasium env for sequential atom placement
- `HeavyFirst` — main environment: places heaviest atom first, then builds molecule
- `MolecularEnvironment`, `tmqmEnv`, `ConstrainedMolecularEnvironment` — variants
- **State**: `(canvas, bag)` — canvas = placed atoms with 3D positions, bag = remaining formula
- **Action**: `(element_index, (x, y, z))` — choose element and 3D position
- **Validity constraints**: minimum atomic distances, "all covered" check (H/F/Cl/Br must be near a heavy atom)

### Reward System (`src/rl/reward.py`)
`InteractionReward` computes physics-based rewards using xTB and RDKit:
- `rew_atomisation` (A): atomization energy via xTB
- `rew_formation` (F): per-step energy gain
- `rew_valid` (V): RDKit-based molecular validity
- `rew_basin`, `rew_rae`, `rew_dipole`, ring penalties — additional terms
Reward coefficients define agent variants: **A**, **F**, **AV**, **FV**, **AFV** (as in paper).

### Agent Policy (`src/agents/`)
- `AbstractActorCritic` (`base.py`): abstract base with `step(observations, actions)` returning `{logp, v, ent}`
- `PainnAC` (`painn/agent.py`): primary agent using **PaiNN** (Polarizable Atom Interaction Neural Network) — an equivariant GNN for 3D molecular graphs
- `src/agents/covariant/` and `src/agents/internal/` — alternative agent architectures

### RL Algorithms (`src/rl/rl_algos.py`, `src/rl/losses.py`)
- `PPO` — online algorithm (clipped surrogate objective, KL early stopping)
- `BC` — behavior cloning (offline)
- `MARWIL` — advantage-reweighted imitation learning (offline)
- `PolicyOptimizer` is the abstract base; `create_policy_optimizer()` is the factory

### Spaces (`src/rl/spaces.py`)
- `ObservationSpace`: wraps `CanvasSpace` (placed atoms) + `BagSpace` (remaining formula)
- `ActionSpace = CanvasItemSpace`: `(element_index, 3D_position)`
- `FormulaType = Tuple[Tuple[int, int], ...]` — e.g., `((6, 3), (1, 8), (8, 1))` for C3H8O

### Environment Management (`src/tools/env_util.py`)
`EnvMaker` sets up vectorized environments from the molecule dataset:
- QM7: `zs = [0, 1, 6, 7, 8, 16]`, canvas size 23
- QM9: `zs = [0, 1, 6, 7, 8, 9]`, canvas size 29
- Handles train/eval splits and reference data loading

### Performance & Analysis (`src/performance/`)
- `energetics.py`: xTB-based energy calculation and geometry optimization
- `metrics.py`: `MoleculeAnalyzer` — RDKit-based validity, SMILES, rings
- `cumulative/`: `CumulativeDiscoveryTracker` — WandB logging across training
- `single_cpkt/`: `SingleCheckpointEvaluator` — hold-out formula evaluation

### Analysis Scripts (`scripts/analyse/`)
- `exp1_SB/`: Single-bag evaluation (Q1 in paper)
- `exp2_MB/`: Multi-bag evaluation on hold-out formulas (Q2)
- `exp3_CUM/`: Cumulative discovery campaign (Q3)
- `exp4_CUM_aggr/`, `exp5_scaffold/`: Additional analyses

### Data (`src/data/`, `data/`)
- `ReferenceDataLoader`: loads QM7/QM9 preprocessed datasets from `data/`
- Raw data must be generated first via `scripts/prep/` scripts
- `data/` directory is populated by preprocessing; model checkpoints in `model_objects/` and `runs/`
