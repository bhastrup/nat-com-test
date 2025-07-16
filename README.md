# Rediscovering Chemical Space from First Principles with Reinforcement Learning

This repository contains the training and analysis code used in the study:

**Rediscovering Chemical Space from First Principles with Reinforcement Learning**  
*Bjarke Hastrup, François Cornet, Tejs Vegge, and Arghya Bhowmik*  
Under review at *Nature Communications*.  
Preprint (Version 1) available on [Research Square](https://doi.org/10.21203/rs.3.rs-6900238/v1).



<img src="resources/image_grid.png" width="100%">



## Installation Guide

### 1. Install Conda (if not already installed)
Make sure you have [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/) installed on your system.


### 2. Clone the Repository
```bash
git clone https://github.com/bhastrup/isomer-discovery-rl.git
cd isomer-discovery-rl
```


### 3. Install Mamba (if not already installed)
```bash
conda install -n base -c conda-forge mamba
```

### 4. Create the Environment using Mamba
Run the following command to create the environment from `env.yaml`:
```bash
mamba env create -f env.yaml
```

### 4. Activate the Environment
Once the installation is complete, activate the environment:
```bash
eval "$(mamba shell hook --shell bash)"
mamaba activate rl-env
```

### 5. Verify Installation
Check that the required packages are installed:

```bash
python -c 'import torch, ase, rdkit, streamlit, pandas; print("✅ All core packages are working!")'
```

### 6. Set Up PYTHONPATH (One-Time)

To ensure Python can locate the `src/` package, set the `PYTHONPATH` automatically when activating the environment: Run the following **from the root of the project directory** (where `src/` is located):

```bash
# Set PYTHONPATH when the Conda environment is activated
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo "export PYTHONPATH=\$(pwd)" > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

# Unset it on deactivate
mkdir -p $CONDA_PREFIX/etc/conda/deactivate.d
echo "unset PYTHONPATH" > $CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh
```
and reactivate the *rl-env*.



## Dataset preparation
For QM7 training, run the following commands (takes 1-2 minutes):
``` bash
python scripts/prep/preprocess_data.py
python scripts/prep/split_train_test.py
```
This saves a dataset of QM7 structures into the *data/* folder (along with XTB energies, SMILES representations and additional meta-data).
QM9 is available also - see preprocess_data.py for arguments.


## RL training
### Weights & Biases (WandB) Setup

This project uses [Weights & Biases](https://wandb.ai/) for experiment tracking.

#### 🔐 Login
Before running training or analysis scripts that use WandB, you need to log in from your terminal:
```bash
wandb login
```
This will prompt you to paste an API key from your WandB account settings. If you're running on a remote machine (e.g., cluster), you can use:
```bash
wandb login YOUR_API_KEY
```

### Launch training jobs
Training scripts are available in the *scripts/train/* folder. Specifically, a training job can be launched as
``` bash
python scripts/train/experiments/nat-com-version/a.py
```
This files launches a training of "Agent A" from the paper. The script contains further instructions for how to setup the other agent trainings.



## Analyse trainings (isomer discovery campaings):
Analysis scripts are available in the *scripts/analyse/* folder, which is further subdivided into
* exp1: Containing single bag evaluation (Q1 in the paper).
* exp2: Multibag evaluations on hold-out formulas (Q2 in the paper).
* exp3: Cumulative discovery analysis where the entire training is seen as a discovery campaign (Q3 in the paper).


# Web app
To interact with the trained agents, we provide a Streamlit-based web app that can be launched using
``` bash
streamlit run app_store/About.py
```
<img src="resources/web-app.png" width="100%">

In the sidebar on the left-hand side, navigate to the **Generator** page. This page is split into three columns, *Agents* (model checkpoints), *Environments* (chemical compositions) and *PlayGrounds* which is the inference module that takes an (agent, environment)-pair as input and samples new molecules according to the agent policy.
### How to use:
* Click *New Playground 💫* button in . This creates a new playground named *"Playground 0"* that opens in *edit* mode. Before we can **"Deploy"** this playground and use **Generator 🚀** functionality, we must provide it with agent and env objects.
* *Agents*: Loaded agents are displayed in the **All agents** expander (left column). We have pre-loaded 5 agent checkpoints from the paper, namely agents *A, AV, F, FV and AFV*. These are named according to the abbreviations of the three core reward terms they are trained on. These are *A: Atomization energy, F: Formation energy, V: Validity*. See paper for explanations.
* *Environments*: Loaded envs are displayed in the **All envs** expander (right column).



