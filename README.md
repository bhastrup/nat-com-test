# Isomer Discovery with Reinforcement Learning

This repository contains the training and analysis code used in the study:

**Rediscovering Chemical Space from First Principles with Reinforcement Learning**  
*Bjarke Hastrup, François Cornet, Tejs Vegge, and Arghya Bhowmik*  
Under review at *Nature Communications*.  
Preprint (Version 1) available on [Research Square](https://doi.org/10.21203/rs.3.rs-6900238/v1).



<img src="resources/image-grid.png" width="40%">
<img src="https://raw.githubusercontent.com/bhastrup/nar-com-test/main/resources/image-grid.png" width="40%">



## Installation Guide

### 1. Install Conda (if not already installed)
Make sure you have [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/) installed on your system.


### 2. Clone the Repository
```bash
git clone https://github.com/yourusername/isomer-discovery-rl.git
cd isomer-discovery-rl
```



### 3. Create the Conda Environment
Run the following command to create the environment from `env.yaml`:

```bash
conda env create -f env.yaml
```

### 4. Activate the Environment
Once the installation is complete, activate the environment:

```bash
conda activate renewables
```

### 5. Verify Installation
Check that the required packages are installed:

```bash
python -c "import torch, rdkit, ase, gym, pandas, numpy, matplotlib, scipy, streamlit; print('✅ Environment is ready!')"
```




## Dataset preparation

For QM7 training, run the following commands:
``` bash
python scripts/prep/preprocess_data.py
python scripts/prep/split_train_test.py
```
This saves a dataset of QM7 structures into the data/ folder (along with XTB energies, SMILES representations and additional meta-data).
QM9 is available also - see preprocess_data.py for arguments.


## Launch RL training

Training scripts are available in the scripts/train/ folder. Specifically, a training job can be launched as
``` bash
python scripts/train/experiments/nat-com-version/a.py
```
This files launches a training of "Agent A" from the paper. The script contains further instructions for how to setup the other agent trainings.


## Analyse trainings (isomer discovery campaings):
Analysis scripts are available in scripts/

``` bash
python solutions/part2/run.py
```
