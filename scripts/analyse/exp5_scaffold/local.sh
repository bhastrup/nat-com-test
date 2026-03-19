
# Set default values for argparse options
RUN_DIR="pretrain_runs/A-30k-Fixed/seed_0"
MODEL_NAME="pretrain_run-0_steps-30000.model"
TAG="tuesday-30k-dip-30000"
NUM_EPISODES_CONST=250
LOG_NAME="pretrain_run-0.json"

# Pass the arguments from the bash script to the Python script
python scripts/analyse/exp5_scaffold/dipole_evaluate.py \
    --run_dir $RUN_DIR \
    --model_name $MODEL_NAME \
    --tag $TAG \
    --num_episodes_const $NUM_EPISODES_CONST \
    --log_name $LOG_NAME



# Visualize results (expects: <RUN_DIR>/results/<TAG>/<formula>/{df.csv,atoms.traj})
python scripts/analyse/exp5_scaffold/visuals/visualize.py \
    --run_dirs $RUN_DIR \
    --tag $TAG \
    --sorting_key dipole \
    --smiles_col SMILES \
    --show_relaxed



################################################################################
# 30500 steps

# Set default values for argparse options
RUN_DIR="pretrain_runs/A-30k-Fixed/seed_0"
MODEL_NAME="pretrain_run-0_steps-30500.model"
TAG="tuesday-30k-dip-30500"
NUM_EPISODES_CONST=250
LOG_NAME="pretrain_run-0.json"

# Pass the arguments from the bash script to the Python script
python scripts/analyse/exp5_scaffold/dipole_evaluate.py \
    --run_dir $RUN_DIR \
    --model_name $MODEL_NAME \
    --tag $TAG \
    --num_episodes_const $NUM_EPISODES_CONST \
    --log_name $LOG_NAME


# Visualize results (expects: <RUN_DIR>/results/<TAG>/<formula>/{df.csv,atoms.traj})
python scripts/analyse/exp5_scaffold/visuals/visualize.py \
    --run_dirs $RUN_DIR \
    --tag $TAG \
    --sorting_key dipole \
    --smiles_col SMILES \
    --show_relaxed


################################################################################
# 30600 steps

# Set default values for argparse options
RUN_DIR="pretrain_runs/A-30k-Fixed/seed_0"
MODEL_NAME="pretrain_run-0_steps-30600.model"
TAG="tuesday-30k-dip-30600"
NUM_EPISODES_CONST=250
LOG_NAME="pretrain_run-0.json"

# Pass the arguments from the bash script to the Python script
python scripts/analyse/exp5_scaffold/dipole_evaluate.py \
    --run_dir $RUN_DIR \
    --model_name $MODEL_NAME \
    --tag $TAG \
    --num_episodes_const $NUM_EPISODES_CONST \
    --log_name $LOG_NAME


# Visualize results (expects: <RUN_DIR>/results/<TAG>/<formula>/{df.csv,atoms.traj})
python scripts/analyse/exp5_scaffold/visuals/visualize.py \
    --run_dirs $RUN_DIR \
    --tag $TAG \
    --sorting_key dipole \
    --smiles_col SMILES \
    --show_relaxed


################################################################################
# 30700 steps

# Set default values for argparse options
RUN_DIR="pretrain_runs/A-30k-Fixed/seed_0"
MODEL_NAME="pretrain_run-0_steps-30700.model"
TAG="tuesday-30k-dip-30700"
NUM_EPISODES_CONST=250
LOG_NAME="pretrain_run-0.json"

# Pass the arguments from the bash script to the Python script
python scripts/analyse/exp5_scaffold/dipole_evaluate.py \
    --run_dir $RUN_DIR \
    --model_name $MODEL_NAME \
    --tag $TAG \
    --num_episodes_const $NUM_EPISODES_CONST \
    --log_name $LOG_NAME


# Visualize results (expects: <RUN_DIR>/results/<TAG>/<formula>/{df.csv,atoms.traj})
python scripts/analyse/exp5_scaffold/visuals/visualize.py \
    --run_dirs $RUN_DIR \
    --tag $TAG \
    --sorting_key dipole \
    --smiles_col SMILES \
    --show_relaxed

################################################################################
# 30800 steps

# Set default values for argparse options
RUN_DIR="pretrain_runs/A-30k-Fixed/seed_0"
MODEL_NAME="pretrain_run-0_steps-30800.model"
TAG="tuesday-30k-dip-30800"
NUM_EPISODES_CONST=250
LOG_NAME="pretrain_run-0.json"

# Pass the arguments from the bash script to the Python script
python scripts/analyse/exp5_scaffold/dipole_evaluate.py \
    --run_dir $RUN_DIR \
    --model_name $MODEL_NAME \
    --tag $TAG \
    --num_episodes_const $NUM_EPISODES_CONST \
    --log_name $LOG_NAME


# Visualize results (expects: <RUN_DIR>/results/<TAG>/<formula>/{df.csv,atoms.traj})
python scripts/analyse/exp5_scaffold/visuals/visualize.py \
    --run_dirs $RUN_DIR \
    --tag $TAG \
    --sorting_key dipole \
    --smiles_col SMILES \
    --show_relaxed


################################################################################
# 30900 steps

# Set default values for argparse options
RUN_DIR="pretrain_runs/A-30k-Fixed/seed_0"
MODEL_NAME="pretrain_run-0_steps-30900.model"
TAG="tuesday-30k-dip-30900"
NUM_EPISODES_CONST=250
LOG_NAME="pretrain_run-0.json"

# Pass the arguments from the bash script to the Python script
python scripts/analyse/exp5_scaffold/dipole_evaluate.py \
    --run_dir $RUN_DIR \
    --model_name $MODEL_NAME \
    --tag $TAG \
    --num_episodes_const $NUM_EPISODES_CONST \
    --log_name $LOG_NAME


# Visualize results (expects: <RUN_DIR>/results/<TAG>/<formula>/{df.csv,atoms.traj})
python scripts/analyse/exp5_scaffold/visuals/visualize.py \
    --run_dirs $RUN_DIR \
    --tag $TAG \
    --sorting_key dipole \
    --smiles_col SMILES \
    --show_relaxed

################################################################################
# 31000 steps

# Set default values for argparse options
RUN_DIR="pretrain_runs/A-30k-Fixed/seed_0"
MODEL_NAME="pretrain_run-0_steps-31000.model"
TAG="tuesday-30k-dip-31000"
NUM_EPISODES_CONST=250
LOG_NAME="pretrain_run-0.json"

# Pass the arguments from the bash script to the Python script
python scripts/analyse/exp5_scaffold/dipole_evaluate.py \
    --run_dir $RUN_DIR \
    --model_name $MODEL_NAME \
    --tag $TAG \
    --num_episodes_const $NUM_EPISODES_CONST \
    --log_name $LOG_NAME


# Visualize results (expects: <RUN_DIR>/results/<TAG>/<formula>/{df.csv,atoms.traj})
python scripts/analyse/exp5_scaffold/visuals/visualize.py \
    --run_dirs $RUN_DIR \
    --tag $TAG \
    --sorting_key dipole \
    --smiles_col SMILES \
    --show_relaxed


################################################################################
# 31100 steps

# Set default values for argparse options
RUN_DIR="pretrain_runs/A-30k-Fixed/seed_0"
MODEL_NAME="pretrain_run-0_steps-31100.model"
TAG="tuesday-30k-dip-31100"
NUM_EPISODES_CONST=250
LOG_NAME="pretrain_run-0.json"

# Pass the arguments from the bash script to the Python script
python scripts/analyse/exp5_scaffold/dipole_evaluate.py \
    --run_dir $RUN_DIR \
    --model_name $MODEL_NAME \
    --tag $TAG \
    --num_episodes_const $NUM_EPISODES_CONST \
    --log_name $LOG_NAME


# Visualize results (expects: <RUN_DIR>/results/<TAG>/<formula>/{df.csv,atoms.traj})
python scripts/analyse/exp5_scaffold/visuals/visualize.py \
    --run_dirs $RUN_DIR \
    --tag $TAG \
    --sorting_key dipole \
    --smiles_col SMILES \
    --show_relaxed


################################################################################
# 31200 steps

# Set default values for argparse options
RUN_DIR="pretrain_runs/A-30k-Fixed/seed_0"
MODEL_NAME="pretrain_run-0_steps-31200.model"
TAG="tuesday-30k-dip-31200"
NUM_EPISODES_CONST=250
LOG_NAME="pretrain_run-0.json"

# Pass the arguments from the bash script to the Python script
python scripts/analyse/exp5_scaffold/dipole_evaluate.py \
    --run_dir $RUN_DIR \
    --model_name $MODEL_NAME \
    --tag $TAG \
    --num_episodes_const $NUM_EPISODES_CONST \
    --log_name $LOG_NAME


# Visualize results (expects: <RUN_DIR>/results/<TAG>/<formula>/{df.csv,atoms.traj})
python scripts/analyse/exp5_scaffold/visuals/visualize.py \
    --run_dirs $RUN_DIR \
    --tag $TAG \
    --sorting_key dipole \
    --smiles_col SMILES \
    --show_relaxed

################################################################################
# 31300 steps

# Set default values for argparse options
RUN_DIR="pretrain_runs/A-30k-Fixed/seed_0"
MODEL_NAME="pretrain_run-0_steps-31300.model"
TAG="tuesday-30k-dip-31300"
NUM_EPISODES_CONST=250
LOG_NAME="pretrain_run-0.json"

# Pass the arguments from the bash script to the Python script
python scripts/analyse/exp5_scaffold/dipole_evaluate.py \
    --run_dir $RUN_DIR \
    --model_name $MODEL_NAME \
    --tag $TAG \
    --num_episodes_const $NUM_EPISODES_CONST \
    --log_name $LOG_NAME


# Visualize results (expects: <RUN_DIR>/results/<TAG>/<formula>/{df.csv,atoms.traj})
python scripts/analyse/exp5_scaffold/visuals/visualize.py \
    --run_dirs $RUN_DIR \
    --tag $TAG \
    --sorting_key dipole \
    --smiles_col SMILES \
    --show_relaxed


################################################################################
# 31400 steps

# Set default values for argparse options
RUN_DIR="pretrain_runs/A-30k-Fixed/seed_0"
MODEL_NAME="pretrain_run-0_steps-31400.model"
TAG="tuesday-30k-dip-31400"
NUM_EPISODES_CONST=250
LOG_NAME="pretrain_run-0.json"

# Pass the arguments from the bash script to the Python script
python scripts/analyse/exp5_scaffold/dipole_evaluate.py \
    --run_dir $RUN_DIR \
    --model_name $MODEL_NAME \
    --tag $TAG \
    --num_episodes_const $NUM_EPISODES_CONST \
    --log_name $LOG_NAME


# Visualize results (expects: <RUN_DIR>/results/<TAG>/<formula>/{df.csv,atoms.traj})
python scripts/analyse/exp5_scaffold/visuals/visualize.py \
    --run_dirs $RUN_DIR \
    --tag $TAG \
    --sorting_key dipole \
    --smiles_col SMILES \
    --show_relaxed