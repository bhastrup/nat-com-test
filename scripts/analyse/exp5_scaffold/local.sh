
# Set default values for argparse options
RUN_DIR="pretrain_runs/Agent-AD/seed_0"
MODEL_NAME="pretrain_run-0_steps-31500.model"
TAG="exp5-31500"
NUM_EPISODES_CONST=500
LOG_NAME="pretrain_run-0.json"

# Pass the arguments from the bash script to the Python script
python scripts/analyse/exp5_scaffold/dipole_evaluate.py \
    --run_dir $RUN_DIR \
    --model_name $MODEL_NAME \
    --tag $TAG \
    --num_episodes_const $NUM_EPISODES_CONST \
    --log_name $LOG_NAME
