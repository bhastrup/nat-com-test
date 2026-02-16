
# Set default values for argparse options
RUN_DIR="runs/nat-com-training/A/seed_0"
MODEL_NAME="pretrain_run-0_CP-8_steps-20000.model"
TAG="exp5-evaluate-basic-agent"
NUM_EPISODES_CONST=500
LOG_NAME="pretrain_run-0.json"

# Pass the arguments from the bash script to the Python script
python scripts/analyse/exp5_scaffold/dipole_evaluate.py \
    --run_dir $RUN_DIR \
    --model_name $MODEL_NAME \
    --tag $TAG \
    --num_episodes_const $NUM_EPISODES_CONST \
    --log_name $LOG_NAME
