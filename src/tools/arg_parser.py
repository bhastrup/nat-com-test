import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def build_default_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Command line tool of MolGym')

    # Name and seed
    parser.add_argument('--name', help='experiment name', type=str, default='devs')
    parser.add_argument('--seed', help='run ID', type=int, default=0)

    # Directories
    parser.add_argument('--log_dir', help='directory for log files', type=str, default='logs')
    parser.add_argument('--model_dir', help='directory for model files', type=str, default='models')
    parser.add_argument('--data_dir', help='directory for saved rollouts', type=str, default='data')
    parser.add_argument('--results_dir', help='directory for results', type=str, default='results')

    # Device
    parser.add_argument('--device', help='select device', type=str, choices=['cpu', 'cuda'], default='cuda')

    # Spaces
    parser.add_argument('--canvas_size',
                        help='maximum number of atoms that can be placed on the canvas',
                        type=int,
                        default=25)
    parser.add_argument('--symbols',
                        help='chemical symbols available on canvas and in bag (comma separated)',
                        type=str,
                        default='X,H,C,N,O,F,S')

    # Environment
    parser.add_argument('--formulas',
                        help='list of formulas for environment (comma separated)',
                        type=str,
                        required=False,
                        default=None)
    parser.add_argument('--eval_formulas',
                        help='list of formulas for environment (comma separated) used for evaluation',
                        type=str,
                        required=False)
    # parser.add_argument('--bag_scale', help='maximum bag size', type=int, required=False)
    parser.add_argument('--min_atomic_distance', help='minimum allowed atomic distance', type=float, default=0.6)
    parser.add_argument('--max_solo_distance',
                        help='maximum distance hydrogen or halogens can be away from the nearest heavy atom',
                        type=float,
                        default=2.0)
    parser.add_argument('--min_reward', help='minimum reward given by environment', type=float, default=-1)

    # Model
    parser.add_argument('--model',
                        help='model representation',
                        type=str,
                        default='painn',
                        choices=['internal', 'covariant', 'painn', 'schnet_edge', 'painn_internal_multimodal', 'kappa_entropy', 'benja', 'simple_equiv', 'simple_equiv_multimodal', 'emma', 'reveal_nn'])
    parser.add_argument('--update_edges', help='update edges in SchnetEdgeAC', type=bool, default=False)
    parser.add_argument('--min_mean_distance', help='minimum mean distance', type=float, default=0.8)
    parser.add_argument('--max_mean_distance', help='maximum mean distance', type=float, default=1.8)
    parser.add_argument('--network_width', help='width of FC layers', type=int, default=128)
    parser.add_argument('--num_interactions', help='number of interaction layers in painn', type=int, default=3)
    parser.add_argument('--cutoff', help='cutoff distance for graph connectivity in painn', type=float, default=4.)

    #parser.add_argument('--load_latest', help='load latest checkpoint file', action='store_true', default=False)
    parser.add_argument('--load_latest', help='load latest checkpoint file', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--load_model', help='load checkpoint file', type=str, default=None)
    parser.add_argument('--save_freq', help='save model every <n> iterations', type=int, default=10)
    parser.add_argument('--eval_freq', help='evaluate model every <n> iterations', type=int, default=2000)
    parser.add_argument('--eval_freq_fast', help='evaluate model every <n> iterations', type=int, default=2000)

    # parser.add_argument('--checkpoints', help='when to save unique model object', type=tuple, default=(0, 10, 50))
    parser.add_argument('--num_eval_episodes', help='number of episodes per evaluation', type=int, default=None)

    # Training algorithm
    parser.add_argument('--optimizer',
                        help='Optimizer for parameter optimization',
                        type=str,
                        default='adam',
                        choices=['adam', 'amsgrad'])
    parser.add_argument('--discount', help='discount factor', type=float, default=1.0)
    parser.add_argument('--max_num_steps', dest='max_num_steps', help='maximum number of steps', type=int, default=100000)
    parser.add_argument('--num_steps_per_iter',
                        help='number of optimization steps per iteration',
                        type=int,
                        default=512)
    parser.add_argument('--mini_batch_size', help='mini batch size for training', type=int, default=128)
    parser.add_argument('--num_envs', help='number of environment copies', type=int, default=8)
    parser.add_argument('--clip_ratio', help='PPO clip ratio', type=float, default=0.2)
    parser.add_argument('--learning_rate', help='Learning rate of Adam optimizer', type=float, default=3e-4)
    parser.add_argument('--vf_coef', help='Coefficient for value function loss', type=float, default=0.5)
    parser.add_argument('--entropy_coef', help='Coefficient for entropy loss', type=float, default=0.15)
    parser.add_argument('--max_num_train_iters', help='Maximum number of training iterations', type=int, default=7)
    parser.add_argument('--gradient_clip', help='maximum norm of gradients', type=float, default=0.5)
    parser.add_argument('--lam', help='Lambda for GAE-Lambda', type=float, default=0.97)
    parser.add_argument('--target_kl',
                        help='KL divergence between new and old policies after an update for early stopping',
                        type=float,
                        default=0.05)

    # Logging
    parser.add_argument('--log_level', help='log level', type=str, default='DEBUG')
    # parser.add_argument('--keep_models', help='keep all models', action='store_true', default=False)
    parser.add_argument('--keep_models', help='keep all models', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--save_rollouts',
                        help='which rollouts to save',
                        type=str,
                        default='none',
                        choices=['none', 'train', 'eval', 'all'])

    parser.add_argument('--build_mode', help='fix TM, fix heaviest or nothing', type=str, default='fix_heavy',
                        choices=['none', 'fix_TM', 'fix_heavy'])
    # parser.add_argument('--beta_MARWIL', help='beta for MARWIL', type=float, default=0.5)

    parser.add_argument("--save_to_wandb", help="Log experimont in Weights & Biases", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--wandb_project', help='WandB project', type=str, default='molgym-pretrain')
    parser.add_argument('--wandb_group', help='WandB group', type=str, default='')
    parser.add_argument('--wandb_job_type', help='WandB job type', type=str, default='')
    parser.add_argument('--wandb_name', help='WandB name', type=str, default='')
    parser.add_argument("--wandb_watch_model", help="Log full model with parameters", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--wandb_mode', help='Logging mode', type=str, default='online')

    parser.add_argument('--train_mode',
                        help='how does the agent learn across bags?',
                        type=str,
                        default='singlebag',
                        choices=['singlebag', 'multibag', 'sequential_independent', 'finetuning']) # sequential_transfer

    parser.add_argument('--finetune_checkpoint_dir', help='dir to save further finetunings', type=str, default='finetunings')
    parser.add_argument('--rl_algo', help='training/finetuning rl algo', type=str, default='PPO',
                        choices=['PPO', 'MARWIL'])
    parser.add_argument('--mol_dataset', help='Which dataset to use', type=str, default='QM7', choices=['TMQM', 'QM7', 'QM7_enhanced'])

    parser.add_argument("--hydrogen_delay", help="", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("--no_hydrogen_focus", help="", type=str2bool, nargs='?', const=True, default=False)


    parser.add_argument('--calculator', help='Which XTB calculator to use', type=str, default='GFN2-xTB', choices=['GFN2-xTB', 'gfnff'])
    parser.add_argument("--eval_on_train", help="", type=str2bool, nargs='?', const=True, default=True)


    parser.add_argument('--maxl', help='maximum L in spherical harmonics expansion', type=int, default=4)
    parser.add_argument('--num_cg_levels', help='number of CG layers', type=int, default=3)
    parser.add_argument('--num_channels_hidden', help='number of channels in hidden layers', type=int, default=10)
    parser.add_argument('--num_channels_per_element', help='number of channels per element', type=int, default=4)
    parser.add_argument('--num_gaussians', help='number of Gaussians in GMM', type=int, default=3)
    parser.add_argument('--beta', help='set beta parameter of spherical distribution', required=False, default=None)
    
    return parser
