import argparse
import json


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def build_default_argparser_pretrain() -> argparse.ArgumentParser:
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
    # parser.add_argument('--canvas_size',
    #                     help='maximum number of atoms that can be placed on the canvas',
    #                     type=int,
    #                     default=25)
    # parser.add_argument('--symbols',
    #                     help='chemical symbols available on canvas and in bag (comma separated)',
    #                     type=str,
    #                     default='X,H,C,N,O,F')

    # Environment
    parser.add_argument('--formulas',
                        help='list of formulas for environment (comma separated)',
                        type=str,
                        required=False)
    # parser.add_argument('--eval_formulas',
    #                     help='list of formulas for environment (comma separated) used for evaluation',
    #                     type=str,
    #                     required=False)
    parser.add_argument('--min_atomic_distance', help='minimum allowed atomic distance', type=float, default=0.6)
    parser.add_argument('--max_solo_distance',
                        help='maximum distance hydrogen or halogens can be away from the nearest heavy atom',
                        type=float,
                        default=2.0)
    parser.add_argument('--min_reward', help='minimum reward given by environment', type=float, default=-3)
    parser.add_argument('--calc_rew', help='calc reward during rollouts', type=str2bool, nargs='?', const=True, default=True)

    parser.add_argument('--partial_canvas', help='partial canvas', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--buffer_capacity', help='buffer size for canvas generator', type=int, default=5000)


    # Model
    parser.add_argument('--model', type=str, default='painn', choices=['internal', 'covariant', 'painn'])
    parser.add_argument('--min_mean_distance', help='minimum mean distance', type=float, default=1.0)
    parser.add_argument('--max_mean_distance', help='maximum mean distance', type=float, default=1.6)
    parser.add_argument('--network_width', help='width of FC layers', type=int, default=128)
    parser.add_argument('--num_interactions', help='number of interaction layers in painn', type=int, default=3)
    parser.add_argument('--cutoff', help='cutoff distance for graph connectivity in painn', type=float, default=4.)

    #parser.add_argument('--load_latest', help='load latest checkpoint file', action='store_true', default=False)
    parser.add_argument('--load_latest', help='load latest checkpoint file', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--load_model', help='load checkpoint file', type=str, default=None)
    parser.add_argument('--save_freq', help='save model every <n> iterations', type=int, default=10)
    parser.add_argument('--eval_freq', help='evaluate model every <n> iterations', type=int, default=10)
    parser.add_argument('--eval_freq_fast', help='evaluate model every <n> iterations', type=int, default=5)
    parser.add_argument('--checkpoints', help='when to save unique model object', 
                        type=lambda s: [int(item) for item in s.split(',')], default=[0,100,1000])
    # parser.add_argument('--num_eval_episodes', help='number of episodes per evaluation', type=int, default=None)

    # Training algorithm
    parser.add_argument('--optimizer',
                        help='Optimizer for parameter optimization',
                        type=str,
                        default='adam',
                        choices=['adam', 'amsgrad'])
    # parser.add_argument('--num_steps_per_iter',
    #                     help='number of optimization steps per iteration',
    #                     type=int,
    #                     default=128)
    parser.add_argument('--mini_batch_size', help='mini batch size for training', type=int, default=128)
    parser.add_argument('--num_envs', help='number of environment copies', type=int, default=8)
    # parser.add_argument('--clip_ratio', help='PPO clip ratio', type=float, default=0.2)
    parser.add_argument('--learning_rate', help='Learning rate of Adam optimizer', type=float, default=3e-4)
    parser.add_argument('--vf_coef', help='Coefficient for value function loss', type=float, default=0.5)
    parser.add_argument('--entropy_coef', help='Coefficient for entropy loss', type=float, default=0.15)
    # parser.add_argument('--max_num_train_iters', help='Maximum number of training iterations', type=int, default=7)
    parser.add_argument('--gradient_clip', help='maximum norm of gradients', type=float, default=0.5)
    # parser.add_argument('--lam', help='Lambda for GAE-Lambda', type=float, default=0.97)
    # parser.add_argument('--target_kl',
    #                     help='KL divergence between new and old policies after an update for early stopping',
    #                     type=float,
    #                     default=0.01)

    # Logging
    parser.add_argument('--log_level', help='log level', type=str, default='INFO')
    # parser.add_argument('--keep_models', help='keep all models', action='store_true', default=False)
    parser.add_argument('--keep_models', help='keep all models', type=str2bool, nargs='?', const=True, default=False)
    # parser.add_argument('--save_rollouts',
    #                     help='which rollouts to save',
    #                     type=str,
    #                     default='none',
    #                     choices=['none', 'train', 'eval', 'all'])

    parser.add_argument('--build_mode', help='fix TM, fix heaviest or nothing', type=str, default='none',
                        choices=['none', 'fix_TM', 'fix_heavy'])
    parser.add_argument('--beta_MARWIL', help='beta for MARWIL', type=float, default=0.5)

    # WandB
    parser.add_argument("--save_to_wandb", help="Log experimont in Weights & Biases", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--wandb_project', help='WandB project', type=str, default='')
    parser.add_argument('--wandb_group', help='WandB group', type=str, default='')
    parser.add_argument('--wandb_job_type', help='WandB job type', type=str, default='')
    parser.add_argument('--wandb_name', help='WandB name', type=str, default='')
    parser.add_argument("--wandb_watch_model", help="Log full model with parameters", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--wandb_mode', help='Logging mode', type=str, default='online')

    # Molecule decomposition variables
    parser.add_argument('--decom_method', help='Search algo to break down molecule', type=str, default='bfs', choices=['bfs', 'dfs'])
    parser.add_argument('--decom_cutoff', help='cutoff in BFS/DFS', type=float, default=1.5)
    parser.add_argument("--decom_shuffle", help="shuffle 1", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("--decom_mega_shuffle", help="shuffle 2", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--dataloader_bs', help='effective bs is larger (dataloader_bs*atoms_per_mol)', type=int, default=8)
    parser.add_argument('--dataloader_num_workers', help='for pytorch.DataLoader', type=int, default=0)
    parser.add_argument('--decom_p_random', help='probability of random decomposition', type=float, default=0.5)
    parser.add_argument('--n_atoms_to_place', help='for PartialCanvasEnv', type=int, default=1)



    parser.add_argument('--num_epochs', help='each epoch is a new decomposition of the dataset', type=int, default=50)
    parser.add_argument('--rl_algo_pretrain', help='Pretraining RL algo', type=str, default='MARWIL',
                        choices=['PPO', 'MARWIL'])
    parser.add_argument('--pretrain_every_k_iter', help='Pretrain every k iterations', type=int, default=None)
    parser.add_argument('--grad_steps_offline', help='', type=int, default=1)
    parser.add_argument('--rl_algo_online', help='RL algo for online rollouts during PRETRAINNIG phase', type=str, default='PPO',
                        choices=['PPO', 'MARWIL'])
    

    parser.add_argument('--train_mode',
                        type=str,
                        default='pretrain',
                        choices=['finetuning', 'pretrain', 'tabula_rasa', 'combined'])
    parser.add_argument('--config_ft', help='config for finetuning jobs', type=json.loads, default=None)
    parser.add_argument("--launch_finetune", help="", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--expert_suffix', help='File extension for already preprocessed expert data', type=str, default='pkl')
    parser.add_argument('--mol_dataset', help='Which dataset to use', type=str, default='QM7', choices=['TMQM', 'QM7'])
    parser.add_argument('--split_method', help='How to split reference data into train and test', type=str, default='read_split_from_disk')
    parser.add_argument("--launch_eval_new_job", help="Spawn new job for eval_zero_shot", type=str2bool, nargs='?', const=True, default=False)


    parser.add_argument("--hydrogen_delay", help="", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("--no_hydrogen_focus", help="", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--relax_steps_final', help='Relaxation steps at episode end', type=int, default=0)


    # Painn agant
    parser.add_argument("--rms_norm_update", help="", type=str2bool, nargs='?', const=True, default=False)

    # MultiModal Agent
    parser.add_argument("--bag_in_mpnn", help="", type=str2bool, nargs='?', const=True, default=False)

    # Gated Equivariant Block
    parser.add_argument("--use_GEB", help="", type=str2bool, nargs='?', const=True, default=True)


    parser.add_argument("--multi_modal", help="", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("--num_gaussians_direction", help="", type=int, default=4)
    parser.add_argument("--gmm_coef_direction_type", help="", type=str, default='length', 
                        choices=['length', 'neural_net_s', 'neural_net_v'])
    parser.add_argument("--use_GMM", help="", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("--chunk_s", help="", type=str2bool, nargs='?', const=True, default=True)


    # Dim reductions
    parser.add_argument("--reduced_v_dim", help="", type=int, default=4)
    parser.add_argument("--reduced_s_dim", help="", type=int, default=None)


    # Cormorant
    parser.add_argument('--maxl', help='maximum L in spherical harmonics expansion', type=int, default=4)
    parser.add_argument('--num_cg_levels', help='number of CG layers', type=int, default=3)
    parser.add_argument('--num_channels_hidden', help='number of channels in hidden layers', type=int, default=10)
    parser.add_argument('--num_channels_per_element', help='number of channels per element', type=int, default=4)
    parser.add_argument('--num_gaussians', help='number of Gaussians in GMM', type=int, default=3)
    parser.add_argument('--beta', help='set beta parameter of spherical distribution', required=False, default=None)
    parser.add_argument('--bag_scale', help='maximum bag size', type=int, required=False)




    return parser
