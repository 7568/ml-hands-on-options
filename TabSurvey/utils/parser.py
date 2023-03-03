import configargparse
import yaml


def get_parser():
    # Use parser that can read YML files
    parser = configargparse.ArgumentParser(config_file_parser_class=configargparse.YAMLConfigFileParser,
                                           formatter_class=configargparse.ArgumentDefaultsHelpFormatter)
    parser.add('--log_to_file', action='store_true')
    parser.add('--log_to_file_name', type=str, default="test", help="Direction of optimization.")
    parser.add('--model_name', required=True, help="Name of the model that should be trained")
    parser.add('-config', '--config', required=False, is_config_file_arg=True, help='config file path',
               default="config/h_sh_300_options.yml")  # kddcup99 covertype california_housing adult higgs h_sh_300_options

    parser.add('--dataset', required=True, help="Name of the dataset that will be used")
    parser.add('--objective', required=True, type=str, default="regression", choices=["regression", "classification",
                                                                                      "binary","binary_f1"],
               help="Set the type of the task")

    parser.add('--use_gpu', action="store_true", help="Set to true if GPU is available")
    parser.add('--gpu_ids', type=int, action="append", help="IDs of the GPUs used when data_parallel is true")
    parser.add('--gpu_index', default=0, type=int, help="ID of the GPU used when use_gpu is true")
    parser.add('--data_parallel', action="store_true", help="Distribute the training over multiple GPUs")
    parser.add('--learning_rate', default=0.00001, type=float)

    parser.add('--optimize_hyperparameters', action="store_true",
               help="Search for the best hyperparameters")
    parser.add('--n_trials', type=int, default=100, help="Number of trials for the hyperparameter optimization")
    parser.add('--direction', type=str, default="minimize", choices=['minimize', 'maximize'],
               help="Direction of optimization.")

    parser.add('--num_splits', type=int, default=5, help="Number of splits done for cross validation")
    parser.add('--shuffle', action="store_true", help="Shuffle data during cross-validation")
    parser.add('--seed', type=int, default=123, help="Seed for KFold initialization.")

    parser.add('--scale', action="store_true", help="Normalize input data.")
    parser.add('--target_encode', action="store_true", help="Encode the targets that they start at 0. (0, 1, 2,...)")
    parser.add('--one_hot_encode', action="store_true", help="OneHotEncode the categorical features")

    parser.add('--batch_size', type=int, default=128, help="Batch size used for training")
    parser.add('--val_batch_size', type=int, default=128, help="Batch size used for training and testing")
    parser.add('--early_stopping_rounds', type=int, default=20, help="Number of rounds before early stopping applies.")
    parser.add('--epochs', type=int, default=1000, help="Max number of epochs to train.")
    parser.add('--logging_period', type=int, default=100, help="Number of iteration after which validation is printed.")

    parser.add('--num_features', type=int, required=True, help="Set the total number of features.")
    parser.add('--num_classes', type=int, default=1, help="Set the number of classes in a classification task.")
    parser.add('--cat_idx', type=int, action="append", help="Indices of the categorical features")
    parser.add('--cat_dims', type=int, action="append", help="Cardinality of the categorical features (is set "
                                                             "automatically, when the load_data function is used.")

    parser.add_argument('--use_pretrain_data', default=False, action='store_true')
    parser.add_argument('--pretrain', default=False, action='store_true')
    parser.add_argument('--pretrain_epochs', default=10, type=int)
    parser.add_argument('--pt_tasks', default=['contrastive', 'denoising'], type=str, nargs='*',
                        choices=['contrastive', 'contrastive_sim', 'denoising'])
    parser.add_argument('--pt_aug', default=['mixup', 'cutmix'], type=str, nargs='*', choices=['mixup', 'cutmix', 'gauss_noise'])
    parser.add_argument('--pt_aug_lam', default=0.1, type=float)
    parser.add_argument('--mixup_lam', default=0.3, type=float)
    parser.add_argument('--ssl_avail_y', default=0, type=int)
    parser.add_argument('--pt_projhead_style', default='diff', type=str, choices=['diff', 'same', 'nohead'])
    parser.add_argument('--nce_temp', default=0.7, type=float)
    parser.add_argument('--lam0', default=0.5, type=float)
    parser.add_argument('--lam1', default=10, type=float)
    parser.add_argument('--lam2', default=1, type=float)
    parser.add_argument('--lam3', default=10, type=float)
    parser.add_argument('--final_mlp_style', default='sep', type=str, choices=['common', 'sep'])
    # Todo: Validate the arguments

    return parser


def get_given_parameters_parser():
    parser = get_parser()

    parser.add('-best_params_file', '--best_params_file', is_config_file_arg=True, default="config/best_params.yml",
               help="Parameter file path")
    parser.add('-parameters', '--parameters', type=yaml.safe_load, help="Parameter values")

    return parser


def get_attribution_parser():
    # Use parser that can read YML files
    parser = get_parser()

    parser.add('-paramsfile', '--paramsfile', required=False, is_config_file_arg=True, help='parameter file path',
               default="config/adult_params.yml")  # kddcup99 covertype california_housing adult higgs

    parser.add('-parameters', '--parameters', type=yaml.safe_load, help='parameter values')

    parser.add('--globalbenchmark', action="store_true", help="Run a ablation global attribution benchmark.")
    parser.add('--compareshap', action="store_true", help="Compare attributions to shapley values.")
    parser.add('--strategy', type=str, help="attribution computation strategy string")
    parser.add('--numruns', type=int, help="number of repetitions to run", default=1)
    return parser
