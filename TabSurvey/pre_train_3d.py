import logging
import sys

import numpy as np
import optuna

from utils import logger_conf
from models import str2model
from utils.load_data_3d import load_data
from utils.scorer import get_scorer
from utils.timer import Timer
from utils.io_utils import save_results_to_file, save_hyperparameters_to_file, save_loss_to_file
from utils.parser import get_parser, get_given_parameters_parser

from sklearn.model_selection import KFold, StratifiedKFold  # , train_test_split


def training_validation_testing(model, X, y, training_trading_dates, validation_trading_dates, testing_trading_dates,args, save_model=False):
    # Record some statistics and metrics
    sc = get_scorer(args)
    pretrain_timer = Timer()
    train_timer = Timer()
    test_timer = Timer()

    # for i, (train_index, test_index) in enumerate(kf.split(X, y)):

    X_train, X_validation, X_test = X['training'], X['validation'], X['testing']
    y_train, y_validation, y_test = y['training'], y['validation'], y['testing']

    # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.05, random_state=args.seed)

    # Create a new unfitted version of the model
    curr_model = model.clone()
    print(f'use_pretrain_data : {args.use_pretrain_data}')
    if args.use_pretrain_data:
        model.load_model(filename_extension="pretrain", directory="tmp")
    # print(f'args.use_gpu {args.use_gpu}')
    # if args.use_gpu:
    #     import torch
    #     device = torch.device(f"cuda:{args.gpu_index}" if torch.cuda.is_available() else "cpu")
    #     curr_model.device = device
    #     print(f'curr_model.device {curr_model.device}')
    # Pretrain model
    pretrain_timer.start()
    pretrain_x =  np.concatenate((X_train, X_validation),axis=0)
    pretrain_y = np.concatenate((y_train, y_validation), axis=0)
    pretrain_trading_dates = np.concatenate((training_trading_dates, validation_trading_dates), axis=0)
    curr_model.pretrain(pretrain_x, pretrain_y, pretrain_trading_dates,args.use_pretrain_data)
    pretrain_timer.end()


    # print("Finished cross validation")
    return sc, (pretrain_timer.get_average_time())



def main_once(args):
    print("Train model with given hyperparameters")
    X, y ,training_trading_dates, validation_trading_dates, testing_trading_dates= load_data(args)

    model_name = str2model(args.model_name)

    parameters = args.parameters[args.dataset][args.model_name]
    model = model_name(parameters, args)
    sc, time = training_validation_testing(model, X, y,training_trading_dates, validation_trading_dates, testing_trading_dates, args)
    print('finished training model')
    # print(sc.get_results())
    # print(time)


# python train.py --config config/h_sh_300_options.yml  --model_name LightGBM --n_trials 2 --epochs 30 --log_to_file &
if __name__ == "__main__":
    parser = get_parser()
    arguments = parser.parse_args()
    if arguments.log_to_file:
        logger_conf.init_log(f'{arguments.log_to_file_name}')
    print(arguments)
    # Also load the best parameters
    parser = get_given_parameters_parser()
    arguments = parser.parse_args()
    arguments.gpu_index = 0
    main_once(arguments)
