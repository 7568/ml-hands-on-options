# -*- coding: UTF-8 -*-
"""
Created by louis at 2022/6/30
Description:
"""
import os
import sys

sys.path.append(os.path.dirname("../../*"))
sys.path.append(os.path.dirname("../*"))
from torch.utils.data import DataLoader

from hedging_options.use_crossnet.pytorch_crossnet.cross_model import CrossNetRegressor

from hedging_options.use_crossnet.pytorch_crossnet import utils

import torch.multiprocessing
from hedging_options.use_crossnet.pytorch_crossnet.logger import logger

torch.multiprocessing.set_sharing_strategy('file_system')
import argparse
import numpy as np


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--normal_type', default='mean_norm', type=str, help='mean_norm,min_max_norm')
    parser.add_argument('--device', default='cuda', type=str, help='cuda,cpu')
    parser.add_argument('--cuda_id', default=7, type=int)
    parser.add_argument('--n_steps', default=16, type=int)
    parser.add_argument('--input_dim', default=[35, 100], type=list)
    parser.add_argument('--n_a', default=[35, 100], type=list)
    parser.add_argument('--n_d', default=[35, 100], type=list)
    parser.add_argument('--max_epochs', default=200, type=int)
    parser.add_argument('--output_dim', default=1, type=int)
    parser.add_argument('--clip_value', default=2, type=int)
    parser.add_argument('--one_day_data_numbers', default=100, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--l_r', default=0.02, type=float)
    parser.add_argument('--scheduler_step_size', default=3, type=int)
    parser.add_argument('--scheduler_gamma', default=0.95, type=float)
    parser.add_argument('--num_workers', default=3, type=int)
    parser.add_argument('--drop_last', default=False, type=bool)
    parser.add_argument('--pin_memory', default=True, type=bool)
    parser.add_argument('--redirect_sys_stderr', default=False, type=bool)
    parser.add_argument('--parquet_data_path',
                        default='/home/liyu/data/hedging-option/china-market/h_sh_300/panel_parquet', type=str)
    parser.add_argument('--next_day_features', default=['S0_n', 'S1_n', 'V0_n', 'V1_n', 'On_ret'], type=list)

    return parser


def main(args):
    # utils.init_distributed_mode(args)
    # print("git:\n  {}\n".format(utils.get_sha()))

    if not os.path.exists(args.normal_type):
        os.makedirs(args.normal_type)
    if not os.path.exists(f'{args.normal_type}/log'):
        os.makedirs(f'{args.normal_type}/log')
    if not os.path.exists(f'{args.normal_type}/pt'):
        os.makedirs(f'{args.normal_type}/pt')
    if not os.path.exists(f'{args.normal_type}/pid'):
        os.makedirs(f'{args.normal_type}/pid')

    if args.redirect_sys_stderr:
        sys.stderr = open(f'{args.normal_type}/log/predict_N_STEPS_{args.n_steps}.log', 'a')
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_id)
    logger.set_logger_param(args.normal_type, args.n_steps, args.redirect_sys_stderr, 'predict')

    # from hedging_options.use_crossnet.pytorch_crossnet.augmentations import RegressionSMOTE

    clf = CrossNetRegressor(device_name=args.device, n_steps=args.n_steps, input_dim=args.input_dim,
                            output_dim=args.output_dim, n_a=args.n_a, n_d=args.n_d, lambda_sparse=1e-4, momentum=0.3,
                            clip_value=args.clip_value,
                            optimizer_fn=torch.optim.Adam,
                            optimizer_params=dict(lr=args.l_r),
                            scheduler_params={"gamma": args.scheduler_gamma,
                                            "step_size": args.scheduler_step_size},
                            scheduler_fn=torch.optim.lr_scheduler.StepLR, epsilon=1e-15,
                            next_day_features=args.next_day_features)

    PARQUET_HOME_PATH = f'{args.parquet_data_path}/{args.normal_type}/'
    validate_params = {
        'data_path': f'{PARQUET_HOME_PATH}/validation/',
        'one_day_data_numbers': args.one_day_data_numbers,
        'target_feature': 'ActualDelta',
        'batch_size': args.batch_size,
        'next_day_features': args.next_day_features
    }

    testing_params = {
        'data_path': f'{PARQUET_HOME_PATH}/testing/',
        'one_day_data_numbers': args.one_day_data_numbers,
        'target_feature': 'ActualDelta',
        'batch_size': args.batch_size,
        'next_day_features': args.next_day_features
    }

    VALIDATE_DATALOADER = DataLoader(
        utils.OptionPriceDataset(**validate_params),
        # batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=args.drop_last,
        pin_memory=args.pin_memory
    )
    TESTING_DATALOADER = DataLoader(
        utils.OptionPriceDataset(**testing_params),
        # batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=args.drop_last,
        pin_memory=args.pin_memory
    )

    clf.load_model(
        f'{args.normal_type}/pt/mean_best_model/{args.n_steps}_network.pt')

    validation_results = []
    testing_results = []
    for i in range(100):
        validation_results.append(clf.predict(VALIDATE_DATALOADER))
        testing_results.append(clf.predict(TESTING_DATALOADER))
        logger.debug(f'validation_result : {np.array(validation_results).mean(axis=0)}')
        logger.debug(f'testing_result : {np.array(testing_results).mean(axis=0)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('TabNet training and evaluation script', parents=[get_args_parser()])
    ARGS = parser.parse_args()
    # if args.output_dir:
    #     Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(ARGS)
