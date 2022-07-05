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

from hedging_options.use_tablenet.pytorch_tabnet.tab_model import TabNetRegressor

from hedging_options.use_tablenet.pytorch_tabnet import utils

import torch.multiprocessing
from hedging_options.use_tablenet.pytorch_tabnet.logger import logger

torch.multiprocessing.set_sharing_strategy('file_system')
import argparse


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--normal_type', default='mean_norm', type=str, help='mean_norm,min_max_norm')
    parser.add_argument('--n_steps', default=3, type=int)
    parser.add_argument('--cuda_id', default=0, type=int)
    parser.add_argument('--input_dim', default=[35, 100], type=list)
    parser.add_argument('--n_a', default=[35, 100], type=list)
    parser.add_argument('--n_d', default=[35, 100], type=list)
    parser.add_argument('--max_epochs', default=100, type=int)
    parser.add_argument('--output_dim', default=1, type=int)
    parser.add_argument('--one_day_data_numbers', default=100, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--num_workers', default=10, type=int)

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

    sys.stderr = open(f'{args.normal_type}/log/test002_N_STEPS_{args.n_steps}.log', 'a')

    logger.set_logger_param(args.normal_type, args.n_steps)

    from hedging_options.use_tablenet.pytorch_tabnet.augmentations import RegressionSMOTE

    clf = TabNetRegressor(device_name=f'cuda:{args.cuda_id}', n_steps=args.n_steps, input_dim=args.input_dim,
                          output_dim=args.output_dim, n_a=args.n_a, n_d=args.n_d)

    aug = RegressionSMOTE(p=0.2)
    aug = None
    PARQUET_HOME_PATH = f'/home/liyu/data/hedging-option/china-market/h_sh_300/panel_parquet/{args.normal_type}/'
    train_params = {
        'data_path': f'{PARQUET_HOME_PATH}/training/',
        'one_day_data_numbers': args.one_day_data_numbers,
        'target_feature': 'ActualDelta'
    }
    validate_params = {
        'data_path': f'{PARQUET_HOME_PATH}/validation/',
        'one_day_data_numbers': args.one_day_data_numbers,
        'target_feature': 'ActualDelta'
    }

    TRAIN_DATALOADER = DataLoader(
        utils.OptionPriceDataset(**train_params),
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    VALIDATE_DATALOADER = DataLoader(
        utils.OptionPriceDataset(**validate_params),
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # for i in range(10):
    #     clf._train_epoch(train_dataloader)
    # print(clf)
    clf.fit(
        train_dataloader=TRAIN_DATALOADER,
        validate_dataloader=VALIDATE_DATALOADER,
        eval_name=['train', 'valid'],
        eval_metric=['rmsle', 'mae', 'rmse', 'mse'],
        max_epochs=args.max_epochs,
        patience=50,
        batch_size=args.batch_size, virtual_batch_size=128,
        num_workers=0,
        drop_last=False,
        augmentations=None  # aug
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser('TabNet training and evaluation script', parents=[get_args_parser()])
    ARGS = parser.parse_args()
    # if args.output_dir:
    #     Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(ARGS)
