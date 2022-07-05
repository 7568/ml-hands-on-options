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

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from hedging_options.use_tablenet.pytorch_tabnet import utils

import pandas as pd
import numpy as np
import torch.multiprocessing
from hedging_options.use_tablenet.pytorch_tabnet.logger import logger
torch.multiprocessing.set_sharing_strategy('file_system')

if not os.path.exists('log'):
    os.makedirs('log')
if not os.path.exists('pt'):
    os.makedirs('pt')
if not os.path.exists('pid'):
    os.makedirs('pid')

N_STEPS=int(sys.argv[1])
sys.stderr = open(f'test002_N_STEPS_{N_STEPS}.log', 'a')

logger.set_logger_param(N_STEPS)
np.random.seed(0)

from hedging_options.use_tablenet.pytorch_tabnet.augmentations import RegressionSMOTE

clf = TabNetRegressor(device_name=f'cuda:{int(sys.argv[2])}',n_steps=N_STEPS, input_dim=[35, 100],output_dim=1,
                      n_a=[35,100],n_d=[35,100])

max_epochs = 100 if not os.getenv("CI", False) else 2

aug = RegressionSMOTE(p=0.2)
aug = None
PARQUET_HOME_PATH = f'/home/liyu/data/hedging-option/china-market/h_sh_300/panel_parquet/'
train_params = {
    'data_path': f'{PARQUET_HOME_PATH}/training/',
    'one_day_data_numbers': 100,
    'target_feature': 'ActualDelta'
}
validate_params = {
    'data_path': f'{PARQUET_HOME_PATH}/validation/',
    'one_day_data_numbers': 100,
    'target_feature': 'ActualDelta'
}
batch_size = 64
sampler = None
need_shuffle = False
num_workers = 10
drop_last = True
pin_memory = False

TRAIN_DATALOADER = DataLoader(
    utils.OptionPriceDataset(**train_params),
    batch_size=batch_size,
    sampler=sampler,
    shuffle=need_shuffle,
    num_workers=num_workers,
    drop_last=drop_last,
    pin_memory=pin_memory,
)

VALIDATE_DATALOADER = DataLoader(
    utils.OptionPriceDataset(**validate_params),
    batch_size=1,
    sampler=sampler,
    shuffle=need_shuffle,
    num_workers=num_workers,
    drop_last=drop_last,
    pin_memory=pin_memory,
)

# for i in range(10):
#     clf._train_epoch(train_dataloader)
# print(clf)
clf.fit(
    train_dataloader=TRAIN_DATALOADER,
    validate_dataloader=VALIDATE_DATALOADER,
    eval_name=['train', 'valid'],
    eval_metric=['rmsle', 'mae', 'rmse', 'mse'],
    max_epochs=max_epochs,
    patience=50,
    batch_size=batch_size, virtual_batch_size=128,
    num_workers=0,
    drop_last=False,
    augmentations=None  # aug
)
