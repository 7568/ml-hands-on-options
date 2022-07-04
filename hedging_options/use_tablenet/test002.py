# -*- coding: UTF-8 -*-
"""
Created by louis at 2022/6/30
Description:
"""
from torch.utils.data import DataLoader

from hedging_options.use_tablenet.pytorch_tabnet.tab_model import TabNetRegressor

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from hedging_options.use_tablenet.pytorch_tabnet import utils

import pandas as pd
import numpy as np

np.random.seed(0)

import os

from pathlib import Path

# define your embedding sizes : here just a random choice
cat_emb_dim = [5, 4, 3, 6, 2, 2, 1, 10]

clf = TabNetRegressor(device_name='cpu', input_dim=[100, 20])

max_epochs = 100 if not os.getenv("CI", False) else 2
from hedging_options.use_tablenet.pytorch_tabnet.augmentations import RegressionSMOTE

aug = RegressionSMOTE(p=0.2)
aug = None
PARQUET_HOME_PATH = f'/home/liyu/data/hedging-option/china-market/'
train_params = {
    'data_path': f'{PARQUET_HOME_PATH}',
    'one_day_data_numbers': 100,
    'target_feature': 'ActualDelta'
}
validate_params = {
    'data_path': f'{PARQUET_HOME_PATH}',
    'one_day_data_numbers': 100,
    'target_feature': 'ActualDelta'
}
batch_size = 2
sampler = None
need_shuffle = False
num_workers = 0
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
    batch_size=batch_size,
    sampler=sampler,
    shuffle=need_shuffle,
    num_workers=num_workers,
    drop_last=drop_last,
    pin_memory=pin_memory,
)

# for i in range(10):
#     clf._train_epoch(train_dataloader)
print(clf)
clf.fit(
    train_dataloader=TRAIN_DATALOADER,
    validate_dataloader=VALIDATE_DATALOADER,
    eval_name=['train', 'valid'],
    eval_metric=['rmsle', 'mae', 'rmse', 'mse'],
    max_epochs=max_epochs,
    patience=50,
    batch_size=1024, virtual_batch_size=128,
    num_workers=0,
    drop_last=False,
    augmentations=aug  # aug
)
