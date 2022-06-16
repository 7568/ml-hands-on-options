# -*- coding: UTF-8 -*-
"""
Created by louis at 2022/5/31
Description:
"""
import os
import sys

# Append the library path to PYTHONPATH, so library can be imported.
sys.path.append(os.path.dirname(os.getcwd()))
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd

import random
from hedging_options.library import dataset

from tqdm import tqdm

SEED = 1234

random.seed(SEED)
np.random.seed(SEED)


# /home/liyu/data/hedging-option/china-market/parquet

# 将训练数据存放成 parquet 格式，方便训练的时候获取

def prepare_dataset(tag):
    save_path = f'/home/liyu/data/hedging-option/china-market/parquet/{tag}'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    data_index = pd.read_csv(f'/home/liyu/data/hedging-option/china-market/h_sh_300_{tag}_index.csv')
    train_data = pd.read_csv('/home/liyu/data/hedging-option/china-market/h_sh_300.csv', parse_dates=[
        'TradingDate', 'ExerciseDate'])

    _dataset = dataset.PrepareChineDataSet(train_data, data_index, 15, save_path)

    # Create data loaders.
    dataloader = DataLoader(_dataset, num_workers=50, batch_size=1)
    for ii, (_, __) in tqdm(enumerate(dataloader), total=data_index.shape[0]):
        continue


prepare_dataset('training')
prepare_dataset('validation')
prepare_dataset('test')
