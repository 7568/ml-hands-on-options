# -*- coding: UTF-8 -*-
"""
Created by louis at 2022/6/21
Description:
"""
import os
import sys
sys.path.append(os.path.dirname("../../*"))
sys.path.append(os.path.dirname("../*"))
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd

import random

import time
from hedging_options.library import dataset
import matplotlib.pyplot as plt
import numpy as np
import transformer_net as net
# import transformer_net_v2 as net
# from tqdm import tqdm
import logging

# display grid search loss
import torch
from tqdm import tqdm
import transformer_train_code as train


def display_grid_search_loss(log_path, scenarios):
    log_files = os.listdir(f'{log_path}/')
    print(log_files)
    train_infos = {}
    for log in log_files:
        if not log.startswith('train_3-'):
            continue
        all_log = open(f'{log_path}/{log}', 'r')
        lines = all_log.readlines()
        sub_train_infos = []
        for line in lines:
            if scenarios in line:
                if scenarios == 'MSHE in test':
                    # print(line.split(f'{scenarios} :')[1])
                    put_call = line.split(f'{scenarios} :')[1].split(',')
                    _put = put_call[0].strip()[1:]
                    _call = put_call[1].strip()[:-2]
                    sub_train_infos.append([float(_put), float(_call)])
                else:
                    sub_train_infos.append(float(line.split(f'{scenarios}:')[1]))
        train_infos[log] = sub_train_infos
    fig = plt.figure(figsize=(10,10))

    fig.patch.set_facecolor('white')
    # print(train_infos)
    i = 0
    for k in train_infos.keys():
        print(k)
        ax = fig.add_subplot(1, 1, i + 1)

        ax.patch.set_facecolor('white')
        ax.set_title(k)
        ax.set_xlabel('epoch')
        ax.set_ylabel('loss')
        i += 1
        if scenarios == 'MSHE in test':
            _data = np.array(train_infos[k])
            ax.scatter(range(1, len(train_infos[k]) + 1), _data[:, 0], cmap='Greens')
            ax.scatter(range(1, len(train_infos[k]) + 1), _data[:, 1], cmap='Greens_r')
        else:
            ax.scatter(range(1, len(train_infos[k]) + 1), train_infos[k])
    plt.savefig(f'{scenarios}.png')

def play_attention():
    N_EPOCHS = 100
    CLIP = 1
    # gpu_ids = '0,1,2'
    # os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
    ENC_LAYERS = 6
    DEVICE = 'cpu'

    INTPUT_DIM = 20
    HID_DIM = 56
    ENC_HEADS = 4
    ENC_DROPOUT = 0.1

    IS_TRAIN = True

    # 3,6,9,16
    # print(f'ENC_LAYERS : {ENC_LAYERS}')
    # python transformer-code-comments.py > 0.0005-log &
    train.PREPARE_HOME_PATH = f'/home/liyu/data/hedging-option/china-market/'
    # PREPARE_HOME_PATH = f'/home/zhanghu/liyu/data/'
    train.NUM_WORKERS = 0
    LEARNING_RATE = 0.0002
    BATCH_SIZE = 64

    logger = logging.getLogger()
    logger.setLevel(level=logging.DEBUG)

    train_dataloader, val_dataloader, test_dataloader, training_dataset_length, valid_dataset_length, \
    test_dataset_length = train.load_data(BATCH_SIZE)

    handler = logging.FileHandler(f'attention_{ENC_LAYERS}-{LEARNING_RATE}-{BATCH_SIZE}.log')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    logger.debug(f'{ENC_LAYERS}-{LEARNING_RATE}-{BATCH_SIZE}')
    # Create data loaders.
    # writer = SummaryWriter()
    enc = net.Encoder(INTPUT_DIM, HID_DIM, ENC_LAYERS, ENC_HEADS, ENC_DROPOUT, DEVICE)

    model = net.Seq2Seq(enc, INTPUT_DIM, DEVICE).to(DEVICE)
    logger.debug(f'The model has {net.count_parameters(model):,} trainable parameters')

    if os.path.exists(f'test-{ENC_LAYERS}-{LEARNING_RATE}-{BATCH_SIZE}-model.pt'):
        logger.debug(f'use {ENC_LAYERS}-{LEARNING_RATE}-{BATCH_SIZE}-model.pt parameters')
        model.load_state_dict(torch.load(f'test-{ENC_LAYERS}-{LEARNING_RATE}-{BATCH_SIZE}-model.pt'))
    else:
        logger.debug(f'use transformer_net.initialize_weights parameters')
        model.apply(net.initialize_weights)
    # model = torch.nn.DataParallel(model)

    model.eval()

    # epoch_loss = 0
    all_loss = []
    with torch.no_grad():
        for ii, (datas, results) in tqdm(enumerate(val_dataloader), total=valid_dataset_length / BATCH_SIZE):
            # for ii, (datas, results) in enumerate(val_dataloader):
            datas = datas.float().to(DEVICE)
            results = results.float().to(DEVICE)

            output = model(datas, results)


    return np.array(all_loss).mean()

if __name__ == '__main__':
    LOG_PATH = '/home/zhanghu/liyu/git/ml-hands-on-options/hedging_options/use_transformer/'
    display_grid_search_loss(LOG_PATH, 'Train Loss')
    display_grid_search_loss(LOG_PATH, 'Validate Loss')
    display_grid_search_loss(LOG_PATH, 'MSHE in test')
    # DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # play_attention()
