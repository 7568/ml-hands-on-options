# -*- coding: UTF-8 -*-
"""
Created by louis at 2022/6/20
Description:
"""
import numpy as np

a = [1 / (i + 1) for i in range(1, 16)]
print(len(a))

print(np.sin(np.array([1 / (i + 1) for i in range(1, 16)]) * np.pi / 2)[::-1])

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

# display grid search loss
import torch
import transformer_train_code as train

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
# train.PREPARE_HOME_PATH = f'/home/liyu/data/hedging-option/china-market/'
train.NUM_WORKERS = 0
LEARNING_RATE = 0.0002
BATCH_SIZE = 64

enc = net.Encoder(INTPUT_DIM, HID_DIM, ENC_LAYERS, ENC_HEADS, ENC_DROPOUT, DEVICE)

model = net.Seq2Seq(enc, INTPUT_DIM, DEVICE).to(DEVICE)
print(f'The model has {net.count_parameters(model):,} trainable parameters')

if os.path.exists(f'{ENC_LAYERS}-{LEARNING_RATE}-{BATCH_SIZE}-model.pt'):
    print(f'use {ENC_LAYERS}-{LEARNING_RATE}-{BATCH_SIZE}-model.pt parameters')
    model.load_state_dict(torch.load(f'{ENC_LAYERS}-{LEARNING_RATE}-{BATCH_SIZE}-model.pt'))
else:
    print(f'use transformer_net.initialize_weights parameters')
    model.apply(net.initialize_weights)

torch.save(model.state_dict(), f'test-{ENC_LAYERS}-{LEARNING_RATE}-{BATCH_SIZE}-model.pt',
           _use_new_zipfile_serialization=False)

print('done')
