# -*- coding: UTF-8 -*-
"""
Created by louis at 2022/6/21
Description:
"""
import shutil
import os
import matplotlib.pyplot as plt
import numpy as np

# display grid search loss
def display_grid_search_loss(log_path, scenarios):
    log_files = os.listdir(f'{log_path}/')
    print(log_files)
    train_infos = {}
    for log in log_files:
        if not log.startswith('grid_search_6_'):
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
    fig = plt.figure(figsize=(40, 40))
    # print(train_infos)
    i = 0
    for k in train_infos.keys():
        print(k)
        ax = fig.add_subplot(4, 5, i + 1)
        ax.set_title(k)
        i += 1
        if scenarios == 'MSHE in test':
            _data = np.array(train_infos[k])
            ax.scatter(range(1, len(train_infos[k]) + 1), _data[:, 0], cmap='Greens')
            ax.scatter(range(1, len(train_infos[k]) + 1), _data[:, 1], cmap='Greens_r')
        else:
            ax.scatter(range(1, len(train_infos[k]) + 1), train_infos[k])
    plt.savefig(f'{scenarios}.png')


if __name__ == '__main__':
    LOG_PATH = '/home/zhanghu/liyu/git/ml-hands-on-options/hedging_options/use_transformer/log/'
    display_grid_search_loss(LOG_PATH, 'Train Loss')
    display_grid_search_loss(LOG_PATH, 'Validate Loss')
    display_grid_search_loss(LOG_PATH, 'MSHE in test')
