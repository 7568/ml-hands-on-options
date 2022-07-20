# -*- coding: UTF-8 -*-
"""
Created by louis at 2022/6/28
Description:
"""

from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from hedging_options.library import common as cm

# 查看一天中最多有多少个期权数据
# min : 149 , max : 332 , mean : 225
# 我们在设计输入的时候就按照每天225个期权数据数据，大于225的就随机挑选，小于225的就用0填充

# 总共的天数为：599,其中训练集的天数为：479，验证集的天数为：60，测试集的天数为：60。


DATA_HOME_PATH = '/home/liyu/data/hedging-option/china-market'
if __name__ == '__main__':
    df = pd.read_csv(f'{DATA_HOME_PATH}/h_sh_300/two_day_all_clean_data.csv')
    # df = pd.read_csv(f'{DATA_HOME_PATH}/h_sh_300/training.csv')
    # df = pd.read_csv(f'{DATA_HOME_PATH}/h_sh_300/validation.csv')
    # df = pd.read_csv(f'{DATA_HOME_PATH}/h_sh_300/testing.csv')
    #'2021-06-21_datas.parquet'
    days = df.sort_values(by=['TradingDate'])['TradingDate'].unique()
    print(len(days))
    all_num = []
    for day in tqdm(days, total=len(days)):
        if day=='2021-06-21':
            print(df[df['TradingDate'] == day].shape[0])
        num = df[df['TradingDate'] == day].shape[0]
        all_num.append(num)
    all_num = np.array(all_num)
    print(f'min : {all_num.min()} , max : {all_num.max()} , mean : {all_num.mean()}')
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)

    ax.patch.set_facecolor('white')
    ax.set_title('')
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    # ax.plot(all_num)
    ax.hist(all_num, 50, density=False, facecolor='g', alpha=0.75)
    plt.show()

