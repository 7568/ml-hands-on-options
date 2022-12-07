# -*- coding: UTF-8 -*-
"""
Created by louis at 2022/6/2
Description:
"""

from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import shutil
from functools import wraps

from hedging_options.library import common as cm

import prepare_real_data

prepare_real_data.DATA_HOME_PATH = '/home/liyu/data/hedging-option/latest-china-market'
OPTION_SYMBOL = 'h_sh_300'

if __name__ == '__main__':
    prepare_real_data.depart_data(1)
    prepare_real_data.DATA_HOME_PATH = prepare_real_data.DATA_HOME_PATH + "/" + OPTION_SYMBOL + "/"
    prepare_real_data.check_each_data_num_by_id(1)
    prepare_real_data.combine_all_data(1, 3)
    prepare_real_data.remove_filling_not0_data(3, 4)  # 删除原始表中节假日填充的数据
    prepare_real_data.remove_real_trade_days_less28(4, 5)  # 将合约交易天数小于28天的删除
    prepare_real_data.remove_end5_trade_date_data(5, 6)  # 将每份期权合约交易的最后5天的数据删除
    prepare_real_data.check_volume(6, 7)  # 将成交量为0的数据中存在nan的地方填充0
    prepare_real_data.check_null_by_id(7)  # 查看是否还有nan数据
    # save_by_each_option()  # 便于查看每份期权合约的每天交易信息
    prepare_real_data.hand_category_data(7, 9)
    prepare_real_data.append_before4_days_data(9, 10)  # 将前4天的数据追加到当天，不够4天的用0填充
    prepare_real_data.append_next_price(10, 11)  # 得到下一天的价格数据，包括期权的价格数据和标的资产的价格数据
    # prepare_real_data.append_real_hedging_rate(11, 12)  # 得到得到真实的对冲比例
    prepare_real_data.append_payoff_rate(11, '12_1')  # 得到期权是涨还是跌
    prepare_real_data.check_null_by_id('12_1')
    prepare_real_data.retype_cat_columns('12_1', 13)  # 将分类数据设置成int型
    # get_expand_head()  # 查看填充效果

    prepare_real_data.rename_raw_data(13)
