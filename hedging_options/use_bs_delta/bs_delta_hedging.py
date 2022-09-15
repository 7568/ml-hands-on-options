# -*- coding: UTF-8 -*-
"""
Created by louis at 2022/6/28
Description:
"""
import os
import sys

sys.path.append(os.path.dirname("../../*"))
sys.path.append(os.path.dirname("../*"))

# Append the library path to PYTHONPATH, so library can be imported.
# sys.path.append(os.path.dirname(os.getcwd()))

from hedging_options.no_hedging import no_hedging


def bs_delta_hedge_in_tag(normal_type, tag, clean_data):
    put_data, call_data = no_hedging.get_data(normal_type, tag, clean_data)
    no_hedging.show_hedge_result(put_data, put_data['Delta'], call_data, call_data['Delta'])


PREPARE_HOME_PATH = '/home/liyu/data/hedging-option/china-market/'
if __name__ == '__main__':
    # NORMAL_TYPE = 'min_max_norm'
    NORMAL_TYPE = 'mean_norm'
    # no_hedging.prepare_data(NORMAL_TYPE)
    CLEAN_DATA = False
    print(f'bs_delta_hedge_in_test , clean={CLEAN_DATA}')
    # bs_delta_hedge_in_training(CLEAN_DATA)
    bs_delta_hedge_in_tag(NORMAL_TYPE, 'training', CLEAN_DATA)
    bs_delta_hedge_in_tag(NORMAL_TYPE, 'validation', CLEAN_DATA)
    bs_delta_hedge_in_tag(NORMAL_TYPE, 'testing', CLEAN_DATA)
