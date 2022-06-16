# -*- coding: UTF-8 -*-
"""
Created by louis at 2022/6/13
Description:
"""
import numpy as np

import hedging_options.use_bs_delta.get_bS_delta_hedge as get_bS_delta_hedge
from hedging_options.library import regression_aux as raux

if __name__ == '__main__':
    test_data = get_bS_delta_hedge.get_clean_data()
    kwargs = {
        # 'features': ['delta_bs'],
        'features': ['Delta'],
        'df': test_data}

    res = raux.run_store_lin2(**kwargs)
    _delta = res['delta']
    test_set = test_data.loc[res['delta'].index]

    call_test_set = test_set[test_set['cp_int'] == 0]
    _S1_n = call_test_set['S1_n']
    _S0_n = call_test_set['S0_n']
    _on_ret = call_test_set['on_ret']
    _V1_n = call_test_set['V1_n']
    _V0_n = call_test_set['V0_n']
    call_delta = call_test_set['Delta']
    _var = np.power(100 * (call_delta * _S1_n + _on_ret * (_V0_n - call_delta * _S0_n) - _V1_n) / _S1_n, 2).mean()
    print(_var)
    put_test_set = test_set[test_set['cp_int'] == 1]
    _S1_n = put_test_set['S1_n']
    _S0_n = put_test_set['S0_n']
    _on_ret = put_test_set['on_ret']
    _V1_n = put_test_set['V1_n']
    _V0_n = put_test_set['V0_n']
    put_delta = _delta.loc[put_test_set.index]
    _var = np.power(100 * (put_delta * _S1_n + _on_ret * (_V0_n - put_delta * _S0_n) - _V1_n) / _S1_n, 2).mean()
    print(_var)

    push_mshes = []
    call_mshes = []
    call_data = call_test_set.to_numpy()
    _delta = call_data[:, 8]
    # _delta = 0

    call_mshe = np.power((100 * (_delta * call_data[:, -1] + call_data[:, 12] * (
            call_data[:, 3] - _delta * call_data[:, 4]) - call_data[:, -2])) / call_data[:, -1], 2).mean()
    print(call_mshe)
    put_data = put_test_set.to_numpy()
    _delta = put_data[:, 8]
    # _delta = 0

    put_mshe = np.power((100 * (_delta * put_data[:, -1] + put_data[:, 12] * (
            put_data[:, 3] - _delta * put_data[:, 4]) - put_data[:, -2])) / put_data[:, -1], 2).mean()
    print(put_mshe)

