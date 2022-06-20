# -*- coding: UTF-8 -*-
"""
Created by louis at 2022/5/31
Description:
"""
import os

import numpy as np
import pandas as pd


class PrepareDataset:
    def __init__(self, data, data_index, start_day, save_path):
        self.all_data = data
        self.all_data_index = data_index
        self.start_day = start_day
        self.save_path = save_path

    def __getitem__(self, idx):
        _id, _days = self.all_data_index[idx]
        _id = int(_id)
        # [:, 2:]为了去掉时间和id ， 还要将最后一天的'S1_n','V1_n','implvol1'设置为0，因为是不知道的
        _data = self.all_data[(self.all_data['optionid'] == _id) & (self.all_data['days'] <= _days)
                              & ((_days - self.start_day) < self.all_data['days'])]
        _result = _data[['S0_n', 'V0_n', 'S1_n', 'V1_n', 'cp_int']].to_numpy()[-1]
        _data = _data.drop(['date', 'optionid', 'days', 'S0_n', 'V0_n', 'S1_n', 'V1_n', 'K_n'], axis=1).to_numpy()
        _data[-1, [-3, -2, -1]] = 0
        # pa_table = pa.table({"datas": _data, 'results': _result})
        _path = f'{self.save_path}/{_id}'
        if not os.path.exists(_path):
            try:
                os.makedirs(_path)
            except BaseException as err:
                print(err)
        pd.DataFrame(data=_data, columns=[str(x) for x in range(_data.shape[1])]).to_parquet(
            f'{self.save_path}/{_id}/day_{int(_days)}_datas.parquet')
        pd.DataFrame(data=_result.T, columns=['0']).to_parquet(
            f'{self.save_path}/{_id}/day_{int(_days)}_results.parquet')
        # pa.parquet.write_table(pa_table, f'{self.save_path}/{id}/day_{_days}.parquet')
        # pd.DataFrame(data={'datas': [_data], 'results': [_result]}).to_parquet(f'{self.save_path}/{id}/day'
        #                                                                     f'_{_days}.parquet')
        return _data, _result

    def __len__(self):
        return self.all_data.shape[0]


class PrepareChineDataSet:
    def __init__(self, data, data_index, start_day, save_path):
        self.all_data = data
        self.all_data_index = data_index.to_numpy()
        self.start_day = start_day
        self.save_path = save_path

    def __getitem__(self, idx):
        # print(idx)
        # _data_index = self.all_data_index[idx]
        # security_id = _data_index['SecurityID']
        # _option_list = self.all_data[self.all_data['SecurityID'] == security_id]
        # _option_list = _option_list.sort_values(by=['TradingDate'])

        _id, _days = self.all_data_index[idx]
        # _id = int(_id)
        _option_list = self.all_data[self.all_data['SecurityID'] == _id]
        _option_list = _option_list.sort_values(by=['TradingDate'])
        _option_list = _option_list.iloc[_days - 15: _days]
        _option_list['M'] = _option_list['UnderlyingScrtClose'] / _option_list['StrikePrice']
        _option_list = _option_list[
            ['RisklessRate', 'CallOrPut', 'ClosePrice', 'UnderlyingScrtClose', 'StrikePrice', 'RemainingTerm', 'Delta',
             'Gamma', 'Vega', 'Theta', 'Rho', 'M', 'ImpliedVolatility', 'Delta_1', 'Gamma_1', 'Vega_1', 'Theta_1',
             'Rho_1', 'ClosePrice_1', 'UnderlyingScrtClose_1']]
        _option_list = _option_list.replace({'CallOrPut': {'C': 0, 'P': 1}})
        _data = _option_list.to_numpy()
        # 将 UnderlyingScrtClose 全部设置为100 ，然后 UnderlyingScrtClose_1 按比例缩减
        underlying_scrt_close = _data[:, 3]
        underlying_scrt_close_rate = underlying_scrt_close / 100
        _data[:, -1] = _data[:, -1] / underlying_scrt_close_rate
        _data[:, 2] = _data[:, 2] / underlying_scrt_close_rate
        _data[:, 4] = _data[:, 4] / underlying_scrt_close_rate
        _data[:, -2] = _data[:, -2] / underlying_scrt_close_rate
        # Theta Vega Rho
        _data[:, 8] = _data[:, 8] / underlying_scrt_close_rate
        _data[:, 9] = _data[:, 9] / underlying_scrt_close_rate
        _data[:, 10] = _data[:, 10] / underlying_scrt_close_rate
        _data[:, -3] = _data[:, -3] / underlying_scrt_close_rate
        _data[:, -4] = _data[:, -4] / underlying_scrt_close_rate
        _data[:, -5] = _data[:, -5] / underlying_scrt_close_rate
        _data[:, 3] = 100
        # _data[:, _data[0, :] == 'C'] = 0
        # _data[:, _data[0, :] == 'P'] = 1
        _result = np.array(_data[-1, :])
        if _result[-2] == 0:
            print(f'_id : {_id} , _days : {_days}')
        _data[-1, range(-7, 0)] = 0

        _path = f'{self.save_path}/{_id}'
        if not os.path.exists(_path):
            try:
                os.makedirs(_path)
            except BaseException as err:
                a = 0
                # print(err)
        pd.DataFrame(data=_data, columns=[str(x) for x in range(_data.shape[1])]).to_parquet(
            f'{self.save_path}/{_id}/day_{int(_days)}_datas.parquet')
        pd.DataFrame(data=_result.T, columns=['0']).to_parquet(
            f'{self.save_path}/{_id}/day_{int(_days)}_results.parquet')

        if _result[-2] == 0:
            print(f'_id : {_id} , _days : {_days}')
        return _data, _result

    def __len__(self):
        return self.all_data_index.shape[0]


class Dataset:
    def __init__(self, data_index, parquet_data_path):
        self.all_data_path = parquet_data_path
        self.all_data_index = data_index

    def __getitem__(self, idx):
        # print(idx)
        _id, _days = self.all_data_index[idx]
        _data = pd.read_parquet(f'{self.all_data_path}/{int(_id)}/day_{int(_days)}_datas.parquet').to_numpy()
        _result = pd.read_parquet(f'{self.all_data_path}/{int(_id)}/day_{int(_days)}_results.parquet').to_numpy()

        if _result[-2, 0] == 0:
            print(f'_id : {_id} , _days : {_days}')
        return _data, _result.T

    def __len__(self):
        return self.all_data_index.shape[0]


class Dataset_transformer:
    def __init__(self, data_index, parquet_data_path):
        self.all_data_path = parquet_data_path
        self.all_data_index = data_index
        self.pos = np.sin(np.array([1 / (i + 1) for i in range(1, 16)]) * np.pi / 2)[::-1]

    def __getitem__(self, idx):
        # print(idx)
        # idx=idx%8
        _id, _days = self.all_data_index[idx]
        _data = pd.read_parquet(f'{self.all_data_path}/{int(_id)}/day_{int(_days)}_datas.parquet').to_numpy()
        _result = pd.read_parquet(f'{self.all_data_path}/{int(_id)}/day_{int(_days)}_results.parquet').to_numpy()
        # delete the columns of ImpliedVolatility
        _data = np.delete(_data, 12, 1)
        _result = np.delete(_result, 12, 0)
        _data = np.concatenate((_data, self.pos.reshape(15, 1)), axis=1)
        _data[:, 2] = _data[:, 2] / 100
        _data[:, 3] = _data[:, 3] / 100
        _data[:, 4] = _data[:, 4] / 100
        _data[:, -2] = _data[:, -2] / 100
        _data[:, -3] = _data[:, -3] / 100
        _data[:, -4] = _data[:, -4] / 100
        # put_index = _data[:, 1] == 1
        # if put_index[0]:
        #     _data[put_index, 6] = -_data[put_index, 6] - 1
        #     _data[put_index, 12] = -_data[put_index, 12] - 1
        # put_index = _result[1] == 1
        # if put_index:
        #     _result[6] = -_result[ 6] - 1
        #     _result[12] = -_result[ 12] - 1
        if _result[-2, 0] == 0:
            print(f'_id : {_id} , _days : {_days}')
        # print(_result)
        return _data, _result.T

    def __len__(self):
        return self.all_data_index.shape[0]


if __name__ == '__main__':
    _data = pd.read_parquet(
        f'/home/liyu/data/hedging-option/china-market/parquet/training/210000001792/day_43_datas.parquet')
    _result = pd.read_parquet(
        f'/home/liyu/data/hedging-option/china-market/parquet/training/210000001792/day_43_results.parquet')
    print(_data['ClosePrice_1'])
