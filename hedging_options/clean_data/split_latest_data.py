import os

import pandas as pd
from tqdm import tqdm

from hedging_options.library import common as cm


def remove_file_if_exists(path):
    if os.path.exists(path):
        os.remove(path)


@cm.my_log
def normalize_data(normal_type):
    """
    """
    all_df = pd.read_csv(f'{LATEST_DATA_PATH}/all_raw_data.csv', parse_dates=['TradingDate'])
    normal_data = pd.read_csv(f'{TRAIN_DATA_PATH}/{normal_type}/normal_data.csv')

    no_need_columns = ['SecurityID', 'Filling', 'ContinueSign', 'TradingDayStatusID']
    all_df.drop(columns=no_need_columns, axis=1, inplace=True)

    cat_columns = ['CallOrPut', 'CallOrPut_1', 'CallOrPut_2', 'CallOrPut_3', 'CallOrPut_4',
                   'MainSign', 'MainSign_1', 'MainSign_2', 'MainSign_3', 'MainSign_4', 'TradingDate',
                   'up_and_down', 'up_and_down_1', 'up_and_down_2', 'up_and_down_3', 'up_and_down_4']



    for i in cat_columns:
        if i == 'TradingDate':
            continue
        all_df[i] = all_df[i].astype('int')

    for j in all_df.columns:
        if not (j in cat_columns):
            all_df[j] = all_df[j].astype('float64')
    for k in tqdm(all_df.columns, total=len(all_df.columns)):
        if normal_type == 'no_norm':
            break
        # if k in ['TradingDate', 'C_1', 'S_1', 'real_hedging_rate']:
        if k in ['TradingDate']:
            continue
        if f'{k}_mean' not in normal_data.columns:
            continue
        if all_df[k].dtype == 'float64':
            # for df in [training_df, validation_df, testing_df]:
            mean = normal_data[f'{k}_mean'][0]
            std = normal_data[f'{k}_std'][0]
            all_df[k] = (all_df[k] - mean) / std

    remove_file_if_exists(f'{LATEST_DATA_PATH}/{normal_type}/predict_latest.csv')
    if not os.path.exists(f'{LATEST_DATA_PATH}/{normal_type}/'):
        os.mkdir(f'{LATEST_DATA_PATH}/{normal_type}/')
    all_df.to_csv(f'{LATEST_DATA_PATH}/{normal_type}/predict_latest.csv', index=False)


@cm.my_log
def check_null(normal_type):
    df = pd.read_csv(f'{LATEST_DATA_PATH}/{normal_type}/predict_latest.csv', parse_dates=['TradingDate'])
    trading_date = df.sort_values(by=['TradingDate'])['TradingDate'].unique()
    for date in tqdm(trading_date, total=trading_date.shape[0]):
        if df[df['TradingDate'] == date].isnull().values.any():
            raise Exception(f'{date} error!')

    # print(f'df.shape : {df.shape}')
    # option_ids = df['SecurityID'].unique()
    # for option_id in tqdm(option_ids, total=len(option_ids)):
    #     _options = df[df['SecurityID'] == option_id]
    #     if _options.isnull().values.any():
    #         contain_null_ids.append(option_id)
    # print(f'contain_null_ids : {contain_null_ids}')


TRAIN_DATA_PATH = f'/home/liyu/data/hedging-option/china-market/h_sh_300/'
LATEST_DATA_PATH = f'/home/liyu/data/hedging-option/latest-china-market/h_sh_300/'
if __name__ == '__main__':

    NORMAL_TYPE = 'mean_norm'
    normalize_data(NORMAL_TYPE)
    check_null(NORMAL_TYPE)

