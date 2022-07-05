import random

import pandas as pd
from tqdm import tqdm


# 我们尽量将 training_set , validation_set , test_set 按照交易时间的先后顺序分成4：1：1。
# 总共有 135170 条数据，包含的期权为 1978 个
# 前4份为 training 数据，中间1份为 validation 数据，后面一份为 testing 数据
# 用 training 数据来训练，然后在训练阶段，用 validation 数据来验证，而且在训练中获取 training 数据的时候，
# 根据时间的近远，取到数据的概率由大变小。
# 当模型在 training 数据收敛之后，再把 validation 数据加进来进行训练，形成新的 training数据，再重新分配数据集被取到的概率，
# 此时用 testing 数据进行验证。
# 当模型在在新的 training 数据上收敛之后，我们按照时间顺序来测试 testing 数据集，当测试完一个数据集之后，
# 就把该数据集放入到新的 training 数据中，又形成更新的training数据集，和新的数据被取到的概率，依次迭代下去，一边训练一边测试。
# 在 testing 数据集加入到新的 training 数据中之后，此时不能再用 testing 数据来做验证，所以训练新的 training 数据集的时候只是看模型有没有收敛。


def split_training_validation_test():
    df = pd.read_csv(f'{PREPARE_HOME_PATH}/h_sh_300/mean_normalization_data.csv')
    print(len(df['SecurityID'].unique()))
    trading_date = df.sort_values(by=['TradingDate'])['TradingDate'].unique()
    split_meta = len(trading_date) / 10
    traning_day = trading_date[0:int(8 * split_meta)]
    validation_day = trading_date[int(8 * split_meta):int(9 * split_meta)]
    testing_day = trading_date[int(9 * split_meta):]
    training_set = df[df['TradingDate'].isin(traning_day)]
    validation_set = df[df['TradingDate'].isin(validation_day)]
    test_set = df[df['TradingDate'].isin(testing_day)]

    print(training_set.shape[0])
    print(validation_set.shape[0])
    print(test_set.shape[0])
    if (training_set.shape[0] + validation_set.shape[0] + test_set.shape[0]) != df.shape[0]:
        print('error')
    training_set.to_csv(f'{PREPARE_HOME_PATH}/h_sh_300/training.csv', index=False)
    validation_set.to_csv(f'{PREPARE_HOME_PATH}/h_sh_300/validation.csv', index=False)
    test_set.to_csv(f'{PREPARE_HOME_PATH}/h_sh_300/testing.csv', index=False)


def make_index(tag):
    train_data = pd.read_csv(f'{PREPARE_HOME_PATH}/{tag}.csv', parse_dates=[
        'TradingDate', 'ExerciseDate'])
    option_ids = train_data['SecurityID'].unique()
    # train_data_index=pd.DataFrame(columns=['SecurityID','days'])
    train_data_index = []
    for id in tqdm(option_ids, total=len(option_ids)):
        opotions = train_data[train_data['SecurityID'] == id]
        start = 15
        end = opotions.shape[0] + 1
        for i in range(start, end):
            train_data_index.append([id, i])

    pd.DataFrame(train_data_index, columns=['SecurityID', 'days']).to_csv(
        f'{PREPARE_HOME_PATH}/h_sh_300_{tag}_index.csv', index=False)


def mean_normalization():
    df = pd.read_csv(f'{PREPARE_HOME_PATH}/h_sh_300/two_day_all_clean_data.csv')
    columns = df.columns
    normalization_info = pd.DataFrame(columns=['column', 'mean', 'std'])
    for column in tqdm(columns, total=len(columns)):
        # print(column, df[column].dtypes)
        if column in exclude_features:
            continue
        if df[column].dtypes not in ['float64', 'int64']:
            continue
        _df = df[column]
        print(column,_df.max(),_df.min())
        mean = _df.mean()
        std = _df.std()
        df[column] = (_df - mean) / std
        normalization_info = normalization_info.append({'column': column, 'mean': mean, 'std': std}, ignore_index=True)
    # normalized_df = (df - df.mean()) / df.std()
    df.to_csv(f'{PREPARE_HOME_PATH}/h_sh_300/mean_normalization_data.csv', index=False)
    normalization_info.to_csv(f'{PREPARE_HOME_PATH}/h_sh_300/mean_normalization_info.csv', index=False)
    print('Done')


def min_max_normalization():
    df = pd.read_csv(f'{PREPARE_HOME_PATH}/h_sh_300/two_day_all_clean_data.csv')
    columns = df.columns
    normalization_info = pd.DataFrame(columns=['column', 'max', 'min'])
    for column in tqdm(columns, total=len(columns)):
        # print(column, df[column].dtypes)
        if column in exclude_features:
            continue
        if df[column].dtypes not in ['float64', 'int64']:
            continue
        _df = df[column]
        max = _df.max()
        min = _df.min()
        df[column] = (_df - min) / (max - min)
        normalization_info = normalization_info.append({'column': column, 'max': max, 'min': min}, ignore_index=True)
    # normalized_df = (df - df.mean()) / df.std()
    df.to_csv(f'{PREPARE_HOME_PATH}/h_sh_300/min_max_normalization_data.csv', index=False)
    normalization_info.to_csv(f'{PREPARE_HOME_PATH}/h_sh_300/min_max_normalization_info.csv', index=False)
    print('Done!')


PREPARE_HOME_PATH = f'/home/liyu/data/hedging-option/china-market/'
exclude_features = ['ActualDelta','SecurityID', 'TradingDate', 'Symbol', 'ExchangeCode', 'UnderlyingSecurityID',
                    'UnderlyingSecuritySymbol', 'ShortName', 'DataType', 'HistoricalVolatility',
                     'ImpliedVolatility', 'TheoreticalPrice','ExerciseDate','CallOrPut','on_ret',
                    'RisklessRate']
if __name__ == '__main__':
    min_max_normalization()
    mean_normalization()
    split_training_validation_test()
    # make_index('training')
    # make_index('validation')
    # make_index('testing')
