import os
import random
import shutil
from multiprocessing import Pool, cpu_count
import pandas as pd
import numpy as np
from tqdm import tqdm
from hedging_options.library import common as cm
from hedging_options.library.common import chunks_np


def remove_file_if_exists(path):
    if os.path.exists(path):
        os.remove(path)


# 我们尽量将 training_set , validation_set , test_set 分成4：1：1。
# 总共有 345442 条数据，包含的期权为 4492 个
# 我们先将期权的id打乱，接着从中挨个取出期权，再设置0，1，2随机取样，来判断该期权是否是需要全部拿出来。
# 在获取 validation_set 的时候，从df中取出该期权，如果取样是0就放入 training_set 中，
# 否则就就从第2天到到期那天之间取一个数，小于这个天数的放入training_set，大于这个天数的放入 validation_set
# 每一次都判断 validation_set 长度，如果长度大于 23000 条就跳出循环
# 获取 test_set 方法同理
def split_training_validation_test():
    df = pd.read_csv(f'{PREPARE_HOME_PATH}/scaled_expand_df_data.csv', parse_dates=['TradingDate'])
    option_ids = df['SecurityID'].unique()
    print(len(option_ids))
    random.shuffle(option_ids)
    training_list = []
    training_num = 0
    validation_list = []
    validation_num = 0
    test_list = []
    test_num = 0
    # training_set = pd.DataFrame(columns=df.columns)
    # validation_set = pd.DataFrame(columns=df.columns)
    # test_set = pd.DataFrame(columns=df.columns)
    for option_id in tqdm(option_ids, total=len(option_ids)):
        _options = df[df['SecurityID'] == option_id]
        if validation_num < 23000:
            if random.choice([0, 1, 2]) == 0:
                # training_set = training_set.append(_options)
                training_list.append(_options)
                training_num += _options.shape[0]
            else:
                s = random.choice(range(1, _options.shape[0] - 2))
                sorted_options = _options.sort_values(by='TradingDate')
                # training_set = training_set.append(sorted_options.iloc[:s])
                training_list.append(sorted_options.iloc[:s])
                training_num += sorted_options.iloc[:s].shape[0]
                # validation_set = validation_set.append(sorted_options.iloc[s:])
                validation_list.append(sorted_options.iloc[s:])
                validation_num += sorted_options.iloc[s:].shape[0]
        elif (validation_num >= 23000) & (test_num < 23000):
            if random.choice([0, 1, 2]) == 0:
                # training_set = training_set.append(_options)
                training_list.append(_options)
                training_num += _options.shape[0]
            else:
                s = random.choice(range(1, _options.shape[0] - 2))
                sorted_options = _options.sort_values(by='TradingDate')
                # training_set = training_set.append(sorted_options.iloc[:s])
                training_list.append(sorted_options.iloc[:s])
                training_num += sorted_options.iloc[:s].shape[0]
                # test_set = test_set.append(sorted_options.iloc[s:])
                test_list.append(sorted_options.iloc[s:])
                test_num += sorted_options.iloc[s:].shape[0]
        elif (test_num > 23000) & (validation_num > 23000):
            # training_set = training_set.append(_options)
            training_list.append(_options)
            training_num += _options.shape[0]

    print(f'training_num : {training_num} , validation_num : {validation_num},test_num : {test_num}')
    training_set = pd.concat([pd.DataFrame(i, columns=df.columns) for i in training_list], ignore_index=True)
    validation_set = pd.concat([pd.DataFrame(i, columns=df.columns) for i in validation_list], ignore_index=True)
    test_set = pd.concat([pd.DataFrame(i, columns=df.columns) for i in test_list], ignore_index=True)
    print(training_set.shape[0])
    print(validation_set.shape[0])
    print(test_set.shape[0])
    if (training_set.shape[0] + validation_set.shape[0] + test_set.shape[0]) < df.shape[0]:
        print('error')
    remove_file_if_exists(f'{PREPARE_HOME_PATH}/training.csv')
    remove_file_if_exists(f'{PREPARE_HOME_PATH}/validation.csv')
    remove_file_if_exists(f'{PREPARE_HOME_PATH}/testing.csv')
    training_set.to_csv(f'{PREPARE_HOME_PATH}/training.csv', index=False)
    validation_set.to_csv(f'{PREPARE_HOME_PATH}/validation.csv', index=False)
    test_set.to_csv(f'{PREPARE_HOME_PATH}/testing.csv', index=False)


def make_index(tag):
    train_data = pd.read_csv(f'{PREPARE_HOME_PATH}/{tag}.csv', parse_dates=['TradingDate'])
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


@cm.my_log
def normalize_data2(normal_type):
    PREPARE_HOME_PATH2 = f'/home/liyu/data/hedging-option/20190701-20221124_2/h_sh_300/'
    training_df = pd.read_csv(f'{PREPARE_HOME_PATH}/training.csv', parse_dates=['TradingDate'])
    validation_df = pd.read_csv(f'{PREPARE_HOME_PATH2}/validation.csv', parse_dates=['TradingDate'])
    testing_df = pd.read_csv(f'{PREPARE_HOME_PATH2}/testing.csv', parse_dates=['TradingDate'])

    no_need_columns = ['SecurityID', 'Filling', 'ContinueSign', 'TradingDayStatusID']
    training_df.drop(columns=no_need_columns, axis=1, inplace=True)
    validation_df.drop(columns=no_need_columns, axis=1, inplace=True)
    testing_df.drop(columns=no_need_columns, axis=1, inplace=True)

    cat_columns = ['CallOrPut', 'CallOrPut_1', 'CallOrPut_2', 'CallOrPut_3', 'CallOrPut_4',
                   'MainSign', 'MainSign_1', 'MainSign_2', 'MainSign_3', 'MainSign_4', 'TradingDate',
                   'up_and_down', 'up_and_down_1', 'up_and_down_2', 'up_and_down_3', 'up_and_down_4']
    normal_data = []

    for i in cat_columns:
        if i == 'TradingDate':
            continue
        training_df[i] = training_df[i].astype('int')
        validation_df[i] = validation_df[i].astype('int')
        testing_df[i] = testing_df[i].astype('int')

    for j in training_df.columns:
        if not (j in cat_columns):
            training_df[j] = training_df[j].astype('float64')
            validation_df[j] = validation_df[j].astype('float64')
            testing_df[j] = testing_df[j].astype('float64')
    for k in tqdm(training_df.columns, total=len(training_df.columns)):
        if normal_type == 'no_norm':
            break
        # if k in ['TradingDate', 'C_1', 'S_1', 'real_hedging_rate']:
        if k in ['TradingDate']:
            continue
        if validation_df[k].dtype == 'float64':
            # for df in [training_df, validation_df, testing_df]:
            df = training_df
            _df = df[k]
            max = _df.max()
            min = _df.min()
            if max > 1 or min < -1:
                if normal_type == 'min_max_norm':
                    df[k] = (_df - min) / (max - min)
                    validation_df[k] = (validation_df[k] - min) / (max - min)
                    testing_df[k] = (testing_df[k] - min) / (max - min)
                    normal_data.append([f'{k}_max', max])
                    normal_data.append([f'{k}_min', min])
                else:
                    mean = _df.mean()
                    std = _df.std()
                    if std == 0:
                        df[k] = 0
                        validation_df[k] = 0
                        testing_df[k] = 0
                        normal_data.append([f'{k}_mean', 0])
                        normal_data.append([f'{k}_std', 0])
                    else:
                        df[k] = (_df - mean) / std
                        validation_df[k] = (validation_df[k] - mean) / std
                        testing_df[k] = (testing_df[k] - mean) / std
                        normal_data.append([f'{k}_mean', mean])
                        normal_data.append([f'{k}_std', std])
            else:
                print(f'{k} do not normalize')

    remove_file_if_exists(f'{PREPARE_HOME_PATH}/{normal_type}/training.csv')
    remove_file_if_exists(f'{PREPARE_HOME_PATH}/{normal_type}/validation.csv')
    remove_file_if_exists(f'{PREPARE_HOME_PATH}/{normal_type}/testing.csv')
    remove_file_if_exists(f'{PREPARE_HOME_PATH}/{normal_type}/normal_data.csv')
    if not os.path.exists(f'{PREPARE_HOME_PATH}/{normal_type}/'):
        os.mkdir(f'{PREPARE_HOME_PATH}/{normal_type}/')
    training_df.to_csv(f'{PREPARE_HOME_PATH}/{normal_type}/training.csv', index=False)
    validation_df.to_csv(f'{PREPARE_HOME_PATH}/{normal_type}/validation.csv', index=False)
    testing_df.to_csv(f'{PREPARE_HOME_PATH}/{normal_type}/testing.csv', index=False)
    pd.DataFrame(data=[{i[0]: i[1] for i in normal_data}]).to_csv(f'{PREPARE_HOME_PATH}/{normal_type}/normal_data.csv',
                                                                  index=False)


@cm.my_log
def normalize_data(normal_type):
    """
    """
    training_df = pd.read_csv(f'{PREPARE_HOME_PATH}/training.csv', parse_dates=['TradingDate'])
    validation_df = pd.read_csv(f'{PREPARE_HOME_PATH}/validation.csv', parse_dates=['TradingDate'])
    testing_df = pd.read_csv(f'{PREPARE_HOME_PATH}/testing.csv', parse_dates=['TradingDate'])

    no_need_columns = ['SecurityID', 'Filling', 'ContinueSign', 'TradingDayStatusID']
    training_df.drop(columns=no_need_columns, axis=1, inplace=True)
    validation_df.drop(columns=no_need_columns, axis=1, inplace=True)
    testing_df.drop(columns=no_need_columns, axis=1, inplace=True)

    cat_columns = ['CallOrPut', 'CallOrPut_1', 'CallOrPut_2', 'CallOrPut_3', 'CallOrPut_4',
                   'MainSign', 'MainSign_1', 'MainSign_2', 'MainSign_3', 'MainSign_4', 'TradingDate',
                   'up_and_down', 'up_and_down_1', 'up_and_down_2', 'up_and_down_3', 'up_and_down_4']
    normal_data = []

    for i in cat_columns:
        if i == 'TradingDate':
            continue
        training_df[i] = training_df[i].astype('int')
        validation_df[i] = validation_df[i].astype('int')
        testing_df[i] = testing_df[i].astype('int')

    for j in training_df.columns:
        if not (j in cat_columns):
            training_df[j] = training_df[j].astype('float64')
            validation_df[j] = validation_df[j].astype('float64')
            testing_df[j] = testing_df[j].astype('float64')
    for k in tqdm(training_df.columns, total=len(training_df.columns)):
        if normal_type == 'no_norm':
            break
        # if k in ['TradingDate', 'C_1', 'S_1', 'real_hedging_rate']:
        if k in ['TradingDate']:
            continue
        if validation_df[k].dtype == 'float64':
            # for df in [training_df, validation_df, testing_df]:
            df = training_df
            _df = df[k]
            max = _df.max()
            min = _df.min()
            if max > 1 or min < -1:
                if normal_type == 'min_max_norm':
                    df[k] = (_df - min) / (max - min)
                    validation_df[k] = (validation_df[k] - min) / (max - min)
                    testing_df[k] = (testing_df[k] - min) / (max - min)
                    normal_data.append([f'{k}_max', max])
                    normal_data.append([f'{k}_min', min])
                else:
                    mean = _df.mean()
                    std = _df.std()
                    if std == 0:
                        df[k] = 0
                        validation_df[k] = 0
                        testing_df[k] = 0
                        normal_data.append([f'{k}_mean', 0])
                        normal_data.append([f'{k}_std', 0])
                    else:
                        df[k] = (_df - mean) / std
                        validation_df[k] = (validation_df[k] - mean) / std
                        testing_df[k] = (testing_df[k] - mean) / std
                        normal_data.append([f'{k}_mean', mean])
                        normal_data.append([f'{k}_std', std])
            else:
                print(f'{k} do not normalize')

    remove_file_if_exists(f'{PREPARE_HOME_PATH}/{normal_type}/training.csv')
    remove_file_if_exists(f'{PREPARE_HOME_PATH}/{normal_type}/validation.csv')
    remove_file_if_exists(f'{PREPARE_HOME_PATH}/{normal_type}/testing.csv')
    remove_file_if_exists(f'{PREPARE_HOME_PATH}/{normal_type}/normal_data.csv')
    if not os.path.exists(f'{PREPARE_HOME_PATH}/{normal_type}/'):
        os.mkdir(f'{PREPARE_HOME_PATH}/{normal_type}/')
    training_df.to_csv(f'{PREPARE_HOME_PATH}/{normal_type}/training.csv', index=False)
    validation_df.to_csv(f'{PREPARE_HOME_PATH}/{normal_type}/validation.csv', index=False)
    testing_df.to_csv(f'{PREPARE_HOME_PATH}/{normal_type}/testing.csv', index=False)
    pd.DataFrame(data=[{i[0]: i[1] for i in normal_data}]).to_csv(f'{PREPARE_HOME_PATH}/{normal_type}/normal_data.csv',
                                                                  index=False)


@cm.my_log
def save_head():
    training_df = pd.read_csv(f'{PREPARE_HOME_PATH}/training.csv', parse_dates=['TradingDate'])
    validation_df = pd.read_csv(f'{PREPARE_HOME_PATH}/validation.csv', parse_dates=['TradingDate'])
    testing_df = pd.read_csv(f'{PREPARE_HOME_PATH}/testing.csv', parse_dates=['TradingDate'])

    training_df.head().to_csv(f'{PREPARE_HOME_PATH}/sub_training.csv', index=False)
    validation_df.head().to_csv(f'{PREPARE_HOME_PATH}/sub_validation.csv', index=False)
    testing_df.head().to_csv(f'{PREPARE_HOME_PATH}/sub_testing.csv', index=False)


@cm.my_log
def split_training_validation_test_by_date():
    """
    首先按照时间排序，从小到大，开始为第0天的数据
    然后从5到15之间随机选一个数，比如是8，那么8之前的数据放入训练集中，第8个数据放入测试集中，
    然后将第9天的数据当作第0天的数据开始下一轮的分配
    :return:
    training_data.shape:(276698, 170) , validation_data.shape:(34494, 170) , testing_data.shape:(34250, 170)
    """

    df = pd.read_csv(f'{PREPARE_HOME_PATH}/all_raw_data.csv', parse_dates=['TradingDate'])

    trading_date = df.sort_values(by=['TradingDate'])['TradingDate'].unique()
    training_data_sets = []
    validation_data_sets = []
    testing_data_sets = []
    index = 0
    while index < len(trading_date):
        choice_index = random.choice(range(5, 12))
        if (index + choice_index) > len(trading_date):
            training_data_dates = trading_date[index:]
            training_data_sets.append(df[df['TradingDate'].isin(training_data_dates)])
            index = index + choice_index
            continue
        else:
            training_data_dates = trading_date[index:index + choice_index]
        training_data_sets.append(df[df['TradingDate'].isin(training_data_dates)])
        if (index + choice_index) == len(trading_date):
            validation_data_sets.append(df[df['TradingDate'] == trading_date[index + choice_index]])
            index = index + choice_index + 1
            continue
        choice_validation = random.choice([0, 1])
        if choice_validation == 1:
            validation_data_sets.append(df[df['TradingDate'] == trading_date[index + choice_index]])
            testing_data_sets.append(df[df['TradingDate'] == trading_date[index + choice_index + 1]])
        else:
            testing_data_sets.append(df[df['TradingDate'] == trading_date[index + choice_index]])
            validation_data_sets.append(df[df['TradingDate'] == trading_date[index + choice_index + 1]])
        index = index + choice_index + 2
    training_data = pd.concat(training_data_sets)
    validation_data = pd.concat(validation_data_sets)
    testing_data = pd.concat(testing_data_sets)
    print(f'df.shape:{df.shape}')
    print(f'training_data.shape:{training_data.shape} , validation_data.shape:{validation_data.shape} ,'
          f' testing_data.shape:{testing_data.shape}')
    print(f'all.shape:{training_data.shape[0] + validation_data.shape[0] + testing_data.shape[0]}')
    if (training_data.shape[0] + validation_data.shape[0] + testing_data.shape[0]) != df.shape[0]:
        raise Exception('error')
    remove_file_if_exists(f'{PREPARE_HOME_PATH}/training.csv')
    remove_file_if_exists(f'{PREPARE_HOME_PATH}/validation.csv')
    remove_file_if_exists(f'{PREPARE_HOME_PATH}/testing.csv')
    training_data.to_csv(f'{PREPARE_HOME_PATH}/training.csv', index=False)
    validation_data.to_csv(f'{PREPARE_HOME_PATH}/validation.csv', index=False)
    testing_data.to_csv(f'{PREPARE_HOME_PATH}/testing.csv', index=False)


@cm.my_log
def split_training_validation_test_by_date_3():
    df = pd.read_csv(f'{PREPARE_HOME_PATH}/all_raw_data.csv', parse_dates=['TradingDate'])
    print(f'all data length is {df.shape}')
    remove_file_if_exists(f'{PREPARE_HOME_PATH}/training.csv')
    training_data = df
    training_data.to_csv(f'{PREPARE_HOME_PATH}/training.csv', index=False)
    print(f'training data length is {training_data.shape}')


@cm.my_log
def split_training_validation_test_by_date_2():
    """
    首先按照时间排序，从小到大，开始为第0天的数据
    前 80% 的数据用来训练，后 20% 中前 10% 用来做验证集，后10% 用来做测试集
    :return:
    """

    print(PREPARE_HOME_PATH)
    df = pd.read_csv(f'{PREPARE_HOME_PATH}/all_raw_data.csv', parse_dates=['TradingDate'])
    print(f'all data length is {df.shape}')

    trading_date = df.sort_values(by=['TradingDate'])['TradingDate'].unique()
    training_end_index = int(trading_date.shape[0] * 0.8)
    validation_end_index = int(trading_date.shape[0] * 0.9)
    training_data_dates = trading_date[:training_end_index]
    validation_data_dates = trading_date[training_end_index:validation_end_index]
    testing_data_dates = trading_date[validation_end_index:]
    print(f'training_data_dates start at {training_data_dates[0]}')
    print(f'training_data_dates end at {training_data_dates[-1]}')
    print(f'validation_data_dates start at {validation_data_dates[0]}')
    print(f'validation_data_dates end at {validation_data_dates[-1]}')
    print(f'testing_data_dates start at {testing_data_dates[0]}')
    print(f'testing_data_dates end at {testing_data_dates[-1]}')
    training_data = df[df['TradingDate'].isin(training_data_dates)]
    validation_data = df[df['TradingDate'].isin(validation_data_dates)]
    testing_data = df[df['TradingDate'].isin(testing_data_dates)]
    remove_file_if_exists(f'{PREPARE_HOME_PATH}/training.csv')
    remove_file_if_exists(f'{PREPARE_HOME_PATH}/validation.csv')
    remove_file_if_exists(f'{PREPARE_HOME_PATH}/testing.csv')
    training_data.to_csv(f'{PREPARE_HOME_PATH}/training.csv', index=False)
    validation_data.to_csv(f'{PREPARE_HOME_PATH}/validation.csv', index=False)
    testing_data.to_csv(f'{PREPARE_HOME_PATH}/testing.csv', index=False)

    print(f'training data length is {training_data.shape}')
    print(f'validation data length is {validation_data.shape}')
    print(f'testing data length is {testing_data.shape}')


@cm.my_log
def sub_data_by_date(traing_end_date, normal_type):
    df = pd.read_csv(f'{PREPARE_HOME_PATH}/all_raw_data.csv', parse_dates=['TradingDate'])
    no_need_columns = ['SecurityID', 'Filling', 'ContinueSign', 'TradingDayStatusID']
    df.drop(columns=no_need_columns, axis=1, inplace=True)
    print(f'all data length is {df.shape}')

    # trading_date = df.sort_values(by=['TradingDate'])['TradingDate'].unique()
    training_df = df[df['TradingDate'] < pd.Timestamp(traing_end_date)]
    for k in tqdm(df.columns, total=len(df.columns)):

        if normal_type == 'no_norm':
            break
        # if k in ['TradingDate', 'C_1', 'S_1', 'real_hedging_rate']:
        if k in ['TradingDate']:
            continue
        if training_df[k].dtype == 'float64':
            # for df in [training_df, validation_df, testing_df]:
            _df = training_df[k]
            max = _df.max()
            min = _df.min()
            if max > 1 or min < -1:
                if normal_type == 'min_max_norm':
                    training_df[k] = (training_df[k] - min) / (max - min)

                else:
                    mean = float(_df.mean())
                    std = float(_df.std())
                    if std == 0:
                        print(f'{k} std is o!')
                        r = np.array(training_df[k])
                        print(f'real std is {r.std()}')
                        training_df[k] = 0
                    else:
                        training_df[k] = (training_df[k] - mean) / std

            else:
                print(f'{k} do not normalize')

    remove_file_if_exists(f'{PREPARE_HOME_PATH}/{normal_type}/training.csv')
    if not os.path.exists(f'{PREPARE_HOME_PATH}/{normal_type}/'):
        os.mkdir(f'{PREPARE_HOME_PATH}/{normal_type}/')
    training_df.to_csv(f'{PREPARE_HOME_PATH}/{normal_type}/training.csv', index=False)


@cm.my_log
def check_null(normal_type):
    training_df = pd.read_csv(f'{PREPARE_HOME_PATH}/{normal_type}/training.csv', parse_dates=['TradingDate'])
    validation_df = pd.read_csv(f'{PREPARE_HOME_PATH}/{normal_type}/validation.csv', parse_dates=['TradingDate'])
    testing_df = pd.read_csv(f'{PREPARE_HOME_PATH}/{normal_type}/testing.csv', parse_dates=['TradingDate'])
    df = pd.concat((training_df, validation_df, testing_df))
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


def do_append_less_rate(param):
    column_all_data = param['column_all_data']
    column_chunk_data = param['column_chunk_data']
    zeros = []
    for index, j in tqdm(enumerate(column_chunk_data), total=len(column_chunk_data)):
        zeros.append((((column_all_data < j).sum()) / len(column_all_data)) + 1)
    return np.array(zeros)


def create_train_mirror_data(normal_type):
    """
    新建一个train数据集对应的概率密度数据
    :return:
    """
    training_df = pd.read_csv(f'{PREPARE_HOME_PATH}/{normal_type}/training.csv', parse_dates=['TradingDate'])
    before_0_column = ['StrikePrice', 'ClosePrice', 'UnderlyingScrtClose', 'RisklessRate', 'HistoricalVolatility',
                       'ImpliedVolatility',
                       'TheoreticalPrice', 'Delta', 'Gamma', 'Vega', 'Theta', 'Rho', 'DividendYeild', 'MainSign',
                       'OpenPrice', 'PositionChange', 'PreClosePrice', 'PrePosition', 'RemainingTerm', 'PreSettlePrice',
                       'HighPrice', 'LowPrice', 'SettlePrice', 'Change1', 'Change2', 'Volume', 'Position', 'Amount',
                       'AvgPrice', 'ClosePriceChangeRatio', 'SettlePriceChangeRatio', 'Amplitude', 'LimitUp',
                       'LimitDown', 'MaintainingMargin', 'ChangeRatio', 'CallOrPut']

    X = training_df[before_0_column].to_numpy()
    r_n, c_n = X.shape
    cpu_num = int(cpu_count() * 0.8)
    mirror = np.zeros_like(X, dtype=np.float32)

    for i in range(c_n):
        c_i = X[:, i]
        data_chunks = chunks_np(c_i, cpu_num)
        param = []
        for data_chunk in tqdm(data_chunks, total=len(data_chunks)):
            ARGS_ = dict(column_chunk_data=data_chunk, column_all_data=c_i)
            param.append(ARGS_)
        with Pool(cpu_num) as p:
            r = p.map(do_append_less_rate, param)
            p.close()
            p.join()
            mirror[:, i] = np.concatenate(r)
    np.save(f'/home/liyu/data/hedging-option/china-market/h_sh_300/hedging_options_mirror.npy', mirror)
    print(mirror[0, :])
    print('done')


# PREPARE_HOME_PATH = f'/home/liyu/data/hedging-option/china-market/h_sh_300/'
# PREPARE_HOME_PATH = f'/home/liyu/data/hedging-option/20140101-20160229/h_sh_300/'
# PREPARE_HOME_PATH = f'/home/liyu/data/hedging-option/20140101-20220321/h_sh_300/'
# PREPARE_HOME_PATH = f'/home/liyu/data/hedging-option/20140101-20221124_2/h_sh_300/'
# PREPARE_HOME_PATH = f'/home/liyu/data/hedging-option/20160301-20190531/h_sh_300/'
# PREPARE_HOME_PATH = f'/home/liyu/data/hedging-option/20160701-20221124/h_sh_300/'
# PREPARE_HOME_PATH = f'/home/liyu/data/hedging-option/20190601-20221123/h_sh_300/'
# PREPARE_HOME_PATH = f'/home/liyu/data/hedging-option/20190701-20221124/h_sh_300/'
up_rete = 0  # 1.0 0, 0.1 , 0.05, 0.01
PREPARE_HOME_PATH = f'/home/liyu/data/hedging-option/20190701-20221124_{up_rete}/h_sh_300'
if __name__ == '__main__':
    NORMAL_TYPE = 'mean_norm'
    for i in [1.0,0, 0.1, 0.05, 0.01]:
        PREPARE_HOME_PATH = f'/home/liyu/data/hedging-option/20190701-20221124_{i}/h_sh_300'
        split_training_validation_test_by_date_2()
        normalize_data(NORMAL_TYPE)
        check_null(NORMAL_TYPE)

if __name__ == '__main1__':
    NORMAL_TYPE = 'mean_norm'
    # sub_data_by_date('2022-07-21',NORMAL_TYPE) # 将原始数据截取至 20220721 ， 因为20220721之后的数据为testing
    # split_training_validation_test()
    # split_training_validation_test_by_date()
    split_training_validation_test_by_date_2()  # 根据时间前后划分，比例为8：1：1
    # split_training_validation_test_by_date_3()  # 将20140101-20220321数据作为训练集
    # save_head() # 方便查看，不修改数据
    normalize_data(NORMAL_TYPE)
    # normalize_data2(NORMAL_TYPE)  # 以20140101-20220321中的训练集作为标准，将20190701-20221124中的验证集和测试集拿过来，再进行归一化，且保存在20140101-20220321
    check_null(NORMAL_TYPE)
    # create_train_mirror_data(NORMAL_TYPE)
    # NORMAL_TYPE = 'min_max_norm'
    # normalize_data(NORMAL_TYPE)
    # NORMAL_TYPE = 'no_norm'
    # normalize_data(NORMAL_TYPE)
    # make_index('training')
    # make_index('validation')
    # make_index('testing')
