import os
import random
import shutil
from multiprocessing import Pool, cpu_count
import pandas as pd
import numpy as np
from tqdm import tqdm
from hedging_options.library import common as cm


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
def normalize_data(normal_type):
    """
    """
    training_df = pd.read_csv(f'{PREPARE_HOME_PATH}/training.csv', parse_dates=['TradingDate'])
    validation_df = pd.read_csv(f'{PREPARE_HOME_PATH}/validation.csv', parse_dates=['TradingDate'])
    testing_df = pd.read_csv(f'{PREPARE_HOME_PATH}/testing.csv', parse_dates=['TradingDate'])

    no_need_columns = ['SecurityID', 'Filling', 'TradingDate', 'ImpliedVolatility', 'ContinueSign',
                       'TradingDayStatusID']
    training_df.drop(columns=no_need_columns, axis=1, inplace=True)
    validation_df.drop(columns=no_need_columns, axis=1, inplace=True)
    testing_df.drop(columns=no_need_columns, axis=1, inplace=True)

    cat_solumns = ['CallOrPut', 'MainSign']
    for i in cat_solumns:
        training_df[i] = training_df[i].astype('int')
        validation_df[i] = validation_df[i].astype('int')
        testing_df[i] = testing_df[i].astype('int')

    for j in training_df.columns:
        if not (j in cat_solumns):
            training_df[j] = training_df[j].astype('float64')
            validation_df[j] = validation_df[j].astype('float64')
            testing_df[j] = testing_df[j].astype('float64')
    for k in tqdm(training_df.columns, total=len(training_df.columns)):
        if normal_type == 'no_norm':
            break
        if k in ['target', 'real_hedging_rate']:
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
                else:
                    mean = _df.mean()
                    std = _df.std()
                    df[k] = (_df - mean) / std
                    validation_df[k] = (validation_df[k] - mean) / std
                    testing_df[k] = (testing_df[k] - mean) / std

    remove_file_if_exists(f'{PREPARE_HOME_PATH}/{normal_type}/training.csv')
    remove_file_if_exists(f'{PREPARE_HOME_PATH}/{normal_type}/validation.csv')
    remove_file_if_exists(f'{PREPARE_HOME_PATH}/{normal_type}/testing.csv')
    if not os.path.exists(f'{PREPARE_HOME_PATH}/{normal_type}/'):
        os.mkdir(f'{PREPARE_HOME_PATH}/{normal_type}/')
    training_df.to_csv(f'{PREPARE_HOME_PATH}/{normal_type}/training.csv', index=False)
    validation_df.to_csv(f'{PREPARE_HOME_PATH}/{normal_type}/validation.csv', index=False)
    testing_df.to_csv(f'{PREPARE_HOME_PATH}/{normal_type}/testing.csv', index=False)
    testing_df.head().to_csv(f'{PREPARE_HOME_PATH}/{normal_type}/testing_head.csv', index=False)


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

    df = pd.read_csv(f'{PREPARE_HOME_PATH}/all_expand_df_data.csv', parse_dates=['TradingDate'])

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
            testing_data_sets.append(df[df['TradingDate'] == trading_date[index + choice_index+1]])
        else:
            testing_data_sets.append(df[df['TradingDate'] == trading_date[index + choice_index]])
            validation_data_sets.append(df[df['TradingDate'] == trading_date[index + choice_index+1]])
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

PREPARE_HOME_PATH = f'/home/liyu/data/hedging-option/china-market/h_sh_300/'
if __name__ == '__main__':
    # split_training_validation_test()
    split_training_validation_test_by_date()
    # save_head() # 方便查看，不修改数据
    NORMAL_TYPE = 'mean_norm'
    normalize_data(NORMAL_TYPE)
    # NORMAL_TYPE = 'min_max_norm'
    # normalize_data(NORMAL_TYPE)
    # NORMAL_TYPE = 'no_norm'
    # normalize_data(NORMAL_TYPE)
    # make_index('training')
    # make_index('validation')
    # make_index('testing')
