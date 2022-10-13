import os
import random
import shutil
from multiprocessing import Pool, cpu_count
import pandas as pd
import numpy as np
from tqdm import tqdm
from hedging_options.library import common as cm


# Load_Clean_aux.py loads the clean data and implements some extra cleaning, before running linear regressions or ANNs.
# Append the library path to PYTHONPATH, so library can be imported.
# sys.path.append(os.path.dirname(os.getcwd()))


#        ['index', 'SecurityID', 'TradingDate', 'Symbol', 'ExchangeCode',
#        'UnderlyingSecurityID', 'UnderlyingSecuritySymbol', 'ShortName',
#        'CallOrPut', 'StrikePrice', 'ExerciseDate', 'ClosePrice',
#        'UnderlyingScrtClose', 'RemainingTerm', 'RisklessRate',
#        'HistoricalVolatility', 'ImpliedVolatility', 'TheoreticalPrice',
#        'Delta', 'Gamma', 'Vega', 'Theta', 'Rho', 'DividendYeild', 'DataType',
#        'ImpliedVolatility_1', 'Delta_1', 'Gamma_1', 'Vega_1', 'Theta_1',
#        'Rho_1', 'ClosePrice_1', 'UnderlyingScrtClose_1']
# Remove samples of period time less 30 day from training and validation sets.
# 由于将来我们的数据还要分成训练集和验证集，所以时间跨度只有15天是不够的，因此这里将时间跨度设置为25天,小于25天且大于15天的全部用来训练。
# df_large, df_small = laux.choose_period_span_for_china(df, 25)
# print(f'df_large.shape : {df_large.shape}')
# print(f'df_small.shape : {df_small.shape}')
# bl = ['SecurityID', 'TradingDate', 'CallOrPut', 'ClosePrice_1', 'UnderlyingScrtClose_1', 'Delta', 'Gamma', 'Vega',
#       'Theta', 'Rho', 'Delta_1', 'Gamma_1', 'Vega_1', 'Theta_1', 'Rho_1']
# df_large_train = df_large.loc[bl]
# df_large_ = df_small.loc[bl]
# df_train = laux.make_features(df_large_train)
# df_large = laux.make_features(df_large_train)
# del df

# 我们尽量将 training_set , validation_set , test_set 分成4：1：1。
# 总共有 345442 条数据，包含的期权为 4492 个
# 我们先将期权的id打乱，接着从中挨个取出期权，再设置0，1，2随机取样，来判断该期权是否是需要全部拿出来。
# 在获取 validation_set 的时候，从df中取出该期权，如果取样是0就放入 training_set 中，
# 否则就就从第2天到到期那天之间取一个数，小于这个天数的放入training_set，大于这个天数的放入 validation_set
# 每一次都判断 validation_set 长度，如果长度大于 23000 条就跳出循环
# 获取 test_set 方法同理

def split_training_validation_test():
    df = pd.read_csv(f'{PREPARE_HOME_PATH}/all_raw_data.csv', parse_dates=['TradingDate'])
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
    os.remove(f'{PREPARE_HOME_PATH}/training.csv')
    os.remove(f'{PREPARE_HOME_PATH}/validation.csv')
    os.remove(f'{PREPARE_HOME_PATH}/testing.csv')
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




PREPARE_HOME_PATH = f'/home/liyu/data/hedging-option/china-market/h_sh_300/'
if __name__ == '__main__':

    split_training_validation_test()
    # make_index('training')
    # make_index('validation')
    # make_index('testing')
