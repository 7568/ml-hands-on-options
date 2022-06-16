import random

import pandas as pd
from tqdm import tqdm


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
# 总共有 239447 条数据，包含的期权为4164个
# 我们先将期权的id打乱，接着从中挨个取出期权，再设置0，1，2随机取样，来判断该期权是否是需要全部拿出来。
# 在获取 validation_set 的时候，从df中取出该期权，如果取样是0就放入 training_set 中，
# 否则就判断该期权的交易天数是否大于25，如果小于25就直接放入 validation_set，如果大于25，就从25天到到期那天之间取一个数，小于这个天数的放入training_set，大于这个天数的放入 validation_set
# 每一次都判断 validation_set 长度，如果长度大于 23000 条就跳出循环
# 获取 test_set 方法同理

def split_training_validation_test():
    df = pd.read_csv('/home/liyu/data/hedging-option/china-market/h_sh_300.csv', parse_dates=[
        'TradingDate', 'ExerciseDate'])
    option_ids = df['SecurityID'].unique()
    print(len(option_ids))
    random.shuffle(option_ids)
    training_set = pd.DataFrame(columns=df.columns)
    validation_set = pd.DataFrame(columns=df.columns)
    test_set = pd.DataFrame(columns=df.columns)
    for option_id in tqdm(option_ids, total=len(option_ids)):
        _options = df[df['SecurityID'] == option_id]
        if validation_set.shape[0] < 23000:
            if random.choice([0, 1, 2]) == 0:
                training_set = training_set.append(_options)
            else:
                if _options.shape[0] < 25:
                    validation_set = validation_set.append(_options)
                else:
                    s = random.choice(range(20, _options.shape[0] - 1))
                    training_set = training_set.append(_options.iloc[:s])
                    validation_set = validation_set.append(_options.iloc[s:])
        elif (validation_set.shape[0] >= 23000) & (test_set.shape[0] < 23000):
            if random.choice([0, 1, 2]) == 0:
                training_set = training_set.append(_options)
            else:
                if _options.shape[0] < 25:
                    test_set = test_set.append(_options)
                else:
                    s = random.choice(range(20, _options.shape[0] - 1))
                    training_set = training_set.append(_options.iloc[:s])
                    test_set = test_set.append(_options.iloc[s:])
        elif (test_set.shape[0] > 23000) & (validation_set.shape[0] > 23000):
            training_set = training_set.append(_options)

    print(training_set.shape[0])
    print(validation_set.shape[0])
    print(test_set.shape[0])
    if (training_set.shape[0] + validation_set.shape[0] + test_set.shape[0]) < df.shape[0]:
        print('error')
    training_set.to_csv('/home/liyu/data/hedging-option/china-market/h_sh_300_training.csv', index=False)
    validation_set.to_csv('/home/liyu/data/hedging-option/china-market/h_sh_300_validation.csv', index=False)
    test_set.to_csv('/home/liyu/data/hedging-option/china-market/h_sh_300_test.csv', index=False)


def make_index(tag):

    train_data = pd.read_csv(f'/home/liyu/data/hedging-option/china-market/h_sh_300_{tag}.csv', parse_dates=[
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
        f'/home/liyu/data/hedging-option/china-market/h_sh_300_{tag}_index.csv', index=False)



if __name__ == '__main__':
    # split_training_validation_test()
    make_index('training')
    make_index('validation')
    make_index('test')
