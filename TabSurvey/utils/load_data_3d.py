import sklearn.datasets
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

import numpy as np
import pandas as pd


def discretize_colum(data_clm, num_values=10):
    """ Discretize a column by quantiles """
    r = np.argsort(data_clm)
    bin_sz = (len(r) / num_values) + 1  # make sure all quantiles are in range 0-(num_quarts-1)
    q = r // bin_sz
    return q


def reformat_data_2(training_df, validation_df, testing_df, not_use_pre_data=False):
    """使用当天和前几天的 up_and_down，将每天的数据属性的位置对其"""
    columns = ['CallOrPut', 'StrikePrice', 'ClosePrice', 'UnderlyingScrtClose','RemainingTerm', 'RisklessRate',
               'HistoricalVolatility', 'ImpliedVolatility','TheoreticalPrice', 'Delta', 'Gamma', 'Vega', 'Theta', 'Rho',
               'DividendYeild','OpenPrice', 'HighPrice', 'LowPrice', 'SettlePrice', 'Change1', 'Change2','Volume',
               'Position', 'Amount', 'PreClosePrice', 'PreSettlePrice','PrePosition', 'PositionChange', 'MainSign',
               'AvgPrice','ClosePriceChangeRatio', 'SettlePriceChangeRatio', 'Amplitude', 'LimitUp','LimitDown',
               'MaintainingMargin', 'ChangeRatio', 'up_and_down']

    day_0_training_df = training_df[columns].copy()
    day_0_training_df.loc[:,'up_and_down']=0
    day_0_validation_df = validation_df[columns].copy()
    day_0_validation_df.loc[:,'up_and_down']=0
    day_0_testing_df = testing_df[columns].copy()
    day_0_testing_df.loc[:,'up_and_down']=0
    for i in range(1,5):
        day_i_columns = [c + f'_{i}' for c in columns]
        day_i_training_df = training_df[day_i_columns].copy()
        day_i_validation_df = validation_df[day_i_columns].copy()
        day_i_testing_df = testing_df[day_i_columns].copy()
        day_0_training_df = pd.concat((day_0_training_df,day_i_training_df),axis=1)
        day_0_validation_df = pd.concat((day_0_validation_df,day_i_validation_df),axis=1)
        day_0_testing_df = pd.concat((day_0_testing_df,day_i_testing_df),axis=1)
    target_fea = 'up_and_down'
    train_y = training_df[target_fea]
    validation_y = validation_df[target_fea]
    testing_y = testing_df[target_fea]

    return day_0_training_df, train_y, day_0_validation_df, validation_y, day_0_testing_df, testing_y


def reformat_data(training_df, validation_df, testing_df, not_use_pre_data=False):
    """
    训练的时候，前4天的 up_and_down 的值可见，当天的不可见，且设置为-1
    :param training_df:
    :param validation_df:
    :param testing_df:
    :param not_use_pre_data:
    :return:
    """

    target_fea = 'up_and_down'
    train_x = training_df.copy()
    print(training_df.columns)
    train_x = train_x.iloc[:, :-5]
    train_y = training_df[target_fea]

    validation_x = validation_df.copy()
    validation_x = validation_x.iloc[:, :-5]
    validation_y = validation_df[target_fea]

    testing_x = testing_df.copy()
    testing_x = testing_x.iloc[:, :-5]
    testing_y = testing_df[target_fea]

    # latest_x = latest_df.copy()
    # latest_x.loc[:, target_fea] = -1
    # latest_y = latest_df[target_fea]
    if not_use_pre_data:
        train_x = train_x.iloc[:, :int(train_x.shape[1] / 5)]
        validation_x = validation_x.iloc[:, :int(validation_x.shape[1] / 5)]
        testing_x = testing_x.iloc[:, :int(testing_x.shape[1] / 5)]
        # latest_x = latest_x.iloc[:, :int(latest_x.shape[1] / 5)]
    # cat_features = ['CallOrPut', 'MainSign']
    # for i in range(1, 5):
    #     cat_features.append(f'CallOrPut_{i}')
    #     cat_features.append(f'MainSign_{i}')
    # for f in cat_features:
    #     print(f'{f} : {testing_df.columns.get_loc(f)}')
    return train_x, train_y, validation_x, validation_y, testing_x, testing_y


def load_h_sh_300_options(pretrain=False):
    print(f'pretrain : {pretrain}')
    NORMAL_TYPE = 'mean_norm'
    # if pretrain:
    if False:
        PREPARE_HOME_PATH = '/home/liyu/data/hedging-option/20160701-20221124/h_sh_300/'
        training_df = pd.read_csv(f'{PREPARE_HOME_PATH}/{NORMAL_TYPE}/training.csv')
        PREPARE_HOME_PATH_2 = '/home/liyu/data/hedging-option/20190701-20221124/h_sh_300/'
        validation_df = pd.read_csv(f'{PREPARE_HOME_PATH_2}/{NORMAL_TYPE}/validation.csv')
        testing_df = pd.read_csv(f'{PREPARE_HOME_PATH_2}/{NORMAL_TYPE}/testing.csv')
    else:
        PREPARE_HOME_PATH = '/home/liyu/data/hedging-option/20190701-20221124/h_sh_300/'
        training_df = pd.read_csv(f'{PREPARE_HOME_PATH}/{NORMAL_TYPE}/training.csv')
        validation_df = pd.read_csv(f'{PREPARE_HOME_PATH}/{NORMAL_TYPE}/validation.csv')
        testing_df = pd.read_csv(f'{PREPARE_HOME_PATH}/{NORMAL_TYPE}/testing.csv')

    training_trading_dates = training_df['TradingDate']
    validation_trading_dates = validation_df['TradingDate']
    testing_trading_dates = testing_df['TradingDate']
    no_need_columns = ['TradingDate', 'C_1']
    training_df.drop(columns=no_need_columns, axis=1, inplace=True)
    validation_df.drop(columns=no_need_columns, axis=1, inplace=True)
    testing_df.drop(columns=no_need_columns, axis=1, inplace=True)
    cat_features = ['CallOrPut', 'MainSign', 'up_and_down']
    for i in range(1, 5):
        cat_features.append(f'CallOrPut_{i}')
        cat_features.append(f'MainSign_{i}')
        cat_features.append(f'up_and_down_{i}')
    train_x, train_y, validation_x, validation_y, testing_x, testing_y = reformat_data_2(
        training_df, validation_df, testing_df, not_use_pre_data=False)
    # pd.DataFrame().to_numpy()
    X = {
        'training': train_x.to_numpy(),
        'validation': validation_x.to_numpy(),
        'testing': testing_x.to_numpy(),
    }
    y = {
        'training': train_y.to_numpy(),
        'validation': validation_y.to_numpy(),
        'testing': testing_y.to_numpy(),
    }

    return X, y, training_trading_dates, validation_trading_dates, testing_trading_dates


def load_data(args):
    print("Loading dataset " + args.dataset + "...")
    if args.dataset == "H_sh_300_options":  # h_sh_300_options dataset
        X, y, training_trading_dates, validation_trading_dates, testing_trading_dates = load_h_sh_300_options(
            args.pretrain)
        # Preprocess target
        if args.target_encode:
            le = LabelEncoder()
            y['training'] = le.fit_transform(y['training'])
            y['validation'] = le.fit_transform(y['validation'])
            y['testing'] = le.fit_transform(y['testing'])

            # Setting this if classification task
            if args.objective == "classification":
                args.num_classes = len(le.classes_)
                print("Having", args.num_classes, "classes as target.")

        num_idx = []
        args.cat_dims = []

        # Preprocess data
        for i in range(args.num_features):
            if args.cat_idx and i in args.cat_idx:
                le = LabelEncoder()
                X['training'][:, i] = le.fit_transform(X['training'][:, i])
                X['validation'][:, i] = le.fit_transform(X['validation'][:, i])
                X['testing'][:, i] = le.fit_transform(X['testing'][:, i])

                # Setting this?
                if len(le.classes_) == 1:
                    args.cat_dims.append(len(le.classes_) + 1)
                else:
                    args.cat_dims.append(len(le.classes_))

            else:
                num_idx.append(i)

        if args.scale:
            print("Scaling the data...")
            scaler = StandardScaler()
            X['training'][:, num_idx] = scaler.fit_transform(X['training'][:, num_idx])
            X['validation'][:, num_idx] = scaler.fit_transform(X['validation'][:, num_idx])
            X['testing'][:, num_idx] = scaler.fit_transform(X['testing'][:, num_idx])

        if args.one_hot_encode:
            ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
            new_x1 = ohe.fit_transform(X[:, args.cat_idx])
            new_x2 = X[:, num_idx]
            X = np.concatenate([new_x1, new_x2], axis=1)
            print("New Shape:", X.shape)
        return X, y, training_trading_dates, validation_trading_dates, testing_trading_dates

    elif args.dataset == "CaliforniaHousing":  # Regression dataset
        X, y = sklearn.datasets.fetch_california_housing(return_X_y=True)

    elif args.dataset == "Covertype":  # Multi-class classification dataset
        X, y = sklearn.datasets.fetch_covtype(return_X_y=True)
        # X, y = X[:10000, :], y[:10000]  # only take 10000 samples from dataset

    elif args.dataset == "KddCup99":  # Multi-class classification dataset with categorical data
        X, y = sklearn.datasets.fetch_kddcup99(return_X_y=True)
        X, y = X[:10000, :], y[:10000]  # only take 10000 samples from dataset

        # filter out all target classes, that occur less than 1%
        target_counts = np.unique(y, return_counts=True)
        smaller1 = int(X.shape[0] * 0.01)
        small_idx = np.where(target_counts[1] < smaller1)
        small_tar = target_counts[0][small_idx]
        for tar in small_tar:
            idx = np.where(y == tar)
            y[idx] = b"others"

        # new_target_counts = np.unique(y, return_counts=True)
        # print(new_target_counts)

        '''
        # filter out all target classes, that occur less than 100
        target_counts = np.unique(y, return_counts=True)
        small_idx = np.where(target_counts[1] < 100)
        small_tar = target_counts[0][small_idx]
        for tar in small_tar:
            idx = np.where(y == tar)
            y[idx] = b"others"

        # new_target_counts = np.unique(y, return_counts=True)
        # print(new_target_counts)
        '''
    elif args.dataset == "Adult" or args.dataset == "AdultCat":  # Binary classification dataset with categorical data, if you pass AdultCat, the numerical columns will be discretized.
        url_data = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"

        features = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital-status', 'occupation',
                    'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']
        label = "income"
        columns = features + [label]
        df = pd.read_csv(url_data, names=columns)

        # Fill NaN with something better?
        df.fillna(0, inplace=True)
        if args.dataset == "AdultCat":
            columns_to_discr = [('age', 10), ('fnlwgt', 25), ('capital-gain', 10), ('capital-loss', 10),
                                ('hours-per-week', 10)]
            for clm, nvals in columns_to_discr:
                df[clm] = discretize_colum(df[clm], num_values=nvals)
                df[clm] = df[clm].astype(int).astype(str)
            df['education_num'] = df['education_num'].astype(int).astype(str)
            args.cat_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
        X = df[features].to_numpy()
        y = df[label].to_numpy()

    elif args.dataset == "HIGGS":  # Binary classification dataset with one categorical feature
        path = "/home/liyu/data/tabular-data/HIGGS.csv.gz"
        df = pd.read_csv(path, header=None)
        df.columns = ['x' + str(i) for i in range(df.shape[1])]
        num_col = list(df.drop(['x0', 'x21'], 1).columns)
        cat_col = ['x21']
        label_col = 'x0'

        def fe(x):
            if x > 2:
                return 1
            elif x > 1:
                return 0
            else:
                return 2

        df.x21 = df.x21.apply(fe)

        # Fill NaN with something better?
        df.fillna(0, inplace=True)

        X = df[num_col + cat_col].to_numpy()
        y = df[label_col].to_numpy()

    elif args.dataset == "Heloc":  # Binary classification dataset without categorical data
        path = "heloc_cleaned.csv"  # Missing values already filtered
        df = pd.read_csv(path)
        label_col = 'RiskPerformance'

        X = df.drop(label_col, axis=1).to_numpy()
        y = df[label_col].to_numpy()

    else:
        raise AttributeError("Dataset \"" + args.dataset + "\" not available")

    print("Dataset loaded!")

    # Preprocess target
    if args.target_encode:
        le = LabelEncoder()
        y = le.fit_transform(y)

        # Setting this if classification task
        if args.objective == "classification":
            args.num_classes = len(le.classes_)
            print("Having", args.num_classes, "classes as target.")

    num_idx = []
    args.cat_dims = []

    # Preprocess data
    for i in range(args.num_features):
        if args.cat_idx and i in args.cat_idx:
            le = LabelEncoder()
            X[:, i] = le.fit_transform(X[:, i])

            # Setting this?
            if len(le.classes_)==1:
                args.cat_dims.append(len(le.classes_)+1)
            else:
                args.cat_dims.append(len(le.classes_))
        else:
            num_idx.append(i)

    if args.scale:
        print("Scaling the data...")
        scaler = StandardScaler()
        X[:, num_idx] = scaler.fit_transform(X[:, num_idx])

    if args.one_hot_encode:
        ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
        new_x1 = ohe.fit_transform(X[:, args.cat_idx])
        new_x2 = X[:, num_idx]
        X = np.concatenate([new_x1, new_x2], axis=1)
        print("New Shape:", X.shape)

    return X, y
