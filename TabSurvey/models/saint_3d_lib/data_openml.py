import numpy as np
import pandas as pd
from torch.utils.data import Dataset

"""
每次batch都是一天内的数据
"""


def data_split(X, y, nan_mask):  # indices
    x_d = {
        'data': X,
        'mask': nan_mask.values
    }

    if x_d['data'].shape != x_d['mask'].shape:
        raise 'Shape of data not same as that of nan mask!'

    y_d = {
        'data': y.reshape(-1, 1)
    }
    return x_d, y_d


def data_prep(X, y):
    temp = pd.DataFrame(X).fillna("MissingValue")
    nan_mask = temp.ne("MissingValue").astype(int)
    X, y = data_split(X, y, nan_mask)
    return X, y


class DataSetCatCon(Dataset):

    def __init__(self, X, Y, cat_cols, task='regression', continuous_mean_std=None, trading_dates=None):
        self.trading_dates = trading_dates
        self.unique_trading_dates = np.sort(np.unique(self.trading_dates))
        X_mask = X['mask'].copy()
        X = X['data'].copy()

        # Added this to handle data without categorical features
        if cat_cols is not None:
            cat_cols = list(cat_cols)
            con_cols = list(set(np.arange(X.shape[1])) - set(cat_cols))
        else:
            con_cols = list(np.arange(X.shape[1]))
            cat_cols = []

        self.X1 = X[:, cat_cols].copy().astype(np.int64)  # categorical columns
        self.X2 = X[:, con_cols].copy().astype(np.float32)  # numerical columns
        self.X1_mask = X_mask[:, cat_cols].copy().astype(np.int64)  # categorical columns
        self.X2_mask = X_mask[:, con_cols].copy().astype(np.int64)  # numerical columns
        if task == 'regression':
            self.y = Y['data'].astype(np.float32)
        else:
            self.y = Y['data']  # .astype(np.float32)
        self.cls = np.zeros_like(self.y, dtype=int)
        self.cls_mask = np.ones_like(self.y, dtype=int)
        if continuous_mean_std is not None:
            mean, std = continuous_mean_std
            self.X2 = (self.X2 - mean) / std

    def __len__(self):
        return len(self.unique_trading_dates)
        # return 1

    def __getitem__(self, idx):
        # print(idx)
        trading_date = self.unique_trading_dates[int(idx)]
        data_idx = np.where(self.trading_dates == trading_date)
        # date_idx = np.random.permutation(len(np.array(date_idx).squeeze()))[:100]

        x_1 = []
        x_2 = []
        y = []
        mask_x1 = []
        mask_x2 = []
        for _idx in np.array(data_idx).squeeze():
            # x_1.append(np.concatenate((self.cls[_idx], self.X1[_idx])))
            x_1.append(self.X1[_idx])
            x_2.append(self.X2[_idx])
            y.append(self.y[_idx])
            # mask_x1.append(np.concatenate((self.cls_mask[_idx], self.X1_mask[_idx])))
            mask_x1.append( self.X1_mask[_idx])
            mask_x2.append(self.X2_mask[_idx])
        return np.array(x_1), np.array(x_2), np.array(y), np.array(mask_x1), np.array(mask_x2)


class DataSetCatCon_pretrain(Dataset):

    def __init__(self, X, Y, cat_cols, task='regression', continuous_mean_std=None, trading_dates=None):
        self.trading_dates = trading_dates
        self.unique_trading_dates = np.sort(np.unique(self.trading_dates))
        X_mask = X['mask'].copy()
        X = X['data'].copy()

        # Added this to handle data without categorical features
        if cat_cols is not None:
            cat_cols = list(cat_cols)
            con_cols = list(set(np.arange(X.shape[1])) - set(cat_cols))
        else:
            con_cols = list(np.arange(X.shape[1]))
            cat_cols = []

        self.X1 = X[:, cat_cols].copy().astype(np.int64)  # categorical columns
        self.X2 = X[:, con_cols].copy().astype(np.float32)  # numerical columns
        self.X1_mask = X_mask[:, cat_cols].copy().astype(np.int64)  # categorical columns
        self.X2_mask = X_mask[:, con_cols].copy().astype(np.int64)  # numerical columns
        if task == 'regression':
            self.y = Y['data'].astype(np.float32)
        else:
            self.y = Y['data']  # .astype(np.float32)
        self.cls = np.zeros_like(self.y, dtype=int)
        self.cls_mask = np.ones_like(self.y, dtype=int)
        if continuous_mean_std is not None:
            mean, std = continuous_mean_std
            self.X2 = (self.X2 - mean) / std

    def __len__(self):
        return len(self.unique_trading_dates)
        # return int(len(self.unique_trading_dates) / 30)

    def __getitem__(self, idx):
        # print(idx)
        trading_date = self.unique_trading_dates[int(idx)]
        data_idx = np.where(self.trading_dates == trading_date)
        # date_idx = np.random.permutation(len(np.array(date_idx).squeeze()))[:100]

        x_1 = []
        x_2 = []
        y = []
        mask_x1 = []
        mask_x2 = []
        for _idx in np.array(data_idx).squeeze():
            # x_1.append(np.concatenate((self.cls[_idx], self.X1[_idx])))
            x_1.append(self.X1[_idx])
            x_2.append(self.X2[_idx])
            y.append(self.y[_idx])
            # mask_x1.append(np.concatenate((self.cls_mask[_idx], self.X1_mask[_idx])))
            mask_x1.append( self.X1_mask[_idx])
            mask_x2.append(self.X2_mask[_idx])
        return np.array(x_1), np.array(x_2), np.array(y), np.array(mask_x1), np.array(mask_x2)