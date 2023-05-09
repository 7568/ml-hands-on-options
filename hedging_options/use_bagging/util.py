# -*- coding: UTF-8 -*-
"""
Created by louis at 2022/10/31
Description:
"""
import os
import sys
import logging

import lightgbm.basic
from sklearn.metrics import confusion_matrix, auc, accuracy_score,f1_score
import numpy as np


def remove_file_if_exists(path):
    if os.path.exists(path):
        os.remove(path)


def touch(fname, times=None):
    fhandle = open(fname, 'a')
    try:
        os.utime(fname, times)
    finally:
        fhandle.close()


def init_log(file_name='tmp'):
    if not os.path.exists(f'log/'):
        os.mkdir(f'log/')
    if not os.path.exists(f'log/{file_name}_std_out.log'):
        touch(f'log/{file_name}_std_out.log')
    if not os.path.exists(f'log/{file_name}_debug_info.log'):
        touch(f'log/{file_name}_debug_info.log')
    sys.stderr = open(f'log/{file_name}_std_out.log', 'a')
    sys.stdout = open(f'log/{file_name}_std_out.log', 'a')
    logger = logging.getLogger()
    logger.setLevel(level=logging.DEBUG)
    handler = logging.FileHandler(f'log/{file_name}_debug_info.log')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def binary_eval_accuracy(y_true, y_test_hat):
    tn, fp, fn, tp = confusion_matrix(y_true, y_test_hat).ravel()
    print('0：不涨 ， 1：涨')
    print('tn, fp, fn, tp', tn, fp, fn, tp)

    print(f'test中为1的比例 : {y_true.sum() / len(y_true)}')
    print(f'test中为0的比例 : {(1 - y_true).sum() / len(y_true)}')

    # error_in_test = mean_squared_error(y_test_hat, np.array(testing_df[target_fea]).reshape(-1, 1))
    print(f'查准率 - 预测为1 且实际为1 ，看涨的准确率: {tp / (tp + fp)}')
    print(f'查全率 - 实际为1，预测为1 : {tp / (tp + fn)}')
    f_1 = f1_score(y_true, y_test_hat, average="binary")
    print(f'F1 = {f_1}')
    # print(f'AUC：{auc(y_true,y_test_hat)}')
    print(f'总体准确率：{accuracy_score(y_true, y_test_hat)}')


def mutil_class_eval_accuracy(y_true, y_test_hat):
    _0_0, _0_1, _0_2, _1_0, _1_1, _1_2, _2_0, _2_1, _2_2 = confusion_matrix(y_true, y_test_hat).ravel()
    print('0：不涨不跌 ， 1：涨，2：跌')
    print('_0_1 ：表示预测为1，但是实际上为0')
    print('_0_0,_0_1,_0_2,_1_0,_1_1,_1_2,_2_0,_2_1,_2_2 : ', _0_0, _0_1, _0_2, _1_0, _1_1, _1_2, _2_0, _2_1, _2_2)

    print(f'test中为0的比例 : {len(y_true[y_true == 0]) / len(y_true)}')
    print(f'test中为1的比例 : {len(y_true[y_true == 1]) / len(y_true)}')
    print(f'test中为2的比例 : {len(y_true[y_true == 2]) / len(y_true)}')

    # error_in_test = mean_squared_error(y_test_hat, np.array(testing_df[target_fea]).reshape(-1, 1))
    print(f'预测为1 且实际为1 ，看涨的准确率: {_1_1 / (_0_1 + _1_1 + _2_1)}')
    # print(f'真实为2中，预测为2 ，看跌的查全率: {_2_2 / (_2_0 + _2_1 + _2_2)}')
    print(f'预测为1，实际为2 ，重大损失的比例: {_2_1 / (_0_1 + _1_1 + _2_1)}')


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
    testing_target_fea = ['up_and_down', 'RemainingTerm', 'MoneyNess', 'CallOrPut']
    train_x = training_df.copy()
    # train_x = train_x.iloc[:,:-5]
    train_y = training_df[target_fea]

    validation_x = validation_df.copy()
    # validation_x = validation_x.iloc[:,:-5]
    validation_y = validation_df[target_fea]

    testing_x = testing_df.copy()
    # testing_x = testing_x.iloc[:,:-5]
    testing_spot = testing_df['UnderlyingScrtClose'].to_numpy()
    testing_strike = testing_df['StrikePrice'].to_numpy()
    testing_money_ness = testing_spot / testing_strike
    testing_df.loc[:, 'MoneyNess'] = testing_money_ness

    testing_y = testing_df[target_fea]
    testing_y_2 = testing_df[testing_target_fea]

    # latest_x = latest_df.copy()
    # latest_x.loc[:, target_fea] = -1
    # latest_y = latest_df[target_fea]
    if not_use_pre_data:
        train_x = train_x.iloc[:, :int(train_x.shape[1] / 5)]
        validation_x = validation_x.iloc[:, :int(validation_x.shape[1] / 5)]
        testing_x = testing_x.iloc[:, :int(testing_x.shape[1] / 5)]
        # latest_x = latest_x.iloc[:, :int(latest_x.shape[1] / 5)]
    train_x.loc[:, target_fea] = 0
    validation_x.loc[:, target_fea] = 0
    testing_x.loc[:, target_fea] = 0
    testing_y_2.loc[:, 'RemainingTerm'] = np.round((testing_y_2['RemainingTerm'] * 365).to_numpy())
    return train_x, train_y, validation_x, validation_y, testing_x, testing_y,testing_y_2

def mse_loss(y_pred, y_val):
    """
    在xgboost中自定义mseloss
    """
    # l(y_val, y_pred) = (y_val-y_pred)**2
    if type(y_val) is lightgbm.basic.Dataset:
        y_val = y_val.get_label()
    grad = 2*(y_val-y_pred)
    hess = np.repeat(2,y_val.shape[0])
    return grad, hess


def _f1_score_2(y_test_hat, y_true):
    f_1 = f1_score(y_true, y_test_hat, average="binary")
    print(f'F1 = {f_1}')
    return f_1


def detail_result_analysis(real_testing_y,predictions):
    new_y = np.column_stack([real_testing_y, predictions])
    c_set = new_y[new_y[:, -2] == 0]
    p_set = new_y[new_y[:, -2] == 1]

    c_m_less_097 = c_set[c_set[:, -3] < 0.97]
    p_m_less_097 = p_set[p_set[:, -3] < 0.97]

    c_m_more_097_less_103 = c_set[(c_set[:, -3] >= 0.97) & (c_set[:, -3] < 1.03)]
    p_m_more_097_less_103 = p_set[(p_set[:, -3] >= 0.97) & (p_set[:, -3] < 1.03)]

    c_m_more_103 = c_set[c_set[:, -3] >= 1.03]
    p_m_more_103 = p_set[p_set[:, -3] >= 1.03]

    print('++++++++++++++++     c_m_less_097     ++++++++++++++++++++++++++++')
    f1 = _f1_score_2(c_m_less_097[:, 0], c_m_less_097[:, -1]).item()
    print('++++++++++++++++     p_m_less_097     +++++++++++++++++++')
    f1 = _f1_score_2(p_m_less_097[:, 0], p_m_less_097[:, -1]).item()
    print('++++++++++++++++     c_m_more_097_less_103     ++++++++++++++++++')
    f1 = _f1_score_2(c_m_more_097_less_103[:, 0], c_m_more_097_less_103[:, -1]).item()
    print('+++++++++++++++     p_m_more_097_less_103     ++++++++++++++++++++')
    f1 = _f1_score_2(p_m_more_097_less_103[:, 0], p_m_more_097_less_103[:, -1]).item()
    print('++++++++++++++++     c_m_more_103     ++++++++++++++++++++++')
    f1 = _f1_score_2(c_m_more_103[:, 0], c_m_more_103[:, -1]).item()
    print('++++++++++++++++     p_m_more_103     +++++++++++++++++++++')
    f1 = _f1_score_2(p_m_more_103[:, 0], p_m_more_103[:, -1]).item()
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

    max_day = 570
    # for index, df in enumerate([c_set, p_set]):
    span_days = 30

    for d in range(0, 210, span_days):
        c_maturity = c_set[(c_set[:, 1] >= d) & (c_set[:, 1] < (d + span_days))]
        p_maturity = p_set[(p_set[:, 1] >= d) & (p_set[:, 1] < (d + span_days))]
        print('+++++++++        c          ++++++++++++++++++++++')
        print(f'{d}~{d + span_days}')
        f1 = _f1_score_2(c_maturity[:, 0], c_maturity[:, -1]).item()
        print('+++++++++++        p           +++++++++++++++++')
        print(f'{d}~{d + span_days}')
        f1 = _f1_score_2(p_maturity[:, 0], p_maturity[:, -1]).item()
        print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

    if max_day > 270:
        d = 210
        c_maturity = c_set[(c_set[:, 1] >= d) & (c_set[:, 1] < (max_day))]
        p_maturity = p_set[(p_set[:, 1] >= d) & (p_set[:, 1] < (max_day))]
        print('+++++++++        c          ++++++++++++++++++++++')
        print(f'{d}~{max_day}')
        f1 = _f1_score_2(c_maturity[:, 0], c_maturity[:, -1]).item()
        print('+++++++++++        p           +++++++++++++++++')
        print(f'{d}~{max_day}')
        f1 = _f1_score_2(p_maturity[:, 0], p_maturity[:, -1]).item()
        print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')