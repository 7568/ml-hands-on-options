# -*- coding: UTF-8 -*-
"""
Created by louis at 2022/10/31
Description:
"""
import os
import sys
import logging
from sklearn.metrics import confusion_matrix


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

def binary_eval_accuracy(y_true,y_test_hat):
    tn, fp, fn, tp = confusion_matrix(y_true, y_test_hat).ravel()
    print('0：不涨 ， 1：涨')
    print('tn, fp, fn, tp', tn, fp, fn, tp)

    print(f'test中为1的比例 : {y_true.sum() / len(y_true)}')
    print(f'test中为0的比例 : {(1 - y_true).sum() / len(y_true)}')

    # error_in_test = mean_squared_error(y_test_hat, np.array(testing_df[target_fea]).reshape(-1, 1))
    print(f'查准率 - 预测为1 且实际为1 ，看涨的准确率: {tp / (tp + fp)}')
    print(f'查全率 - 实际为1，预测为1 : {tp / (tp + fn)}')
    print(f'F1 = {(2 * tp) / (len(y_true) + tp - tn)}')


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
