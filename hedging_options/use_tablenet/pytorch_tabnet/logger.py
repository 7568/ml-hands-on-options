# -*- coding: UTF-8 -*-
"""
Created by louis at 2022/7/5
Description:
"""
import logging


class Logger:

    def __init__(self):
        self.logger = logging.getLogger()
        self.logger.setLevel(level=logging.DEBUG)

    def set_logger_param(self, normal_type,n_steps):
        handler = logging.FileHandler(f'{normal_type}/log/train_test002_{n_steps}.log')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def info(self, msg):
        self.logger.info(msg)

    def debug(self, msg):
        self.logger.debug(msg)

    def error(self, msg):
        self.logger.error(msg)


logger = Logger()
