# -*- coding: UTF-8 -*-
"""
Created by louis at 2022/7/5
Description:
"""
import logging


class Logger:

    def __init__(self):
        self.redirect_sys_stderr = False
        self.logger = logging.getLogger()
        self.logger.setLevel(level=logging.DEBUG)

    def set_logger_param(self, normal_type, n_steps, redirect_sys_stderr,tag):
        handler = logging.FileHandler(f'{normal_type}/log/sys_out_{tag}_{n_steps}.log')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.redirect_sys_stderr = redirect_sys_stderr

    def info(self, msg):
        if self.redirect_sys_stderr:
            self.logger.info(msg)
        else:
            print(msg)

    def debug(self, msg):
        if self.redirect_sys_stderr:
            self.logger.debug(msg)
        else:
            print(msg)

    def error(self, msg):
        if self.redirect_sys_stderr:
            self.logger.error(msg)
        else:
            print(msg)


logger = Logger()
