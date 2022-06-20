# -*- coding: UTF-8 -*-
"""
Created by louis at 2022/6/20
Description:
"""
import numpy as np

a = [1/(i+1) for i in range(1,16)]
print(len(a))

print(np.sin(np.array([1/(i+1) for i in range(1,16)])*np.pi/2)[::-1])
