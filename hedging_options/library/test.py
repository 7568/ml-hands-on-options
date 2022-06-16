# -*- coding: UTF-8 -*-
"""
Created by louis at 2022/5/30
Description:
"""
import numpy as np
a=[]
# a.append([1,2])
# a.append([3,4])
aa=[x for x in range(1,6)]
bb = [11,12,13,14,15]
aa = np.append(aa,aa)
bb = np.append(bb,bb)
a = np.array([aa,bb]).T
print(a)
print(a.shape)

from tqdm import tqdm
for i in tqdm(range(10000)):
    print(i)