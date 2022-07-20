# -*- coding: UTF-8 -*-
"""
Created by louis at 2022/6/30
Description:
"""
import numpy as np
import matplotlib.pyplot as plt

# all_points = 1-np.cos(np.arange(1 / 430, 1, 1 / 430) * np.pi / 2)
import torch

all_points = np.power(np.arange(1 / 430, 1, 1 / 430) * np.pi / 2,1)
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1, 1, 1)

ax.set_title('')
ax.set_xlabel('epoch')
ax.set_ylabel('loss')

ax.scatter(range(1, len(all_points) + 1), all_points, cmap='Greens')
plt.show()
all_rates = all_points / np.sum(all_points)
# 让最小的数缩放成1，然后其他的数依次以相同的比例缩放
all_rates = np.rint(all_rates/all_rates[0])
# print(np.rint(all_rates))
all_index = np.array([])
for i, _rate in enumerate(all_rates):
    all_index = np.append(np.repeat(i, _rate), all_index)
np.random.shuffle(all_index)
print(all_index.shape)
print(all_index.max())
print(all_index.min())
print(all_index[-100:])


a=[1,2,3]
print(a[:-1])

a = torch.ones((2,3,4))
print(torch.sum(a,dim=[1,2]))



