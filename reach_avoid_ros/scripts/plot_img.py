#coding:utf-8
import matplotlib.pyplot as plt
# plt.rcParams['font.sans-serif']=['SimHei'] 
# plt.rcParams['axes.unicode_minus']=False
import numpy as np

import matplotlib.patches as patches
import torch.utils.data as Data
import torch
from model import GaussianPolicy, QNetwork, DeterministicPolicy
from gym import spaces

# a=np.load("rl_lj_0.npy")
# a=np.load("rl_zj_3.npy")
a=np.load("record.npy")
print(a)
# a=np.load("rl_zj_.npy")
length = a.shape[0]
index = length
# index = 120
print(length)
x = a[:index,:6]
# u = a[:index,6:]
print(x)
ax = x[:,0]
ay = x[:,1]
dx = x[:,3]
dy = x[:,4]

figure, axes = plt.subplots()

axes.add_patch(patches.Rectangle(np.array([0.5,0.5]), 1, 1,color='yellow', linewidth=3.0,fill=False,label='target'))
plt.plot(dx,dy,color='red',linewidth=4.0,linestyle=':',label='attacker')
plt.plot(ax,ay,color='blue',linewidth=4.0,linestyle='--',label='defender')
plt.legend()
plt.xlim((-1.5,1.5))
plt.ylim((-1.5,1.5))
plt.xlabel('x/m')
plt.ylabel('y/m')
plt.grid()
plt.show()