import numpy as np
import matplotlib.pyplot as plt
import torch.utils.data as Data
import torch
from model import GaussianPolicy, QNetwork, DeterministicPolicy
from gym import spaces

a=np.load("record.npy")
length = a.shape[0]
index = (length//10)*9

x = torch.from_numpy(a[:index,:6].astype(np.float32))
u = torch.from_numpy(a[:index,6:].astype(np.float32))

x_eval = torch.from_numpy(a[:index,:6].astype(np.float32))
u_eval = torch.from_numpy(a[:index,6:].astype(np.float32))
# # y = np.linspace(-5, 5, 100)\
# # time = 120
# print(x.shape)
torch_dataset = Data.TensorDataset(x, u)

BATCH_SIZE = 200
loader = Data.DataLoader(
    dataset=torch_dataset,      # torch TensorDataset format
    batch_size=BATCH_SIZE,      # mini batch size
    shuffle=True,               # 要不要打乱数据 (打乱比较好)
)


action_space = spaces.Box(low=-np.array([0.5,2.0]), high=+np.array([0.5,2.0]), shape=(2,), dtype=np.float32)
net = DeterministicPolicy(6, 2, 256, action_space).to('cuda')

optimizer = torch.optim.SGD(net.parameters(), lr=0.05)
loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss



for epoch in range(1000):   # 训练所有!整套!数据 3 次
    for step, (batch_x, batch_u) in enumerate(loader):  # 每一步 loader 释放一小批数据用来学习
        # print(batch_x.shape)
        prediction = net(batch_x.cuda())     # input x and predict based on x

        loss = loss_func(prediction, batch_u.cuda())     # must be (1. nn output, 2. target)

        optimizer.zero_grad()   # clear gradients for next train
        loss.backward()         # backpropagation, compute gradients
        optimizer.step()        # apply gradients
        if (step==0):
            prediction_eval = net(x_eval.cuda())
            loss_eval = loss_func(prediction_eval, u_eval.cuda()) 
            print('Epoch: ', epoch, '| Step: ', step, '| loss : ', loss_eval)
    if(epoch%100==0):
        torch.save(net.state_dict(), "models/pretrain")