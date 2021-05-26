from gym import spaces, core
import torch
import os
import numpy as np
import rps.robotarium as robotarium
import matplotlib.pyplot as plt
from controller import CLF_CBF_controller,MPC_controller, MPC_controller_defence,Fast_controller_defence,Fast_Catch
from model import DeterministicPolicy
class ReachAvoidAgent_Def(core.Env):

    def __init__(self):
        self.u_range = np.array([0.5,2.0])
        self.action_space = spaces.Box(low=-self.u_range, high=+self.u_range, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=+np.inf, shape=(6,), dtype=np.float32)
        
        a = np.array([-1 + np.random.rand() * 0.4 - 0.2, np.random.rand() * 0.8 - 0.4 ,0])
        d = np.array([0.5 + np.random.rand() * 0.4 - 0.2 ,np.random.rand() * 0.4 - 0.2 ,np.random.rand() * 0.4 - 0.2 ])
        self.initial_conditions = np.array([a,d]).T 
        self.agents = robotarium.Robotarium(number_of_robots=2, show_figure=False, initial_conditions=self.initial_conditions,sim_in_real_time=False)
        self.states = self.agents.get_poses().T
        self.agents.step()
        self._max_episode_steps = 400
        self.times = 0
        self.net = DeterministicPolicy(6, 2, 256, self.action_space).to('cuda')
        self.net.load_state_dict(torch.load("models/save/pretrain"))

    def step(self, action, k=1):

        dxu = np.zeros([2,2])
        
        noise = np.array([ np.random.rand() * 0.1 - 0.05, np.random.rand() * 0.4 - 0.2 ])
        
        
        inp = torch.FloatTensor(self.states.flatten()).to("cuda").unsqueeze(0)
        dxu[0] = self.net(inp).detach().cpu().numpy()[0]

        dxu[1] = action
        # if(k==1):
        #     dxu[1] = Fast_controller_defence(self.states[0],self.states[1]) + noise 
        # if(k==2):
        #     dxu[1] = Fast_Catch(self.states[0],self.states[1]) + noise
        # if(k==3):
        #     dxu[1] = np.array([ 0.2 + np.random.rand() * 0.1 - 0.05, np.random.rand() * 2 - 1 ])
        # dxu[1] = np.array([ 0,0])
        # dxu[1] = Fast_controller_defence(self.states[0],self.states[1])
        # dxu[1] = Fast_Catch
        self.agents.set_velocities(np.arange(2), dxu.T)
        self.agents.step()

        self.states = self.agents.get_poses().T

        reward = self._get_reward()
        done = self._get_done()
        
        state = self.states.flatten()
        self.times += 1
        return state, reward, done, {}


    def _get_reward(self):
        a_state = self.states[0]
        d_state = self.states[1]
        r = 0
        distance = np.sqrt(np.sum(np.square(a_state[:2] - d_state[:2])))
        goal_cost = self.states[0,0]

        r += 2*goal_cost
        if (goal_cost>=1):
            r += 2000
        if (goal_cost>=0):
            r += goal_cost*10
        if (distance<0.2):
            r -= 1000
        else: 
            r += 0.01*distance

        if (a_state[1]>1 or a_state[1]<-1 or a_state[0]<-1.5):
            r -= 1000
        
        return -r

    def _get_done(self):
        a_state = self.states[0]
        d_state = self.states[1]
        distance = np.sqrt(np.sum(np.square(a_state[:2] - d_state[:2])))
        # print(distance)
        if (distance<0.2 or a_state[0]>1):
            done_n = True
        else:
            done_n = False
        if(self.times > self._max_episode_steps):
            done_n = True
        if (a_state[1]>1 or a_state[1]<-1 or a_state[0]<-1.5):
            done_n = True
        return done_n

    def reset(self, if_show_figure=False):
        plt.close()
        self.times = 0
        # a = np.array([-1 + np.random.rand() * 0.4 - 0.2, np.random.rand() * 0.8 - 0.4 , np.random.rand() * 0.4 - 0.2 ])
        # d = np.array([0.5 + np.random.rand() * 0.4 - 0.2 ,np.random.rand() * 0.4 - 0.2 , np.pi + np.random.rand() * 0.4 - 0.2 ])

        a = np.array([-1 + np.random.rand() * 1 - 0.5, np.random.rand() * 1.4 - 0.7 , np.random.rand() * 1.4 - 0.7 ])
        d = np.array([ 0.1 + np.random.rand() * 1 - 0.5 ,np.random.rand() * 1.4 - 0.7 , np.pi + np.random.rand() * 1.4 - 0.7 ])

        self.initial_conditions = np.array([a,d]).T 
        self.agents = robotarium.Robotarium(number_of_robots=2, show_figure=if_show_figure, initial_conditions=self.initial_conditions,sim_in_real_time=False)
        self.states = self.agents.get_poses().T
        self.agents.step()
        self.agents.get_poses()
        # dxu = np.zeros([2,2])
        # self.agents.set_velocities(np.arange(2), dxu.T)
        # initial_conditions = np.array([[-1.5,0,0],[1,0,-np.pi]]).T 
        # self.agents.set_poses(initial_conditions)
        # self.agents._iterations=0
        # self.states = self.agents.get_poses().T
        state = self.states.flatten()
        return state

    def render(self, mode='human'):
        return None

    def close(self):
        return None
