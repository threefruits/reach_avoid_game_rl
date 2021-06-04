import torch
import os
import numpy as np
from controller import CLF_CBF_controller,MPC_controller, MPC_controller_defence,Fast_controller_defence,Fast_Catch
from ra_env import ReachAvoidAgent
from sac import SAC
import argparse

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env-name', default="HalfCheetah-v2",
                    help='Mujoco Gym environment (default: HalfCheetah-v2)')
parser.add_argument('--policy', default="Deterministic",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy a policy every 10 episode (default: True)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                    help='Automaically adjust α (default: False)')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=1000001, metavar='N',
                    help='maximum number of steps (default: 1000000)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=1000, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
parser.add_argument('--cuda', action="store_true",default=True,
                    help='run on CUDA (default: False)')
args = parser.parse_args()

if __name__ == '__main__':
    env = ReachAvoidAgent()
    i =0 
    state = env.reset()
    record = []
    
    # agent = SAC(env.observation_space.shape[0], env.action_space, args)
    # agent.load_model("models/pretrain",None)
    # agent.load_model("models/save2/mpc_actor","models/save2/mpc_critic")

    p = np.random.rand()
    k=0
    if (p>0.4):
        k=1
    elif (0.4<=p<0.8):
        k=2
    else:
        k=3
    while True:
        # print(env.action_space.shape)
        # act_n = []
        # act_n = np.array([0,0])
        act_n = MPC_controller(state[:3],state[3:])
        # act_n = env.action_space.sample()
        # act_n = agent.select_action(state, evaluate=True)

        record.append( np.append(state,np.array([act_n[0],act_n[1]])) )

        
        obs_n, reward_n, done_n, _ = env.step(act_n,k)
        state = obs_n
        i=i+1
        # np.save("record.npy",record)

        if(done_n == True):
            print(env.times)
            env.reset(True)
            p = np.random.rand()
            k=0
            if (p>0.4):
                k=2
            elif (0.4<=p<0.8):
                k=2
            else:
                k=3
            
            
        # print(done_n)
        # env.render()
        
