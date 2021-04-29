import rps.robotarium as robotarium
from rps.utilities.transformations import *
from rps.utilities.barrier_certificates import *
from rps.utilities.misc import *
from rps.utilities.controllers import *
from controller import CLF_CBF_controller,MPC_controller, MPC_controller_defence, Fast_controller_defence, Fast_Catch
import numpy as np
import time
import argparse
from sac import SAC
from gym import spaces
from ra_env import ReachAvoidAgent

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env-name', default="HalfCheetah-v2",
                    help='Mujoco Gym environment (default: HalfCheetah-v2)')
parser.add_argument('--policy', default="Gaussian",
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

env = ReachAvoidAgent()
agent = SAC(env.observation_space.shape[0], env.action_space, args)
agent.load_model("models/sac_actor_ra_35.0","models/sac_critic_ra_35.0")


def compute_a(states):
    state = states
    return np.array([1,0])


## notice theta [-pi,pi]
def compute_d(states):
    u = np.zeros(2)
    u[0] = 0
    u[1] = 0
    # print(states[1,2])
    return u

# Instantiate Robotarium object
N = 2
# initial_conditions = np.array(np.mat('1 0.5; 0.8 -0.3 ; 0 0 '))
# initial_conditions = np.array([[-1,0,0],[0.5,0,-np.pi]]).T 



# a = np.array([-1 + np.random.rand() * 0.4 - 0.2, np.random.rand() * 0.8 - 0.4 ,0])
# d = np.array([0.5 + np.random.rand() * 0.4 - 0.2 ,np.random.rand() * 0.4 - 0.2 , np.pi + np.random.rand() * 0.4 - 0.2 ])
a = np.array([-1.2, -0.2 ,-0.2])
d = np.array([0.4, 0.15, -3.14 ])
initial_conditions = np.array([a,d]).T 
r = robotarium.Robotarium(number_of_robots=N, show_figure=True, initial_conditions=initial_conditions,sim_in_real_time=False)

# Define goal points by removing orientation from poses
goal_points = np.array(np.mat('0.5 1; -0.3 0.8 ; -0.6 0 '))

# generate_initial_conditions(N)

# Create unicycle pose controller
unicycle_pose_controller = create_clf_unicycle_pose_controller()

# Create barrier certificates to avoid collision
uni_barrier_cert = create_unicycle_barrier_certificate()

# define x initially
x = r.get_poses().T
r.step()

record = np.zeros([400,6])
# np.save("filename.npy",a)
# b = np.load("filename.npy")
i=0
# While the number of robots at the required poses is less
# than N...
while (np.size(at_pose(x.T, goal_points)) != N):

    # Get poses of agents
    x = r.get_poses().T
    record[i] = x.flatten()
    # print(x)
    # Create unicycle control inputs
    # dxu = unicycle_pose_controller(x.T, goal_points)
    # print(dxu)
    # # Create safe control inputs (i.e., no collisions)
    # dxu = uni_barrier_cert(dxu, x.T)
    
    u = MPC_controller(x[0],x[1])
    # u = agent.select_action(x.flatten(), evaluate=True)
    # u = CLF_CBF_controller(x[0],x[1])
    # u2 = MPC_controller_defence(x[0],x[1])
    # u2 = Fast_Catch(x[0],x[1])
    u2 = Fast_controller_defence(x[0],x[1])
    
    dxu = np.zeros([2,2])
    dxu[0] = np.array([u[0],u[1]])
    dxu[1] = np.array([u2[0],u2[1]])
    # # Set the velocities
    r.set_velocities(np.arange(N), dxu.T)
    # r.set_velocities(np.array([0,1]), np.array([[0.1,-0.1],[0.1,-0.1]]) )
    # Iterate the simulation
    i+=1
    if(i<390):
        np.save("record.npy",record)
    r.step()

#Call at end of script to print debug information and for your script to run on the Robotarium server properly
r.call_at_scripts_end()
