#!/usr/bin/env python
import numpy as np 
import sys
import rospy
import math
import matplotlib.pyplot as plt
import sys, select, os

from gym import spaces, core
import torch
from model import DeterministicPolicy
from controller import CLF_CBF_controller,MPC_controller, MPC_controller_defence,Fast_controller_defence,Fast_Catch

if os.name == 'nt':
    import msvcrt
else:
    import tty, termios
if os.name != 'nt':
    settings = termios.tcgetattr(sys.stdin)
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Twist
from nav_msgs.msg import Path
from std_msgs.msg import Float32MultiArray,Float32,Int16
from scipy.interpolate import interp1d

control_mode = 1 # 0 for manual 1 for auto 

def quart_to_rpy(x,y,z,w):
    r = math.atan2(2*(w*x+y*z),1-2*(x*x+y*y))
    p = math.asin(2*(w*y-z*x))
    y = math.atan2(2*(w*z+x*y),1-2*(z*z+y*y))
    return y


class Controller():
    def __init__(self):

        self.rate = rospy.Rate(100)
        rospy.Subscriber('gazebo/model_states',ModelStates,self.loc_callback)

        self.states = np.zeros(6)

        self.pub0 = rospy.Publisher('/tb3_0/cmd_vel',Twist,queue_size=10)
        self.pub1 = rospy.Publisher('/tb3_1/cmd_vel',Twist,queue_size=10)
        
        self.control_cmd0 = Twist()
        self.control_cmd1 = Twist()
        
        self.u_range = np.array([0.5,2.0])
        self.action_space = spaces.Box(low=-self.u_range, high=+self.u_range, shape=(2,), dtype=np.float32)
        self.attacker = DeterministicPolicy(6, 2, 256, self.action_space).to('cuda')
        self.attacker.load_state_dict(torch.load("pretrain"))
        self.defender = DeterministicPolicy(6, 2, 256, self.action_space).to('cuda')
        self.defender.load_state_dict(torch.load("d_actor"))
        if( control_mode == 1):
            self.auto()
        else:
            self.manual()
        
    def loc_callback(self,msg):
        self.states[0] = msg.pose[2].position.x
        self.states[1] = msg.pose[2].position.y
        yaw1 = quart_to_rpy(msg.pose[2].orientation.x, msg.pose[2].orientation.y,msg.pose[2].orientation.z,msg.pose[2].orientation.w)
        self.states[2] = yaw1
        self.states[3] = msg.pose[3].position.x
        self.states[4] = msg.pose[3].position.y
        yaw2 = quart_to_rpy(msg.pose[3].orientation.x, msg.pose[3].orientation.y,msg.pose[3].orientation.z,msg.pose[3].orientation.w)

        self.states[5] = yaw2


    def cmd_0(self,data):
        self.control_cmd0.linear.x = data[0]
        self.control_cmd0.angular.z = data[1]
        self.pub0.publish(self.control_cmd0)

    def cmd_1(self,data):
        self.control_cmd1.linear.x = data[0]
        self.control_cmd1.angular.z = data[1]
        self.pub1.publish(self.control_cmd1)

    def auto(self):
        while not rospy.is_shutdown():
            # input_a = torch.FloatTensor(self.states.flatten()).to("cuda").unsqueeze(0)
            # dxu0 = self.attacker(input_a).detach().cpu().numpy()[0]
            input_d = torch.FloatTensor(self.states.flatten()).to("cuda").unsqueeze(0)
            dxu1 = self.defender(input_d).detach().cpu().numpy()[0]
            dxu0 = MPC_controller(self.states[:3],self.states[3:])
            # dxu1 = Fast_controller_defence(self.states[:3],self.states[3:])

            self.cmd_0(dxu0)
            self.cmd_1(dxu1)
            
            # print(dxu0)

            self.rate.sleep()


    def getKey(self):
        if os.name == 'nt':
            return msvcrt.getch()

        tty.setraw(sys.stdin.fileno())
        rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
        if rlist:
            key = sys.stdin.read(1)
        else:
            key = ''

        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        return key

    def manual(self):
        
        data = np.array([ 0.0, 0.0])
        while not rospy.is_shutdown():
            print(self.states)
            key = self.getKey()
            if key == 'w':
                if(data[0]< 0.5):
                    data[0] = data[0] + 0.05
                else:
                    data = data
            elif key == 'x':
                if(data[0]> -0.5):
                    data[0] = data[0] - 0.05
                else: 
                    data = data
            elif key == 'a':
                if(data[1]< 2):
                    data[1] += 0.2
                else:
                    data = data
            elif key == 'd':
                if(data[1]> -2):
                    data[1] -= 0.2
                else:
                    data = data
            elif key == 's':
                data = np.array([ 0.0,   0.0])
            elif (key == '\x03'):
                break
            else:
                data = data

            self.cmd_0(data)
            self.rate.sleep()
        
if __name__=='__main__':
    rospy.init_node('control')
    controller = Controller()

	# rospy.spin()
    