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

control_mode = 0 # 0 for manual 1 for auto 

def quart_to_rpy(x,y,z,w):
    r = math.atan2(2*(w*x+y*z),1-2*(x*x+y*y))
    p = math.asin(2*(w*y-z*x))
    y = math.atan2(2*(w*z+x*y),1-2*(z*z+y*y))
    return y


class Controller():
    def __init__(self):

        self.rate = rospy.Rate(100)
        rospy.Subscriber('robot_0/pose',Float32MultiArray,self.loc_callback_0)
        rospy.Subscriber('robot_1/pose',Float32MultiArray,self.loc_callback_1)
        self.states = np.zeros(6)

        self.pub0 = rospy.Publisher('/robot_0/cmd_vel',Twist,queue_size=10)
        self.pub1 = rospy.Publisher('/robot_1/cmd_vel',Twist,queue_size=10)

        # self.listener = TransformListener()
        # self.__timer_current = rospy.Timer(rospy.Duration(0.01), self.loc)

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
    
    def check_goal(self, a_state):
        if( 0.5<=a_state[0]<=1.5 and 0.5<=a_state[1]<=1.5 ):
            return True
        else:
            return False

    def _get_done(self):
        a_state = self.states[:3]
        d_state = self.states[3:]
        distance = np.sqrt(np.sum(np.square(a_state[:2] - d_state[:2])))
        # print(distance)
        if (distance<0.10 or self.check_goal(a_state)):
            done_n = True
        else:
            done_n = False
        if (a_state[1]>1.4 or a_state[1]<-1.4 or a_state[0]<-1.4 or a_state[0]>1.4 ):
            done_n = True
        if (d_state[1]>1.4 or d_state[1]<-1.4 or d_state[0]<-1.4 or d_state[0]>1.4 ):
            done_n = True
        return done_n

    def loc_callback_0(self,msg):
        for i in range(3):
            self.states[i] = msg.data[i]

    def loc_callback_1(self,msg):
        for i in range(3):
            self.states[i+3] = msg.data[i]

    def cmd_0(self,data):
        self.control_cmd0.linear.x = data[0]
        self.control_cmd0.angular.z = data[1]
        self.pub0.publish(self.control_cmd0)

    def cmd_1(self,data):
        self.control_cmd1.linear.x = data[0]
        self.control_cmd1.angular.z = data[1]
        self.pub1.publish(self.control_cmd1)

    def auto(self):
        record = []
        while not rospy.is_shutdown():
            input_a = torch.FloatTensor(self.states.flatten()).to("cuda").unsqueeze(0)
            dxu0 = self.attacker(input_a).detach().cpu().numpy()[0]
            input_d = torch.FloatTensor(self.states.flatten()).to("cuda").unsqueeze(0)
            dxu1 = self.defender(input_d).detach().cpu().numpy()[0]
            # dxu0 = MPC_controller(self.states[:3],self.states[3:])
            # dxu1 = Fast_controller_defence(self.states[:3],self.states[3:])
            # dxu1 = Fast_Catch(self.states[:3],self.states[3:])
            a = np.copy(self.states)
            record.append(a)
            # self.cmd_0(dxu0)
            # self.cmd_1(dxu1)
            if(self._get_done()==True):
                dxu1=np.array([ 0.0, 0.0])
                dxu0=np.array([ 0.0, 0.0])
                print("stop!")
            self.cmd_0(dxu0)
            self.cmd_1(dxu1)
            np.save("record.npy",record)
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
        record = []
        data = np.array([ 0.0, 0.0])
        data_2 = np.array([ 0.0, 0.0])
        while not rospy.is_shutdown():
            print(data,data_2)
            # print(record)
            a = np.copy(self.states)
            record.append(a)
            
            key = self.getKey()
            if key == 'w':
                if(data[0]< 0.5):
                    data[0] = data[0] + 0.1
                else:
                    data = data
            elif key == 'x':
                if(data[0]> -0.5):
                    data[0] = data[0] - 0.1
                else: 
                    data = data
            elif key == 'a':
                if(data[1]< 2):
                    data[1] += 0.5
                else:
                    data = data
            elif key == 'd':
                if(data[1]> -2):
                    data[1] -= 0.5
                else:
                    data = data
            elif key == 's':
                data = np.array([ 0.0,   0.0])
            elif (key == '\x03'):
                break
            else:
                data = data

            # if(self._get_done()==True):
            #     data=np.array([ 0.0, 0.0])
            #     print("stop!")
            np.save("record.npy",record)

            

            self.cmd_0(data)
            self.rate.sleep()
        
if __name__=='__main__':
    rospy.init_node('control')
    controller = Controller()

	# rospy.spin()
    