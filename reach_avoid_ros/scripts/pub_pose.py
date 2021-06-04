import numpy as np 
import sys
import rospy
import math
import matplotlib.pyplot as plt
import sys, select, os

import tf

from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Twist
from nav_msgs.msg import Path
from std_msgs.msg import Float32MultiArray,Float32,Int16


control_mode = 0 # 0 for manual 1 for auto 

def quart_to_rpy(x,y,z,w):
    r = math.atan2(2*(w*x+y*z),1-2*(x*x+y*y))
    p = math.asin(2*(w*y-z*x))
    y = math.atan2(2*(w*z+x*y),1-2*(z*z+y*y))
    return y


class Controller():
    def __init__(self):
        self.rate = rospy.Rate(100)
        # self.rate = rospy.Rate(100)
        self.states = np.zeros(6)
        self.pub0 = rospy.Publisher('/robot_0/pose',Float32MultiArray,queue_size=10)
        self.pub1 = rospy.Publisher('/robot_1/pose',Float32MultiArray,queue_size=10)
        
        self.listener = tf.TransformListener()
        # self.__timer_current = rospy.Timer(rospy.Duration(0.01), self.get_current_state)
        rospy.sleep(1)

        while not rospy.is_shutdown():
            self.get_current_state()

    def get_current_state(self):
        try:
            data = np.array([ 0.0, 0.0,0.0])
            data1 = np.array([ 0.0, 0.0,0.0])
            (trans,rot) = self.listener.lookupTransform("map","robot_0/base_footprint", rospy.Time(0))
            (trans1,rot1) = self.listener.lookupTransform("map","robot_1/base_footprint", rospy.Time(0))
            data[0]=trans[0]
            data[1]=trans[1]
            data[2]=quart_to_rpy(rot[0],rot[1],rot[2],rot[3])
            data1[0]=trans1[0]
            data1[1]=trans1[1]
            data1[2]=quart_to_rpy(rot1[0],rot1[1],rot1[2],rot1[3])
            print(data,data1)
            self.cmd_0(data)
            self.cmd_1(data1)
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            pass
        

        

    def cmd_0(self,data):
        pose0 = Float32MultiArray()
        for j in range(3):
            pose0.data.append(data[j])
        self.pub0.publish(pose0)

    def cmd_1(self,data):
        pose0 = Float32MultiArray()
        for j in range(3):
            pose0.data.append(data[j])
        self.pub1.publish(pose0)
        
if __name__=='__main__':
    rospy.init_node('lala')
    controller = Controller()

	# rospy.spin()
    