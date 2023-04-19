import numpy as np
import rospy
import matplotlib.pyplot as plt

from SensorStream import *

class StateEstimation():
    def __init__(self, init_state):
        self.state = init_state
        

    def deadReckon(self, state_dot, dt):
        pass

    def odomEstimation(self, odom_pose):
        self.state = odom_pose
        

if __name__ == '__main__':
    # setting for the sensor listener
    sensors_listener = SensorListener()

    # task 
    loop_hz = 60
    rate = rospy.Rate(loop_hz)

    init_state = np.zeros(3)
    se = StateEstimation(init_state)
    se_list = []
    while not rospy.is_shutdown():
        if not sensors_listener.sensor_initialzed:
            continue
        
        # ---- state estimation ----
        se.odomEstimation(sensors_listener.odom_pose)
        se_list.append(np.copy(se.state))
        rate.sleep()

    # plot the trajectory
    states = np.array(se_list)
    plt.scatter(states[:, 0], states[:, 1])
    plt.show()
    print(states.shape)
    print(states)
