#!/usr/bin/env python

from PIL import Image as pilImage
import numpy as np
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Imu, Image
from nav_msgs.msg import Odometry
import cv2
import matplotlib.pyplot as plt

class SensorListener():

    def __init__(self):

        self.rgb_topic_str = "/camera/rgb/image_raw"
        self.depth_topic_str = "/camera/depth/image_raw"
        self.odom_topic_str = "/odom"

        # the video stream
        self.rgb_image = None
        self.depth_image = None
        self.odom_linear = None
        self.odom_angular = None
        self.odom_pose = None
        self.time = None

        #
        self.sensor_initialzed = False
        self.in_callback = False

        # start listener
        self.initListener()

    def initListener(self): 
        print("initializing the sensor listener node...")
        # rgb camera
        rospy.init_node('sensors')
        rospy.Subscriber(self.rgb_topic_str, Image, self.rgbCallback)
        rospy.Subscriber(self.depth_topic_str, Image, self.depthCallback)
        rospy.Subscriber(self.odom_topic_str, Odometry, self.odomCallback)

    def updateTime(self, msg):
        self.last_time_secs = msg.header.stamp.secs
        self.last_time_nsecs = msg.header.stamp.nsecs
        self.last_state_transition_time = (self.last_time_secs + self.last_time_nsecs*1e-9)

    def rgbCallback(self, data):
        # check if the callback is lock by others.
        if self.in_callback:
            return # if the listener is locked by other callback, skip this one.
        else:
            self.in_callback = True # lock the rgb sensor callback
        
        # convert the image
        im = np.frombuffer(data.data, dtype=np.uint8).reshape(data.height, data.width, -1)
        img = pilImage.fromarray(im, "RGB")
        
        # record the lastest image
        self.rgb_image = np.array(img)

        # update the time line
        self.updateTime(data)

        # release the callback
        self.in_callback = False

        # sensor initialized
        if (self.rgb_image is not None) and \
            (self.depth_image is not None) and \
                (self.odom_linear is not None):
                self.sensor_initialzed = True
        # release the callback
        self.in_callback = False

    def depthCallback(self, data):
        # check if the callback is lock by others.
        if self.in_callback:
            return # if the listener is locked by other callback, skip this one.
        else:
            self.in_callback = True # lock the rgb sensor callback

        # convert the data to depth image
        im = np.frombuffer(data.data, dtype=np.uint8)
        depthIm = im.reshape(data.height, data.width, 4)
        depthIm = depthIm[:,:,2]
        img = pilImage.fromarray(depthIm, "L")

        # record the lastest depth image
        self.depth_image = np.asanyarray(img)[:, :, np.newaxis]
        # self.depth_image = depthIm[:, :, np.newaxis]
        # update the time line
        self.updateTime(data)

        # sensor initialized
        if (self.rgb_image is not None) and \
            (self.depth_image is not None) and \
                (self.odom_linear is not None):
                self.sensor_initialzed = True
        # release the callback
        self.in_callback = False

    def odomCallback(self, data):
        # check if the callback is lock by others.
        if self.in_callback:
            return # if the listener is locked by other callback, skip this one.
        else:
            self.in_callback = True # lock the rgb sensor callback

        # log the odom linear
        self.odom_linear = np.asanyarray([data.twist.twist.linear.x, 
                                          data.twist.twist.linear.y, 
                                          data.twist.twist.linear.z])
        self.odom_angular = np.asanyarray([data.twist.twist.angular.x, 
                                           data.twist.twist.angular.y, 
                                           data.twist.twist.angular.z])

        self.odom_pose = np.asanyarray([data.pose.pose.position.x, 
                                        data.pose.pose.position.y, 
                                        data.pose.pose.position.z])

        # update the time line
        self.updateTime(data)

        # sensor initialized
        if (self.rgb_image is not None) and \
            (self.depth_image is not None) and \
                (self.odom_linear is not None):
                self.sensor_initialzed = True

        # release the callback
        self.in_callback = False

if __name__ == '__main__':
    # setting for the sensor listener
    sensors_listener = SensorListener()

    # task 
    loop_hz = 60
    rate = rospy.Rate(loop_hz)
    
    while not rospy.is_shutdown():
        if not sensors_listener.sensor_initialzed:
            continue

        # ---- Show images ------
        cv2.namedWindow('RGB', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RGB', sensors_listener.rgb_image)
        cv2.namedWindow('Depth', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Depth', sensors_listener.depth_image)
        Key = cv2.waitKey(1)
        if Key == 27:
            break

        # ----- state estimation -------


        rate.sleep()


