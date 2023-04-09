#!/usr/bin/env python

from PIL import Image as pilImage
import numpy as np
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Imu, Image

class Listeners():

    def __init__(self):
        self.initListener()

    def initListener(self): 
        # rgb camera
        topic_name = "/camera/rgb/image_raw"
        rospy.init_node('listener', anonymous=True)
        rospy.Subscriber(topic_name, Image, self.rgbCallback)

        # depth camera
        topic_name = "/camera/depth/image_raw"
        rospy.init_node('listener2', anonymous=True)
        rospy.Subscriber(topic_name, Image, self.depthCallback)

        rospy.spin()

    def rgbCallback(self, data):
        # handle rgb image
        #  im = np.frombuffer(data.data, dtype=np.uint8).reshape(data.height, data.width, -1)
        #  img = pilImage.fromarray(im, "RGB")
        #  img.show()
        #  rospy.loginfo(data)
        print("rgb callback")

    def depthCallback(self, data):
        # handle depth image
        #  im = np.frombuffer(data.data, dtype=np.uint8).reshape(data.height, data.width, -1)
        #  print(im.shape)
        #  img = pilImage.fromarray(im, "L")
        #  img.show()
        #  rospy.loginfo(data)
        print("depth callback")


#  def callback(data): 

    # handle rgb image
    #  im = np.frombuffer(data.data, dtype=np.uint8).reshape(data.height, data.width, -1)
    #  img = pilImage.fromarray(im, "RGB")
    #  img.show()
    #  rospy.loginfo(data)

    # handle depth image
    #  im = np.frombuffer(data.data, dtype=np.uint8).reshape(data.height, data.width, -1)
    #  print(im.shape)
    #  img = pilImage.fromarray(im, "L")
    #  img.show()
    #  rospy.loginfo(data)

#  def listener(): 

    # rgb camera
    #  topic_name = "/camera/rgb/image_raw"
    #  rospy.init_node('listener', anonymous=True)
    #  rospy.Subscriber(topic_name, Image, callback)

    # depth camera
    #  topic_name = "/camera/depth/image_raw"
    #  rospy.init_node('listener', anonymous=True)
    #  rospy.Subscriber(topic_name, Image, callback)

    # imu data
    #  topic_name = "/imu/data"
    #  rospy.init_node('listener', anonymous=True)
    #  rospy.Subscriber(topic_name, Imu, callback)

    #  rospy.spin()

if __name__ == '__main__': 
    l = Listeners()
    #  listener()
    #  print("hello")

