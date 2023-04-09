#!/usr/bin/env python

from cv_bridge import CvBridge
from PIL import Image as pilImage
import numpy as np
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Imu, Image
import time
import sys

class Listeners():

    def __init__(self):
        self.depthListener()

    def rgbListener(self): 
        # rgb camera
        topic_name = "/camera/rgb/image_raw"
        rospy.init_node('listener', anonymous=True)
        rospy.Subscriber(topic_name, Image, self.rgbCallback)
        rospy.spin()

    def rgbCallback(self, data):
        # handle rgb image
        #  im = np.frombuffer(data.data, dtype=np.uint8).reshape(data.height, data.width, -1)
        #  img = pilImage.fromarray(im, "RGB")
        #  img.show()
        #  rospy.loginfo(data)
        print("rgb callback")
        time.sleep(10)

    def depthListener(self): 
        # depth camera
        topic_name = "/camera/depth/image_raw"
        rospy.init_node('listener', anonymous=True)
        rospy.Subscriber(topic_name, Image, self.depthCallback)
        rospy.spin()


    def depthCallback(self, data):
        # handle depth image
        #  im = np.frombuffer(data.data, dtype=np.uint8).reshape(data.height, data.width, -1)
        #  print(im.shape)

        #  np.set_printoptions(threshold=sys.maxsize)
        #  get every 4th val

        #  im = np.frombuffer(data.data, dtype=np.uint8)[3::4]
        im = np.frombuffer(data.data, dtype=np.uint8)
        print(im.size)
        depthIm = im.reshape(data.height, data.width, 4)
        depthIm = depthIm[:,:,2]
        #  print(depthIm)

        print(depthIm.shape)
        img = pilImage.fromarray(depthIm, "L")
        img.show()
        time.sleep(10)

        #  bridge = CvBridge()
        #  depth_image = bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
        #  depth_array = np.array(depth_image, dtype=np.float32)
        #  im = pilImage.fromarray(depth_array, "L")
        #  im.show()
        #  time.sleep(10)


        # remove every 4th val
        #  im = np.frombuffer(data.data, dtype=np.uint8)
        #  im = np.delete(im, np.arange(3, im.size, 4))
        #  print(im.shape)

        #  depthIm = im.reshape(data.height, data.width, -1)
        #  print(depthIm.shape)
        #  if (True): 
        #      print(depthIm.shape)
        #      img = pilImage.fromarray(depthIm, "RGB")
        #      img.show()
        #      time.sleep(10)


        #  rospy.loginfo(data)
        #  print("depth callback")


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

