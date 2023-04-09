#!/usr/bin/env python

from cv_bridge import CvBridge

from PIL import Image as pilImage
#  import PIL as pil
import numpy as np
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Imu, Image


def callback(data): 
    #  rospy.loginfo(rospy.get_called_id() + " I heard %s", data.data)

    #  bridge = CvBridge()
    #  cv_image = bridge.imgmsg_to_cv2(data.data, desired_encoding='passthrough')
    #  print(PIL.__version__)
    im = np.frombuffer(data.data, dtype=np.uint8).reshape(data.height, data.width, -1)
    img = pilImage.fromarray(im, "RGB")

    img.show()

    rospy.loginfo(data)


def listener(): 

    # rgb camera
    topic_name = "/camera/rgb/image_raw"
    rospy.init_node('listener', anonymous=True)
    rospy.Subscriber(topic_name, Image, callback)

    # imu data
    #  topic_name = "/imu/data"
    #  rospy.init_node('listener', anonymous=True)
    #  rospy.Subscriber(topic_name, Imu, callback)

    rospy.spin()

if __name__ == '__main__': 
    listener()
