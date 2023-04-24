# Copyright (c) 2022 Boston Dynamics, Inc.  All rights reserved.
#
# Downloading, reproducing, distributing or otherwise using the SDK Software
# is subject to the terms and conditions of the Boston Dynamics Software
# Development Kit License (20191101-BDSDK-SL).

"""Simple image display example."""

import argparse
import rospy
import logging
import sys
import time
#  from multiprocessing import Barrier, Process, Queue, Value
from threading import BrokenBarrierError, Thread

import cv2
import numpy as np
from scipy import ndimage

import bosdyn.client
import bosdyn.client.util
from bosdyn.api import image_pb2
from bosdyn.client.image import ImageClient, build_image_request
from bosdyn.client.time_sync import TimedOutError
from bosdyn.client.frame_helpers import VISION_FRAME_NAME, get_vision_tform_body


_LOGGER = logging.getLogger(__name__)


class SensorListener():

    def __init__(self, robot, image_task, robot_state_task, time_delay):
        # the video stream
        self.rgb_image = None
        self.depth_image = None
        self.odom_pose = None
        self.time = None

        self.sensor_initialzed = False

        self.JPEG_QUALITY_PERCENT = 50 # 0~100
        self.CAPTURE_DELAY = 100 # time [ms] to wait before next capture
        self.IMAGE_SERVICE = ImageClient.default_service_name
        self.DISABLE_FULL_SCREEN = True
        self.AUTO_ROTATE = True

        self.robot = robot
        self.image_task = image_task
        self.robot_state_task = robot_state_task
        self.time_delay = time_delay # substitute for time between capture

        # start listener
        self.initListener()

    def initListener(self): 
        rospy.init_node('sensors')
        # Start image capture process
        image_capture_thread = Thread(target=self.capture_images, daemon=True)
        #  image_capture_thread = Process(target=self.capture_images,
                # daemon=True)
        image_capture_thread.start()

        # Start odom capture process
        #  state_capture_thread = Process(target=self.capture_state,
        state_capture_thread = Thread(target=self.capture_state,
                daemon=True)
        state_capture_thread.start()

    def capture_state(self): 
        while True:
            # print("in the state capture")
            robot_state_resp = self.robot_state_task.proto
            # start_time = time.time()

            if not robot_state_resp: 
                return

            vo_tform_robot = get_vision_tform_body(robot_state_resp.kinematic_state.transforms_snapshot)

            self.odom_pose = vo_tform_robot
            # print("odom print: ", self.odom_pose)
            acquisition_time = robot_state_resp.kinematic_state.acquisition_timestamp
            self.time = acquisition_time.seconds + acquisition_time.nanos * 1e-9

            # print("time in state: ", self.time)

            if (self.rgb_image is not None) and \
                (self.depth_image is not None) and \
                    (self.odom_pose is not None):
                    self.sensor_initialzed = True
            time.sleep(0.01)
            # time.sleep(sleep_between_capture)

    def capture_images(self): 
        while True:
            # print("in the image capture")
            get_im_resp = self.image_task.proto
            start_time = time.time()
            if not get_im_resp:
                return

            for im_resp in get_im_resp: 
                image, _ = self.image_to_opencv(im_resp)
                acquisition_time = im_resp.shot.acquisition_time
                image_time = acquisition_time.seconds + acquisition_time.nanos * 1e-9

                if im_resp.source.image_type == image_pb2.ImageSource.IMAGE_TYPE_DEPTH: 
                    self.depth_image = image
                else: 
                    self.rgb_image = image
                self.time = image_time


            if (self.rgb_image is not None) and \
                (self.depth_image is not None) and \
                    (self.odom_pose is not None):
                    self.sensor_initialzed = True
            # time.sleep(sleep_between_capture)
            time.sleep(0.01)



    def image_to_opencv(self, image, auto_rotate=True):
        """Convert an image proto message to an openCV image."""
        num_channels = 1  # Assume a default of 1 byte encodings.
        if image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_DEPTH_U16:
            dtype = np.uint16
            extension = ".png"
        else:
            dtype = np.uint8
            if image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_RGB_U8:
                num_channels = 3
            elif image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_RGBA_U8:
                num_channels = 4
            elif image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_GREYSCALE_U8:
                num_channels = 1
            elif image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_GREYSCALE_U16:
                num_channels = 1
                dtype = np.uint16
            extension = ".jpg"

        img = np.frombuffer(image.shot.image.data, dtype=dtype)
        if image.shot.image.format == image_pb2.Image.FORMAT_RAW:
            try:
                # Attempt to reshape array into a RGB rows X cols shape.
                img = img.reshape((image.shot.image.rows, image.shot.image.cols, num_channels))
            except ValueError:
                # Unable to reshape the image data, trying a regular decode.
                img = cv2.imdecode(img, -1)
        else:
            img = cv2.imdecode(img, -1)

        #  if auto_rotate:
        #      img = ndimage.rotate(img, ROTATION_ANGLE[image.source.name])

        return img, extension

