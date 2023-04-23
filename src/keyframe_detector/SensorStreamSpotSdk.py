# Copyright (c) 2022 Boston Dynamics, Inc.  All rights reserved.
#
# Downloading, reproducing, distributing or otherwise using the SDK Software
# is subject to the terms and conditions of the Boston Dynamics Software
# Development Kit License (20191101-BDSDK-SL).

"""Simple image display example."""

import argparse
import logging
import sys
import time
from multiprocessing import Barrier, Process, Queue, Value
from threading import BrokenBarrierError, Thread

import cv2
import numpy as np
from scipy import ndimage

import bosdyn.client
import bosdyn.client.util
from bosdyn.api import image_pb2
from bosdyn.client.image import ImageClient, build_image_request
from bosdyn.client.time_sync import TimedOutError


_LOGGER = logging.getLogger(__name__)


class SensorListener():

    def __init__(self, robot, image_task, time_delay):
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
        self.time_delay = time_delay # substitute for time between capture

        # start listener
        self.initListener(self.image_task)

    def initListener(): 

        # Start image capture process
        image_capture_thread = Process(target=capture_images,
                args=(self.image_task, self.time_delay),
                daemon=True)
        image_capture_thread.start()

        self.sensor_initialzed = True

    def capture_images(image_task, sleep_between_capture): 
        get_im_resp = image_task.proto
        start_time = time.time()
        if not get_im_resp:
            continue

        for im_resp in get_im_resp: 
            image, _ = image_to_opencv(images[i], AUTO_ROTATE)
            acquisition_time = im_resp.shot.acquisition_time
            image_time = acquisition_time.seconds + acquisition_time.nanos * 1e-9

            if img.source.image_type == image_pb2.ImageSource.IMAGE_TYPE_DEPTH: 
                self.depth_image = image
            else: 
                self.rgb_image = image
            self.time = image_time

        #  if (self.rgb_image is not None) and \
        #      (self.depth_image is not None) and \
        #          (self.odom_linear is not None):
        #          self.sensor_initialzed = True

        if (self.rgb_image is not None) and \
                (self.depth_image is not None): 
                    self.sensor_initialzed = True

        time.sleep(sleep_between_capture)


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

    def reset_image_client(self):
        """Recreate the ImageClient from the robot object."""
        del self.robot.service_clients_by_name['image']
        del self.robot.channels_by_authority['api.spot.robot']
        return self.robot.ensure_client('image')

        

#  VALUE_FOR_Q_KEYSTROKE = 113
#  VALUE_FOR_ESC_KEYSTROKE = 27

#  ROTATION_ANGLE = {
#      'back_fisheye_image': 0,
#      'frontleft_fisheye_image': -78,
#      'frontright_fisheye_image': -102,
#      'left_fisheye_image': 0,
#      'right_fisheye_image': 180
#  }




#  def main(argv):


#      image_client = robot.ensure_client(IMAGE_SERVICE)
#      requests = [
#          build_image_request(source, quality_percent=JPEG_QUALITY_PERCENT)
#          for source in self.image_sources
#      ]

#      for image_source in options.image_sources:
#          cv2.namedWindow(image_source, cv2.WINDOW_NORMAL)
#          if len(options.image_sources) > 1 or DISABLE_FULL_SCREEN:
#              cv2.setWindowProperty(image_source, cv2.WND_PROP_AUTOSIZE, cv2.WINDOW_AUTOSIZE)
#          else:
#              cv2.setWindowProperty(image_source, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

#      keystroke = None
#      timeout_count_before_reset = 0
#      while keystroke != VALUE_FOR_Q_KEYSTROKE and keystroke != VALUE_FOR_ESC_KEYSTROKE:
#          try:
#              images_future = image_client.get_image_async(requests, timeout=0.5)
#              while not images_future.done():
#                  keystroke = cv2.waitKey(25)
#                  print(keystroke)
#                  if keystroke == VALUE_FOR_ESC_KEYSTROKE or keystroke == VALUE_FOR_Q_KEYSTROKE:
#                      sys.exit(1)
#              images = images_future.result()
#          except TimedOutError as time_err:
#              if timeout_count_before_reset == 5:
#                  # To attempt to handle bad comms and continue the live image stream, try recreating the
#                  # image client after having an RPC timeout 5 times.
#                  _LOGGER.info("Resetting image client after 5+ timeout errors.")
#                  image_client = reset_image_client()
#                  timeout_count_before_reset = 0
#              else:
#                  timeout_count_before_reset += 1
#          except Exception as err:
#              _LOGGER.warning(err)
#              continue
#          for i in range(len(images)):
#              image, _ = image_to_opencv(images[i], AUTO_ROTATE)
#              cv2.imshow(images[i].source.name, image)
#          keystroke = cv2.waitKey(CAPTURE_DELAY)


#  if __name__ == "__main__":
#      if not main(sys.argv[1:]):
#          sys.exit(1)

