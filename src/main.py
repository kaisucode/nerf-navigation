import bosdyn.client
import bosdyn.client.util
from bosdyn.api import basic_command_pb2
from bosdyn.api import geometry_pb2 as geo
from bosdyn.api.basic_command_pb2 import RobotCommandFeedbackStatus
from bosdyn.client import math_helpers
from bosdyn.client.frame_helpers import (BODY_FRAME_NAME, ODOM_FRAME_NAME, VISION_FRAME_NAME,
                                         get_se2_a_tform_b)
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
from bosdyn.client.robot_command import (RobotCommandBuilder, RobotCommandClient,
                                         block_for_trajectory_cmd, blocking_stand)
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.async_tasks import AsyncPeriodicQuery, AsyncTasks
from bosdyn.client.image import ImageClient, build_image_request

from bosdyn.api import gripper_camera_param_pb2, header_pb2
from bosdyn.client.gripper_camera_param import GripperCameraParamClient


import logging
from keyframe_detector.KeyframeDetection import *
import rospy

import cv2
from keyframe_detector.SensorStreamSpotSdk import * 
import time

HOSTNAME = "gouger.rlab.cs.brown.edu"
image_sources = ["hand_color_image", "hand_depth_in_hand_color_frame"] # sources for depth and rgb image
sensor_time_delay = 0/60.0
LOGGER = logging.getLogger(__name__)

def _update_thread(async_task):
    while True:
        async_task.update()
        time.sleep(0.01)

class AsyncImage(AsyncPeriodicQuery):
    """Grab image."""

    def __init__(self, image_client, image_sources, loop_rate):
        # Period is set to be about 15 FPS
        super(AsyncImage, self).__init__('images', image_client, LOGGER, period_sec=1/loop_rate)
        self.image_sources = image_sources

    def _start_query(self):
        return self._client.get_image_from_sources_async(self.image_sources)

class AsyncRobotState(AsyncPeriodicQuery):
    """Grab robot state."""

    def __init__(self, robot_state_client, loop_rate):
        # period is set to be about the same rate as detections on the CORE AI
        super(AsyncRobotState, self).__init__('robot_state', robot_state_client, LOGGER,
                                              period_sec=1/loop_rate)

    def _start_query(self):
        return self._client.get_robot_state_async()

def main(argv): 
    # define the loop rate
    loop_rate = 60

    #####################################
    # Create robot object.
    #####################################
    sdk = bosdyn.client.create_standard_sdk('RobotCommandMaster')
    robot = sdk.create_robot(HOSTNAME)
    bosdyn.client.util.authenticate(robot)

    # Check that an estop is connected with the robot so that the robot commands can be executed.
    assert not robot.is_estopped(), "Robot is estopped. Please use an external E-Stop client, " \
                                    "such as the estop SDK example, to configure E-Stop."

    # Create the lease client.
    lease_client = robot.ensure_client(LeaseClient.default_service_name)

    # camera param
    # gripper_camera_param_client = robot.ensure_client(GripperCameraParamClient.default_service_name)
    # camera_mode = gripper_camera_param_pb2.GripperCameraParams.MODE_640_480_120FPS_UYVY
    # params = gripper_camera_param_pb2.GripperCameraParams(camera_mode=camera_mode)
    # request = gripper_camera_param_pb2.GripperCameraParamRequest(params=params)
    # response = gripper_camera_param_client.set_camera_params(request)
    # if response.header.error and response.header.error.code != header_pb2.CommonError.CODE_OK:
    #     print('Got an error:')
    #     print(response.header.error)
    
    # Setup clients for the robot
    # image client
    robot_image_client = robot.ensure_client(ImageClient.default_service_name)
    # motion client
    robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)
    robot_command_client = robot.ensure_client(RobotCommandClient.default_service_name)

    # async image 
    print("begin tasks query")
    image_task = AsyncImage(robot_image_client, image_sources, loop_rate)
    robot_state_task = AsyncRobotState(robot_state_client, loop_rate)
    task_list = [image_task, robot_state_task]
    _async_tasks = AsyncTasks(task_list)

    update_thread = Thread(target=_update_thread, args=[_async_tasks])
    update_thread.daemon = True
    update_thread.start()

    # Wait for the first responses.
    while any(task.proto is None for task in task_list):
        time.sleep(0.1)
    print("sensor ready")


    ## pass image_task for SensorListener to access
    sensors_listener = SensorListener(robot, image_task, robot_state_task, sensor_time_delay)
    print("listener opened")


    # ------------------- Start Task ------------------------
    # init the keyframe detector
    kfd = KeyframeDetector()

    # task settings
    loop_hz = 60
    rate = rospy.Rate(loop_hz)

    rgb_data = []
    depth_data = []
    odom_position_data = []
    odom_orientation_data = []
    timstamp_data = []

    # task begins
    frame_id = 0
    while not rospy.is_shutdown():
        if not sensors_listener.sensor_initialzed:
            continue
        # ---- keyframe detection ----
        rgb = sensors_listener.rgb_image
        depth = sensors_listener.depth_image
        odom = sensors_listener.odom_pose
        timstamp = sensors_listener.time

        # bebug
        # print("odom position" , odom.position.x)
        # print("odom position datatype" , type(odom.position.x))
        # print("odom rotation" , odom.rotation.w)
        # print("odom rotation datatype" , type(odom.rotation.w))

        # print(timstamp)

        success = kfd.run(rgb, frame_id)

        if success:
            rgb_data.append(rgb)
            depth_data.append(depth)
            odom_position_data.append(
                np.array([odom.position.x, odom.position.y, odom.position.z])
                )
            odom_orientation_data.append(
                np.array([odom.rotation.w, odom.rotation.x, odom.rotation.y, odom.rotation.z])
                )
            timstamp_data.append(timstamp)


        print("length of the keyframe buffer", kfd.num_keyframe)
        frame_id += 1
    
    
    # write the data
    all_rgb_data = np.array(rgb_data)
    all_depth_data = np.array(depth_data)
    all_odom_position_data = np.array(odom_position_data)
    all_odom_orientation_data = np.array(odom_orientation_data)
    all_timestamp_data = np.array(timstamp_data)


    print("rgb shape: ", all_rgb_data.shape)
    print("depth shape: ", all_depth_data.shape)
    print("odom position shape: ", all_odom_position_data.shape)
    print("odom orientation shape: ", all_odom_orientation_data.shape)
    print("timestamp shape: ", all_timestamp_data.shape)
    np.savez("./spot_data", 
             all_rgb_data, 
             all_depth_data, 
             all_odom_position_data, 
             all_odom_orientation_data, 
             all_timestamp_data)


        # rate.sleep()


    # # enter main loop
    # while True: 
    #     # robot_state_resp = robot_state_task.proto
    #     # acquisition_time = robot_state_resp.kinematic_state.acquisition_timestamp
    #     # time_temp = acquisition_time.seconds + acquisition_time.nanos * 1e-9
    #     # print("time in the main: ", time_temp)

    #     if not sensors_listener.sensor_initialzed: 
    #         continue
    #     print("in the loop")
    #     # ---- Show images ------
    #     cv2.namedWindow('RGB', cv2.WINDOW_AUTOSIZE)
    #     cv2.imshow('RGB', sensors_listener.rgb_image)
    #     cv2.namedWindow('Depth', cv2.WINDOW_AUTOSIZE)
    #     cv2.imshow('Depth', sensors_listener.depth_image)
    #     Key = cv2.waitKey(50)
    #     if Key == 27:
    #         cv2.destroyAllWindows()
    #         break

if __name__ == "__main__":
    if not main(sys.argv[1:]):
        sys.exit(1)
