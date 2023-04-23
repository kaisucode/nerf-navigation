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

from keyframe_detector.SensorStreamSpotSdk import * 

HOSTNAME = "gouger.rlab.cs.brown.edu"
image_sources = ["hand_color_image", "hand_depth"] # sources for depth and rgb image
sensor_time_delay = 1.0/60.0

def _update_thread(async_task):
    while True:
        async_task.update()
        time.sleep(0.01)

class AsyncImage(AsyncPeriodicQuery):
    """Grab image."""

    def __init__(self, image_client, image_sources):
        # Period is set to be about 15 FPS
        super(AsyncImage, self).__init__('images', image_client, LOGGER, period_sec=0.067)
        self.image_sources = image_sources

    def _start_query(self):
        return self._client.get_image_from_sources_async(self.image_sources)

class AsyncRobotState(AsyncPeriodicQuery):
    """Grab robot state."""

    def __init__(self, robot_state_client):
        # period is set to be about the same rate as detections on the CORE AI
        super(AsyncRobotState, self).__init__('robot_state', robot_state_client, LOGGER,
                                              period_sec=0.02)

    def _start_query(self):
        return self._client.get_robot_state_async()

def main(argv): 

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

    # Setup clients for the robot
    # image client
    robot_image_client = robot.ensure_client(imageClient.default_service_name)
    # motion client
    robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)
    robot_command_client = robot.ensure_client(RobotCommandClient.default_service_name)

    # async image 
    image_task = AsyncImage(image_client, image_sources)
    robot_state_task = AsyncRobotState(robot_state_client)
    task_list = [image_task, robot_state_task]
    _async_tasks = AsyncTasks(task_list)

    update_thread = Thread(target=_update_thread, args=[_async_tasks])
    update_thread.daemon = True
    update_thread.start()

    # Wait for the first responses.
    while any(task.proto is None for task in task_list):
        time.sleep(0.1)


    ## !! pass image_task for SensorListener to access
    sensors_listener = SensorListener(robot, image_task, sensor_time_delay)


if __name__ == "__main__":
    if not main(sys.argv[1:]):
        sys.exit(1)
