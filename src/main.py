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

#####################################
# Create robot object.
#####################################
sdk = bosdyn.client.create_standard_sdk('RobotCommandMaster')
robot = sdk.create_robot(options.hostname)
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


