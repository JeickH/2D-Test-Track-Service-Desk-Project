#!/usr/bin/env python3
# =============================================================================
"""
Code Information:
    Maintainer: Eng. John Alberto Betancourt G
	Mail: john@kiwicampus.com
	Kiwi Campus / Computer & Ai Vision Team
"""

# =============================================================================
import time
import sys
import os

import numpy as np

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import qos_profile_sensor_data
from rclpy.logging import get_logger
from rclpy.node import Node

from usr_srvs.srv import Move
from usr_srvs.srv import Turn

from std_msgs.msg import Int8, Bool

from usr_msgs.msg import Kiwibot as kiwibot_status

from utils.python_utils import printlog

# =============================================================================
def setProcessName(name: str) -> None:
    """!
    Function for seting the process name
    @see name 'str' defining the process name
    """
    if sys.platform in ["linux2", "linux"]:
        import ctypes

        libc = ctypes.cdll.LoadLibrary("libc.so.6")
        libc.prctl(15, name, 0, 0, 0)
    else:
        raise Exception(
            "Can not set the process name on non-linux systems: " + str(sys.platform)
        )


# =============================================================================
class KiwibotNode(Node):
    def __init__(self) -> None:
        """
            Class constructor for Kiwibot Node
        Args:
        Returns:
        """

        # ---------------------------------------------------------------------
        Node.__init__(self, node_name="kiwibot_node")

        # Allow callbacks to be executed in parallel without restriction.
        self.callback_group = ReentrantCallbackGroup()

        self._TURN_PRINT_WAYPOINT = int(os.getenv("TURN_PRINT_WAYPOINT", default=0))
        self._FORWARE_PRINT_WAYPOINT = int(
            os.getenv("FORWARE_PRINT_WAYPOINT", default=0)
        )

        # ---------------------------------------------------------------------
        self.status = kiwibot_status()
        self.status.yaw = float(os.getenv("BOT_INITIAL_YAW", default=0.0))
        self.status.pos_x = int(os.getenv("BOT_INITIAL_X", default=917))
        self.status.pos_y = int(os.getenv("BOT_INITIAL_Y", default=1047))
        """ kiwibot_status:
            int8 pos_x      # x axis position in the map
            int8 pos_y      # y axis position in the map
            float32 dist    # distance traveled by robot
            float32 speed   # speed m/s
            float32 time    # time since robot is moving
            float32 yaw     # time since robot is moving
            bool moving     # Robot is moving
        """

        # ---------------------------------------------------------------------
        # Publishers
        self.pub_bot_status = self.create_publisher(
            msg_type=kiwibot_status,
            topic="/kiwibot/status",
            qos_profile=qos_profile_sensor_data,
            callback_group=self.callback_group,
        )

        self.pub_speaker = self.create_publisher(
            msg_type=Int8,
            topic="/device/speaker/command",
            qos_profile=qos_profile_sensor_data,
            callback_group=self.callback_group,
        )

        # Subscribers
        # cancel routine msg subscriber
        self.sub_cancel_rout = self.create_subscription(
            msg_type=Bool,
            topic="/graphics/cancel_routine",
            callback=self.cb_cancel_rout,
            qos_profile=qos_profile_sensor_data,
            callback_group=self.callback_group,
        )
        self.sub_cancel_rout

        # ---------------------------------------------------------------------
        # Services

        # service to turn the robot
        self.srv_robot_turn = self.create_service(
            Turn,
            "/robot/turn",
            self.cb_srv_robot_turn,
            callback_group=self.callback_group,
        )

        # service to move the robot
        self.srv_robot_move = self.create_service(
            Move,
            "/robot/move",
            self.cb_srv_robot_move,
            callback_group=self.callback_group,
        )

    def cb_cancel_rout(self, data):
        self.cancel_rout = data.data

    def cb_srv_robot_turn(self, request, response) -> Turn:
        """
            Callback to update kiwibot state information when turning
            request service
        Args:
            request: `usr_srvs.srv._turn.Turn_Request` request of turning
            references to turn the robot
        Returns:
            response: `usr_srvs.srv._turn.Turn_Response` turning request
            successfully completed or not
        """

        try:

            for idx, turn_ref in enumerate(request.turn_ref[:-1]):

                if self._TURN_PRINT_WAYPOINT:
                    printlog(msg=turn_ref, msg_type="INFO")

                self.status.speed = 0.0
                self.status.yaw += (
                    request.turn_ref[idx + 1].yaw - request.turn_ref[idx].yaw
                )
                self.status.time += turn_ref.dt
                self.status.moving = True

                if self.status.yaw >= 360:
                    self.status.yaw += -360
                #
                if not self.cancel_rout:
                    self.pub_bot_status.publish(self.status)
                    time.sleep(turn_ref.dt)
                else:
                    self.pub_speaker.publish(Int8(data=0))
                    break

                time.sleep(turn_ref.dt)

            self.status.moving = False
            self.pub_bot_status.publish(self.status)
            response.completed = True

        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            printlog(
                msg="{}, {}, {}, {}".format(e, exc_type, fname, exc_tb.tb_lineno),
                msg_type="ERROR",
            )
            response.completed = False

        return response

    def cb_srv_robot_move(self, request, response) -> Move:
        """
            Callback to update kiwibot state information when move
            request service
        Args:
            request: `usr_srvs.srv._move.Move_Request` request of turning
            references to move the robot
        Returns:
            response: `usr_srvs.srv._move.Move_Response` Moving request
            successfully completed or not
        """

        try:
            for wp in request.waypoints:

                if self._FORWARE_PRINT_WAYPOINT:
                    printlog(msg=wp, msg_type="INFO")

                # Updating robot status
                abs_dist = (
                    np.sqrt(
                        pow(self.status.pos_x - wp.x, 2)
                        + pow(self.status.pos_y - wp.y, 2)
                    )
                    * 0.00847619047
                )
                self.status.speed = abs_dist / wp.dt
                self.status.dist += abs_dist

                self.status.pos_x = wp.x
                self.status.pos_y = wp.y
                self.status.moving = True
                self.status.time += wp.dt

                # cancel routine condition
                if not self.cancel_rout:
                    self.pub_bot_status.publish(self.status)
                    time.sleep(wp.dt)
                else:
                    self.pub_speaker.publish(Int8(data=0))
                    break

                time.sleep(wp.dt)

            self.status.speed = 0.0
            self.status.moving = False
            self.pub_bot_status.publish(self.status)
            self.pub_speaker.publish(Int8(data=1))
            response.completed = True

        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            printlog(
                msg="{}, {}, {}, {}".format(e, exc_type, fname, exc_tb.tb_lineno),
                msg_type="ERROR",
            )
            response.completed = False

        return response


# =============================================================================
def main(args=None):
    """!
    Main Functions of Local Console Node
    """
    # Initialize ROS communications for a given context.
    setProcessName("kiwibot-node")
    rclpy.init(args=args)

    # Execute work and block until the context associated with the
    # executor is shutdown.
    kiwibot_node = KiwibotNode()

    # Runs callbacks in a pool of threads.
    executor = MultiThreadedExecutor()

    # Execute work and block until the context associated with the
    # executor is shutdown. Callbacks will be executed by the provided
    # executor.
    rclpy.spin(kiwibot_node, executor)

    # Clear thread
    kiwibot_node.clear()

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    kiwibot_node.destroy_node()
    rclpy.shutdown()


# =============================================================================
if __name__ == "__main__":
    main()

# =============================================================================
