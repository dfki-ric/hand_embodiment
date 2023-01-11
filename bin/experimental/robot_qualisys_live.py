"""Write data streamed from Qualisys MoCap system to file."""
# Install dependency:
# python -m pip install qtm

import asyncio
from time import sleep

import numpy as np
import qtm
import json
import argparse
import pytransform3d.rotations as pr
import pytransform3d.transformations as pt
from ur_policy_control import robot_controller
from hand_embodiment.pipelines import MoCapToRobot
from hand_embodiment.command_line import (
    add_hand_argument, add_animation_arguments, add_configuration_arguments,
    add_playback_control_arguments)
from hand_embodiment.target_configurations import TARGET_CONFIG
from ur_policy_control import robot_controller_mockup
from ur_policy_control import constants
from ur_policy_control import kinematics
from ur_policy_control.commands.move_to_pose import MoveToJointAngles
from ur_policy_control.transformations import pos_rvec_from_transform

use_simulation = True
calc_finger_interval = 100

class OnPacket:
    def __init__(self, verbose=False):
        self.verbose = verbose

        # label order from AIM model
        self.labels = [
            "hand_top",
            "hand_left",
            "hand_right",
            "thumb_tip",
            "thumb_middle",
            "little_tip",
            "little_middle",
            "ring_tip",
            "ring_middle",
            "middle_middle",
            "middle_tip",
            "index_tip",
            "index_middle",
        ]
        if not use_simulation:
            self.robot = robot_controller.RobotController("192.168.1.103", use_arm=True, use_hand=False)
        else:
            self.robot = robot_controller_mockup.RobotControllerMockup("192.168.1.103",
                                                                       robot=constants.Robot.MiaHandOnUR10)

        self.robot_kinematics = kinematics.UniversalRobotAndMiaHandKinematics(constants.Robot.MiaHandOnUR10)
        args = self.parse_args()

        finger_names = ['middle', 'little', 'index', 'thumb', 'ring']

        self.pipeline = MoCapToRobot(args.hand, args.mano_config, finger_names,
                                     record_mapping_config=args.record_mapping_config,
                                     verbose=self.verbose, measure_time=args.measure_time,
                                     robot_config=args.robot_config)
        self.mocap_origin2origin = None

        if use_simulation:
            move_to_start = MoveToJointAngles(self.robot, self.robot_kinematics,
                                              np.deg2rad([-90.0, -90.0, -90.0, 180.0, -90.0, 90.0]))
            move_to_start.execute()

        self.robot.idle(2)

        self.hand_pose_markers = None
        self.finger_markers = None

        self.visual_frame = None
        self.palm_0_to_base = None

    def parse_args(self):
        parser = argparse.ArgumentParser()
        add_hand_argument(parser)

        # add_configuration_arguments(parser)

        # TODO We need to which repo should be the root.
        #  When urpolcity control is the root the normal mano path doesn't work.
        parser.add_argument(
            "--mano-config", type=str,
            default="../hand_embodiment/examples/config/mano/20210520_april.yaml",
            help="MANO configuration file.")
        parser.add_argument(
            "--record-mapping-config", type=str, default=None,
            help="Record mapping configuration file.")
        parser.add_argument(
            "--robot-config", type=str, default=None,
            help="Target system configuration file.")

        add_playback_control_arguments(parser)
        parser.add_argument(
            "--interpolate-missing-markers", action="store_true",
            help="Interpolate NaNs.")
        parser.add_argument(
            "--show-mano", action="store_true", help="Show MANO mesh")
        parser.add_argument(
            "--mia-thumb-adducted", action="store_true",
            help="Adduct thumb of Mia hand.")
        parser.add_argument(
            "--measure-time", action="store_true",
            help="Measure time of record and embodiment mapping.")
        add_animation_arguments(parser)
        return parser.parse_args()


    finger_step_counter = 0
    def __call__(self, packet):
        """Callback function that is called everytime a data packet arrives from QTM."""
        print("Framenumber: {}".format(packet.framenumber))

        calc_hand = True

        header, markers = packet.get_3d_markers()
        if self.verbose:
            print("Component info: {}".format(header))

        result = {}
        for i, label, marker in zip(range(len(markers)), self.labels, markers):
            if self.verbose:
                print(f"{marker.x:.1f} {marker.y:.1f} {marker.z:.1f} - {label}")
            result[label] = (marker.x / 1000.0, marker.y / 1000.0, marker.z / 1000.0)

        if calc_hand:
            self.hand_pose_markers = np.array([result["hand_top"], result["hand_left"], result["hand_right"]])
            self.finger_markers = {"thumb": np.array([result["thumb_tip"], result["thumb_middle"]]),
                                   "index": np.array([result["index_tip"], result["index_middle"]]),
                                   "middle": np.array([result["middle_tip"], result["middle_tip"]]),
                                   "ring": np.array([result["ring_tip"], result["ring_middle"]]),
                                   "little": np.array([result["little_tip"], result["little_middle"]])}

            self.pipeline.estimate_hand(self.hand_pose_markers, self.finger_markers)

            joint_angles = self.pipeline.estimate_joints()
            joint_angles = [joint_angles["index"][0], joint_angles["middle"][0], joint_angles["thumb"][0]]

            self.robot.set_finger_angles(joint_angles)

        palm_t_to_qualisys = self.pipeline.estimate_end_effector()

        # Needed to flip the frame of the hand
        scale_matrix = np.identity(4)
        scale_matrix[2, 2] *= -1
        scale_matrix[1, 1] *= -1
        # scale_matrix *= 0.5

        if self.mocap_origin2origin is None:
            self.mocap_origin2origin = palm_t_to_qualisys.dot(scale_matrix)
            self.mocap_origin2origin = np.linalg.inv(self.mocap_origin2origin)

            ee_0_to_base = self.robot_kinematics.forward(self.robot.get_current_joint_angles())
            self.palm_0_to_base = self.robot_kinematics.translate_ee_pose_to_palm_pose(ee_0_to_base)

            if use_simulation:
                self.visual_frame = robot_controller_mockup.VisualizePose.show_in_base(self.robot,
                                                                                       pose=self.palm_0_to_base)

        if use_simulation:
            self.visual_frame.update_pose(self.palm_0_to_base)

        palm_t_to_0 = self.mocap_origin2origin.dot(palm_t_to_qualisys)
        palm_t_to_0 = palm_t_to_0.dot(scale_matrix)

        palm_t_to_base = pt.concat(palm_t_to_0, self.palm_0_to_base, False, False)
        ee_t_to_base = self.robot_kinematics.translate_palm_pose_to_ee_pose(palm_t_to_base)

        self.robot.servo_l(pos_rvec_from_transform(ee_t_to_base))


        if use_simulation:
            self.robot._step_simulation()


async def setup(ip, frequency=None):
    """ Main function """
    connection = await qtm.connect(ip)
    if connection is None:
        return

    if frequency is None:
        frames = "allframes"
    else:
        frames = "frequency:%d" % frequency

    components = ["3d"]
    # '2d', '2dlin', '3d', '3dres', '3dnolabels', '3dnolabelsres', 'analog',
    # 'analogsingle', 'force', 'forcesingle', '6d', '6dres', '6deuler',
    # '6deulerres', 'gazevector', 'eyetracker', 'image', 'timecode', 'skeleton',
    # 'skeleton:global'

    await connection.stream_frames(
        frames=frames, components=components, on_packet=OnPacket(False))


if __name__ == "__main__":
    import sys

    # print("Sleeping 3 sec")
    # sleep(3)

    # ip = sys.argv[-1]
    ip = "10.250.6.39"
    asyncio.ensure_future(setup(ip, frequency=5))
    asyncio.get_event_loop().run_forever()
