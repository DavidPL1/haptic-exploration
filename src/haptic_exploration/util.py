import numpy as np

from dataclasses import dataclass
from pyquaternion import Quaternion
from typing import List, Tuple
from geometry_msgs.msg import PoseStamped, Transform


@dataclass
class Pose:
    point: np.ndarray
    orientation: np.ndarray

    @staticmethod
    def from_transform(transform: Transform) -> "Pose":
        trans, rot = transform.translation, transform.rotation
        point = np.array([trans.x, trans.y, trans.z])
        orientation = np.array([rot.x, rot.y, rot.z, rot.w])
        return Pose(point, orientation)
    
    @staticmethod
    def from_ros_pose(ros_pose: PoseStamped) -> "Pose":
        position = np.array([ros_pose.pose.position.x, ros_pose.pose.position.y, ros_pose.pose.position.z])
        orientation = np.array([ros_pose.pose.orientation.x, ros_pose.pose.orientation.y, ros_pose.pose.orientation.z, ros_pose.pose.orientation.w])
        return Pose(position, orientation)
    
    def to_ros_pose(self) -> PoseStamped:
        pose_stamped = PoseStamped()
        pos, ori = pose_stamped.pose.position, pose_stamped.pose.orientation
        pos.x, pos.y, pos.z = self.point
        ori.x, ori.y, ori.z, ori.w = self.orientation
        return pose_stamped


@dataclass
class Trajectory:
    start_pose: Pose
    end_pose: Pose


@dataclass
class GlanceParameters:
    target_pose: Pose
    approach_direction: np.ndarray


@dataclass
class GlanceAreaBB:
    x_limits: Tuple[float, float]
    y_limits: Tuple[float, float]
    z_limits: Tuple[float, float]


def get_myrmex_normal(myrmex_orientation): 
    rot_quat = Quaternion(myrmex_orientation)
    base_normal = np.array([0, 0, 1])
    myrmex_normal = rot_quat.rotate(base_normal)
    return myrmex_normal


def calculate_start_pose(glance_parameters: GlanceParameters, distance=0.3):
    approach_direction_normalized = glance_parameters.approach_direction/np.linalg.norm(glance_parameters.approach_direction)
    start_point = glance_parameters.target_pose.point - distance*approach_direction_normalized
    return Pose(start_point, glance_parameters.target_pose.orientation)
