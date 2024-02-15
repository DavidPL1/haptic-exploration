import numpy as np
import haptic_exploration.mujoco_config as mujoco_config

from copy import deepcopy
from pyquaternion import Quaternion
from typing import Tuple
from haptic_exploration.util import Pose, GlanceAreaBB


class GlanceParameters:

    def get_start_target_pose(self, glance_area: GlanceAreaBB) -> Tuple[Pose, Pose]:
        raise NotImplementedError()


class XPhiGlanceParameters(GlanceParameters):

    def __init__(self, x_factor, phi_factor):
        super().__init__()
        self.x_factor = x_factor
        self.phi_factor = phi_factor

    def get_start_target_pose(self, glance_area: GlanceAreaBB) -> Tuple[Pose, Pose]:

        min_z, max_z = glance_area.z_limits
        min_x, max_x = glance_area.x_limits
        start_pose_point = np.array([min_x + self.x_factor*(max_x - min_x), 0, max_z + mujoco_config.basic_objects_z_buffer])
        min_phi, max_phi = -90, 90
        phi = min_phi + self.phi_factor * (max_phi - min_phi)
        start_pose_quat = Quaternion(np.array([1, 0, 0, 0])) * Quaternion(axis=[0, 1, 0], angle=phi * np.pi / 180)
        start_pose = Pose(start_pose_point, start_pose_quat)

        target_pose = deepcopy(start_pose)
        target_pose.point[2] = min_z

        return start_pose, target_pose

    def __str__(self):
        return f"x: {self.x_factor:.3f}, phi: {self.phi_factor:.3f}"


class XYGlanceParameters(GlanceParameters):

    def __init__(self, x_factor, y_factor):
        super().__init__()
        self.x_factor = x_factor
        self.y_factor = y_factor

    def get_start_target_pose(self, glance_area: GlanceAreaBB) -> Tuple[Pose, Pose]:

        min_z, max_z = glance_area.z_limits
        min_x, max_x = glance_area.x_limits
        min_y, max_y = glance_area.y_limits
        x = min_x + self.x_factor * (max_x - min_x)
        y = min_y + self.y_factor * (max_y - min_y)
        start_pose = Pose(np.array([x, y, max_z]), Quaternion(np.array([1, 0, 0, 0])))

        target_pose = deepcopy(start_pose)
        target_pose.point[2] = min_z

        return start_pose, target_pose

    def __str__(self):
        return f"x: {self.x_factor:.3f}, y: {self.y_factor:.3f}"


class YCBGlanceParameters(GlanceParameters):

    def __init__(self, x_factor=0.5, y_factor=0.5, x_angle_factor=0.5, y_angle_factor=0.5):
        super().__init__()
        self.x_factor = x_factor
        self.y_factor = y_factor
        self.x_angle_factor = x_angle_factor
        self.y_angle_factor = y_angle_factor

    def get_start_target_pose(self, glance_area: GlanceAreaBB) -> Tuple[Pose, Pose]:

        min_z, max_z = glance_area.z_limits
        min_x, max_x = glance_area.x_limits
        min_y, max_y = glance_area.y_limits
        x = min_x + self.x_factor * (max_x - min_x)
        y = min_y + self.y_factor * (max_y - min_y)
        start_pose_point = np.array([x, y, max_z + mujoco_config.ycb_objects_z_buffer])

        min_angle, max_angle = -np.pi/2, np.pi/2
        x_angle = min_angle + self.x_angle_factor * (max_angle - min_angle)
        y_angle = min_angle + self.y_angle_factor * (max_angle - min_angle)
        start_pose_quat = Quaternion(np.array([1, 0, 0, 0])) * Quaternion(axis=[1, 0, 0], angle=x_angle) * Quaternion(axis=[0, 1, 0], angle=y_angle)

        start_pose = Pose(start_pose_point, start_pose_quat)
        target_pose = deepcopy(start_pose)
        target_pose.point[2] = min_z

        return start_pose, target_pose

    def __str__(self):
        return f"x: {self.x_factor:.3f}, y: {self.y_factor:.3f}, x_angle: {self.x_factor:.3f}, y_angle: {self.y_factor:.3f}"
