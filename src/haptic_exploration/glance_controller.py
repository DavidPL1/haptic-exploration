import numpy as np
import haptic_exploration.mujoco_config as mujoco_config

from fpstimer import FPSTimer
from haptic_exploration.util import Pose, GlanceAreaBB
from haptic_exploration.ros_client import MujocoRosClient
from haptic_exploration.object_controller import BaseObjectController
from haptic_exploration.glance_parameters import GlanceParameters


class GlancePressureMonitor:
    
    def __init__(self) -> None:
        self.max_values = np.zeros((64,))
        self.max_values_sum = 0
        self.max_values_pose = None
        self.velocity_counter = 0
    
    def add(self, values, pose_linvel, mocap_pose) -> bool:
        pose, linvel = pose_linvel
        values_sum = values.sum()
        max_cell_value = values.max()

        if values_sum >= self.max_values_sum:
            self.max_values = values
            self.max_values_sum = values_sum
            self.max_values_pose = pose

        if np.linalg.norm(linvel) < mujoco_config.glance_velocity_threshold:
            self.velocity_counter += 1
        else:
            self.velocity_counter = 0

        c1 = values_sum > mujoco_config.glance_sum_pressure_threshold
        c2 = max_cell_value > mujoco_config.glance_cell_pressure_threshold
        c3 = self.velocity_counter > mujoco_config.glance_velocity_threshold_count
        c4 = np.linalg.norm(pose.point - mocap_pose.point) > mujoco_config.glance_mocap_distance_threshold
        return c1 or c2 or c3 or c4


class MocapGlanceController(MujocoRosClient):

    def __init__(self, object_controller: BaseObjectController, glance_area: GlanceAreaBB) -> None:
        super().__init__("mocap_glance_controller")

        self.object_controller = object_controller
        self.glance_area = glance_area

    def set_object(self, object_id):
        if object_id is None or object_id < 0:
            self.object_controller.clear_object(self)
        else:
            self.object_controller.set_object(object_id, self)

    def clear_object(self):
        self.object_controller.clear_object(self)

    def perform_glance(self, glance_params: GlanceParameters, rt=False):

        sim_step_size = 30
        fps_timer = FPSTimer(1000/sim_step_size) if rt else None

        start_pose, target_pose = glance_params.get_start_target_pose(self.glance_area)
        total_glance_steps = int(1000 * abs(start_pose.point[2] - target_pose.point[2]) / mujoco_config.mocap_velocity)

        # wait for myrmex to reach starting pose
        self.set_mocap_body(mujoco_config.MYRMEX_MOCAP_BODY, start_pose)
        while True:
            self.perform_steps(10)
            if fps_timer is not None:
                fps_timer.sleep()
            myrmex_pose, _ = self.get_body_pose_linvel(mujoco_config.MYRMEX_BODY)
            if np.linalg.norm(start_pose.point - myrmex_pose.point) < 0.02:
                break
        self.current_myrmex_state = None

        # perform glance
        glance_monitor = GlancePressureMonitor()
        for elapsed_steps in self.perform_steps_chunked(total_glance_steps, sim_step_size):
            if fps_timer is not None:
                fps_timer.sleep()

            fraction = elapsed_steps/total_glance_steps
            interpolated_point = (1-fraction) * start_pose.point + fraction * target_pose.point
            mocap_pose = Pose(interpolated_point, target_pose.orientation)
            self.set_mocap_body(mujoco_config.MYRMEX_MOCAP_BODY, mocap_pose)

            if self.current_myrmex_state is not None:
                vel = self.get_body_pose_linvel(mujoco_config.MYRMEX_BODY)
                if glance_monitor.add(np.array(self.current_myrmex_state.sensors[0].values), vel, mocap_pose):
                    break

        # compute coordinates relative to BB
        pose = glance_monitor.max_values_pose
        pose.point[0] = (pose.point[0] - self.glance_area.x_limits[0]) / (self.glance_area.x_limits[1] - self.glance_area.x_limits[0])
        pose.point[1] = (pose.point[1] - self.glance_area.y_limits[0]) / (self.glance_area.y_limits[1] - self.glance_area.y_limits[0])
        pose.point[2] = (pose.point[2] - self.glance_area.z_limits[0]) / (self.glance_area.z_limits[1] - self.glance_area.z_limits[0])
        return glance_monitor.max_values, glance_monitor.max_values_pose
