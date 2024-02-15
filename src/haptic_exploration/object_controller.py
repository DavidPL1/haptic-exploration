import numpy as np
import haptic_exploration.mujoco_config as mujoco_config

from typing import Union, List
from copy import deepcopy
from haptic_exploration.util import Pose
from haptic_exploration.ros_client import MujocoRosClient


class BaseObjectController:

    def __init__(self, num_objects):
        self.num_objects = num_objects

    def set_object(self, object_id: int, mujoco_client: MujocoRosClient):
        raise NotImplementedError()

    def clear_object(self, mujoco_client: MujocoRosClient):
        raise NotImplementedError()

    def get_current_object(self) -> Union[int, None]:
        raise NotImplementedError()


class SimpleObjectController(BaseObjectController):

    def __init__(self):
        super().__init__(len(mujoco_config.basic_objects))
        self.current_object_id = None

    def set_object(self, object_id: int, mujoco_client: MujocoRosClient):
        if self.current_object_id != object_id:
            self.clear_object(mujoco_client)

            if object_id is not None:
                mujoco_client.set_body_pose(f"{object_id}_body", mujoco_config.simple_glance_object_pose)
                self.current_object_id = object_id

    def clear_object(self, mujoco_client: MujocoRosClient):
        if self.current_object_id is not None:
            object_x = mujoco_config.simple_inactive_object_x + self.current_object_id * 0.15
            inactive_object_pose = Pose(np.array([object_x, 0, 0]), np.array([0, 0, 0, 0]))
            mujoco_client.set_body_pose(f"{self.current_object_id}_body", inactive_object_pose)
            self.current_object_id = None

    def get_current_object(self) -> Union[int, None]:
        return self.current_object_id


class CompositeObjectController(BaseObjectController):

    def __init__(self, composite_objects: List[np.ndarray]):
        super().__init__(len(composite_objects))
        self.composite_objects = composite_objects
        self.current_object_id = None

    def set_object(self, object_id: int, mujoco_client: MujocoRosClient):
        if self.current_object_id != object_id:
            self.clear_object(mujoco_client)

            if object_id is not None:
                object_features = self.composite_objects[object_id]
                for position_idx, target_feature_idx in enumerate(object_features):
                    self.set_feature(target_feature_idx, position_idx, mujoco_client)
                self.current_object_id = object_id

    def clear_object(self, mujoco_client: MujocoRosClient):
        if self.current_object_id is not None:
            for position_idx, feature_idx in enumerate(self.composite_objects[self.current_object_id]):
                reset_pose = self.get_reset_pose(position_idx, feature_idx)
                body_name = self.get_body_name(position_idx, feature_idx)
                mujoco_client.set_body_pose(body_name, reset_pose)
            self.current_object_id = None

    def set_feature(self, feature_idx: int, position_idx: int, mujoco_client: MujocoRosClient):
        active_pose = self.get_active_pose(position_idx)
        body_name = self.get_body_name(position_idx, feature_idx)
        mujoco_client.set_body_pose(body_name, active_pose)

    def get_reset_pose(self, position_idx, feature_idx):
        pose = deepcopy(mujoco_config.composite_inactive_base_pose)
        pose.point += np.asarray([0.2 * feature_idx, 0.2 * position_idx, 0])
        return pose

    def get_active_pose(self, position_idx):
        pose = deepcopy(mujoco_config.composite_active_base_pose)
        pose.point += mujoco_config.composite_active_relative_positions[position_idx]
        return pose

    def get_body_name(self, position_idx, feature_idx):
        return f"f{feature_idx}_{position_idx}"

    def get_current_object(self) -> Union[int, None]:
        return self.current_object_id


class YCBObjectController(BaseObjectController):

    def __init__(self, ycb_objects):
        super().__init__(len(ycb_objects))
        self.ycb_objects = ycb_objects
        self.current_object_id = None

    def set_object(self, object_id: int, mujoco_client: MujocoRosClient):
        if self.current_object_id != object_id:
            self.clear_object(mujoco_client)

            if object_id is not None:
                mujoco_client.set_body_pose(f"{object_id}_body", mujoco_config.ycb_glance_object_pose)
                self.current_object_id = object_id

    def clear_object(self, mujoco_client: MujocoRosClient):
        if self.current_object_id is not None:
            object_y = mujoco_config.ycb_inactive_object_y
            inactive_object_pose = Pose(np.array([0, object_y, 0]), np.array([0, 0, 0, 0]))
            mujoco_client.set_body_pose(f"{self.current_object_id}_body", inactive_object_pose)
            self.current_object_id = None

    def get_current_object(self) -> Union[int, None]:
        return self.current_object_id
