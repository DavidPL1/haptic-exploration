import rospy
from rospkg import RosPack
import xacro

import os.path as osp
import tempfile
import getpass
import numpy as np

from typing import Union

import haptic_exploration.mujoco_config as mujoco_config
from haptic_exploration.config import ObjectSet
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


class YCBObjectController(BaseObjectController):

    def __init__(self):
        super().__init__(len(mujoco_config.ycb_objects))
        self.current_object_id = None

    def _build_model(self, object_id):
        object_rotation = np.array([0, 0, 0])
        if object_id in mujoco_config.ycb_objects_custom_rotation:
            for dim_name, rot in mujoco_config.ycb_objects_custom_rotation[object_id].items():
                object_rotation[["x", "y", "z"].index(dim_name)] = rot
        mesh_rot = object_rotation / 180 * np.pi

        mesh_pos = np.array([0, 0, 0.03])
        if object_id in mujoco_config.ycb_objects_custom_position:
            mesh_pos += np.array(mujoco_config.ycb_objects_custom_position[object_id])
        default_mesh, custom_meshes = mujoco_config.ycb_objects_default_mesh, mujoco_config.ycb_objects_custom_meshes
        mesh_type = custom_meshes[object_id] if object_id in custom_meshes else default_mesh
        mesh_path = f"{mujoco_config.ycb_objects[object_id]}/{mesh_type.value}/nontextured.stl"
        show_surfaces = rospy.get_param("~show_surfaces", False)

        arg_map = dict(
            mesh_subpath=mesh_path,
            visualize_surfaces=f'{int(show_surfaces)}',
            mesh_pos=' '.join(map(str, mesh_pos)),
            mesh_rot=' '.join(map(str, mesh_rot)),
        )

        outfile = osp.join(tempfile.gettempdir(), f'ycb_exploration_{getpass.getuser()}.xml')

        xacro_base = osp.join(RosPack().get_path('haptic_exploration'), 'assets', 'xacro', 'generic_ycb_exploration.xml.xacro')
        doc = xacro.process_file(xacro_base, mappings=arg_map).toprettyxml(indent='  ')
        with open(outfile, 'w') as f:
            f.write(doc)
        return outfile

    def set_object(self, object_id: int, mujoco_ros_client: MujocoRosClient):
        model_filepath = self._build_model(object_id)
        mujoco_ros_client.load_model(model_filepath)
        self.current_object_id = object_id

    def clear_object(self, mujoco_client: MujocoRosClient):
        pass

    def get_current_object(self) -> Union[int, None]:
        return self.current_object_id


def get_object_controller(object_set: ObjectSet):
    spec = {
        ObjectSet.YCB: YCBObjectController
    }
    if object_set in spec:
        return spec[object_set]()
    else:
        raise Exception("invalid object set")
