from haptic_exploration.ros_client import MujocoRosClient
from controller_manager_msgs.srv import ListControllers, LoadController, SwitchController

from geometry_msgs.msg import PoseStamped
import dynamic_reconfigure.client

import rospy
import numpy as np

import copy
import time

class PandaController(MujocoRosClient):
    
    def __init__(self, node_name="panda_controller") -> None:
        super().__init__(node_name)
        self.load_count = 1

        self.pub = rospy.Publisher('target', PoseStamped, queue_size=10)
        self.pub_control = rospy.Publisher('/cartesian_impedance_example_controller/equilibrium_pose', PoseStamped, queue_size=10)

    def load_model(self, model_filepath):
        ret = super().load_model(model_filepath)
        self.set_pause(False)
        time.sleep(.5)
        self.ensure_controller_started()
        self.set_pause(True)
        return ret

    def ensure_controller_started(self):
        self.switch_controllers(start=["franka_state_controller", "cartesian_impedance_example_controller"], stop=["effort_joint_trajectory_controller"])
        time.sleep(0.2)
        ns = '/cartesian_impedance_example_controller/dynamic_reconfigure_compliance_param_node'
        client = dynamic_reconfigure.client.Client(ns, timeout=1, config_callback=None)
        client.update_configuration({
            'translational_stiffness': 400.,
            'rotational_stiffness': 30.,
            'nullspace_stiffness': 0.
        })

    def set_target_pose(self, target):
        self.pub.publish(target.to_ros_pose())
        # Controller expects in link_0 frame without checking and we know the static transform from world (no rotation)
        target_t = copy.deepcopy(target)
        target_t.point += np.array([0.5, 0, -0.08])
        self.pub_control.publish(target_t.to_ros_pose())

    @staticmethod
    def switch_controllers(start, stop=None, ns="/controller_manager"):
        def call(ns, cls, **kwargs):
            rospy.wait_for_service(ns, timeout=1)
            service = rospy.ServiceProxy(ns, cls)
            return service(**kwargs)

        loaded = call(ns + "/list_controllers", ListControllers)
        loaded = [c.name for c in loaded.controller]

        for name in start:
            if name not in loaded:
                call(ns + "/load_controller", LoadController, name=name)

        if not call(
            ns + "/switch_controller",
            SwitchController,
            start_controllers=start,
            stop_controllers=stop,
            strictness=1,
            start_asap=False,
            timeout=0.0,
        ).ok:
            raise RuntimeError("Failed to switch controller")