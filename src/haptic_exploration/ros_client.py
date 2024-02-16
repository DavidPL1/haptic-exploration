import rospy
import actionlib
import numpy as np

from rospy.exceptions import ROSException
from typing import Iterator, Tuple

from haptic_exploration.util import Pose
from mujoco_ros_msgs.msg import StepAction, StepGoal, MocapState
from mujoco_ros_msgs.srv import SetBodyState, SetBodyStateRequest, GetBodyState, GetBodyStateRequest, GetBodyStateResponse, SetPause, SetPauseRequest, Reload, ReloadRequest, GetSimInfo, GetSimInfoRequest
from mujoco_contact_surface_sensors.srv import GetTactileState, GetTactileStateRequest
from tactile_msgs.msg import TactileState
from std_srvs.srv import Empty, SetBool, SetBoolRequest


class MujocoRosClient:

    def __init__(self, node_name) -> None:
        rospy.init_node(node_name)


        self.mocap_state_publisher = rospy.Publisher("/mujoco_server/mocap_poses", MocapState, queue_size=100)

        self.step_action_client = actionlib.SimpleActionClient("/mujoco_server/step", StepAction)
        self.step_action_client.wait_for_server()

        self.reset_client = rospy.ServiceProxy("/mujoco_server/reset", Empty)
        self.set_body_state_client = rospy.ServiceProxy("mujoco_server/set_body_state", SetBodyState)
        self.get_body_state_client = rospy.ServiceProxy("/mujoco_server/get_body_state", GetBodyState)
        self.tactile_set_pause_client = rospy.ServiceProxy("/tactile_module_16x16_v2/set_pause", SetBool)
        self.get_myrmex_data_client = rospy.ServiceProxy("/tactile_module_16x16_v2/get_state", GetTactileState)
        self.reload_client = rospy.ServiceProxy('/mujoco_server/reload', Reload)
        self.set_pause_client = rospy.ServiceProxy('/mujoco_server/set_pause', SetPause)
        self.get_sim_info = rospy.ServiceProxy('/mujoco_server/get_sim_info', GetSimInfo)

        self.load_count = 1

        service_clients = [
            self.reset_client,
            self.set_body_state_client,
            self.get_body_state_client,
            self.tactile_set_pause_client,
            self.reload_client,
            self.set_pause_client,
            self.get_sim_info
        ]
        for service_client in service_clients:
            try:
                service_client.wait_for_service(2)
            except ROSException:
                rospy.logerr(f"Timeout for service {service_client.resolved_name}")

        rospy.rostime.wallsleep(0.5)

    def set_pause(self, paused):
        resp = self.set_pause_client(SetPauseRequest(paused=paused))
        return resp.success

    def toggle_myrmex(self, paused):
        # self.tactile_set_pause_client = rospy.ServiceProxy("/tactile_module_16x16_v2/set_pause", SetBool)
        # self.tactile_set_pause_client.wait_for_service(2)
        set_tactile_pause_request = SetBoolRequest()
        set_tactile_pause_request.data = paused
        self.tactile_set_pause_client.call(set_tactile_pause_request)

    def get_myrmex_data(self):
        return np.array(self.get_myrmex_data_client(GetTactileStateRequest()).tactile_state.sensors[0].values)

    def perform_steps_chunked(self, total_steps: int, step_chunk_size: int) -> Iterator[int]:
        for chunk in range(total_steps // step_chunk_size):
            self.perform_steps(num_steps=step_chunk_size)
            yield (chunk + 1) * step_chunk_size

    def perform_steps(self, num_steps: int) -> None:
        self.step_action_client.send_goal_and_wait(StepGoal(num_steps=num_steps))
        return self.step_action_client.get_result()

    def set_mocap_body(self, mocap_body_name: str, pose: Pose):
        mocap_state = MocapState()
        mocap_state.name = [mocap_body_name]
        mocap_state.pose = [pose.to_ros_pose()]
        mocap_state.pose[0].header.frame_id = "world"
        self.mocap_state_publisher.publish(mocap_state)

    def set_body_pose(self, body_name: str, pose: Pose):
        set_body_state_request = SetBodyStateRequest()
        set_body_state_request.state.name = body_name
        set_body_state_request.state.pose = pose.to_ros_pose()
        set_body_state_request.set_pose = True
        self.set_body_state_client.call(set_body_state_request)

    def get_body_pose_linvel(self, body_name: str) -> Tuple[Pose, np.ndarray]:
        request = GetBodyStateRequest()
        request.name = body_name
        while True:
            try:
                response: GetBodyStateResponse = self.get_body_state_client.call(request)
                pose = Pose.from_ros_pose(response.state.pose)
                linvel = response.state.twist.twist.linear
                linvel = np.array([linvel.x, linvel.y, linvel.z])
                return pose, linvel
            except:
                if rospy.is_shutdown():
                    break
    
    def pre_wait_for_sim(self):
        resp = self.get_sim_info(GetSimInfoRequest())
        self.load_count = resp.state.load_count + 1


    def wait_for_sim(self):
        for _ in range(600):
            resp = self.get_sim_info(GetSimInfoRequest())
            if resp.state.load_count == self.load_count:
                self.load_count = resp.state.load_count
                return True
        return False
            

    def load_model(self, filepath: str):
        self.set_pause(False)
        rospy.sleep(0.2)
        self.set_pause(True)

        self.pre_wait_for_sim()
        resp = self.reload_client(ReloadRequest(model=filepath))
        loaded = self.wait_for_sim()
        return resp.success and loaded