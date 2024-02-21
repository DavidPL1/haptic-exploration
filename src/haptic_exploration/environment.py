import numpy as np
import gymnasium as gym

from gymnasium import spaces
from haptic_exploration.actions import DiscreteActionSpace, ContinuousActionSpace, HybridActionSpace, ParameterizedActionSpace
from haptic_exploration.data import GlanceTable
from haptic_exploration.preprocessing import resize_pressure, pressure_normalization_binary
from haptic_exploration.glance_controller import MocapGlanceController
from haptic_exploration.object_controller import SimpleObjectController, CompositeObjectController, YCBObjectController
from haptic_exploration.glance_parameters import GlanceParameters


class HapticExplorationEnv(gym.Env):
    metadata = {}

    def __init__(self, num_objects, num_params, position_dim, pressure_dim, max_steps=11, glance_reward=-0.05, true_reward=1, false_reward=-1, not_glanced_penalty=0, not_classified_penalty=-1, out_bounds_penalty=-0.1, first_obs="empty", verbose=False):

        # env config
        #self.glance_table = glance_table
        self.max_steps = max_steps
        #self.num_objects = len(glance_table.id_label)
        self.num_objects = num_objects
        self.num_params = num_params
        self.glance_reward = glance_reward
        self.true_reward = true_reward
        self.false_reward = false_reward
        self.not_glanced_penalty = not_glanced_penalty
        self.not_classified_penalty = not_classified_penalty
        self.out_bounds_penalty = out_bounds_penalty
        self.first_obs = first_obs  # one of: "empty", "random", "empty_signal"
        self.verbose = verbose

        # action space
        self.action_decision_space = DiscreteActionSpace(2)  # 0: perform glance, 1: classify object
        #self.glance_param_space = ContinuousActionSpace(glance_table.n_params)  # something like x, phi or x, y
        self.glance_param_space = ContinuousActionSpace(num_params)  # something like x, phi or x, y
        #self.classification_param_space = DiscreteActionSpace(self.num_objects)  # class id
        #self.custom_action_space = ParameterizedActionSpace(self.action_decision_space, [self.glance_param_space, self.classification_param_space])
        #self.custom_action_space = HybridActionSpace([self.action_decision_space, self.glance_param_space])
        #self.custom_action_space = self.action_decision_space
        #self.action_space = self.custom_action_space.gym_space

        # observation space
        assert first_obs in ["empty", "random", "empty_signal"]
        self.obs_offset = 1 if first_obs in ["random", "empty_signal"] else 0
        #self.pressure_shape = (self.max_steps + self.obs_offset, glance_table.pressure_table.shape[-1])
        #self.position_shape = (self.max_steps + self.obs_offset, glance_table.position_table.shape[-1])
        self.pressure_shape = (self.max_steps + self.obs_offset, pressure_dim)
        self.position_shape = (self.max_steps + self.obs_offset, position_dim)
        self.observation_space = spaces.Tuple([
            spaces.Sequence(spaces.Box(0, 1, shape=self.pressure_shape)),
            spaces.Sequence(spaces.Box(0, 1, shape=self.position_shape))
        ])

        # env state
        self.object_id = None
        self.step_count = None
        self.terminated = None
        self.glance_pressures = None
        self.glance_positions = None
        self.solved = None

    def reset(self, seed=None, options=None, object_id=None):
        super().reset(seed=seed)

        self.object_id = self.np_random.integers(0, self.num_objects) if object_id is None else object_id
        self._set_object()
        self.step_count = 0
        self.terminated = False
        self.glance_pressures = np.zeros(self.pressure_shape, dtype=np.float32)
        self.glance_positions = np.zeros(self.position_shape, dtype=np.float32)
        self.solved = False

        if self.first_obs == "random":
            glance_params = tuple((np.random.rand(self.num_params) - 1)*2)
            #pressure, position = self.glance_table.get_pressure_position(self.object_id, tuple(glance_params), zero_centered=True)
            pressure, position = self._get_pressure_position(tuple(glance_params))
            self.glance_pressures[0] = pressure
            self.glance_positions[0] = position
        elif self.first_obs == "empty_signal":
            self.glance_pressures[0] = np.zeros(self.pressure_shape[1:]) - 1
            self.glance_positions[0] = np.zeros(self.position_shape[1:]) - 1

        return self._get_obs(), self._get_info()

    def step(self, action):
        if self.terminated:
            raise gym.error.ResetNeeded()

        decision, parameters = action
        if decision == 0:
            parameters_clipped = tuple(np.clip(parameters[0].detach().numpy(), -1, 1).tolist())
            reward = self._process_glance(parameters_clipped)
            if ((parameters > 1) | (parameters < -1)).any():
                reward += self.out_bounds_penalty
        else:
            reward = self._process_classification(parameters)

        self.step_count += 1
        num_glances = self.step_count + (1 if self.first_obs == "random" else 0)

        self.terminated = decision == 1 or self.step_count >= self.max_steps
        if self.terminated and decision == 0:
            reward += self.not_classified_penalty
        if self.terminated and num_glances == 1:
            reward += self.not_glanced_penalty
        truncated = False

        if self.terminated:
            self.step_count -= 1

        return self._get_obs(), reward, self.terminated, truncated, self._get_info()

    def render(self):
        return

    def close(self):
        return

    def _get_info(self):
        return {
            "solved": self.solved,
            "object_id": self.object_id,
            "num_glances": self.step_count + (1 if self.first_obs == "random" else 0),
            "num_performed_glances": self.step_count
        }

    def _get_obs(self):
        end = self.obs_offset + self.step_count
        start = 1 if self.first_obs == "empty_signal" and self.step_count > 0 else 0
        return self.glance_positions[start:end], self.glance_pressures[start:end]

    def _process_glance(self, glance_params):
        #pressure, position = self.glance_table.get_pressure_position(self.object_id, tuple(glance_params), zero_centered=True)
        pressure, position = self._get_pressure_position(tuple(glance_params))
        self.glance_pressures[self.step_count + self.obs_offset] = pressure
        self.glance_positions[self.step_count + self.obs_offset] = position

        from haptic_exploration.ml_util import to2D, print_pressure
        if self.verbose:
            print()
            print()
            print(f"*** GLANCE {glance_params} *** ")
            print("OBJECT: ", self.object_id)
            print_pressure(pressure, glance_params=glance_params)
            print(position)

        return self.glance_reward

    def _process_classification(self, classification):
        self.solved = classification == self.object_id
        return self.true_reward if self.solved else self.false_reward

    def _get_pressure_position(self, glance_params):
        raise NotImplementedError()

    def _set_object(self):
        raise NotImplementedError()


class HapticExplorationTableEnv(HapticExplorationEnv):

    def __init__(self, glance_table: GlanceTable, add_noise=True, add_offset=False, **kwargs):
        super().__init__(len(glance_table.id_label), glance_table.n_params, glance_table.position_table.shape[-1], glance_table.pressure_table.shape[-1], **kwargs)
        self.glance_table = glance_table
        self.add_noise = add_noise
        self.apply_offset = add_offset

    def _set_object(self):
        self.offset = self.glance_table.generate_offset(self.object_id) if self.apply_offset else (0, 0)

    def _get_pressure_position(self, glance_params):
        return self.glance_table.get_pressure_position(self.object_id, tuple(glance_params), zero_centered=True, add_noise=self.add_noise, offset=self.offset)


class HapticExplorationSimEnv(HapticExplorationEnv):
    metadata = {}

    def __init__(self, glance_controller: MocapGlanceController, num_params=2, position_dim=7, pressure_dim=256, rt=False, **kwargs):
        super().__init__(glance_controller.object_controller.num_objects, num_params, position_dim, pressure_dim, **kwargs)
        self.glance_controller = glance_controller
        self.rt = rt

    def _set_object(self):
        self.glance_controller.set_object(self.object_id)

    def _get_pressure_position(self, glance_params):

        factors = tuple((p+1)/2 for p in glance_params)
        if isinstance(self.glance_controller.object_controller, SimpleObjectController):
            controller_glance_params = GlanceParameters(x_factor=factors[0], y_angle_factor=factors[1])
        elif isinstance(self.glance_controller.object_controller, CompositeObjectController):
            controller_glance_params = GlanceParameters(x_factor=factors[0], y_factor=factors[1])
        elif isinstance(self.glance_controller.object_controller, YCBObjectController):
            controller_glance_params = GlanceParameters(x_factor=factors[0], y_factor=factors[1], x_angle_factor=factors[2])
        else:
            raise Exception("unsupported object controller")

        pressure, pose = self.glance_controller.perform_glance(controller_glance_params, rt=self.rt)
        position = np.concatenate([pose.point, pose.orientation])

        pressure = pressure_normalization_binary(pressure)
        return pressure, position
