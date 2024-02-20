import torch
import numpy as np

from math import sqrt, pi
from torchvision.transforms import Resize
from pyquaternion import Quaternion
from haptic_exploration.util import GlanceAreaBB


def pressure_normalization_constant(pressure_values):
    return pressure_values / 0.05


def pressure_normalization_binary(pressure_values):
    return pressure_values.astype(bool).astype(np.float32)


def resize_pressure(glance_values, resolution):
    old_resolution = round(sqrt(glance_values.shape[0]))
    reshaped = torch.Tensor(np.reshape(glance_values, (old_resolution, old_resolution)))
    reshaped = reshaped.unsqueeze(0)
    resized = Resize((resolution, resolution), antialias=True)(torch.Tensor(reshaped))
    resized = resized.squeeze()
    return np.array(resized).reshape((resolution ** 2,))


def apply_position_noise(position, glance_area: GlanceAreaBB, translation_std_m=0.002, rotation_std_deg=1.5):
    """
    - translation_std_m is in real world units (meter): 0.01m -> 1.0cm, 0.005m -> 0.5cm
        -> we have to scale translation noise with bb size, given by bb limits in world coords
    - rotation_std is in degree
    """
    translation, rotation = position[:3], position[3:]

    component_scales = [limits[1] - limits[0] for limits in [glance_area.x_limits, glance_area.y_limits, glance_area.z_limits]]
    translation_std_bb = translation_std_m / np.array(component_scales)
    translation_noise = np.random.normal(0, translation_std_bb, 3)
    translation_new = translation + translation_noise

    axis = np.random.random(3)
    axis /= np.linalg.norm(axis)
    rotation_std_rad = rotation_std_deg * (pi/180)
    angle = np.random.normal(0, rotation_std_rad, 1)
    rotation_quat = Quaternion(rotation)
    rotation_noise = Quaternion(axis=axis, angle=angle)
    rotation_quat_new = rotation_quat * rotation_noise

    return np.concatenate((translation_new, rotation_quat_new.elements))

