import time
import sys
import os
import numpy as np
import functools
import itertools
import pickle
from pathlib import Path
from math import sqrt

import rospy.rostime
from tqdm import tqdm
from dataclasses import dataclass
from typing import Dict, List, Tuple
from datetime import datetime
from rospkg import RosPack

from haptic_exploration.config import ObjectSet
from haptic_exploration.util import GlanceAreaBB, deg2rad
from haptic_exploration.glance_controller import MocapGlanceController, PandaGlanceController
from haptic_exploration.object_controller import get_object_controller
from haptic_exploration.glance_parameters import GlanceParameters


@dataclass
class SamplingConfig:
    object_set: ObjectSet
    object_dict: Dict[int, str]
    param_resolution: List[Tuple[str, int]]
    glance_area: GlanceAreaBB
    max_angle: float
    z_clearance: float
    object_rotation_delta: float


def get_sampling_dir(sc: SamplingConfig):
    sampling_resolutions = [sampling_res for _, sampling_res in sc.param_resolution]

    sampling_resolutions_str = 'x'.join([str(sample_res) for sample_res in sampling_resolutions])
    sampling_dir = Path(
        RosPack().get_path("haptic_exploration"),
        "haptic_sampling",
        "_".join([sc.object_set.value, sampling_resolutions_str])
    )
    return sampling_dir


def sample_objects(sc: SamplingConfig, use_panda=False, footprints=None, myrmex_dim=16):
    if use_panda:
        glance_controller = PandaGlanceController(get_object_controller(sc.object_set), sc.glance_area, sc.max_angle, sc.z_clearance)
        time.sleep(1)
    else:
        glance_controller = MocapGlanceController(get_object_controller(sc.object_set), sc.glance_area, sc.max_angle, sc.z_clearance)

    names = [name for name, _ in sc.param_resolution]
    sampling_resolutions = [sampling_res for _, sampling_res in sc.param_resolution]

    sampling_resolutions_str = 'x'.join([str(sample_res) for sample_res in sampling_resolutions])
    sampling_dir = get_sampling_dir(sc)
    sampling_dir.mkdir(parents=True, exist_ok=True)

    limits = [sc.glance_area.x_limits, sc.glance_area.y_limits] + [(-sc.max_angle, sc.max_angle)] * (len(names) - 2)
    dim_values = [list(zip(np.arange(sampling_res), np.linspace(limit_low, limit_high, sampling_res), np.linspace(0, 1, sampling_res))) for sampling_res, (limit_low, limit_high) in zip(sampling_resolutions, limits)]
    param_spec = [(name, [v[1] for v in values]) for name, values in zip(names, dim_values)]
    sensor_diff = 0.02  # adjust for smaller myrmex sensor (2x2cm) as footprints are calculated for 10x10cm sensor
    object_radii = {object_name: max(-x0, x1, -y0, y1) - sensor_diff for object_name, ((x0, x1), (y0, y1)) in footprints.items()}
    error = 1e-5

    rotations = np.linspace(0, deg2rad(90), round(deg2rad(90) / sc.object_rotation_delta + 1)) if sc.object_rotation_delta > 0 else [0]

    if footprints is not None:
        print("Footprints:")
        for object_name, footprint in footprints.items():
            print(f"Object {object_name}: indices {footprint}, radius {object_radii[object_name]}")

    if not glance_controller.wait_for_sim():
        rospy.logerr("Model could not be loaded!")
        sys.exit(-1)

    with tqdm(total=len(sc.object_dict)*functools.reduce(lambda a, b: a*b, sampling_resolutions)*len(rotations)) as pbar:

        for i, (object_index, object_name) in enumerate(sorted(sc.object_dict.items(), key=lambda e: e[0])):
            pbar.set_description(f"Sampling {len(sc.object_dict)} objects with resolution {sampling_resolutions_str}")

            object_radius = object_radii[object_name]
            for rotation_idx, rotation in enumerate(rotations):

                glance_controller.set_object(object_index, rotation=rotation)

                object_pressure_table = np.zeros([len(rotations)] + sampling_resolutions + [myrmex_dim], dtype=float)
                object_position_table = np.zeros([len(rotations)] + sampling_resolutions + [7], dtype=float)

                for arg_spec in itertools.product(*dim_values):
                    indices, values, value_factors = zip(*arg_spec)

                    x, y = values[:2]
                    dist = max(abs(x), abs(y))
                    if not (dist <= object_radius + error):
                        continue

                    rotation_deg = rotation * 180 / np.pi
                    pbar.set_postfix_str(f"{object_name} ({i+1}/{len(sc.object_dict)}), theta={rotation_deg:.2f} ({rotation_idx+1}/{len(rotations)-1}) " + ", ".join(f'{name}={value:.3f}' for name, value in zip(names, values)))

                    glance_kwargs = {f"{name}_factor": v_factor for name, v_factor in zip(names, value_factors)}
                    glance_params = GlanceParameters(**glance_kwargs)
                    max_values, pose = glance_controller.perform_glance(glance_params)
                    object_pressure_table[(rotation_idx,) + tuple(indices)] = max_values
                    object_position_table[(rotation_idx,) + tuple(indices)] = np.concatenate([pose.point, pose.orientation])
                    pbar.update()

                with open(sampling_dir / f"{object_name}.pkl", "wb") as file:
                    object_data = object_name, param_spec, object_pressure_table, object_position_table, sc, object_radii
                    pickle.dump(object_data, file)
