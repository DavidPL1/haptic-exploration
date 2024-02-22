import time
import sys
import os
import numpy as np
import functools
import itertools
import pickle
from pathlib import Path

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
    param_resolution: List[Tuple[int, int]]
    glance_area: GlanceAreaBB
    max_angle: float
    z_clearance: float

def get_sampling_dir(sc: SamplingConfig):
    sampling_resolutions = [sampling_res for _, sampling_res in sc.param_resolution]

    sampling_resolutions_str = 'x'.join([str(sample_res) for sample_res in sampling_resolutions])
    sampling_dir = Path(
        RosPack().get_path("haptic_exploration"),
        "haptic_sampling",
        "_".join([sc.object_set.value, sampling_resolutions_str])
    )
    return sampling_dir

def sample_objects(sc: SamplingConfig, use_panda=False):
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

    limits = [sc.max_angle if "angle" in name else 1 for name in names]
    dim_values = [list(zip(np.arange(sampling_res), np.linspace(-limit, limit, sampling_res), np.linspace(0, 1, sampling_res))) for sampling_res, limit in zip(sampling_resolutions, limits)]
    param_spec = [(name, [v[2] for v in values]) for name, values in zip(names, dim_values)] # TODO: v[1] instead of v[2] for scaled values

    if not glance_controller.wait_for_sim():
        rospy.logerr("Model could not be loaded!")
        sys.exit(-1)

    with tqdm(total=len(sc.object_dict)*functools.reduce(lambda a, b: a*b, sampling_resolutions)) as pbar:

        for i, (object_index, object_name) in enumerate(sorted(sc.object_dict.items(), key=lambda e: e[0])):
            pbar.set_description(f"Sampling {len(sc.object_dict)} objects with resolution {sampling_resolutions_str}")

            glance_controller.set_object(object_index)

            object_pressure_table = np.zeros(sampling_resolutions + [256], dtype=float)
            object_position_table = np.zeros(sampling_resolutions + [7], dtype=float)

            for arg_spec in itertools.product(*dim_values):
                indices, values, value_factors = zip(*arg_spec)
                pbar.set_postfix_str(f"{object_name} ({i+1}/{len(sc.object_dict)}), " + ", ".join(f'{name}={value:.3f}' for name, value in zip(names, values)))

                glance_kwargs = {f"{name}_factor": v_factor for name, v_factor in zip(names, value_factors)}
                glance_params = GlanceParameters(**glance_kwargs)
                max_values, pose = glance_controller.perform_glance(glance_params)
                object_pressure_table[tuple(indices)] = max_values
                object_position_table[tuple(indices)] = np.concatenate([pose.point, pose.orientation])
                pbar.update()

            with open(sampling_dir / f"{object_name}.pkl", "wb") as file:
                object_data = object_name, param_spec, object_pressure_table, object_position_table
                pickle.dump(object_data, file)
