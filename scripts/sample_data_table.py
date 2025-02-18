import json
import pickle
import numpy as np
import functools
import itertools
from datetime import datetime
from tqdm import tqdm
from pathlib import Path
from haptic_exploration.config import ObjectSet, GLANCE_AREA, SAMPLING_SPEC

from haptic_exploration.glance_controller import MocapGlanceController
from haptic_exploration.glance_parameters import GlanceParameters
from haptic_exploration.ml_util import print_pressure
from haptic_exploration.util import deg2rad


def sample(glance_controller, objects, dim_spec, max_angle):

    sample_sizes = [sample_size for _, sample_size in dim_spec]
    sample_sizes_str = 'x'.join([str(sample_size) for sample_size in sample_sizes])
    sampling_id = datetime.now().strftime(f"%Y_%m_%d_%H-%M-%S_simple_{sample_sizes_str}")
    sampling_dir = Path(f"../haptic_sampling/{sampling_id}")
    sampling_dir.mkdir(parents=True, exist_ok=True)

    names = [name for name, _ in dim_spec]
    limits = [max_angle if "angle" in name else 1 for name in names]
    dim_values = [list(zip(np.arange(sample_size), np.linspace(-limit, limit, sample_size), np.linspace(0, 1, sample_size))) for (name, sample_size), limit in zip(dim_spec, limits)]
    param_spec = [(spec[0], [v[2] for v in values]) for spec, values in zip(dim_spec, dim_values)]

    with tqdm(total=len(objects)*functools.reduce(lambda a, b: a*b, sample_sizes)) as pbar:
        pbar.set_description(f"Sample Glances {', '.join(names)} ({sample_sizes_str})")

        for object_index, object_name in objects.items():
            glance_controller.set_object(object_index)
            object_pressure_table = np.zeros(sample_sizes + [256], dtype=float)
            object_position_table = np.zeros(sample_sizes + [7], dtype=float)

            for arg_spec in itertools.product(*dim_values):
                indices, values, value_factors = zip(*arg_spec)
                pbar.set_postfix_str(f"object={object_name}, id={object_index}, {', '.join(f'{name}={value:.3f}' for name, value in zip(names, values))}")

                glance_kwargs = {f"{name}_factor": v_factor for name, v_factor in zip(names, value_factors)}
                glance_params = GlanceParameters(**glance_kwargs)
                max_values, pose = glance_controller.perform_glance(glance_params)
                object_pressure_table[tuple(indices)] = max_values
                object_position_table[tuple(indices)] = np.concatenate([pose.point, pose.orientation])
                pbar.update()

                #print_pressure(max_values, glance_params=glance_params)
                #print(pose)

            with open(sampling_dir / f"{object_name}.pkl", "wb") as file:
                object_data = object_name, param_spec, object_pressure_table, object_position_table
                pickle.dump(object_data, file)


def sample_objects(object_set: ObjectSet):

    glance_area = GLANCE_AREA[object_set]
    object_controller_class, objects, max_angle_deg, z_clearance, dim_spec, object_controller_kwargs = SAMPLING_SPEC[object_set]
    glc = MocapGlanceController(object_controller_class(**object_controller_kwargs), glance_area, deg2rad(max_angle_deg), z_clearance)

    sample(glc, objects, dim_spec, deg2rad(max_angle_deg))


if __name__ == "__main__":
    sample_objects(ObjectSet.Composite)
