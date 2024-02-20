import os
import json
import pickle
import numpy as np

from pathlib import Path
from typing import Tuple
from collections import defaultdict
from haptic_exploration.preprocessing import resize_pressure, pressure_normalization_binary, apply_position_noise
from haptic_exploration.config import OBJECT_PATHS, GLANCE_AREA, TRANSLATION_STD_M, ROTATION_STD_DEG


class HapticGlance:
    """ Represents a Haptic Glance with parameters, sensor pose and pressure values"""

    def __init__(self, parameters: dict, pose: Tuple[np.ndarray, np.ndarray], values: np.ndarray) -> None:
        self.parameters = parameters
        self.pose = pose
        self.values = values

    @staticmethod
    def from_json(glance_json) -> "HapticGlance":
        parameters = {param: np.float32(value) for param, value in glance_json["parameters"].items()}
        pose = (np.array(glance_json["pose"]["point"], dtype=np.float32), np.array(glance_json["pose"]["orientation"], dtype=np.float32))
        values = np.array(glance_json["values"], dtype=np.float32)
        return HapticGlance(parameters, pose, values)


def load_glance_json(filepath) -> HapticGlance:
    """ Load a glance from json file into a python dict """

    with open(filepath, "r") as glance_file:
        glance_json = json.load(glance_file)
        return HapticGlance.from_json(glance_json)


def load_data_json(path, resolution=None) -> dict:
    """ Load presampled glances of multiple objects from json format into dict """

    print("LOADING GLANCE DATA FROM JSON... ", end="")

    json_data_dir = Path(path) / "json"

    data = defaultdict(list)
    for object_dir in json_data_dir.iterdir():
        for glance_json_file in object_dir.iterdir():
            glance = load_glance_json(glance_json_file)
            if resolution is not None:
                glance.values = resize_pressure(glance.values, resolution)
            data[object_dir.name].append(glance)

    print("DONE")

    return data


class GlanceTable:

    def __init__(self, object_set, pressure_normalization=pressure_normalization_binary):

        self.object_set = object_set
        object_dir = OBJECT_PATHS[object_set]

        object_data = []

        for filename in os.listdir(os.path.join(object_dir, "pkl")):
            if filename.endswith(".pkl"):
                with open(os.path.join(object_dir, "pkl", filename), "rb") as object_file:
                    object_data.append(pickle.load(object_file))

        try:
            object_data = sorted(object_data, key=lambda d: int(d[0].split("_")[0]))
        except:
            object_data = sorted(object_data, key=lambda d: d[0])

        self.id_label = {idx: object_label for idx, (object_label, _, _, _) in enumerate(object_data)}
        self.label_id = {label: idx for idx, label in self.id_label.items()}
        self.num_objects = len(self.id_label)

        _, param_spec, pressure_table, position_table = object_data[0]
        self.n_params = len(param_spec)
        self.param_names = [name for name, _ in param_spec]
        self.param_values = [values for _, values in param_spec]
        self.param_ranges = [(min(values), max(values)) for values in self.param_values]
        self.param_resolution = [len(values) for _, values in param_spec]

        self.pressure_table = np.zeros((len(object_data),) + pressure_table.shape)
        self.position_table = np.zeros((len(object_data),) + position_table.shape)

        for object_label, _, pressure_table, position_table in object_data:
            object_id = self.label_id[object_label]
            self.pressure_table[object_id] = pressure_table
            self.position_table[object_id] = position_table

        if pressure_normalization is not None:
            self.pressure_table = pressure_normalization(self.pressure_table)

        self.glance_area = GLANCE_AREA[object_set]

    def _get_param_indices(self, params_normalized):
        return tuple(round(param_normalized * (param_resolution - 1)) for param_normalized, param_resolution in zip(params_normalized, self.param_resolution))

    def get_pressure_position(self, object_id, params_normalized, zero_centered=False, add_noise=False):
        if zero_centered:
            params_normalized = [(param+1)/2 for param in params_normalized]
        indices = (object_id,) + self._get_param_indices(params_normalized)
        pressure = self.pressure_table[indices]
        position = self.position_table[indices]
        if add_noise:
            position = apply_position_noise(position, self.glance_area, TRANSLATION_STD_M, ROTATION_STD_DEG)
        return pressure, position

    def get_params(self, params_normalized):
        exact_params = tuple(self.param_values[param][param_idx] for param, param_idx in enumerate(self._get_param_indices(params_normalized)))
        table_params = tuple(param_min + (param_max - param_min) * param_normalized for param_normalized, (param_min, param_max) in zip(params_normalized, self.param_ranges))
        return exact_params, table_params
