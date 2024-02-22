import os
import json
import pickle
import numpy as np

from random import randint
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

        for filename in os.listdir(object_dir):
            if filename.endswith(".pkl"):
                with open(os.path.join(object_dir, filename), "rb") as object_file:
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

        self.object_indices = dict()
        self.object_nonzero_xy_count = dict()
        self.object_nonzero_total_count = dict()
        for object_id in self.id_label.keys():
            x0, x1 = np.where(self.pressure_table[object_id].sum(axis=(1, 2, 3)) > 0)[0][[0, -1]]
            y0, y1 = np.where(self.pressure_table[object_id].sum(axis=(0, 2, 3)) > 0)[0][[0, -1]]
            self.object_indices[object_id] = (x0, x1), (y0, y1)
            self.object_nonzero_total_count[object_id] = self.pressure_table[object_id].sum(axis=(3))
            self.object_nonzero_xy_count[object_id] = self.pressure_table[object_id].sum(axis=(2, 3))
            print(f"{self.id_label[object_id]} indices: x={x0, x1}, y={y0, y1}, footprint={x1-x0, y1-y0}, nonzero_xy={(np.sum(self.object_nonzero_xy_count[object_id] > 0)/self.object_nonzero_xy_count[object_id].size*100):.2f}%")

    def _get_indices(self, params_normalized):
        return tuple(round(param_normalized * (param_resolution - 1)) for param_normalized, param_resolution in zip(params_normalized, self.param_resolution))

    def get_pressure_position(self, object_id, params_normalized, zero_centered=False, add_noise=False, offset=(0, 0)):
        if zero_centered:
            params_normalized = [(param+1)/2 for param in params_normalized]
        indices = self._get_indices(params_normalized)
        return self.get_pressure_position_indices(object_id, indices, add_noise=add_noise, offset=offset)

    def get_pressure_position_indices(self, object_id, indices, add_noise=False, offset=(0, 0)):

        indices1 = [idx-idx_o for idx, idx_o in zip(indices, offset)] + list(indices[len(offset):])
        boundary_offset = np.zeros(2, dtype=int)
        for i, (idx, res) in enumerate(zip(indices1[:2], self.param_resolution)):
            if idx < 0:
                boundary_offset[i] = -idx
            elif idx >= res:
                boundary_offset[i] = - idx - 1 + res
        indices2 = [idx+idx_b for idx, idx_b in zip(indices1, boundary_offset)] + list(indices1[len(boundary_offset):])
        indices2 = tuple(indices2)

        position_offset = np.array([(idx_o - idx_b) * (1/(res-1)) for idx_o, idx_b, res in zip(offset, boundary_offset, self.param_resolution)])

        indices2 = (object_id,) + indices2
        pressure = self.pressure_table[indices2].copy()
        position = self.position_table[indices2].copy()
        position[:position_offset.shape[0]] += position_offset
        if add_noise:
            position = apply_position_noise(position, self.glance_area, TRANSLATION_STD_M, ROTATION_STD_DEG)
        return pressure, position

    def get_params(self, params_normalized):
        exact_params = tuple(self.param_values[param][param_idx] for param, param_idx in enumerate(self._get_indices(params_normalized)))
        table_params = tuple(param_min + (param_max - param_min) * param_normalized for param_normalized, (param_min, param_max) in zip(params_normalized, self.param_ranges))
        return exact_params, table_params

    def generate_offset(self, object_id):
        max_x, max_y = self.param_resolution[:2]
        (x0, x1), (y0, y1) = self.object_indices[object_id]

        return randint(-x0, max_x -1 - x1), randint(-y0, max_y - 1 - y1)
