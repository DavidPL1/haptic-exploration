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
from haptic_exploration.ml_util import rotation_matrix_2d


def crop_table(table, rescale_position=False):
    # calc center square crop
    xy_dim = table.shape[1:3]
    square_dim = min(xy_dim)
    (xl, xh), (yl, yh) = [(int((dim - square_dim) / 2), int(dim - (dim - square_dim) / 2)) for dim in xy_dim]

    if rescale_position:
        for dim_id, dim_size in enumerate(xy_dim):
            if dim_size > square_dim:
                factor = (dim_size - 1) / (square_dim - 1)
                table[:, :, :, dim_id] = (table[:, :, :, dim_id] - 0.5) * factor + 0.5

    return table[:, xl:xh, yl:yh, :]


class GlanceTable:

    def __init__(self, object_set, pressure_normalization=pressure_normalization_binary):

        self.object_set = object_set
        object_dir = OBJECT_PATHS[object_set]

        object_data_all = []
        empty_object_data = None

        for filename in os.listdir(object_dir):
            if filename.endswith(".pkl"):
                with open(os.path.join(object_dir, filename), "rb") as object_file:
                    object_data = pickle.load(object_file)
                    if object_data[0] == "empty":
                        empty_object_data = object_data
                    else:
                        object_data_all.append(object_data)

        try:
            object_data_all = sorted(object_data_all, key=lambda d: int(d[0].split("_")[0]))
        except:
            object_data_all = sorted(object_data_all, key=lambda d: d[0])

        # stuff for both non-rotation and rotation sets
        self.id_label = {idx: object_data[0] for idx, object_data in enumerate(object_data_all)}
        self.label_id = {label: idx for idx, label in self.id_label.items()}
        self.num_objects = len(self.id_label)

        _, param_spec, pressure_table0, position_table0 = object_data_all[0][:4]
        if "rot" in object_set.value:
            pressure_table0 = crop_table(pressure_table0.copy())
            position_table0 = crop_table(position_table0.copy(), rescale_position=True)
            # TODO: adjust next stuff according to cropped tables
        self.n_params = len(param_spec)
        self.param_names = [name for name, _ in param_spec]
        # self.param_values = [values for _, values in param_spec]
        # self.param_ranges = [(min(values), max(values)) for values in self.param_values]
        if "rot" in object_set.value:
            self.param_resolution = pressure_table0.shape[1:self.n_params+1]
        else:
            self.param_resolution = [len(values) for _, values in param_spec]

        # handle rotation
        if "rot" in object_set.value:
            self.rot = True
            self.sc = object_data_all[0][4]

            self.n_rotations = pressure_table0.shape[0] * 4
            n_rotations_quarter = round(self.n_rotations/4)

            self.pressure_table = np.zeros((len(object_data_all),) + (self.n_rotations,) + pressure_table0.shape[1:])
            self.position_table = np.zeros((len(object_data_all),) + (self.n_rotations,) + position_table0.shape[1:])

            empty_pressure_table = crop_table(np.repeat(np.expand_dims(empty_object_data[2][0], 0), n_rotations_quarter, axis=0))
            empty_position_table = crop_table(np.repeat(np.expand_dims(empty_object_data[3][0], 0), n_rotations_quarter, axis=0), rescale_position=True)

            for object_data in object_data_all:

                object_label, _, object_pressure_table_footprint, object_position_table_footprint = object_data[:4]
                object_id = self.label_id[object_label]

                footprint_mask = object_position_table_footprint.sum(axis=-1) != 0
                footprint_mask = crop_table(np.expand_dims(footprint_mask, -1)).squeeze().astype("float")
                object_pressure_table_footprint = crop_table(object_pressure_table_footprint)
                object_position_table_footprint = crop_table(object_position_table_footprint, rescale_position=True)

                object_pressure_table = empty_pressure_table.copy()
                object_position_table = empty_position_table.copy()

                footprint_pressure_mask = np.repeat(np.expand_dims(footprint_mask, -1), object_pressure_table.shape[-1], axis=3)
                footprint_position_mask = np.repeat(np.expand_dims(footprint_mask, -1), object_position_table.shape[-1], axis=3)
                object_pressure_table -= np.multiply(object_pressure_table, footprint_pressure_mask)
                object_position_table -= np.multiply(object_position_table, footprint_position_mask)

                object_pressure_table += np.multiply(object_pressure_table_footprint, footprint_pressure_mask)
                object_position_table += np.multiply(object_position_table_footprint, footprint_position_mask)

                self.pressure_table[object_id, :n_rotations_quarter] = object_pressure_table
                self.position_table[object_id, :n_rotations_quarter] = object_position_table

                for i in range(1, 4):

                    quarter_start, quarter_end = n_rotations_quarter*i, n_rotations_quarter*(i+1)

                    # init with default first (to init zero glances outside of square)
                    self.pressure_table[object_id, quarter_start:quarter_end] = object_pressure_table
                    self.position_table[object_id, quarter_start:quarter_end] = object_position_table

                    # PRESSURE
                    rotated_index = np.rot90(object_pressure_table, k=i, axes=(1, 2))
                    d_sensor = round(np.sqrt(object_pressure_table.shape[-1]))
                    rotated_index_square = rotated_index.reshape(*object_pressure_table.shape[:-1], d_sensor, d_sensor)
                    rotated_index_pressure_square = np.rot90(rotated_index_square, k=i, axes=(3, 4))
                    rotated_index_pressure = rotated_index_pressure_square.reshape(*object_pressure_table.shape)
                    self.pressure_table[object_id, quarter_start:quarter_end] = rotated_index_pressure

                    # POSITION
                    # first, rotate indices of position
                    initial_xy = object_position_table[:, :, :, :2]
                    rotated_xy = np.rot90(initial_xy, k=i, axes=(1, 2))
                    # rotate x,y coordinate around center
                    R = rotation_matrix_2d(i * np.pi/2)
                    flattened_xy = rotated_xy.reshape(-1, 2)
                    flattened_xy_rotated = R.dot(flattened_xy.T - 0.5).T + 0.5
                    reshaped_xy_rotated = flattened_xy_rotated.reshape(*object_position_table.shape[:3], 2)
                    self.position_table[object_id, quarter_start:quarter_end, :, :, :2] = reshaped_xy_rotated

        else:
            self.rot = False

            self.pressure_table = np.zeros((len(object_data_all),) + pressure_table0.shape)
            self.position_table = np.zeros((len(object_data_all),) + position_table0.shape)

            for object_data in object_data_all:
                object_label, _, object_pressure_table, object_position_table = object_data[:4]
                object_id = self.label_id[object_label]
                self.pressure_table[object_id] = object_pressure_table
                self.position_table[object_id] = object_position_table

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

                print(f"{self.id_label[object_id]} indices: x={x0, x1}, y={y0, y1}, footprint={x1-x0+1, y1-y0+1}, nonzero_xy={(np.sum(self.object_nonzero_xy_count[object_id] > 0)/self.object_nonzero_xy_count[object_id].size*100):.2f}%")

    def _get_rotation_index(self, rotation):
        rotation_idx = round(rotation/(2*np.pi) * self.n_rotations)
        rotation_idx = rotation_idx % self.n_rotations
        return rotation_idx

    def _get_indices(self, params_normalized):
        return tuple(round(param_normalized * (param_resolution - 1)) for param_normalized, param_resolution in zip(params_normalized, self.param_resolution))

    def get_pressure_position(self, object_id, params_normalized, zero_centered=False, add_noise=False, offset=(0, 0), rotation=0):
        if zero_centered:
            params_normalized = [(param+1)/2 for param in params_normalized]
        indices = self._get_indices(params_normalized)
        return self.get_pressure_position_indices(object_id, indices, add_noise=add_noise, offset=offset, rotation=rotation)

    def get_pressure_position_indices(self, object_id, indices, add_noise=False, offset=(0, 0), rotation=0):

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

        if "rot" in self.object_set.value:
            rotation_idx = self._get_rotation_index(rotation)
            indices2 = (object_id, rotation_idx) + indices2
        else:
            indices2 = (object_id,) + indices2
        pressure = self.pressure_table[indices2].copy()
        position = self.position_table[indices2].copy()
        position[:position_offset.shape[0]] += position_offset
        if add_noise:
            position = apply_position_noise(position, self.glance_area, TRANSLATION_STD_M, ROTATION_STD_DEG)
        return pressure, position

    """
    def get_params(self, params_normalized):
        exact_params = tuple(self.param_values[param][param_idx] for param, param_idx in enumerate(self._get_indices(params_normalized)))
        table_params = tuple(param_min + (param_max - param_min) * param_normalized for param_normalized, (param_min, param_max) in zip(params_normalized, self.param_ranges))
        return exact_params, table_params
    """

    def generate_offset(self, object_id):
        max_x, max_y = self.param_resolution[:2]
        (x0, x1), (y0, y1) = self.object_indices[object_id]

        return randint(-x0, max_x -1 - x1), randint(-y0, max_y - 1 - y1)
