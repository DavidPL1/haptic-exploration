import numpy as np
import random

from tqdm import tqdm
from torch.utils.data import Dataset
from haptic_exploration.preprocessing import pressure_normalization_binary
from haptic_exploration.ml_util import DataProperties, get_empty_glance, unzip
from haptic_exploration.composite import get_feature_position_param


def generate_random_glance_sequences(glance_table, num_glances, num_object_samples, add_noise=False):

    random_glance_sequences = {cls_idx: [] for cls_idx in range(len(glance_table.id_label.keys()))}

    with tqdm(total=len(glance_table.id_label)*num_object_samples, desc="Generate Random Glance Sequences") as pbar:
        for object_idx in glance_table.id_label.keys():
            for i in range(num_object_samples):
                if num_glances == 0:
                    pressure_sequence, position_sequence, params_sequence = [], [], []
                else:
                    params_sequence = [tuple(random.random() for _ in range(glance_table.n_params)) for _ in range(num_glances)]
                    pressure_position_sequence = [glance_table.get_pressure_position(object_idx, params, add_noise=add_noise) for params in params_sequence]
                    pressure_sequence, position_sequence = unzip(pressure_position_sequence)
                random_glance_sequences[object_idx].append((pressure_sequence, position_sequence, params_sequence))
                pbar.update()

    return random_glance_sequences


def generate_position_glance_sequences(glance_table, num_glances, num_object_samples, add_empty_glance=False):

    position_glance_sequences = {cls_idx: [] for cls_idx in range(len(glance_table.id_label.keys()))}

    positions = list(range(4))
    with tqdm(total=len(glance_table.id_label)*num_object_samples, desc="Generate Position Glance Sequences") as pbar:
        for object_idx in glance_table.id_label.keys():
            for i in range(num_object_samples):
                positions = random.sample(positions, k=num_glances)
                params_sequence = [get_feature_position_param(position_idx) for position_idx in positions]
                pressure_sequence, position_sequence = list(zip(*[glance_table.get_pressure_position(object_idx, param) for param in params_sequence]))
                pressure_sequence = list(pressure_sequence)
                position_sequence = list(position_sequence)
                if add_empty_glance:
                    empty_position, empty_pressure = get_empty_glance(glance_table.position_table.shape[-1], glance_table.pressure_table.shape[-1])
                    pressure_sequence = [empty_pressure] + pressure_sequence
                    position_sequence = [empty_position] + position_sequence
                position_glance_sequences[object_idx].append((pressure_sequence, position_sequence, params_sequence))
                pbar.update()

    return position_glance_sequences


def generate_pressure_input(glance, pressure_normalization=None):
    """ Generate the pressure input """

    pressure_values = glance.values

    if pressure_normalization is None:
        return glance.values
    else:
        return pressure_normalization(pressure_values)


def generate_position_input(glance):
    """ Generate the position input """

    return np.concatenate(glance.pose)


class HapticGlanceDataset(Dataset):
    """ Dataset containing haptic glance sequences """

    def __init__(self, object_glance_sequences, labels_map, pressure_normalization=None) -> None:

        self.labels_map = labels_map

        self.data = []
        for object_id in self.labels_map.keys():
            for pressure_sequence, position_sequence, params in object_glance_sequences[object_id]:
                if pressure_normalization is not None:
                    pressure_sequence = [pressure_normalization(pressure_values) if pressure_values[0] >= 0 else pressure_values for pressure_values in pressure_sequence]
                    self.data.append((np.array(position_sequence).astype(np.float32), np.array(pressure_sequence).astype(np.float32), object_id))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def generate_dataset(
        type,
        glance_table,
        num_glances,
        num_samples_train,
        num_samples_test,
        pressure_normalization=pressure_normalization_binary,
        add_noise=False):

    assert type in ["random", "position"]

    if type == "random":
        print("GENERATE RANDOM SEQUENCE DATASET... ", end="")
        # generate random glance sequences
        sequences_train = generate_random_glance_sequences(glance_table, num_glances, num_samples_train, add_noise=add_noise)
        sequences_test = generate_random_glance_sequences(glance_table, num_glances, num_samples_test, add_noise=add_noise)
    else:
        print("GENERATE POSITION SEQUENCE DATASET FOR COMPOSITE... ", end="")
        # generate random glance sequences
        sequences_train = generate_position_glance_sequences(glance_table, num_glances, num_samples_train)
        sequences_test = generate_position_glance_sequences(glance_table, num_glances, num_samples_test)

    # create actual datasets
    dataset_train = HapticGlanceDataset(sequences_train, glance_table.id_label, pressure_normalization=pressure_normalization)
    dataset_test = HapticGlanceDataset(sequences_test, glance_table.id_label, pressure_normalization=pressure_normalization)

    # compute data properties
    position_sequence, pressure_sequence, _ = dataset_train[0]
    data_properties = DataProperties(glance_table.position_table.shape[-1], glance_table.pressure_table.shape[-1], [num_glances], len(glance_table.id_label.keys()), len(glance_table.position_table.shape) - 2)

    print("Train samples:", len(dataset_train), "Test samples:", len(dataset_test))
    print("Labels:", glance_table.id_label)
    print("Properties:", data_properties)

    return dataset_train, dataset_test, data_properties
