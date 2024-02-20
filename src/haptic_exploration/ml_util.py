import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import random
import pickle

from math import sqrt
from enum import Enum
from torch import nn
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import List, Iterable
from copy import deepcopy
from torchinfo import summary

from haptic_exploration.model import ModelParameters
from haptic_exploration.config import MODEL_SAVE_PATH, WAIT_PLOTS


plt_colors = ["blue", "green", "red", "cyan", "magenta", "yellow"]


class ModelType(Enum):
    LSTM = "lstm"
    Transformer = "transformer"


@dataclass
class DataProperties:

    input_position_dim: int
    input_pressure_dim: int
    n_glances: List[int]
    n_objects: int
    n_glance_params: int


# only used for torchinfo summary compatibility
class PackInputModel(nn.Module):

    def __init__(self, model):
        super(PackInputModel, self).__init__()
        self.model = model

    def forward(self, x1, x2):
        return self.model((x1, x2))


class ModelTrainingMonitor:

    def __init__(self):
        self.losses = []
        self.accuracies = []
        # (model_weights, loss, accuracy, index)
        self.best_model_loss = None
        self.best_model_accuracy = None

    def process_episode(self, model_weights, val_loss, val_accuracy):
        self.losses.append(val_loss)
        self.accuracies.append(val_accuracy)
        if self.best_model_loss is None or val_loss < self.best_model_loss[1]:
            self.best_model_loss = (deepcopy(model_weights), val_loss, val_accuracy, len(self.losses) - 1)
        if self.best_model_accuracy is None or val_accuracy >= self.best_model_accuracy[2]:
            self.best_model_accuracy = (deepcopy(model_weights), val_loss, val_accuracy, len(self.accuracies) - 1)

    def print_results(self):
        print(f"##### TRAINING RESULTS ({len(self.losses)} episodes) #####")
        print("Validation Accuracies: ", end="")
        print(", ".join(f"{ep}: {(acc*100):.2f}%" for ep, acc in zip(range(len(self.accuracies)), self.accuracies)))
        print("Validation Losses: ", end="")
        print(", ".join(f"{ep}: {loss:.3f}" for ep, loss in zip(range(len(self.accuracies)), self.losses)))
        print("Best Accuracy:", f"{(self.best_model_accuracy[2]*100):.2f}%", f"(episode {self.best_model_accuracy[3]})")
        print("Best Loss:", self.best_model_loss[1], f"(episode {self.best_model_loss[3]})")
        print()


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")
    return device


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def print_summary(model, model_params: ModelParameters):

    print("SUMMARY OF MODEL:", type(model))
    position_shape = (1, 3, model_params.position_input_dim)
    pressure_shape = (1, 3, model_params.pressure_input_dim)
    summary(PackInputModel(model), [position_shape, pressure_shape], depth=10, device="cpu")


def get_model_params_env(table_env, position_embedded_dim, pressure_embedded_dim, base_output_dim, n_glances):
    return ModelParameters(table_env.glance_table.position_table.shape[-1],
                           table_env.glance_table.pressure_table.shape[-1],
                           position_embedded_dim,
                           pressure_embedded_dim,
                           position_embedded_dim + pressure_embedded_dim,
                           base_output_dim,
                           n_glances,
                           table_env.num_objects,
                           len(table_env.glance_table.param_names))


def get_model_params_dataset(data_properties: DataProperties, position_embedded_dim, pressure_embedded_dim, base_output_dim):
    return ModelParameters(data_properties.input_position_dim,
                           data_properties.input_pressure_dim,
                           position_embedded_dim,
                           pressure_embedded_dim,
                           position_embedded_dim + pressure_embedded_dim,
                           base_output_dim,
                           data_properties.n_glances,
                           data_properties.n_objects,
                           data_properties.n_glance_params)


def get_time_str():
    return datetime.now().strftime(f"%Y_%m_%d_%H-%M-%S")


def save_model_weights(model, object_set, save_description):
    dir = os.path.join(MODEL_SAVE_PATH, object_set.value, "cls")
    Path(dir).mkdir(parents=True, exist_ok=True)
    state_dict = model.state_dict()
    torch.save(state_dict, os.path.join(dir, f"weights_{get_time_str()}_{save_description}"))


def save_best_model_weights(model, training_monitor, object_set, save_description):
    best_model_weights, best_loss, best_accuracy, index = training_monitor.best_model_accuracy
    model.load_state_dict(best_model_weights)
    print("Saving model with best accuracy: ", f"loss: {best_loss:.2f}", f"accuracy: {(best_accuracy*100):.2f}%", f"episode: {index}")
    save_model_weights(model, object_set, save_description)


def load_model_weights(model, filepath, strict=True):
    state_dict = torch.load(os.path.join(MODEL_SAVE_PATH, filepath))
    model.load_state_dict(state_dict, strict=strict)
    model.eval()


def save_rl(ac_stats, object_set, save_description):
    dir = os.path.join(MODEL_SAVE_PATH, object_set.value, "rl")
    Path(dir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(dir, f"{get_time_str()}_{save_description}.pkl"), 'wb') as file:
        pickle.dump(ac_stats, file, pickle.HIGHEST_PROTOCOL)


def load_rl(filename):
    with open(os.path.join(MODEL_SAVE_PATH, filename), "rb") as file:
        data = pickle.load(file)
    return data


def get_best_checkpoints(checkpoints, validation=True):
    select = lambda cp: cp.validation_stats if validation else cp.training_stats
    checkpoints = [cp for cp in checkpoints if select(cp) is not None]
    best_acc = max(select(cp).accuracy for cp in checkpoints)
    best_acc_cps = [cp for cp in checkpoints if select(cp).accuracy == best_acc]
    best_n_glances = min(select(cp).avg_n_glances for cp in best_acc_cps)
    best_cps = [cp for cp in checkpoints if select(cp).accuracy == best_acc and select(cp).avg_n_glances == best_n_glances]
    return best_cps


def set_sns(figsize=(10, 4), font_scale=1.1):
    sns.set_theme()
    sns.set(rc={'figure.figsize': figsize}, font_scale=font_scale)


def plot_n_glances_hist(n_glances_list, percent=True, xticks=None):

    set_sns(figsize=(3.5, 7.2), font_scale=1.2)

    n_glances_list = np.array(n_glances_list)
    if xticks is None:
        xticks = np.arange(11)

    dp = sns.displot(
        n_glances_list,
        discrete=True,
        stat="percent" if percent else "frequency",
        height=3.5,
        aspect=1.3,
        color="#2C72DB"
    )

    dp.set(xlabel="N", ylabel="Frequency")
    ax = dp.axes[0][0]
    if percent:
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=100.0))
    ax.set_xticks(xticks)

    plt.tight_layout()
    plt.show()


def get_empty_glance(position_dim, pressure_dim):
    value = -1
    return np.zeros(shape=(position_dim,)) + value, np.zeros(shape=(pressure_dim,)) + value


def extract_action_params(action_params):
    flattened_params = []
    if isinstance(action_params, (tuple, list)):
        for item in action_params:
            flattened_params.extend(extract_action_params(item))
    elif action_params is not None:
        flattened_params.append(action_params)
    return flattened_params


def unzip(l: Iterable):
    return list(zip(*l))


def xtr(*attributes: str):
    if len(attributes) == 1:
        return lambda o: getattr(o, attributes[0])
    else:
        return lambda o: [getattr(o, attribute) for attribute in attributes]


def xtri(*indices: int):
    if len(indices) == 1:
        return lambda o: o[indices[0]]
    else:
        return lambda o: [o[i] for i in indices]


def mapl(f, l):
    return list(map(f, l))


def get_action_param(action_param):
    if isinstance(action_param, (list, tuple)):
        for x in action_param:
            subparam = get_action_param(x)
            if subparam is not None:
                return subparam
    return action_param


def to2D(pressure_values):
    dim = int(sqrt(pressure_values.shape[0]))
    return np.reshape(pressure_values, (dim, dim))


def print_pressure(pressure_values, print_values=False, glance_params=None):
    pressure_values_2D = to2D(pressure_values)
    print()
    if glance_params is not None:
        print("Params:", str(glance_params))
    print(" # " * (pressure_values_2D.shape[0]+2))
    for i in range(pressure_values_2D.shape[0]):
        print(" # ", end="")
        for k in range(pressure_values_2D.shape[0]):
            if pressure_values_2D[i, k] != 0:
                if print_values:
                    c = str(int(pressure_values_2D[i, k])).rjust(3)
                else:
                    c = " O "
            else:
                c = " . "
            print(c, end="")
        print(" # ")
    print(" # " * (pressure_values_2D.shape[0]+2))
