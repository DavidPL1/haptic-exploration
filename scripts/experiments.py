import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import pickle
import os
import pandas as pd
import numpy as np

from collections import defaultdict

from spec import get_trained_ac
from haptic_exploration.composite import get_feature_position_param, calculate_object_n
from haptic_exploration.ml_util import ModelType, load_rl, get_action_param, get_best_checkpoints
from haptic_exploration.config import ObjectSet, EXPERIMENTS_DIR


def load_rl_exp(experiment_dir, filename):
    with open(EXPERIMENTS_DIR / experiment_dir / filename, "rb") as file:
        data = pickle.load(file)
    return data


def accuracy_from_checkpoints(checkpoints, validation=False):
    def xtr(cp):
        if validation:
            return cp.validation_stats.accuracy
        else:
            return cp.training_stats.accuracy
    return [xtr(cp) for cp in checkpoints]


def n_glances_from_checkpoints(checkpoints, validation=False):
    def xtr(cp):
        if validation:
            return cp.validation_stats.avg_n_glances
        else:
            return cp.training_stats.avg_n_glances
    return [xtr(cp) for cp in checkpoints]


def rewards_from_checkpoints(checkpoints, validation=False):
    def xtr(cp):
        if validation:
            return cp.validation_stats.avg_reward
        else:
            return cp.training_stats.avg_reward
    return [xtr(cp) for cp in checkpoints]


def set_sns(figsize=(10, 4), font_scale=1.1):
    sns.set_theme()
    sns.set(rc={'figure.figsize': figsize}, font_scale=font_scale)


"""
********** PLOT FUNCTIONS **********
"""


def plot_training_accuracies(names, rl_data, num_episodes, validation=False, cutoff_epoch=None, optimal_n=None):

    set_sns(figsize=(10, 4))

    if cutoff_epoch is not None:
        rl_data = [(v1, v2, cp[:cutoff_epoch]) for v1, v2, cp in rl_data]

    accuracies = [accuracy_from_checkpoints(cps) for _, _, cps in rl_data]
    #print("Best training accuracies (individual):")
    #for name, training_accuracies in zip(names, accuracies):
    #    print(f"{name}: {(max(training_accuracies)*100):.2f}%")

    accuracy_dict = defaultdict(list)
    for name, accuracy in zip(names, accuracies):
        accuracy_dict[name] += [max(accuracy)]
    print("Best training accuracies (grouped):")
    for name, grouped_accuracies in accuracy_dict.items():
        print(f"{name}: {(sum(grouped_accuracies)/len(grouped_accuracies)*100):.2f}%")

    accuracies = np.array(accuracies).T
    xs = np.arange(accuracies.shape[0]) * num_episodes + num_episodes
    data = pd.DataFrame(accuracies, xs, columns=names)

    ax = sns.lineplot(data=data, linewidth=2.5, dashes=False)
    ax.set(xlabel="Number of Episodes", ylabel="Classification Accuracy")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    plt.tight_layout()
    plt.legend(loc="lower right")
    plt.show()


def plot_training_combine_accuracies_n(names, rl_data, num_episodes, validation=False, cutoff_epoch=None, optimal_n=None):

    set_sns(figsize=(10, 4.2), font_scale=1.3)

    if cutoff_epoch is not None:
        rl_data = [(v1, v2, cp[:cutoff_epoch]) for v1, v2, cp in rl_data]

    accuracies = [accuracy_from_checkpoints(cps, validation=validation) for _, _, cps in rl_data]
    n_glances = [n_glances_from_checkpoints(cps, validation=validation) for _, _, cps in rl_data]

    palette = sns.color_palette()
    acc_color = palette[0]
    n_color = palette[1]

    columns1 = [f"Accuracy {name}" for name in names]
    columns2 = [f"Avg. N {name}" for name in names]

    accuracies = np.array(accuracies, dtype=float).T
    xs = np.arange(accuracies.shape[0]) * num_episodes + num_episodes
    data = pd.DataFrame(accuracies, xs, columns=columns1)

    ax = sns.lineplot(data=data, linewidth=2.5, dashes=False)
    ax.set(xlabel="Number of Episodes", ylabel="Classification Accuracy")
    ax.set_ylim([-0.08, 1.1])
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    ax.legend([], [], frameon=False)

    n_glances = np.array(n_glances).T
    data2 = pd.DataFrame(n_glances, xs, columns=columns2)
    twin_ax = plt.twinx()
    twin_ax.set(ylabel="Number of Glances N")
    ax2 = sns.lineplot(data=data2, linestyle="--", linewidth=3.0, ax=twin_ax)

    if optimal_n is not None:
        ax2.axhline(optimal_n, ls='--', color="gray")
        ax2.text(20000, optimal_n+0.15, f"Optimal Policy (N={optimal_n:.2f})")

    ax1_lines, ax2_lines = ax.get_lines(), ax2.get_lines()
    lines, labels = [], []
    for line in ax2_lines:
        line.set_linestyle("--")
    for line in ax1_lines + ax2_lines:
        label = line.get_label()
        if label in columns1 + columns2:
            lines.append(line)
            labels.append(label)
    ncol = 2 if len(lines) >= 4 else 1
    plt.legend(lines, labels, ncol=ncol, loc="lower right", fontsize="x-small")
    ax2.grid()

    plt.tight_layout()
    plt.show()


def plot_training_rewards(names, rl_data, num_episodes, validation=False, cutoff_epoch=None, optimal_reward=None):

    set_sns(figsize=(10, 3.5), font_scale=1.3)

    if cutoff_epoch is not None:
        rl_data = [(v1, v2, cp[:cutoff_epoch]) for v1, v2, cp in rl_data]

    rewards = [rewards_from_checkpoints(cps, validation=validation) for _, _, cps in rl_data]

    rewards = np.array(rewards).T
    xs = np.arange(rewards.shape[0]) * num_episodes + num_episodes
    data = pd.DataFrame(rewards, xs, columns=names)

    ax = sns.lineplot(data=data, linewidth=3.0, dashes=False)
    ax.set(xlabel="Number of Episodes", ylabel="Average Reward")
    y_min = min(-1, rewards.min()) - 0.15
    ax.set_ylim([y_min, 1.2])

    if optimal_reward is not None:
        ax.axhline(optimal_reward, ls='--', color="gray")
        ax.text(1000, optimal_reward + 0.07, f"Optimal Policy (R={optimal_reward:.3f})")

    plt.tight_layout()
    plt.legend(loc="lower right", fontsize="small")
    plt.show()


def plot_standard_deviations(names, rl_data, num_episodes, cutoff_epoch=None):

    set_sns(figsize=(10, 3))

    if cutoff_epoch is not None:
        rl_data = [(v1, v2, cp[:cutoff_epoch]) for v1, v2, cp in rl_data]

    stds = []
    stds_names = []
    for name, data in zip(names, rl_data):
        cps = data[2]
        std0 = []
        std1 = []
        for cp in cps:
            action_param = get_action_param(cp.action_params)
            std0.append(np.exp(action_param[0].item()))
            std1.append(np.exp(action_param[1].item()))
        stds.append(std0)
        stds.append(std1)
        stds_names += [f"x", f"y"]

    stds = np.array(stds).T
    xs = np.arange(stds.shape[0]) * num_episodes + num_episodes
    data_std = pd.DataFrame(stds, xs, columns=stds_names)
    ax = sns.lineplot(data=data_std, palette="Set2", linewidth=2.5, dashes=False)
    ax.set(xlabel="Number of Episodes", ylabel="Standard Deviation")
    plt.tight_layout()
    plt.legend(loc="upper right", fontsize="medium")
    plt.show()


def plot_glance_parameters(parameters, axis_names=("x", "y"), axis_ranges=((0, 1), (0, 1)), feature_positions=True, kde=True):

    set_sns(figsize=(4.4, 4.2), font_scale=1.2)

    parameters = np.array(parameters).clip(min=-1, max=1)
    (min_x, max_x), (min_y, max_y) = axis_ranges
    scale = 0.15/2
    off0 = (max_x - min_x)*scale
    off1 = (max_y - min_y)*scale

    from scipy.interpolate import interp1d
    m0 = interp1d([-1, 1], [min_x, max_x], bounds_error=False)
    m1 = interp1d([-1, 1], [min_y, max_y], bounds_error=False)
    parameters[:, 0] = m0(parameters[:, 0])
    parameters[:, 1] = m1(parameters[:, 1])

    plt.xlim(*axis_ranges[0])
    plt.ylim(*axis_ranges[1])
    ax = sns.scatterplot(x=parameters[:, 0], y=parameters[:, 1], s=50, linewidth=0.7, edgecolor="black", alpha=0.4)
    if kde:
        ax2 = sns.kdeplot(
            x=parameters[:, 0],
            y=parameters[:, 1],
            #clip=((min_x, max_x), (min_y, max_y)),
            cmap=sns.color_palette("rocket_r", as_cmap=True),
            levels=1000,
            fill=True,
            alpha=0.3,
            thresh=0,
            #s=50,
            linewidth=0.7,
            edgecolor="black")
    #else:
    #    ax = sns.scatterplot(x=parameters[:, 0], y=parameters[:, 1])
    #sns.histplot(x=parameters[:, 0], y=parameters[:, 1], bins=20, thresh=None, fill=True, cmap="mako")

    ax.set_xlabel(axis_names[0], fontsize=17)
    ax.set_ylabel(axis_names[1], fontsize=17)
    ax.set_xlim([min_x - off0, max_x + off0])
    ax.set_ylim([min_y - off1, max_y + off1])

    # draw exploration area
    area_color = "yellowgreen"
    linewidth = 5
    ax.hlines(y=min_y, xmin=min_x, xmax=max_x, linewidth=linewidth, ls="--", color=area_color)
    ax.hlines(y=max_y, xmin=min_x, xmax=max_x, linewidth=linewidth, ls="--", color=area_color)
    ax.vlines(x=min_x, ymin=min_y, ymax=max_y, linewidth=linewidth, ls="--", color=area_color)
    ax.vlines(x=max_x, ymin=min_y, ymax=max_y, linewidth=linewidth, ls="--", color=area_color)

    # draw feature positions
    if feature_positions:
        feature_position_params = np.array([get_feature_position_param(feature, zero_centered=True) for feature in range(4)])
        plt.plot(m0(feature_position_params[:, 0]), m1(feature_position_params[:, 1]), "o", markersize=15, markeredgewidth=1.5, markeredgecolor="black", c="r", zorder=10)

    ax.set_xticks(np.linspace(min_x, max_x, 5))
    ax.set_yticks(np.linspace(min_y, max_y, 5))

    plt.tight_layout()
    plt.show()


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


def plot_n_glances_object(object_n_dict):

    set_sns(figsize=(3.5, 7.2), font_scale=1.3)

    object_n_list = []
    for object_id, count in object_n_dict.items():
        object_n_list += [object_id]*count

    dp = sns.displot(
        object_n_list,
        discrete=True,
        height=3.5,
        aspect=1.6,
        color="seagreen"
    )

    dp.set(xlabel="Composite Object", ylabel="N")
    ax = dp.axes[0][0]
    ax.set_xticks(np.arange(len(object_n_dict)))

    plt.tight_layout()
    plt.show()


"""
********** COMPOSITE EXPERIMENTS **********
"""



def composite_visualize_training():
    """
    Plots the training accuracy of the Haptic Transformer for N=1,2,3,4
    """

    num_episodes = 1000

    #object_set, model_type, action_type = "hybrid", shared_architecture = True, init_pretrained = True, freeze_core = False, n_glances = -1
    spec_all = [
        ("Transformer hybrid", (ObjectSet.Composite, ModelType.Transformer, "hybrid", True, True, False, -1)),
        #("Transformer param", (ObjectSet.Composite, ModelType.Transformer, "parameterized", True, True, False, -1)),
        ("LSTM hybrid", (ObjectSet.Composite, ModelType.LSTM, "hybrid", True, True, False, -1)),
        #("LSTM param", (ObjectSet.Composite, ModelType.LSTM, "parameterized", True, True, False, -1)),
    ]

    rl_data_all = []
    names = []

    stats = dict()

    for name, spec in spec_all:
        spec_rl_data = [load_rl(file) for file in get_trained_ac(*spec)]
        #spec_rl_data = spec_rl_data[:1]
        rl_data_all += spec_rl_data
        names += [f"{name}" for i, _ in enumerate(spec_rl_data)]

        stat = {
            "best_reward_validation": [],
            "best_acc_validation": [],
            "best_n_validation": [],
            "best_epoch_validation": [],
            "best_reward_training": [],
            "best_acc_training": [],
            "best_n_training": [],
            "best_epoch_training": []
        }

        for rl_data in spec_rl_data:
            best_cps_validation = get_best_checkpoints(rl_data[2], validation=True)
            best_cp_validation = next(cp for cp in reversed(best_cps_validation))

            best_cps_training = get_best_checkpoints(rl_data[2], validation=False)
            best_cp_training = next(cp for cp in reversed(best_cps_training))

            stat["best_reward_validation"].append(best_cp_validation.validation_stats.avg_reward)
            stat["best_acc_validation"].append(best_cp_validation.validation_stats.accuracy)
            stat["best_n_validation"].append(best_cp_validation.validation_stats.avg_n_glances)
            stat["best_epoch_validation"].append(best_cp_validation.i_epoch)

            stat["best_reward_training"].append(best_cp_training.training_stats.avg_reward)
            stat["best_acc_training"].append(best_cp_training.training_stats.accuracy)
            stat["best_n_training"].append(best_cp_training.training_stats.avg_n_glances)
            stat["best_epoch_training"].append(best_cp_training.i_epoch)

        stats[name] = stat

    for name, stat in stats.items():
        print("MODEL:", name)
        for key, value in stat.items():
            value_str = [f"{v:.4f}" for v in value]
            print(f"{key}: {value_str} (AVG: {(sum(value)/len(value)):.4f})")


    glance_penalty = -0.05
    optimal_n = 2.54
    optimal_reward = 1 + glance_penalty*optimal_n

    plot_training_rewards(names, rl_data_all, num_episodes, optimal_reward=optimal_reward, validation=False)
    #plot_training_combine_accuracies_n(names, rl_data_all, num_episodes, optimal_n=optimal_n, validation=False)
    #plot_standard_deviations(names, rl_data_all, num_episodes)


def visualize_composite_objects():
    object_n_dict, feature_dict = calculate_object_n()

    plot_n_glances_object(object_n_dict)

    n_glances_list = []
    for object_id, count in object_n_dict.items():
        n_glances_list += [count]
    plot_n_glances_hist(n_glances_list, percent=False, xticks=np.arange(start=min(n_glances_list), stop=max(n_glances_list)+1))


def visualize_training_history():

    print("FILE:", EXPERIMENTS_DIR / "epoch_stats.pkl")
    with open(EXPERIMENTS_DIR / "epoch_stats.pkl", "rb") as object_file:
        epoch_n_dict, epoch_params_dict = pickle.load(object_file)
    print("Available epochs:", epoch_n_dict.keys())

    n_glances_0 = epoch_n_dict[0]
    plot_n_glances_hist(n_glances_0)

    plot_glance_parameters(epoch_params_dict[0])



if __name__ == "__main__":
    #basic_1_random_first_training_transformer()
    #basic_2_learn_first_pretrained_training_transformer()
    #basic_2_visualize_glance_parameters()
    #visualize_composite_objects()
    #composite_visualize_training()
    visualize_training_history()
