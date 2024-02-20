import matplotlib.pyplot as plt
from typing import Iterable, List

from haptic_exploration.ml_util import xtr, unzip, plt_colors, ModelTrainingMonitor
from haptic_exploration.config import WAIT_PLOTS
from haptic_exploration.actor_critic import EpochStats, ActorCriticCheckpoint


def summarize_training(checkpoints: Iterable[ActorCriticCheckpoint]):
    stats = unzip(map(xtr("training_stats", "validation_stats"), checkpoints))
    plot_stats(stats, ["training", "validation"])


def plot_stats(stats_all: Iterable[List[EpochStats]], stats_names: Iterable[str], attributes: Iterable[str] = ["accuracy", "avg_n_glances", "avg_reward"]):
    for attribute in attributes:
        stats_attributes = [list(map(xtr(attribute), stats)) for stats in stats_all]
        for stats_attribute, stats_name, color in zip(stats_attributes, stats_names, plt_colors):
            xs = range(len(stats_attribute))
            plt.plot(xs, stats_attribute, label=stats_name, color=color)
        plt.title(attribute)
        plt.legend()
        plt.show(block=WAIT_PLOTS)


def plot_training_performance(training_monitor: ModelTrainingMonitor, descr=""):

    x = range(1, len(training_monitor.losses) + 1)
    y1 = [acc*100 for acc in training_monitor.accuracies]
    y2 = training_monitor.losses

    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('episodes')
    ax1.set_ylabel('accuracy', color=color)
    ax1.plot(x, y1, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    plt.xticks(x)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('loss', color=color)  # we already handled the x-label with ax1
    ax2.plot(x, y2, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title("Training: " + descr)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show(block=WAIT_PLOTS)
