import numpy as np
from random import shuffle
from itertools import count
from collections import namedtuple
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tqdm import trange
from typing import Any, List, Iterable
from dataclasses import dataclass
from collections import OrderedDict
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import seaborn as sns

from haptic_exploration.ml_util import extract_action_params, get_best_checkpoints, plot_n_glances_hist
from haptic_exploration.config import WAIT_PLOTS
from haptic_exploration.composite import get_feature_position_param
from haptic_exploration.actions import ActionSpace, DiscreteActionSpace, ContinuousActionSpace, HybridActionSpace, ParameterizedActionSpace
from haptic_exploration.environment import HapticExplorationEnv


eps = np.finfo(np.float32).eps.item()

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])


@dataclass
class ActorCriticHyperparameters:
    batch_size: int = 1
    gamma: float = 1
    cls_lr: float = 3e-5
    action_lr: float = 3e-5
    value_lr: float = 3e-5
    std_lr: float = 5e-4
    train_cls: bool = True
    train_policy: bool = True
    n_glances: int = 4
    optimal_glances: bool = False
    random_glances: bool = False
    method: str = "reinforce"

    def __post_init__(self):
        assert self.method in ["reinforce", "ac"]


@dataclass
class EpochStats:
    accuracy: float
    avg_n_glances: float
    avg_reward: float
    num_episodes: int
    num_not_classified: int
    accuracy_classified: float
    n_glances_hist: dict


@dataclass
class ActorCriticCheckpoint:
    i_epoch: int
    training_stats: EpochStats
    validation_stats: EpochStats
    cls_model_weights: dict
    action_model_weights: dict
    value_model_weights: dict
    action_params: Any
    shared_model_weights: dict



class ActorCritic:

    def __init__(self, env: HapticExplorationEnv, shared_architecture: bool, model_spec, action_parameters: Any, action_space: ActionSpace, hyperparameters: ActorCriticHyperparameters, save_checkpoints=True, store_weight_interval=20):

        self.shared_architecture = shared_architecture

        self.hp = hyperparameters
        self.store_weight_interval = store_weight_interval
        self.save_checkpoints = save_checkpoints
        self.last_stored_checkpoint_epoch = 0

        # Env
        self.env = env
        self.action_space = action_space

        # Models
        if shared_architecture:
            self.shared_model = model_spec
        else:
            self.cls_model, self.action_model, self.value_model = model_spec

        self.action_parameters = action_parameters
        action_params = extract_action_params(action_parameters)

        # init optimizer
        if self.shared_architecture:
            core_lr = (self.hp.cls_lr + self.hp.action_lr)/2
            optimizer_params = [
                {'params': self.shared_model.model.model[0].parameters(), "lr": core_lr},
                {'params': self.shared_model.model.model[1].classification_output.parameters(), "lr": self.hp.cls_lr},
                {'params': self.shared_model.model.model[1].action_output.parameters(), "lr": self.hp.action_lr},
                {'params': self.shared_model.model.model[1].value_output.parameters(), "lr": self.hp.value_lr}
            ]
        else:
            optimizer_params = [
                {'params': self.cls_model.parameters(), "lr": self.hp.cls_lr},
                {'params': self.action_model.parameters(), "lr": self.hp.action_lr},
                {'params': self.value_model.parameters(), "lr": self.hp.value_lr}
            ]
        if len(action_params) > 0:
            optimizer_params += [{'params': action_params, "lr": self.hp.std_lr},]

        self.optimizer = optim.Adam(optimizer_params)

        # Action & Reward Buffer
        self.init_episode()

        self.device = "cpu"
        self.cross_entropy_loss = nn.CrossEntropyLoss()

        self.init_epoch_stats()

        self.checkpoints = []

    def init_episode(self):
        self.saved_actions = []
        self.rewards = []
        self.saved_classification = None
        self.optimal_glance_positions = list(range(4))
        shuffle(self.optimal_glance_positions)

    def init_batch(self):
        self.batched_actions = []
        self.batched_rewards = []
        self.batched_classifications = []

    def init_epoch_stats(self):
        # training statistics
        self.epoch_num_episodes = 0
        self.epoch_reward = 0
        self.epoch_correct = 0
        self.epoch_n_glances = []
        self.epoch_n_glances_sum = 0
        self.epoch_ys = []
        self.epoch_yps = []
        self.epoch_decisions = []
        self.epoch_glance_params = []
        self.epoch_zero_glances = []

    def store_checkpoint(self, i_epoch, training_stats, validation_stats, save_weights=False):
        if i_epoch in [-1, 0]:
            save_weights = True
        else:
            best_cps = get_best_checkpoints(self.checkpoints)
            last_best_cp = best_cps[-1]
            best_acc, best_n_glances = last_best_cp.validation_stats.accuracy, last_best_cp.validation_stats.avg_n_glances
            if validation_stats.accuracy > best_acc:
                save_weights = True
            if validation_stats.accuracy == best_acc and validation_stats.avg_n_glances < best_n_glances:
                save_weights = True
            if (i_epoch % self.store_weight_interval) == 0:
                save_weights = True

        if self.save_checkpoints:
            if save_weights:
                if self.shared_architecture:
                    state = [
                        None,
                        None,
                        None,
                        deepcopy(self.action_parameters),
                        deepcopy(self.shared_model.state_dict())]
                else:
                    state = [
                        deepcopy(self.cls_model.state_dict()),
                        deepcopy(self.action_model.state_dict()),
                        deepcopy(self.value_model.state_dict()),
                        deepcopy(self.action_parameters),
                        None
                    ]
            else:
                state = [None, None, None, deepcopy(self.action_parameters), None]

            self.checkpoints.append(ActorCriticCheckpoint(
                i_epoch,
                training_stats,
                validation_stats,
                *state
            ))

    def select_action(self, action_logits, state_value, deterministic=False):

        self.action_space.proba_distribution(action_logits, self.action_parameters)
        action = self.action_space.get_actions(deterministic=deterministic)
        action_log_prob = self.action_space.log_prob(action)

        self.saved_actions.append(SavedAction(action_log_prob, state_value))
        return action

    def get_classification(self, cls_logits):
        self.saved_classification = cls_logits
        return cls_logits.argmax().item()

    def get_action(self, i_episode, num_glances, cls_output, action_output, value_output, deterministic=False):

        decision_action = 0 if num_glances < self.hp.n_glances else 1
        glance_params = None

        if isinstance(self.action_space, DiscreteActionSpace):
            decision_action = self.select_action(action_output, value_output, deterministic=deterministic).item()
        elif isinstance(self.action_space, ContinuousActionSpace):
            glance_params = self.select_action(action_output, value_output, deterministic=deterministic)
        elif isinstance(self.action_space, HybridActionSpace):
            raw_action = self.select_action(action_output, value_output, deterministic=deterministic)
            decision_action, glance_params = raw_action[0].item(), raw_action[1]
        elif isinstance(self.action_space, ParameterizedActionSpace):
            raw_action = self.select_action(action_output, value_output, deterministic=deterministic)
            decision_action, glance_params = raw_action[0].item(), raw_action[1]
        else:
            raise Exception("unsupported action space")

        if self.hp.random_glances:
            glance_params = (torch.rand(1, 2) - 0.5) * 2

        if self.hp.optimal_glances and isinstance(self.action_space, DiscreteActionSpace):
            position_idx = self.optimal_glance_positions[i_episode % 4]
            glance_params = torch.Tensor(get_feature_position_param(position_idx, zero_centered=True)).unsqueeze(0)

        if decision_action == 0:
            action = (0, glance_params)
        else:
            classification = self.get_classification(cls_output)
            action = (1, classification)

        return action

    def train_episode(self):
        """
        Training code. Calculates actor and critic loss and performs backprop.
        """

        policy_losses = []  # list to save actor (policy) loss
        value_losses = []  # list to save critic (value) loss

        for saved_actions, rewards in zip(self.batched_actions, self.batched_rewards):

            R = 0
            #saved_actions = self.saved_actions
            #rewards = self.rewards
            returns = [] # list to save the true values

            # calculate the true value using rewards returned from the environment
            for r in rewards[::-1]:
                # calculate the discounted value
                R = r + self.hp.gamma * R
                returns.insert(0, R)

            returns = torch.tensor(returns)

            for i in range(len(saved_actions)):
                log_prob, value = saved_actions[i]
                next_value = saved_actions[+1][1].item() if i < len(saved_actions) - 1 else 0
                G_t = returns[i]
                reward = rewards[i]

                if self.hp.method == "reinforce":
                    ret = G_t
                else:  # "ac"
                    ret = reward + next_value

                advantage = ret - value.item()

                # calculate actor (policy) loss
                policy_losses.append(-log_prob * advantage)

                # calculate critic (value) loss using L1 smooth loss
                value_losses.append(F.smooth_l1_loss(value, torch.tensor([[ret]])))

        # reset gradients
        self.optimizer.zero_grad()

        # construct loss
        total_loss = 0
        if self.hp.train_policy:
            # sum up all the values of policy_losses and value_losses
            total_loss += torch.stack(policy_losses).sum() #/ self.hp.batch_size
            total_loss += torch.stack(value_losses).sum() #/ self.hp.batch_size
        if self.hp.train_cls and self.saved_classification is not None:
            total_loss += self.cross_entropy_loss(self.saved_classification, torch.Tensor([self.env.object_id]).long())
            self.saved_classification = None

        # backward pass
        total_loss.backward()

        # apply weight updates
        self.optimizer.step()

        # reset rewards and action buffer
        del self.rewards[:]
        del self.saved_actions[:]

    def get_outputs(self, obs):
        model_input = torch.Tensor(obs[0]).unsqueeze(0).to("cpu"), torch.Tensor(obs[1]).unsqueeze(0).to("cpu")
        if self.shared_architecture:
            return self.shared_model(model_input)
        else:
            return self.cls_model(model_input), self.action_model(model_input), self.value_model(model_input)

    def run_episode(self, object_id=None, deterministic=False):
        self.init_episode()
        # reset environment and episode reward
        obs, info = self.env.reset(object_id=object_id)
        episode_reward = 0

        for i_episode in count(0):

            cls_output, action_output, value_output = self.get_outputs(obs)
            action = self.get_action(i_episode, info["num_glances"], cls_output, action_output, value_output, deterministic=deterministic)
            obs, reward, terminated, _, info = self.env.step(action)

            self.rewards.append(float(reward))
            episode_reward += reward
            self.epoch_decisions.append(action[0])
            if action[0] == 0:
                self.epoch_glance_params.append(action[1].squeeze().tolist())
                pass

            if terminated:
                break

        self.batched_actions.append(self.saved_actions)
        self.batched_rewards.append(self.rewards)
        self.batched_classifications.append(self.saved_classification)

        self.epoch_num_episodes += 1
        self.epoch_reward += episode_reward
        self.epoch_correct += 1 if info["solved"] else 0
        self.epoch_n_glances.append(info["num_glances"])
        self.epoch_n_glances_sum += info["num_glances"]
        self.epoch_ys.append(info["object_id"])
        self.epoch_yps.append(action[1] if action[0] == 1 else -1)
        self.epoch_zero_glances.append(sum([1 if np.abs(p).sum() == 0 else 0 for p in obs[1]]))

    def analyse_epoch(self, epoch_descr="", cm=True, draw_plots=True):

        accuracy = self.epoch_correct/self.epoch_num_episodes
        avg_reward = self.epoch_reward/self.epoch_num_episodes
        avg_n_glances = self.epoch_n_glances_sum/self.epoch_num_episodes
        num_decisions0, num_decisions1 = (self.epoch_decisions.count(decision) for decision in range(2))
        num_not_classified = sum(1 for yp in self.epoch_yps if yp < 0)
        num_classified = self.epoch_num_episodes - num_not_classified
        accuracy_classified = 0 if num_classified == 0 else self.epoch_correct/num_classified
        n_glances_hist = OrderedDict([(n, self.epoch_n_glances.count(n)) for n in range(self.env.max_steps)])
        zero_glances_hist = OrderedDict([(n, self.epoch_zero_glances.count(n)) for n in range(self.env.max_steps)])

        print()
        print(f"*** {self.epoch_num_episodes} episodes ***")
        print(f"Epoch: {epoch_descr}")
        print(f"Accuracy: {(100*accuracy):.2f}%")
        print(f"Avg reward: {avg_reward:.2f}")
        print(f"Avg n_glances: {avg_n_glances:.2f}")
        print(f"Decisions: 0={(100*num_decisions0/(num_decisions0+num_decisions1)):.2f}% ({num_decisions0}), 1={(100*num_decisions1/(num_decisions0+num_decisions1)):.2f}% ({num_decisions1})")
        print(f"Not classified: {(100*num_not_classified/self.epoch_num_episodes):.2f}%")
        print(f"Accuracy of classified: {(100*accuracy_classified):.2f}%")
        print(f"n_glances hist:", ", ".join(f"{n}: {count}" for n, count in n_glances_hist.items()))
        print(f"Action std:", self.action_parameters)
        print(f"Zero Glances hist:", ", ".join(f"{n}: {count}" for n, count in zero_glances_hist.items()))

        if draw_plots:
            # confusion matrix
            if cm:
                sns.reset_orig()
                cm = confusion_matrix(np.array(self.epoch_ys, dtype=np.float32), np.array(self.epoch_yps, dtype=np.float32), labels=range(-1, self.env.num_objects))
                ConfusionMatrixDisplay(cm, display_labels=range(-1, self.env.num_objects)).plot()
                plt.suptitle(f"confusion matrix (epoch {epoch_descr})")
                plt.xlabel('Predicted class')
                plt.ylabel('True class')
                plt.show(block=WAIT_PLOTS)

            # n glance histogram
            plot_n_glances_hist(self.epoch_n_glances, xticks=np.arange(self.env.max_steps+1))

        return EpochStats(accuracy, avg_n_glances, avg_reward, self.epoch_num_episodes, num_not_classified, accuracy_classified, n_glances_hist)

    def validate_epoch(self, epoch_desc, num_eval_episodes, deterministic=True):

        self.env.add_noise = True

        if self.shared_architecture:
            self.shared_model.eval()
        else:
            self.cls_model.eval()
            self.action_model.eval()
            self.value_model.eval()

        self.init_epoch_stats()
        self.init_batch()

        with torch.no_grad():
            pbar = trange(num_eval_episodes, desc=epoch_desc)
            for i_episode in pbar:
                self.run_episode(object_id=i_episode % self.env.num_objects, deterministic=deterministic)

    def train_epoch(self, epoch_desc, num_episodes):

        self.env.add_noise = True

        if self.shared_architecture:
            self.shared_model.train()
        else:
            self.cls_model.train()
            self.action_model.train()
            self.value_model.train()

        self.init_epoch_stats()

        pbar = trange(num_episodes, desc=epoch_desc)
        for i_batch in range(num_episodes // self.hp.batch_size):

            batch_objects = list(range(self.env.num_objects))
            shuffle(batch_objects)

            for i_batch_episode in range(self.hp.batch_size):
                self.init_batch()
                self.run_episode(object_id=batch_objects[i_batch_episode % self.env.num_objects])
                pbar.update()

            self.train_episode()

            i_episode = i_batch * self.hp.batch_size + i_batch_episode
            pbar.set_postfix_str(
                f"accuracy={(100 * self.epoch_correct / (i_episode + 1)):.2f}%, n_glances={(self.epoch_n_glances_sum / (i_episode + 1)):.2f}, reward={(self.epoch_reward / (i_episode + 1)):.2f}")

    def train(self, num_epochs, num_episodes, draw_plots=False) -> List[ActorCriticCheckpoint]:

        for i_epoch in range(num_epochs):

            self.train_epoch(f"RL Training Epoch {i_epoch + 1}/{num_epochs}", num_episodes)
            training_stats = self.analyse_epoch(f"training {i_epoch}", draw_plots=draw_plots)

            self.validate_epoch(f"RL Validation Epoch {i_epoch + 1}/{num_epochs}", self.env.num_objects)
            validation_stats = self.analyse_epoch(f"validation {i_epoch}", cm=True, draw_plots=draw_plots)

            if self.save_checkpoints:
                self.store_checkpoint(i_epoch, training_stats, validation_stats, save_weights=(i_epoch == num_epochs-1))

        return self.checkpoints

    def evaluate(self, eval_desc="Evaluation", deterministic=True):
        num_eval_episodes = self.env.num_objects if deterministic else 1000
        self.validate_epoch(eval_desc, num_eval_episodes, deterministic=deterministic)
        validation_stats = self.analyse_epoch(eval_desc, cm=True)
        return validation_stats
