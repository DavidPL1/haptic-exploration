import collections
from abc import ABC, abstractmethod
from typing import Any, Optional, List, Iterable, Tuple, Dict, Union
from dataclasses import dataclass

import gymnasium as gym
import torch
from torch import nn
from torch.distributions import Normal, Categorical


# ********** Action Nets ********** #


class DiscreteActionNet(nn.Module):

    def __init__(self, latent_dim, action_dim, net_dims=()):
        super().__init__()

        last_dim = latent_dim
        layers = []
        for net_dim in net_dims:
            layers.append(nn.Linear(last_dim, net_dim))
            layers.append(nn.ReLU())
            last_dim = net_dim
        layers.append(nn.Linear(last_dim, action_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, latent) -> torch.Tensor:
        return self.net(latent)


class ContinuousActionNet(nn.Module):

    def __init__(self, latent_dim, action_dim, net_dims=()):
        super().__init__()

        last_dim = latent_dim
        layers = []
        for net_dim in net_dims:
            layers.append(nn.Linear(last_dim, net_dim))
            layers.append(nn.ReLU())
            last_dim = net_dim
        layers.append(nn.Linear(last_dim, action_dim))
        layers.append(nn.Tanh())

        self.net = nn.Sequential(*layers)

    def forward(self, latent) -> torch.Tensor:
        return self.net(latent)


class HybridActionNet(nn.Module):

    def __init__(self, hybrid_action_nets: Iterable[Union[DiscreteActionNet, ContinuousActionNet]]):
        super().__init__()
        self.hybrid_actions_nets = nn.ModuleList(hybrid_action_nets)

    def forward(self, latent) -> list:
        assert len(latent) == len(self.hybrid_actions_nets)

        return [net(latent) for net, latent in zip(self.hybrid_actions_nets, latent)]


class ParametrizedActionNet(nn.Module):

    def __init__(self, discrete_action_net: DiscreteActionNet, hybrid_parameter_action_nets: Iterable[Union[DiscreteActionNet, ContinuousActionNet, HybridActionNet]]):
        super().__init__()
        self.discrete_action_net = discrete_action_net
        self.hybrid_parameter_action_nets = nn.ModuleList(hybrid_parameter_action_nets)

    def forward(self, latent) -> tuple:
        latent_discrete, latent_parameters = latent
        assert len(latent_parameters) == len(self.hybrid_parameter_action_nets)

        discrete_out = self.discrete_action_net(latent_discrete)
        parameter_outs = [parameter_net(latent_parameter) for parameter_net, latent_parameter in zip(self.hybrid_parameter_action_nets, latent_parameters)]
        return discrete_out, parameter_outs



# ********** Action Spaces ********** #


@dataclass
class ActionSpaceConfig:
    log_std_init: float = -1.5


class ActionSpace(ABC):

    def __init__(self, action_size: int, logit_size: int, gym_space: gym.Space, space_config: ActionSpaceConfig = None):
        self.action_size = action_size
        self.logit_size = logit_size
        self.gym_space = gym_space
        self.latent_dim = None
        self.distribution = None
        self.space_config = space_config or ActionSpaceConfig()

    """
    # encoding and decoding only necessary when actions are required to be tensors (e.g. for storing in buffer in sb3)
    def encode_action(self, action_structure) -> torch.Tensor:
        raise NotImplementedError()
    def decode_action(self, action_array) -> Any:
        raise NotImplementedError()
    """

    @abstractmethod
    def build_action_net(self, latent_dim: Any) -> Tuple[nn.Module, Any]:
        raise NotImplementedError()

    @abstractmethod
    def proba_distribution(self, action_logits: Any, params: Any):
        raise NotImplementedError()

    @abstractmethod
    def log_prob(self, actions) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def sample(self) -> Any:
        raise NotImplementedError()

    @abstractmethod
    def mode(self) -> Any:
        raise NotImplementedError()

    def get_actions(self, deterministic: bool = False) -> Any:
        if deterministic:
            return self.mode()
        return self.sample()


class DiscreteActionSpace(ActionSpace):

    def __init__(self, n, start=0, net_dims=(), space_config=None):
        self.n = n
        self.start = start
        self.net_dims = net_dims
        gym_space = gym.spaces.Discrete(self.n, start=self.start)
        super().__init__(1, n, gym_space, space_config)

    def build_action_net(self, latent_dim: int) -> Tuple[DiscreteActionNet, None]:
        self.latent_dim = latent_dim
        return DiscreteActionNet(latent_dim, self.n, net_dims=self.net_dims), None

    def proba_distribution(self, action_logits, params):
        self.distribution = Categorical(logits=action_logits)
        return self

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions)

    def sample(self) -> torch.Tensor:
        return self.distribution.sample()

    def mode(self) -> torch.Tensor:
        return torch.argmax(self.distribution.probs, dim=1)


class ContinuousActionSpace(ActionSpace):

    def __init__(self, n_actions, low=-1, high=1, net_dims=(), space_config=None):
        self.n_actions = n_actions
        self.low = low
        self.high = high
        self.net_dims = net_dims
        gym_space = gym.spaces.Box(low=low, high=high, shape=(n_actions,))
        super().__init__(n_actions, n_actions, gym_space, space_config)

    def build_action_net(self, latent_dim: int) -> Tuple[ContinuousActionNet, nn.Parameter]:
        self.latent_dim = latent_dim
        action_net = ContinuousActionNet(latent_dim, self.n_actions, net_dims=self.net_dims)
        log_std = nn.Parameter(torch.ones(self.n_actions) * self.space_config.log_std_init, requires_grad=True)
        return action_net, log_std

    def proba_distribution(self, action_logits, params):
        mean_actions, log_std = action_logits, params
        action_std = torch.ones_like(mean_actions) * log_std.exp()
        self.distribution = Normal(mean_actions, action_std)
        return self

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        log_prob = self.distribution.log_prob(actions)
        return log_prob.sum(dim=1)

    def sample(self) -> torch.Tensor:
        return self.distribution.sample()

    def mode(self) -> torch.Tensor:
        return self.distribution.mean


class HybridActionSpace(ActionSpace):

    def __init__(self, subspaces: Iterable[ActionSpace], space_config=None):
        self.subspaces = list(subspaces)
        action_size = sum(subspace.action_size for subspace in subspaces)
        logit_size = sum(subspace.logit_size for subspace in subspaces)
        gym_space = gym.spaces.Tuple(subspace.gym_space for subspace in subspaces)
        super().__init__(action_size, logit_size, gym_space, space_config)

    def build_action_net(self, latent_dim: Any) -> Tuple[nn.Module, nn.Parameter]:
        self.latent_dim = latent_dim
        assert len(latent_dim) == len(self.subspaces)
        subspace_nets, subspace_parameters = zip(*[subspace.build_action_net(latent_dim_subspace) for subspace, latent_dim_subspace in zip(self.subspaces, latent_dim)])
        return HybridActionNet(subspace_nets), subspace_parameters

    def proba_distribution(self, action_logits, params):
        for subspace, subspace_action_logits, subspace_params in zip(self.subspaces, action_logits, params):
            subspace.proba_distribution(subspace_action_logits, subspace_params)
        return self

    def log_prob(self, actions) -> torch.Tensor:
        return sum(subspace.log_prob(subspace_actions) for subspace, subspace_actions in zip(self.subspaces, actions))

    def sample(self) -> Any:
        return [subspace.sample() for subspace in self.subspaces]

    def mode(self) -> Any:
        return [subspace.mode() for subspace in self.subspaces]


class ParameterizedActionSpace(ActionSpace):

    def __init__(self, action_type_space: DiscreteActionSpace, parameter_spaces: Union[Iterable[ActionSpace], Dict[int, ActionSpace]], space_config=None):
        self.action_type_space = action_type_space
        self.parameter_spaces = parameter_spaces
        if not isinstance(self.parameter_spaces, dict):
            self.parameter_spaces = {key: space for key, space in enumerate(self.parameter_spaces)}
        self.parameter_spaces = collections.OrderedDict(sorted(self.parameter_spaces.items()))

        action_size = self.action_type_space.action_size + sum(parameter_space.action_size for parameter_space in self.parameter_spaces.values())
        logit_size = self.action_type_space.logit_size + sum(parameter_space.logit_size for parameter_space in self.parameter_spaces.values())
        gym_space = gym.spaces.Tuple((self.action_type_space.gym_space, gym.spaces.Tuple([parameter_space.gym_space for parameter_space in self.parameter_spaces.values()])))
        super().__init__(action_size, logit_size, gym_space, space_config)

    def build_action_net(self, latent_dim: Any) -> Tuple[nn.Module, Any]:
        action_type_latent_dim, parameters_latent_dims = latent_dim
        action_type_space_net, action_type_space_params = self.action_type_space.build_action_net(action_type_latent_dim)
        parameter_spaces_nets, parameter_spaces_params = zip(*[parameter_space.build_action_net(latent_dim_subspace) for parameter_space, latent_dim_subspace in zip(self.parameter_spaces.values(), parameters_latent_dims)])
        return ParametrizedActionNet(action_type_space_net, parameter_spaces_nets), (action_type_space_params, parameter_spaces_params)

    def proba_distribution(self, action_logits: Any, params: Any):
        action_type_logits, parameters_logits = action_logits
        action_type_params, parameters_params = params
        self.action_type_space.proba_distribution(action_type_logits, action_type_params)
        for parameter_space, parameter_logits, parameter_params in zip(self.parameter_spaces.values(), parameters_logits, parameters_params):
            parameter_space.proba_distribution(parameter_logits, parameter_params)
        return self

    def log_prob(self, actions) -> torch.Tensor:
        action_type, action_param = actions
        log_prob = self.action_type_space.log_prob(action_type)
        if action_type.item() in self.parameter_spaces:
            log_prob += self.parameter_spaces[action_type.item()].log_prob(action_param)
        return log_prob

    def sample(self) -> Any:
        action_type = self.action_type_space.sample()
        if action_type.item() in self.parameter_spaces:
            action_param = self.parameter_spaces[action_type.item()].sample()
        else:
            action_param = None
        return action_type, action_param

    def mode(self) -> Any:
        action_type = self.action_type_space.mode()
        if action_type.item() in self.parameter_spaces:
            action_param = self.parameter_spaces[action_type.item()].mode()
        else:
            action_param = None
        return action_type, action_param


def main():

    space = ParameterizedActionSpace(DiscreteActionSpace(2), [ContinuousActionSpace(3), DiscreteActionSpace(7)])
    latent_dim = (16, [32, 16])
    action_net, params = space.build_action_net(latent_dim)
    latent = [torch.rand(7, 32), torch.rand(7, 16)]

    pass

    # hybrid_test
    space = HybridActionSpace([DiscreteActionSpace(5), ContinuousActionSpace(2)])
    latent_dim = [32, 16]
    action_net, params = space.build_action_net(latent_dim)
    latent = [torch.rand(7, 32), torch.rand(7, 16)]
    action_logits = action_net(latent)
    space.proba_distribution(action_logits, params)
    action = space.sample()
    log_prob = space.log_prob(action)

    pass

if __name__ == "__main__":
    main()
