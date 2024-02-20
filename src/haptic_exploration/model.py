import torch

from torch import nn
from dataclasses import dataclass
from typing import List


########## MODEL PARAMETERS ##########


@dataclass
class ModelParameters:
    position_input_dim: int
    pressure_input_dim: int
    position_embedded_dim: int
    pressure_embedded_dim: int
    total_embedding_dim: int
    core_hidden_dim: int
    n_glances: List[int]
    n_objects: int
    n_glance_params: int


@dataclass
class TransformerParameters:
    n_encoder_layers: int
    n_heads: int
    output_cls_token: bool
    output_glance_token: bool


@dataclass
class MLPParameters:
    neuron_dims: List[int]
    activation: str
    dropout: float


@dataclass
class LSTMParameters:
    num_stacked_layers: int


########## EMBEDDING ##########


class PressureEmbedding(nn.Module):

    def __init__(self, model_params: ModelParameters):
        super().__init__()

        self.model_params = model_params
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=1, padding=1),
            nn.MaxPool2d(2, stride=2, padding=0),
            nn.Conv2d(8, 8, 3, stride=1, padding=1),
            nn.MaxPool2d(2, stride=2, padding=0),
            nn.Flatten(start_dim=1),
            nn.Linear(8 * 4 * 4, 64),
            nn.Linear(64, model_params.pressure_embedded_dim)
        )

    def forward(self, x):
        x_2d = x.reshape(-1, 1, 16, 16)
        embedded = self.net(x_2d)
        return embedded.reshape(*x.shape[:2], self.model_params.pressure_embedded_dim)


class GlanceEmbedding(nn.Module):
    """ Embeds and concatenates position and pressure of a glance """

    def __init__(self, model_params: ModelParameters, add_empty_embedding=True, empty_embedding_learnable=False):
        super().__init__()

        self.add_empty_embedding = add_empty_embedding

        self.position_embedding = nn.Linear(model_params.position_input_dim, model_params.position_embedded_dim)
        self.pressure_embedding = PressureEmbedding(model_params)

        if add_empty_embedding:
            if empty_embedding_learnable:
                self.empty_embedding = nn.Parameter(torch.rand(1, model_params.total_embedding_dim))
            else:
                self.empty_embedding = nn.Parameter(torch.zeros(1, model_params.total_embedding_dim) - 1, requires_grad=False)

    def forward(self, position_pressure_batch):
        position_batch, pressure_batch = position_pressure_batch
        if self.add_empty_embedding:
            empty_embedding_batch = self.empty_embedding.repeat(position_batch.shape[0], 1, 1)
            if position_batch.shape[1] == 0:
                return empty_embedding_batch

        embedded_positions = self.position_embedding(position_batch)
        embedded_pressures = self.pressure_embedding(pressure_batch)
        concatenated_glance_embeddings = torch.cat((embedded_positions, embedded_pressures), 2)

        if self.add_empty_embedding:
            return torch.cat([empty_embedding_batch, concatenated_glance_embeddings], dim=1)
        else:
            return concatenated_glance_embeddings


########## CORE ##########


class HapticTransformerCore(nn.Module):
    """ Transformer Encoder processing sequences of haptic glances """

    def __init__(self, model_params: ModelParameters, core_params: TransformerParameters):
        super().__init__()

        # transformer encoder parameters
        self.hidden_dim = model_params.core_hidden_dim
        self.n_encoder_layers = core_params.n_encoder_layers
        self.n_heads = core_params.n_heads
        self.add_cls_token = core_params.output_cls_token
        self.add_glance_token = core_params.output_glance_token

        assert self.add_cls_token or self.add_glance_token, "either cls or glance token must be added"

        self.custom_tokens = []
        if self.add_cls_token:
            self.cls_token = nn.Parameter(torch.rand(1, self.hidden_dim))
            self.custom_tokens.append(self.cls_token)
        if self.add_glance_token:
            self.glance_token = nn.Parameter(torch.rand(1, self.hidden_dim))
            self.custom_tokens.append(self.glance_token)

        # transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_dim, nhead=self.n_heads, dim_feedforward=4*self.hidden_dim, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=self.n_encoder_layers)

    def forward(self, embedded_glances):

        custom_tokens = torch.vstack(self.custom_tokens)
        tokens_batch = torch.stack([torch.vstack((custom_tokens, embedded_glances[i])) for i in range(len(embedded_glances))])

        # TODO: tokens = torch.cat((self.class_token.expand(n, 1, -1), tokens), dim=1)
        # feed transformer encoder
        out_tokens_batch = self.transformer_encoder(tokens_batch)

        if self.add_cls_token and self.add_glance_token:
            return out_tokens_batch[:, 0], out_tokens_batch[:, 1]
        else:
            return out_tokens_batch[:, 0]


class HapticLSTMCore(nn.Module):
    """ LSTM unit processing sequences of haptic glances """

    def __init__(self, model_params: ModelParameters, core_params: LSTMParameters):
        super().__init__()

        self.lstm = nn.LSTM(model_params.total_embedding_dim, model_params.core_hidden_dim, num_layers=core_params.num_stacked_layers, batch_first=True)
        #self.hidden = torch.rand(1, 1, model_params.core_hidden_dim), torch.rand(1, 1, model_params.core_hidden_dim)

    def forward(self, embedded_glances):
        hidden_states, _ = self.lstm(embedded_glances)
        return hidden_states[:, -1]


class HapticMLPCore(nn.Module):

    def __init__(self, model_params: ModelParameters, core_params: MLPParameters):
        super().__init__()

        self.neuron_dims = core_params.neuron_dims
        self.activation = core_params.activation
        self.dropout = core_params.dropout

        layer_dims = zip(self.neuron_dims, self.neuron_dims[1:])
        linear_layers = [nn.Linear(input_dim, output_dim) for input_dim, output_dim in layer_dims]
        activation_layers = [nn.ReLU() for _ in linear_layers]
        self.layers = nn.Sequential(*(module for pair in zip(linear_layers, activation_layers) for module in pair))

    def forward(self, embedded_glances):
        concatenated_glances = embedded_glances.flatten(start_dim=1)
        out = self.layers(concatenated_glances)
        return out


########## OUTPUT ##########


class ClassificationOutput(nn.Module):

    def __init__(self, model_params: ModelParameters):
        super().__init__()

        self.input_dim = model_params.core_hidden_dim
        self.n_classes = model_params.n_objects

        self.classification_layer = nn.Sequential(
            nn.Linear(self.input_dim, self.n_classes),
        )

    def forward(self, x):
        return self.classification_layer(x)


class ValueOutput(nn.Module):

    def __init__(self, model_params: ModelParameters):
        super().__init__()

        self.input_dim = model_params.core_hidden_dim

        self.value_layer = nn.Linear(self.input_dim, 1)

    def forward(self, x):
        return self.value_layer(x)


class HybridActionOutput(nn.Module):

    def __init__(self, model_params: ModelParameters, hybrid_action_net):
        super().__init__()

        self.hybrid_action_net = hybrid_action_net

    def forward(self, x):
        return self.hybrid_action_net([x for _ in self.hybrid_action_net.hybrid_actions_nets])


class ParameterizedActionOutput(nn.Module):

    def __init__(self, model_params: ModelParameters, parameterized_action_net):
        super().__init__()

        self.parameterized_action_net = parameterized_action_net

    def forward(self, x):
        return self.parameterized_action_net([x, [x for _ in self.parameterized_action_net.hybrid_parameter_action_nets]])


class SingleActionOutput(nn.Module):

    def __init__(self, model_params: ModelParameters, action_net):
        super().__init__()

        self.action_net = action_net

    def forward(self, x):
        return self.action_net(x)


class CombinedOutput(nn.Module):

    def __init__(self, model_params: ModelParameters, build_action_output):
        super().__init__()

        self.classification_output = ClassificationOutput(model_params)
        self.action_output = build_action_output(model_params)
        self.value_output = ValueOutput(model_params)

    def forward(self, x):
        return self.classification_output(x), self.action_output(x), self.value_output(x)


########## COMPLETE MODELS ##########


class HapticModel(nn.Module):

    def __init__(self, base_model, output_model):
        super().__init__()

        self.model = nn.Sequential(
            base_model,
            output_model
        )

    def forward(self, x):
        return self.model(x)


class HapticTransformer(nn.Module):

    def __init__(self, output_class, model_params, core_params):
        super().__init__()

        base_model = nn.Sequential(
            GlanceEmbedding(model_params),
            HapticTransformerCore(model_params, core_params)
        )
        output_model = output_class(model_params)

        self.model = HapticModel(base_model, output_model)

    def forward(self, x):
        return self.model(x)


class HapticLSTM(nn.Module):

    def __init__(self, output_class, model_params, core_params):
        super().__init__()

        base_model = nn.Sequential(
            GlanceEmbedding(model_params),
            HapticLSTMCore(model_params, core_params)
        )
        output_model = output_class(model_params)

        self.model = HapticModel(base_model, output_model)

    def forward(self, x):
        return self.model(x)


class HapticMLP(nn.Module):

    def __init__(self, output_class, model_params, core_params):
        super().__init__()

        base_model = nn.Sequential(
            GlanceEmbedding(model_params),
            HapticMLPCore(model_params, core_params)
        )
        output_model = output_class(model_params)

        self.model = HapticModel(base_model, output_model)

    def forward(self, x):
        return self.model(x)


########## BUILD SPECIFIC VARIANTS FOR CLASSIFICATION, GLANCE PREDICTION OR BOTH ##########

def get_build_action(action_type, action_net):
    if action_type in ["decision", "glance"]:
        def build_output(model_params):
            return SingleActionOutput(model_params, action_net)
    elif action_type == "hybrid":
        def build_output(model_params):
            return HybridActionOutput(model_params, action_net)
    elif action_type == "parameterized":
        def build_output(model_params):
            return ParameterizedActionOutput(model_params, action_net)
    else:
        raise Exception("invalid action type")
    return build_output


def build_model_cls(core_class, core_params):

    def build_model(model_params):
        return core_class(ClassificationOutput, model_params, core_params)
    return build_model


def build_model_action(core_class, core_params, action_type, action_net):
    def build_model(model_params):
        build_action_output = get_build_action(action_type, action_net)
        return core_class(build_action_output, model_params, core_params)
    return build_model


def build_model_value(core_class, core_params):

    def build_model(model_params):
        return core_class(ValueOutput, model_params, core_params)
    return build_model


def build_model_shared(core_class, core_params, action_type, action_net):
    def build_model(model_params):
        def build_output(model_params):
            build_action_output = get_build_action(action_type, action_net)
            return CombinedOutput(model_params, build_action_output)
        return core_class(build_output, model_params, core_params)
    return build_model
