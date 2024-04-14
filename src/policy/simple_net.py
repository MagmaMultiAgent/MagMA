import torch
import torch.nn as nn
import torch.nn.functional as F
# import torchvision.ops as T
import numpy as np
from impl_config import ActDims, UnitActChannel, UnitActType, EnvParam
from .actor_head import sample_from_categorical
import sys
import time
import hashlib

used_names = set()


def myHash(text: str) -> int:
    text_bytes = text.encode('utf-8')
    hash_str = hashlib.md5(text_bytes).hexdigest()
    hash_int = int(hash_str, 16) % (10 ** 8)
    return hash_int


def seed_init(seed: int, name: str, salt: str = ""):
    name = name + salt
    print(f"Setting seed for '{name}'", file=sys.stderr)
    if name in used_names:
        raise ValueError(f"Name {name} already used")
    else:
        used_names.add(name)
    
    seed = myHash(f"seed{seed}_{name}")
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_activation(activation):
    if activation == "relu":
        activation = nn.ReLU()
    elif activation == "leaky_relu":
        activation = nn.LeakyReLU()
    elif activation == "sigmoid":
        activation = nn.Sigmoid()
    elif activation == "tanh":
        activation = nn.Tanh()
    else:
        activation = None
    return activation


def init_orthogonal(module, weight_init, bias_init, gain=1, scaling=1.0):
    """Helper to initialize a layer weight and bias."""
    weight_init(module.weight.data, gain=gain)
    module.weight.data *= scaling
    if module.bias is not None:
        bias_init(module.bias.data)
    return module

weight_scale = 0.01
init_leaky_relu_ = lambda m: init_orthogonal(m, nn.init.orthogonal_, nn.init.zeros_, nn.init.calculate_gain('leaky_relu'), weight_scale)
init_relu_ = lambda m: init_orthogonal(m, nn.init.orthogonal_, nn.init.zeros_, nn.init.calculate_gain('relu'), weight_scale)
init_sigmoid_ = lambda m: init_orthogonal(m, nn.init.orthogonal_, nn.init.zeros_, nn.init.calculate_gain('sigmoid'), weight_scale)
init_value_ = lambda m: init_orthogonal(m, nn.init.orthogonal_, nn.init.zeros_, 0.01, weight_scale)
init_actor_ = lambda m: init_orthogonal(m, nn.init.orthogonal_, nn.init.zeros_, 0.01, weight_scale)


USE_BATCH_NORM = True
USE_LAYER_NORM = False
USE_SPECTRAL_NORM = True


def MyConv2d(name, in_channels, out_channels, kernel_size=1, stride=1, padding="same", bias=True, dilation=1, spectral_norm=False, batch_norm=False, layer_norm=False, activation="leaky_relu", init_fn=None, seed=None):
    if seed is None:
        print(f"No seed provided for {name}")
        seed = int(time.time())

    name = name + "_conv2d"

    activation = get_activation(activation)

    seed_init(seed, name)
    layer = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, dilation=dilation)

    if spectral_norm:
        layer = nn.utils.spectral_norm(layer)

    if init_fn is not None:
        seed_init(seed, name, "_init")
        layer = init_fn(layer)
    
    if batch_norm:
        seed_init(seed, name, "_batch_norm")
        layer = nn.Sequential(layer, nn.BatchNorm2d(out_channels))
    
    if layer_norm:
        seed_init(seed, name, "_layer_norm")
        layer = nn.Sequential(layer, nn.GroupNorm(1, out_channels))

    if activation is not None:
        layer = nn.Sequential(layer, activation)

    return layer


def MyLinear(name, in_features, out_features, bias=True, spectral_norm=False, batch_norm=False, layer_norm=False, activation="leaky_relu", init_fn=None, seed=None):
    if seed is None:
        print(f"No seed provided for {name}", file=sys.stderr)
        seed = int(time.time())

    name = name + "_linear"

    activation = get_activation(activation)

    seed_init(seed, name)
    layer = nn.Linear(in_features, out_features, bias=bias)

    if spectral_norm:
        layer = nn.utils.spectral_norm(layer)

    if init_fn is not None:
        seed_init(seed, name, "_init")
        layer = init_fn(layer)
    
    if batch_norm:
        seed_init(seed, name, "_batch_norm")
        layer = nn.Sequential(layer, nn.BatchNorm1d(out_features))
    
    if layer_norm:
        seed_init(seed, name, "_layer_norm")
        layer = nn.Sequential(layer, nn.GroupNorm(1, out_features))

    if activation is not None:
        layer = nn.Sequential(layer, activation)

    return layer


def Conv1x1(name, in_channels, out_channels, bias=True, spectral_norm=False, batch_norm=False, layer_norm=False, activation="leaky_relu", init_fn=None, seed=None):
    return MyConv2d(name, in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias, spectral_norm=spectral_norm, batch_norm=batch_norm, layer_norm=layer_norm, activation=activation, init_fn=init_fn, seed=seed)

def Conv3x3(name, in_channels, out_channels, bias=True, spectral_norm=False, batch_norm=False, layer_norm=False, activation="leaky_relu", init_fn=None, seed=None):
    return MyConv2d(name, in_channels, out_channels, kernel_size=3, stride=1, padding="same", bias=bias, spectral_norm=spectral_norm, batch_norm=batch_norm, layer_norm=layer_norm, activation=activation, init_fn=init_fn, seed=seed)

def Conv5x5(name, in_channels, out_channels, bias=True, spectral_norm=False, batch_norm=False, layer_norm=False, activation="leaky_relu", init_fn=None, seed=None):
    return MyConv2d(name, in_channels, out_channels, kernel_size=5, stride=1, padding="same", bias=bias, spectral_norm=spectral_norm, batch_norm=batch_norm, layer_norm=layer_norm, activation=activation, init_fn=init_fn, seed=seed)


def EmbeddingConv(name, in_channels, out_channels, seed=None):
    return Conv1x1(name, in_channels, out_channels, bias=(not USE_BATCH_NORM), spectral_norm=USE_SPECTRAL_NORM, batch_norm=USE_BATCH_NORM, layer_norm=USE_LAYER_NORM, activation="leaky_relu", init_fn=init_leaky_relu_, seed=seed)

def Critic(name, in_channels, out_channels, seed=None):
    return nn.Sequential(
        Conv1x1(name, in_channels, out_channels, bias=True, spectral_norm=False, batch_norm=False, layer_norm=False, activation=None, init_fn=init_value_, seed=seed),
    )

def Actor(name, in_features, out_features, seed=None):
    return nn.Sequential(
        MyLinear(name, in_features, out_features, bias=True, spectral_norm=False, batch_norm=False, layer_norm=False, activation=None, init_fn=init_actor_, seed=seed),
    )


class SELayer(nn.Module):

    def __init__(self, name, channel, reduction=16, seed=None):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            MyLinear(name + "_se_fc", channel, channel // reduction, bias=True, spectral_norm=False, batch_norm=False, layer_norm=False, activation="relu", init_fn=init_relu_, seed=seed),
            MyLinear(name + "_se_fc2", channel // reduction, channel, bias=True, spectral_norm=False, batch_norm=False, layer_norm=False, activation="sigmoid", init_fn=init_sigmoid_, seed=seed),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SEResidual(nn.Module):
    def __init__(self, name, layers, channel, reduction=4, seed=None):
        super(SEResidual, self).__init__()
        _layers = []
        for i in range(layers):
            _layers.append(nn.Sequential(
                Conv3x3(name + F"_residual_conv_{i}", channel, channel, bias=(not USE_BATCH_NORM), spectral_norm=USE_SPECTRAL_NORM, batch_norm=USE_BATCH_NORM, layer_norm=USE_LAYER_NORM, activation="leaky_relu", init_fn=init_leaky_relu_, seed=seed),
            ))
        _layers.append(SELayer(name + f"_residual_se", channel, reduction=reduction, seed=seed))
        self.layers = nn.Sequential(*_layers)

    def forward(self, x):
        _x = self.layers(x)
        x = x + _x
        return x


class ActivationNormalizer(nn.Module):
    def __init__(self):
        super(ActivationNormalizer, self).__init__()
    
    def forward(self, x):
        return x / torch.norm(x, dim=1, keepdim=True)


class SimpleNet(nn.Module):

    def __init__(self, max_entity_number: int, seed: int):
        super(SimpleNet, self).__init__()

        self.max_entity_number = max_entity_number

        # EMBEDDINGS
        """
        Embeddings are used to convert the input features into a lower-dimensional space, and to extract relevant information from the input features.
        """

        self.embedding_dims = 32

        self.embedding_feature_counts = {
            "global": 2,
            "factory": 6,
            "unit": 4,
            "map": 6,
        }
        self.embedding_feature_count = sum(self.embedding_feature_counts.values())

        self.embedding_basic_actor = nn.Sequential(
            EmbeddingConv("hidden_conv_1", self.embedding_feature_count, self.embedding_dims, seed=seed),

            SEResidual("se_residual_1", 2, self.embedding_dims, reduction=4, seed=seed),

            SEResidual("se_residual_2", 2, self.embedding_dims, reduction=4, seed=seed),

            EmbeddingConv("hidden_conv_2", self.embedding_dims, self.embedding_dims, seed=seed),

            SEResidual("se_residual_3", 2, self.embedding_dims, reduction=4, seed=seed),

            SEResidual("se_residual_4", 2, self.embedding_dims, reduction=4, seed=seed),

            EmbeddingConv("hidden_conv_3", self.embedding_dims, self.embedding_dims, seed=seed),

            SEResidual("se_residual_5", 2, self.embedding_dims, reduction=4, seed=seed),

            SEResidual("se_residual_6", 2, self.embedding_dims, reduction=4, seed=seed),

            EmbeddingConv("hidden_conv_4", self.embedding_dims, self.embedding_dims, seed=seed),

             SEResidual("se_residual_7", 2, self.embedding_dims, reduction=4, seed=seed),

            SEResidual("se_residual_8", 2, self.embedding_dims, reduction=4, seed=seed),

            EmbeddingConv("hidden_conv_5", self.embedding_dims, self.embedding_dims, seed=seed),
        )

        self.embedding_basic_value = nn.Sequential(
            EmbeddingConv("_hidden_conv_1", self.embedding_feature_count, self.embedding_dims, seed=seed),

            SEResidual("_se_residual_1", 2, self.embedding_dims, reduction=4, seed=seed),

            SEResidual("_se_residual_2", 2, self.embedding_dims, reduction=4, seed=seed),

            EmbeddingConv("_hidden_conv_2", self.embedding_dims, self.embedding_dims, seed=seed),

            SEResidual("_se_residual_3", 2, self.embedding_dims, reduction=4, seed=seed),

            SEResidual("_se_residual_4", 2, self.embedding_dims, reduction=4, seed=seed),

            EmbeddingConv("_hidden_conv_3", self.embedding_dims, self.embedding_dims, seed=seed),

            SEResidual("_se_residual_5", 2, self.embedding_dims, reduction=4, seed=seed),

            SEResidual("_se_residual_6", 2, self.embedding_dims, reduction=4, seed=seed),

            EmbeddingConv("_hidden_conv_4", self.embedding_dims, self.embedding_dims, seed=seed),

            SEResidual("_se_residual_7", 2, self.embedding_dims, reduction=4, seed=seed),

            SEResidual("_se_residual_8", 2, self.embedding_dims, reduction=4, seed=seed),

            EmbeddingConv("_hidden_conv_5", self.embedding_dims, self.embedding_dims, seed=seed),
        )

        # HEADS

        # critic
        self.critic_feature_count = self.embedding_dims
        self.critic_dim = 4
        self.critic_head = nn.Sequential(
            Conv1x1("critic_1", self.critic_feature_count, self.critic_dim, bias=True, spectral_norm=True, batch_norm=True, layer_norm=False, activation="leaky_relu", init_fn=init_leaky_relu_, seed=seed),
            Critic("critic_o", self.critic_dim, 2, seed=seed),  # 2 channels: 0 for unit, 1 for factory
        )

        # factory
        self.factory_feature_count = self.embedding_dims
        self.factory_head = nn.Sequential(
            Actor("factory_act", self.factory_feature_count, ActDims.factory_act, seed=seed),
        )

        self.unit_feature_count = self.embedding_dims
        self.unit_emb_dim = self.unit_feature_count

        # act type
        self.act_type_feature_count = self.unit_emb_dim
        self.unit_act_type_net = nn.Sequential(
            Actor("unit_act_type", self.act_type_feature_count, len(UnitActType), seed=seed),
        )

        # params
        self.param_heads = nn.ModuleDict({
            unit_act_type.name: nn.ModuleDict({
                "direction": Actor(f"{unit_act_type}_direction", self.unit_emb_dim, ActDims.direction, seed=seed),
                "resource": Actor(f"{unit_act_type}_resource", self.unit_emb_dim, ActDims.resource, seed=seed),
                "amount": Actor(f"{unit_act_type}_amount", self.unit_emb_dim, ActDims.amount, seed=seed),
                "repeat": Actor(f"{unit_act_type}_repeat", self.unit_emb_dim, ActDims.repeat, seed=seed),
            }) for unit_act_type in UnitActType
        })


    def forward(self, global_feature, map_feature, factory_feature, unit_feature, location_feature, va, action=None, is_deterministic=False):
        B, _, H, W = map_feature.shape
        max_group_count = self.max_entity_number

        # Embeddings
        global_feature = global_feature[..., None, None].expand(-1, -1, H, W)
        all_features = torch.cat([global_feature, factory_feature, unit_feature, map_feature], dim=1)
        features_embedded_actor = self.embedding_basic_actor(all_features)
        features_embedded_value = self.embedding_basic_value(all_features)

        # Valid actions
        unit_act_type_va = torch.stack(
            [
                va['move'].flatten(1, 2).any(1),
                va['transfer'].flatten(1, 3).any(1),
                va['pickup'].flatten(1, 2).any(1),
                va['dig'].any(1),
                va['self_destruct'].any(1),
                va['recharge'].any(1),
                va['do_nothing'],
            ],
            axis=1,
        )

        # Locations
        factory_pos = torch.where(va['factory_act'].any(1))
        unit_pos = torch.where(unit_act_type_va.any(1))
        def _gather_from_map(x, pos):
            return x[pos[0], ..., pos[1], pos[2]]
        factory_ids = _gather_from_map(location_feature[:, 0], factory_pos).int()
        unit_ids = (_gather_from_map(location_feature[:, 1], unit_pos)).int()
        unit_indices = unit_pos[0] * max_group_count + unit_ids
        if len(unit_indices) > 0:
            assert unit_indices.max(dim=-1)[0] < (B * max_group_count)
            assert unit_indices.min(dim=-1)[0] >= 0
        factory_indices = factory_pos[0] * max_group_count + factory_ids

        # Critic
        critic_value = torch.zeros((B, max_group_count), device=features_embedded_value.device)
        critic_value = critic_value.view(-1)

        _critic_value = self.critic(features_embedded_value)
        _critic_value_unit = _gather_from_map(_critic_value[:, 0], unit_pos)
        if len(unit_indices) > 0:
            critic_value.scatter_add_(0, unit_indices, _critic_value_unit)
        _critic_value_factory = _gather_from_map(_critic_value[:, 1], factory_pos)
        if len(factory_indices) > 0:
            critic_value.scatter_add_(0, factory_indices, _critic_value_factory)

        critic_value = critic_value.view(B, max_group_count)

        # Actor
        logp, action, entropy = self.actor(features_embedded_actor, va, factory_pos, unit_act_type_va, unit_pos, factory_ids, max_group_count, unit_indices, factory_indices, action, is_deterministic)

        return logp, critic_value, action, entropy

    def critic(self, x):
        critic_value = self.critic_head(x)
        return critic_value

    def actor(self, x, va, factory_pos, unit_act_type_va, unit_pos, factory_ids, max_group_count, unit_indices, factory_indices, action=None, is_deterministic=False):
        B, _, H, W = x.shape

        logp = torch.zeros((B, max_group_count), device=x.device)
        logp = logp.view(-1)
        entropy = torch.zeros((B, max_group_count), device=x.device)
        entropy = entropy.view(-1)
        output_action = {}

        def _gather_from_map(x, pos):
            return x[pos[0], ..., pos[1], pos[2]]

        def _put_into_map(emb, pos):
            shape = (B, ) + emb.shape[1:] + (H, W)
            map = torch.zeros(shape, dtype=emb.dtype, device=emb.device)
            map[pos[0], ..., pos[1], pos[2]] = emb
            return map

        # factory actor
        factory_emb = _gather_from_map(x, factory_pos)

        factory_va = _gather_from_map(va['factory_act'], factory_pos)
        factory_action = action and _gather_from_map(action['factory_act'], factory_pos)
        factory_logp, factory_action, factory_entropy = self.factory_actor(
            factory_emb,
            factory_va,
            factory_action,
            is_deterministic=is_deterministic,
        )

        if len(factory_indices) > 0:
            logp.scatter_add_(0, factory_indices, factory_logp)
            entropy.scatter_add_(0, factory_indices, factory_entropy)
        
        output_action['factory_act'] = _put_into_map(factory_action, factory_pos)

        # unit actor
        unit_emb = _gather_from_map(x, unit_pos)
        
        unit_va = {
            'act_type': _gather_from_map(unit_act_type_va, unit_pos),
            'move': _gather_from_map(va['move'], unit_pos),
            'transfer': _gather_from_map(va['transfer'], unit_pos),
            'pickup': _gather_from_map(va['pickup'], unit_pos),
            'dig': _gather_from_map(va['dig'], unit_pos),
            'self_destruct': _gather_from_map(va['self_destruct'], unit_pos),
            'recharge': _gather_from_map(va['recharge'], unit_pos),
            'do_nothing': _gather_from_map(va['do_nothing'], unit_pos),
        }

        unit_action = action and _gather_from_map(action['unit_act'], unit_pos)
        unit_logp, unit_action, unit_entropy = self.unit_actor(
            unit_emb,
            unit_emb,
            unit_va,
            unit_action,
            is_deterministic=is_deterministic,
        )

        if len(unit_indices) > 0:
            logp.scatter_add_(0, unit_indices, unit_logp)
            entropy.scatter_add_(0, unit_indices, unit_entropy)

        output_action['unit_act'] = _put_into_map(unit_action, unit_pos)

        logp = logp.view(B, max_group_count)
        entropy = entropy.view(B, max_group_count)

        return logp, output_action, entropy


    def factory_actor(self, x, va, action=None, is_deterministic=False):
        logits = self.factory_head(x)
        logp, output_action, entropy = sample_from_categorical(logits, va, action, is_deterministic)
        return logp, output_action, entropy


    def unit_actor(self, x_act, x_param,  va, action=None, is_deterministic=False):
        act_type_logp, act_type, act_type_entropy = sample_from_categorical(
            self.unit_act_type_net(x_act),
            va['act_type'],
            action[:, UnitActChannel.TYPE] if action is not None else None,
            is_deterministic
        )
        logp = act_type_logp
        entropy = act_type_entropy
        output_action = torch.zeros((x_act.shape[0], len(UnitActChannel)), device=x_act.device)

        for type in UnitActType:
            mask = (act_type == type)
            move_logp, move_action, move_entropy = self.get_unit_action(
                x_param[mask],
                va[type.name.lower()][mask],
                type,
                action[mask] if action is not None else None,
                is_deterministic,
            )
            logp[mask] += move_logp
            entropy[mask] += move_entropy
            output_action[mask] = move_action

        return logp, output_action, entropy


    def get_unit_action(self, x, va, unit_act_type, action=None, is_deterministic=False):
        n_units = x.shape[0]
        unit_idx = torch.arange(n_units, device=x.device)

        output_action = torch.zeros((n_units, len(UnitActChannel)), device=x.device)

        params, param_logp, param_entropy = self.get_params(unit_idx, x, unit_act_type, va, action, is_deterministic)

        output_action[:, UnitActChannel.TYPE] = unit_act_type
        output_action[:, UnitActChannel.N] = 1
        for param_name in ["direction", "resource", "amount", "repeat"]:
            if param_name in params and params[param_name] is not None:
                output_action[:, UnitActChannel[param_name.upper()]] = params[param_name]

        return param_logp, output_action, param_entropy


    def get_params(self, unit_idx, x, action_type, va, action=None, is_deterministic=False):
        n_units = x.shape[0]
        
        direction, resource, amount, repeat = None, None, None, None
        direction_logp = torch.zeros(n_units, device=x.device)
        resource_logp = torch.zeros(n_units, device=x.device)
        amount_logp = torch.zeros(n_units, device=x.device)
        repeat_logp = torch.zeros(n_units, device=x.device)
        direction_entropy = torch.zeros(n_units, device=x.device)
        resource_entropy = torch.zeros(n_units, device=x.device)
        amount_entropy = torch.zeros(n_units, device=x.device)
        repeat_entropy = torch.zeros(n_units, device=x.device)

        # direction
        if action_type in [UnitActType.MOVE, UnitActType.TRANSFER]:
            direction, direction_logp, direction_entropy = self.get_direction_param(x, va, action_type, action, is_deterministic)

        # resource
        if action_type in [UnitActType.TRANSFER, UnitActType.PICKUP]:
            resource, resource_logp, resource_entropy = self.get_resource_param(x, va, action_type, unit_idx, direction, action, is_deterministic)

        # amount
        if action_type in [UnitActType.PICKUP, UnitActType.RECHARGE]:
            amount, amount_logp, amount_entropy = self.get_amount_param(x, action_type, action, is_deterministic)

        # repeat
        if action_type in [UnitActType.MOVE, UnitActType.DIG]:
            repeat, repeat_logp, repeat_entropy = self.get_repeat_param(x, va, action_type, unit_idx, direction, resource, action, is_deterministic)

        return {"direction": direction, "resource": resource, "amount": amount, "repeat": repeat}, \
                direction_logp + resource_logp + amount_logp + repeat_logp, \
                direction_entropy + resource_entropy + amount_entropy + repeat_entropy


    def get_direction_param(self, x, va, action_type, action=None, is_deterministic=False):
        direction_va = va.flatten(2).any(dim=-1)
        direction_head = self.param_heads[action_type.name]['direction']
        direction_logp, direction, direction_entropy = sample_from_categorical(
            direction_head(x),
            direction_va,
            action[:, UnitActChannel.DIRECTION] if action is not None else None,
            is_deterministic,
        )
        return direction, direction_logp, direction_entropy


    def get_resource_param(self, x, va, action_type, unit_idx, direction, action=None, is_deterministic=False):
        if action_type in [UnitActType.TRANSFER]:
            resource_va = va[unit_idx, direction].flatten(2).any(-1)
        elif action_type in [UnitActType.PICKUP]:
            resource_va = va.flatten(2).any(-1)
        else:
            resource_va = va
        resource_head = self.param_heads[action_type.name]['resource']
        resource_logp, resource, resource_entropy = sample_from_categorical(
            resource_head(x),
            resource_va,
            action[:, UnitActChannel.RESOURCE] if action is not None else None,
            is_deterministic,
        )
        return resource, resource_logp, resource_entropy


    def get_amount_param(self, x, action_type, action=None, is_deterministic=False):
        amount_head = self.param_heads[action_type.name]['amount']
        amount_logp, amount, amount_entropy = sample_from_categorical(
            amount_head(x),
            torch.tensor(True, device=x.device),
            action[:, UnitActChannel.AMOUNT] if action is not None else None,
            is_deterministic,
        )
        return amount, amount_logp, amount_entropy


    def get_repeat_param(self, x, va, action_type, unit_idx, direction, resource, action=None, is_deterministic=False):
        if action_type in [UnitActType.MOVE]:
            repeat_va = va[unit_idx, direction]
        elif action_type in [UnitActType.PICKUP]:
            repeat_va = va[unit_idx, resource]
        elif action_type in [UnitActType.TRANSFER]:
            repeat_va = va[unit_idx, direction, resource]
        else:
            repeat_va = va

        repeat_head = self.param_heads[action_type.name]['repeat']
        repeat_logp, repeat, repeat_entropy = sample_from_categorical(
            repeat_head(x),
            repeat_va,
            action[:, UnitActChannel.REPEAT] if action is not None else None,
            is_deterministic,
        )

        return repeat, repeat_logp, repeat_entropy
    