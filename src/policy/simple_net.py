import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from impl_config import ActDims, UnitActChannel, UnitActType, EnvParam
from .actor_head import sample_from_categorical
import sys


class SimpleNet(nn.Module):

    def __init__(self):
        super(SimpleNet, self).__init__()

        # Factories
        self.factory_feature_count = 6
        self.factory_feature_dims = 6
        self.factory_head = nn.Linear(self.factory_feature_dims, ActDims.factory_act, bias=True)

        # Units and critic
        self.global_feature_count = 4
        self.map_feature_count = 2
        self.unit_feature_count = 3

        self.large_distance_embedding = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
        )
        self.large_distance_norm = nn.BatchNorm2d(self.map_feature_count)

        self.value_feature_dim = self.map_feature_count + self.unit_feature_count
        self.act_type_feature_dim = self.map_feature_count + self.unit_feature_count

        self.critic_head = nn.Sequential(
            nn.Conv2d(self.value_feature_dim, 1, kernel_size=1, stride=1, padding=0, bias=True),
        )

        self.direction_dim = 4
        self.direction_net = nn.Sequential(
            nn.Conv2d(self.map_feature_count + self.unit_feature_count, self.direction_dim, kernel_size=3, stride=1, padding="same", bias=True),
            nn.LeakyReLU(),
        )

        self.unit_act_type = nn.Linear(self.act_type_feature_dim, len(UnitActType), bias=True)
        self.param_heads = nn.ModuleDict({
            unit_act_type.name: nn.ModuleDict({
                "direction": nn.Linear(self.direction_dim, ActDims.direction, bias=True),
                "resource": nn.Linear(self.direction_dim, ActDims.resource, bias=True),
                "amount": nn.Linear(self.direction_dim, ActDims.amount, bias=True),
                "repeat": nn.Linear(self.direction_dim, ActDims.repeat, bias=True),
            }) for unit_act_type in UnitActType
        })


    def forward(self, global_feature, map_feature, factory_feature, unit_feature, location_feature, va, action=None):
        B, _, H, W = map_feature.shape

        # Embeddings
        global_feature = global_feature[..., None, None].expand(-1, -1, H, W)

        large_embedding = self.large_distance_embedding(map_feature)
        # make rubble zero
        # large_embedding[:, 2] = 0
        large_embedding += map_feature
        # scale between -1 and 1 for each channel
        for i in range(large_embedding.shape[1]):
            large_embedding[:, i] = 2 * (large_embedding[:, i] - large_embedding[:, i].min()) / (large_embedding[:, i].max() - large_embedding[:, i].min()) - 1
        assert large_embedding.shape[2] == H
        assert large_embedding.shape[3] == W

        value_feature = torch.cat([map_feature, unit_feature], dim=1)
        act_type_feature = torch.cat([map_feature, unit_feature], dim=1)

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
        unit_ids = (_gather_from_map(location_feature[:, 1], unit_pos) + 10).int()

        # Critic
        critic_value = torch.zeros((B, 1000), device=value_feature.device)
        _critic_value = self.critic(value_feature)
        _critic_value_unit = _gather_from_map(_critic_value, unit_pos)
        critic_value[unit_pos[0], unit_ids] = _critic_value_unit

        # Actor
        direction_feature = self.direction_net(torch.cat([large_embedding, unit_feature], dim=1))
        logp, action, entropy = self.actor(act_type_feature, direction_feature, factory_feature, va, factory_pos, unit_act_type_va, unit_pos, factory_ids, unit_ids, action)

        return logp, critic_value, action, entropy

    def critic(self, value_feature):
        critic_value = self.critic_head(value_feature)[:, 0]
        return critic_value

    def actor(self, act_type_feature, direction_feature, factory_feature, va, factory_pos, unit_act_type_va, unit_pos, factory_ids, unit_ids, action=None):
        B, _, H, W = act_type_feature.shape

        logp = torch.zeros((B, 1000), device=act_type_feature.device)
        entropy = torch.zeros((B, 1000), device=act_type_feature.device)
        output_action = {}

        def _gather_from_map(x, pos):
            return x[pos[0], ..., pos[1], pos[2]]

        def _put_into_map(emb, pos):
            shape = (B, ) + emb.shape[1:] + (H, W)
            map = torch.zeros(shape, dtype=emb.dtype, device=emb.device)
            map[pos[0], ..., pos[1], pos[2]] = emb
            return map

        # factory actor
        factory_emb = _gather_from_map(factory_feature, factory_pos)

        factory_va = _gather_from_map(va['factory_act'], factory_pos)
        factory_action = action and _gather_from_map(action['factory_act'], factory_pos)
        _, factory_action, _ = self.factory_actor(
            factory_emb,
            factory_va,
            factory_action,
        )
        
        output_action['factory_act'] = _put_into_map(factory_action, factory_pos)

        # unit actor
        unit_emb = _gather_from_map(act_type_feature, unit_pos)
        unit_dir = _gather_from_map(direction_feature, unit_pos)

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
            unit_dir,
            unit_va,
            unit_action,
        )

        logp[unit_pos[0], unit_ids] = unit_logp
        entropy[unit_pos[0], unit_ids] = unit_entropy

        output_action['unit_act'] = _put_into_map(unit_action, unit_pos)

        return logp, output_action, entropy


    def factory_actor(self, x, va, action=None):
        logits = self.factory_head(x)
        logp, output_action, entropy = sample_from_categorical(logits, va, action)
        return logp, output_action, entropy


    def unit_actor(self, x_act, x_param,  va, action=None):
        act_type_logp, act_type, act_type_entropy = sample_from_categorical(
            self.unit_act_type(x_act),
            va['act_type'],
            action[:, UnitActChannel.TYPE] if action is not None else None
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
            )
            logp[mask] += move_logp
            entropy[mask] += move_entropy
            output_action[mask] = move_action

        return logp, output_action, entropy


    def get_unit_action(self, x, va, unit_act_type, action=None):
        n_units = x.shape[0]
        unit_idx = torch.arange(n_units, device=x.device)

        output_action = torch.zeros((n_units, len(UnitActChannel)), device=x.device)

        params, param_logp, param_entropy = self.get_params(unit_idx, x, unit_act_type, va, action)

        output_action[:, UnitActChannel.TYPE] = unit_act_type
        output_action[:, UnitActChannel.N] = 1
        for param_name in ["direction", "resource", "amount", "repeat"]:
            if param_name in params and params[param_name] is not None:
                output_action[:, UnitActChannel[param_name.upper()]] = params[param_name]

        return param_logp, output_action, param_entropy


    def get_params(self, unit_idx, x, action_type, va, action=None):
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
            direction, direction_logp, direction_entropy = self.get_direction_param(x, va, action_type, action)

        # resource
        if action_type in [UnitActType.TRANSFER, UnitActType.PICKUP]:
            resource, resource_logp, resource_entropy = self.get_resource_param(x, va, action_type, unit_idx, direction, action)

        # amount
        if action_type in [UnitActType.TRANSFER, UnitActType.PICKUP, UnitActType.RECHARGE]:
            amount, amount_logp, amount_entropy = self.get_amount_param(x, action_type, action)

        # repeat
        if action_type in [UnitActType.DIG]:
            repeat, repeat_logp, repeat_entropy = self.get_repeat_param(x, va, action_type, unit_idx, direction, resource, action)

        return {"direction": direction, "resource": resource, "amount": amount, "repeat": repeat}, \
                direction_logp + resource_logp + amount_logp + repeat_logp, \
                direction_entropy + resource_entropy + amount_entropy + repeat_entropy


    def get_direction_param(self, x, va, action_type, action=None):
        direction_va = va.flatten(2).any(dim=-1)
        direction_head = self.param_heads[action_type.name]['direction']
        direction_logp, direction, direction_entropy = sample_from_categorical(
            direction_head(x),
            direction_va,
            action[:, UnitActChannel.DIRECTION] if action is not None else None,
        )
        return direction, direction_logp, direction_entropy


    def get_resource_param(self, x, va, action_type, unit_idx, direction, action=None):
        resource_va = va[unit_idx, direction].flatten(2).any(-1)
        resource_head = self.param_heads[action_type.name]['resource']
        resource_logp, resource, resource_entropy = sample_from_categorical(
            resource_head(x),
            resource_va,
            action[:, UnitActChannel.RESOURCE] if action is not None else None,
        )
        return resource, resource_logp, resource_entropy


    def get_amount_param(self, x, action_type, action=None):
        amount_head = self.param_heads[action_type.name]['amount']
        amount_logp, amount, amount_entropy = sample_from_categorical(
            amount_head(x),
            torch.tensor(True, device=x.device),
            action[:, UnitActChannel.AMOUNT] if action is not None else None,
        )
        return amount, amount_logp, amount_entropy


    def get_repeat_param(self, x, va, action_type, unit_idx, direction, resource, action=None):
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
        )

        return repeat, repeat_logp, repeat_entropy