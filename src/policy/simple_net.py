import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from impl_config import ActDims, UnitActChannel, UnitActType, EnvParam
from .actor_head import sample_from_categorical


class SimpleNet(nn.Module):

    def __init__(self):
        super(SimpleNet, self).__init__()

        self.critic_head = nn.Linear(4, 1, bias=True)

        self.factory_head = nn.Linear(4, ActDims.factory_act, bias=True)
        self.unit_act_type = nn.Linear(4, len(UnitActType), bias=True)

        self.param_heads = nn.ModuleDict({
            unit_act_type.name: nn.ModuleDict({
                "direction": nn.Linear(4, ActDims.direction, bias=True),
                "resource": nn.Linear(4, ActDims.resource, bias=True),
                "amount": nn.Linear(4, ActDims.amount, bias=True),
                "repeat": nn.Linear(4, ActDims.repeat, bias=True),
            }) for unit_act_type in UnitActType
        })

    def forward(self, global_feature, map_feature, action_feature, va, action=None):
        B, _, H, W = map_feature.shape

        # TODO: remove and use real input
        x = torch.rand(B, 4, H, W, device=map_feature.device)
        logp, action, entropy = self.actor(x, va, action)

        x = torch.rand(B, 4, device=map_feature.device)
        critic_value = self.critic_head(x)

        return logp, critic_value, action, entropy


    def actor(self, x, va, action=None):
        B, _, H, W = x.shape

        logp = torch.zeros(B, device=x.device)
        entropy = torch.zeros(B, device=x.device)
        output_action = {}

        def _gather_from_map(x, pos):
            return x[pos[0], ..., pos[1], pos[2]]

        def _put_into_map(emb, pos):
            shape = (B, ) + emb.shape[1:] + (H, W)
            map = torch.zeros(shape, dtype=emb.dtype, device=emb.device)
            map[pos[0], ..., pos[1], pos[2]] = emb
            return map

        # factory actor
        factory_pos = torch.where(va['factory_act'].any(1))
        factory_emb = _gather_from_map(x, factory_pos)
        factory_va = _gather_from_map(va['factory_act'], factory_pos)
        factory_action = action and _gather_from_map(action['factory_act'], factory_pos)

        factory_logp, factory_action, factory_entropy = self.factory_actor(
            factory_emb,
            factory_va,
            factory_action,
        )
        logp.scatter_add_(0, factory_pos[0], factory_logp)
        entropy.scatter_add_(0, factory_pos[0], factory_entropy)
        output_action['factory_act'] = _put_into_map(factory_action, factory_pos)

        # unit actor
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
        unit_pos = torch.where(unit_act_type_va.any(1))
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
            unit_va,
            unit_action,
        )
        logp.scatter_add_(0, unit_pos[0], unit_logp)
        entropy.scatter_add_(0, unit_pos[0], unit_entropy)
        output_action['unit_act'] = _put_into_map(unit_action, unit_pos)

        return logp, output_action, entropy


    def factory_actor(self, x, va, action=None):
        logits = self.factory_head(x)
        logp, output_action, entropy = sample_from_categorical(logits, va, action)
        return logp, output_action, entropy


    def unit_actor(self, x, va, action=None):
        act_type_logp, act_type, act_type_entropy = sample_from_categorical(
            self.unit_act_type(x),
            va['act_type'],
            action[:, UnitActChannel.TYPE] if action is not None else None,
        )
        logp = act_type_logp
        entropy = act_type_entropy
        output_action = torch.zeros((x.shape[0], len(UnitActChannel)), device=x.device)

        for type in UnitActType:
            mask = (act_type == type)
            move_logp, move_action, move_entropy = self.get_unit_action(
                x[mask],
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
        if action_type in [UnitActType.MOVE, UnitActType.TRANSFER, UnitActType.PICKUP, UnitActType.DIG, UnitActType.SELF_DESTRUCT, UnitActType.RECHARGE]:
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

