from policy.simple_net import SimpleNet
import torch
import numpy as np


PATH = 'src/focus_on_ice_m_dir__16_0_better_masking_separate_dir_obs_simple_prev_reward_4096_model_655360.pth'

agent = SimpleNet()
agent.load_state_dict(torch.load(PATH,map_location=torch.device('cpu')))

obs = torch.tensor(
    [[
        0,  # factory

        1,  # light
        0,  # heavy

        0,  # ice

        0,  # power
        0,  # cargo ice

        0.5, # distance from ice

        1,  # cloest ice up
        0,  # cloest ice right
        0,  # cloest ice down
        0,  # cloest ice left

        0,  # ice up
        0,  # ice down
        0,  # ice left
        0   # ice right
    ]]
).float()

obs_act = obs[:, :7]
obs_param = obs[:, 7:]

softmax = torch.nn.Softmax(dim=1)

x = agent.unit_act_net(obs_act)
x = agent.unit_act_type(x)
x[:, 1] = -1e10
x[:, 2] = -1e10
x[:, 4] = -1e10
x = softmax(x)

for p, name in zip(x[0], ['move', 'transfer', 'pickup', 'dig', 'self_destruct', 'recharge', 'do_nothing']):
    if name in ['move', 'dig', 'recharge', 'do_nothing']:
        print(f"{name}: {round(p.item()*100, 2)}%")

print("---")
x = agent.unit_param_net(obs_param)
x = agent.param_heads['MOVE']['direction'](x)
x = softmax(x)

for p, name in zip(x[0], ['up', 'right', 'down', 'left']):
    print(f"{name}: {round(p.item()*100, 2)}%")
