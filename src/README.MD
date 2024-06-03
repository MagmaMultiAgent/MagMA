# Run

To verify your installation, you can run the CLI tool by replacing `path/to/bot/main.py` with a path to a bot.

```bash
luxai-s2 path/to/bot/main.py path/to/bot/main.py -v 2 -o replay.json
```

This will turn on logging to level 2, and store the replay file at *replay.json*.

## Train

To use the training code, run [train.py](https://github.com/MagmaMultiAgent/MagMA/blob/main/src/Lux-Agents-S2/train.py) --help for help and to train an agent run:

```bash
python train_<ALGO>.py --n-envs 8 --log-path <PATH> --seed <SEED>
```

Set your `--n-envs` according to your available CPU cores. This will train an RL agent using the PPO algorithm with 8 parallel environments to sample training data from.

## Evaluation

To start evaluating with the CLI tool and eventually submit to the competition, we need to save our best model (stored in <log_path>/models/best_model.zip) to the root directory. Alternatively you can modify `MODEL_WEIGHTS_RELATIVE_PATH` in [agent.py](https://github.com/MagmaMultiAgent/MagMA/blob/main/src/Lux-Agents-S2/agent.py) to point to where the model file is. If you ran the training script above it will save the trained agent to `results/<ALGORITHM>/<DATE>/<ALGORITHM>_<RUN_NUM>/models/best_model.zip`.

Once that is setup, you can test and watch your trained agent on the HTML visualizer by running the following:

```bash
luxai-s2 main.py main.py --out=replay.html
```

Open up `replay.html` and you can look at what your agent was doing during evalutation. Additionally you can visualize the whole episode utilizing the [Lux Eye S2](https://s2vis.lux-ai.org) tool provided by the organizers of the competition.
