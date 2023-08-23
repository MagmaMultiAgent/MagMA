# Run

To verify your installation, you can run the CLI tool by replacing `path/to/bot/main.py` with a path to a bot.

```bash
luxai-s2 path/to/bot/main.py path/to/bot/main.py -v 2 -o replay.json
```

This will turn on logging to level 2, and store the replay file at *replay.json*.

# Train

To use the training code, run [train.py](https://github.com/Getlar/VigIL-Game-Validation/blob/main/src/Lux-Agents-S2/train.py) --help for help and to train an agent run:

```bash
python train.py --n-envs 16 --log-path ../results/<ALGORITHM>/<DATE>/<ALGORITHM>_<RUN_NUM> --seed 999
```

Set your `--n-envs` according to your available CPU cores. This will train an RL agent using the PPO algorithm with 16 parallel environments to sample from.

# Evaluation

To start evaluating with the CLI tool and eventually submit to the competition, we need to save our best model (stored in <log_path>/models/best_model.zip) to the root directory. Alternatively you can modify `MODEL_WEIGHTS_RELATIVE_PATH` in [agent.py](https://github.com/Getlar/VigIL-Game-Validation/blob/main/src/Lux-Agents-S2/agent.py) to point to where the model file is. If you ran the training script above it will save the trained agent to `results/<ALGORITHM>/<DATE>/<ALGORITHM>_<RUN_NUM>/models/best_model.zip`.

Once that is setup, you can test and watch your trained agent on the nice HTML visualizer by running the following:

```bash
luxai-s2 main.py main.py --out=replay.html
```

Open up `replay.html` and you can look at what your agent is doing.

Or create virtual environments to visually assess the agent and produce a `.mp4` video file.

```bash
python train.py --n-envs 16 --log-path ../results/<ALGORITHM>/<DATE>/<ALGORITHM>_<RUN_NUM> --model-path ../results/<ALGORITHM>/<DATE>/<ALGORITHM>_<RUN_NUM>/models/best_model --eval --seed 999
```

