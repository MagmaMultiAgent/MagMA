
<p align="center">
    <img width="100%"  src="docs/images/logo.png" />
</p>

# VigIL - Lux AI Competition

This repository is a record of my research journey, starting from brainstorming **RL and IL** solutions in game environments, diving into RL competitions, and eventually crafting agents for the 2023 Lux AI competition held during the yearly **NeurIPS conference**.

The **Lux AI Challenge** is a competition where competitors design agents to tackle a multi-variable optimization, resource gathering, and allocation problem in a 1v1 scenario against other competitors. In addition to optimization, successful agents must be capable of analyzing their opponents and developing appropriate policies to get the upper hand.

# Getting Started

I highly recommend utilizing Docker to set up and manage the environment on your system. Alternatively, a binary installation is also a viable option, especially considering the excellent performance and reliability of the official [pip](https://pypi.org/project/luxai-s2/) package.

## Docker

For implementing this solution, it's essential to have Docker Desktop installed on your system. Detailed guides for setting up Docker Desktop on [Mac](https://docs.docker.com/desktop/install/mac-install/), [Windows](https://docs.docker.com/desktop/install/windows-install/), and [Linux](https://docs.docker.com/desktop/install/linux-install/) can be accessed through the official Docker website. 

### CPU

Once Docker is properly configured on your system to execute the environment using a **CPU**, you can proceed by using the provided **run script**.

```bash
bash ./docker_run.sh -d cpu -v latest
```

### GPU

To employ [JAX](https://github.com/google/jax) as the backend and execute the environment on a GPU device, follow the script below:

```bash
bash ./docker_run.sh -d gpu -v latest
```

## Binary

You will need Python >=3.7, <3.11 installed on your system. Once installed, you can install the Lux AI season 2 environment and optionally the GPU version with:

```bash
pip install --upgrade luxai_s2
pip install juxai-s2 # installs the GPU version, requires a compatible GPU
```

Due to potential compatibility challenges when installing `gym` alongside `vec_noise` and the `stable_baselines3` package, it is advisable to apply specific version **tags** during installation to prevent potential crashes. To mitigate this, I've streamlined the package selection in the [environment.yml file](https://github.com/Getlar/VigIL-Game-Validation/blob/main/envs/conda/environment.yml), aiming to alleviate strain on the conda environment setup.

To create a conda environment and use it run:

```bash
conda env create -f environment.yml
conda activate luxai_s2
```

To install **additional packages** required to train and run specific agents run the following commands:

```bash
apt update -y && apt upgrade -y && apt install -y build-essential && apt-get install -y manpages-dev # Required if dev tools are missing.
```
```bash
pip install --no-cache-dir setuptools==57.1.0 psutil==5.7.0 pettingzoo==1.12.0 vec_noise==1.1.4 ipykernel pygame termcolor wheel==0.38.4 notebook tensorboard
```
```bash
pip install stable_baselines3==1.7.0 gym==0.21 --upgrade luxai_s2 
```
# Run

To verify your installation, you can run the CLI tool by replacing `path/to/bot/main.py` with a path to a bot.

```bash
luxai-s2 path/to/bot/main.py path/to/bot/main.py -v 2 -o replay.json
```

This will turn on logging to level 2, and store the replay file at *replay.json*.

# Train

To use the training code, run [train.py](https://github.com/Getlar/VigIL-Game-Validation/blob/main/src/Lux-Agents-S2/train.py) --help for help and to train an agent run:

```bash
python src/Lux-Agents-S2/train.py --n-envs 4 --log-path logs/exp_1  --seed 660
```

Set your `--n-envs` according to your available CPU cores. This will train an RL agent using the PPO algorithm with 4 parallel environments to sample from.

# Evaluation

To start evaluating with the CLI tool and eventually submit to the competition, we need to save our best model (stored in <log_path>/models/best_model.zip) to the root directory. Alternatively you can modify `MODEL_WEIGHTS_RELATIVE_PATH` in [agent.py](https://github.com/Getlar/VigIL-Game-Validation/blob/main/src/Lux-Agents-S2/agent.py) to point to where the model file is. If you ran the training script above it will save the trained agent to `logs/exp_1/models/best_model.zip`.

Once that is setup, you can test and watch your trained agent on the nice HTML visualizer by running the following:

```bash
luxai-s2 main.py main.py --out=replay.html
```

Open up `replay.html` and you can look at what your agent is doing.

# Core Contributors

I would like to extend my heartfelt gratitude to [Gulyás László](https://github.com/lesIII) for their invaluable guidance and insightful mentorship throughout the course of this project.

I am also thankful to **Eötvös Lóránd University** for providing the necessary resources and environment that facilitated the development of this project.