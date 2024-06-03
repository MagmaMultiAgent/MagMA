<p align="center">
    <img width="100%"  src="docs/images/logo.png"/>
</p>

[![Docker](https://github.com/MagmaMultiAgent/MagMA/actions/workflows/docker-publish.yml/badge.svg?branch=v0.0.3-Monolithic)](https://github.com/MagmaMultiAgent/MagMA/actions/workflows/docker-publish.yml)&nbsp;[![Pylint](https://github.com/MagmaMultiAgent/MagMA/actions/workflows/pylint.yml/badge.svg?branch=v0.0.3-Monolithic)](https://github.com/MagmaMultiAgent/MagMA/actions/workflows/pylint.yml)&nbsp;[![CodeQL Advanced](https://github.com/MagmaMultiAgent/MagMA/actions/workflows/codeql.yml/badge.svg?branch=v0.0.3-Monolithic)](https://github.com/MagmaMultiAgent/MagMA/actions/workflows/codeql.yml)&nbsp;[![PyTest](https://github.com/MagmaMultiAgent/MagMA/actions/workflows/pytest.yml/badge.svg?branch=v0.0.3-Monolithic)](https://github.com/MagmaMultiAgent/MagMA/actions/workflows/pytest.yml)&nbsp;![badge](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/Getlar/5a256487d767cc5d606d29bd521a18ae/raw/algo.json)&nbsp;![badge](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/Getlar/59b429b8fd9c4a525959c66dfa4ab97e/raw/TDK.json)&nbsp;![badge](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/Getlar/68de4e5ec35fcfe2a3f5583059a90521/raw/status.json)&nbsp;![GitHub issues](https://img.shields.io/github/issues/Getlar/VigIL-Game-Validation)&nbsp;![GitHub](https://img.shields.io/github/license/Getlar/VigIL-Game-Validation)&nbsp;![GitHub release (with filter)](https://img.shields.io/github/v/release/Getlar/VigIL-Game-Validation)&nbsp;

# MagMA - Monolithic Approach

In our **monolithic solution**, we introduce a central decision-maker agent, serving as the exclusive learning entity with oversight over entities that provide information but lack direct influence on the whole learning process. This framework enables the central decision-maker to supervise all units on the map concurrently, leveraging a global trajectory and reward system for updates. To tackle the challenge of fluctuating numbers of learning agents ([Piccoli 2023](https://arxiv.org/abs/2302.12308); [J. K. Terry et al. 2020](http://www.arxiv.org/abs/2008.08932)), such as factories or units, we implemented a single-trajectory approach for each episode. This method ensures fixed episode lengths, independent of factors like unit deaths or creations, resulting in a singular termination flag at the conclusion of each episode.

In our context, **direct influence** refers to the scenario where the actions of a single entity directly impact the central decision-maker. However, in the case of the monolithic approach, this direct influence does not occur. Instead, the central brain only perceives global changes in the environment without attributing them to specific agents. We refer to this as indirect influence on the learning agent. This distinction significantly simplifies the functioning of the learning agent. For instance, when there is a positive change in the environment, such as ice collection, it is reinforced without the need to identify which agent was responsible for the action.

In our approach, we utilize a single actor and a single critic, forming a single-task learning setup ([Eysenbach et al. 2023](https://arxiv.org/abs/2307.12968); [Gai & Wang 2024](https://arxiv.org/abs/2404.12639); [Mysore et al. 2022](https://openreview.net/pdf?id=rJvY_5OzoI)) that optimizes a unified global objective. The single actor generates actions for every pixel on the map, creating a global action tensor that encompasses the entire map. From the raw observations, we filter out pixels containing units and factories to form a tensor of shape *M × |A|*, where **M** is the number of factories and units, and |A| represents the size of the action space. Similarly, the single critic approximates the singular value of the entire game state. This is what we refer to as a **monolithic framework for PPO**: it includes global observations, a single decision maker, a single global trajectory for every episode, a global reward system, a single actor generating a global action tensor, and a single critic evaluating the global state of the game.


# Getting Started

I highly recommend utilizing Docker to set up and manage the environment on your system. Alternatively, a **Conda** installation is also a viable option, especially considering the excellent performance and reliability of the official [pip](https://pypi.org/project/luxai-s2/) package.

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

### DevContainers

Efficiently develop and test code within a Visual Studio Container by cloning the project and using `Ctrl+Shift+P` in VS Code's command palette.

```bash
> Dev Containers: Reopen in Container
```

On a Mac, using a Dev Container can lead to problems due to image incompatibility with `ARM processors`. For Macs, it's better to utilize [dockerRun](https://github.com/Getlar/VigIL-Game-Validation/blob/main/docker_run.sh). If you're on an `x86-64 processor`, opt for the `VS Code` dev container.

## Conda

You will need `Python 3.9` installed on your system. Once installed. To create a conda environment and use it run:

```bash
conda env create -f envs/conda/environment.yml
conda activate luxai_s2

pip install --no-cache-dir ipykernel
python -m ipykernel install --user --name=luxai_s2 --display-name "LuxAI S2"
```

To install **the rest of the packages** required to train and run specific agents run the following commands:

### Pip Packages

```bash
pip install --no-cache-dir --upgrade pip
pip install --no-cache-dir -r envs/pip/requirements_GH.txt

```

### Install JAX support: (Optional)

```bash
pip install --no-cache-dir juxai-s2
```

# Core Contributors

I would like to extend my heartfelt gratitude to [Gulyás László](https://github.com/lesIII) for their invaluable guidance and insightful mentorship throughout the course of this project.

I am also thankful to **Eötvös Loránd University** and **The Department of Artificial Intelligence** for providing the necessary resources and environment that facilitated the development of this project.
