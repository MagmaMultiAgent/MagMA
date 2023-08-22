
<p align="center">
    <img width="100%"  src="docs/images/logo.png" />
</p>

[![Docker Publish](https://github.com/Getlar/VigIL-Game-Validation/actions/workflows/docker-publish.yml/badge.svg?branch=main)](https://github.com/Getlar/VigIL-Game-Validation/actions/workflows/docker-publish.yml)&nbsp;[![Pylint](https://github.com/Getlar/VigIL-Game-Validation/actions/workflows/pylint.yml/badge.svg)](https://github.com/Getlar/VigIL-Game-Validation/actions/workflows/pylint.yml)&nbsp;[![CodeQL](https://github.com/Getlar/VigIL-Game-Validation/actions/workflows/github-code-scanning/codeql/badge.svg?branch=main)](https://github.com/Getlar/VigIL-Game-Validation/actions/workflows/github-code-scanning/codeql)&nbsp;![Version](https://img.shields.io/badge/python-v3.8-blue)&nbsp;![GitHub issues](https://img.shields.io/github/issues/Getlar/VigIL-Game-Validation)&nbsp;![GitHub](https://img.shields.io/github/license/Getlar/VigIL-Game-Validation)![GitHub release (with filter)](https://img.shields.io/github/v/release/Getlar/VigIL-Game-Validation)&nbsp;![badge](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/Getlar/5a256487d767cc5d606d29bd521a18ae/raw/algo.json)

# VigIL - Lux AI Competition

This repository contains a thesis research exploring **AI-based Game Validation**, along with a partial submission and template for the second edition of the annual **NeurIPS** conference in 2023.

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

### DevContainers

Efficiently develop and test code within a Visual Studio Container by cloning the project and using `Ctrl+Shift+P` in VS Code's command palette.

```bash
> Dev Containers: Reopen in Container
```

On a Mac, using a Dev Container can lead to problems due to image incompatibility with `ARM processors`. For Macs, it's better to utilize [dockerRun](https://github.com/Getlar/VigIL-Game-Validation/blob/main/docker_run.sh). If you're on an `x86-64 processor`, opt for the `VS Code` dev container.

## Binary

You will need `Python 3.8` installed on your system. Once installed, you can install the Lux AI season 2 environment and optionally the GPU version with:

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

#### Devtools:
```bash
apt update -y && apt upgrade -y && apt install -y build-essential && apt-get install -y manpages-dev # Required if dev tools are missing.
```
#### Base packages:
```bash
pip install setuptools==57.1.0 psutil==5.7.0 \ 
    pettingzoo==1.12.0 vec_noise==1.1.4 ipykernel moviepy \
    pygame termcolor wheel==0.38.4 notebook tensorboard
```
#### Lux packages:
```bash
pip install stable_baselines3==1.7.0 gym==0.21 --upgrade luxai_s2
```

#### Install JAX support: (Optional)
```bash
pip install --no-cache-dir juxai-s2
```

# Core Contributors

I would like to extend my heartfelt gratitude to [Gulyás László](https://github.com/lesIII) for their invaluable guidance and insightful mentorship throughout the course of this project.

I am also thankful to **Eötvös Lóránd University** for providing the necessary resources and environment that facilitated the development of this project.

@software{Lux_AI_Challenge_S1,
  author = {Tao, Stone and Doerschuk-Tiberi, Bovard},
  doi = {https://doi.org/10.5281/zenodo.7988163},
  month = {10},
  title = {{Lux AI Challenge Season 2}},
  url = {https://github.com/Lux-AI-Challenge/Lux-Design-S2},
  version = {1.0.0},
  year = {2023}
}