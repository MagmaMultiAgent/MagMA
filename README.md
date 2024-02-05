[![Docker Publish](https://github.com/Getlar/VigIL-Game-Validation/actions/workflows/docker-publish.yml/badge.svg?branch=v0.0.1-CNN-mixed)](https://github.com/Getlar/VigIL-Game-Validation/actions/workflows/docker-publish.yml)&nbsp;[![Pylint](https://github.com/Getlar/VigIL-Game-Validation/actions/workflows/pylint.yml/badge.svg?branch=v0.0.1-CNN-mixed)](https://github.com/Getlar/VigIL-Game-Validation/actions/workflows/pylint.yml)&nbsp;[![CodeQL Advanced](https://github.com/Getlar/VigIL-Game-Validation/actions/workflows/github-code-scanning/codeql/badge.svg?branch=v0.0.1-CNN-mixed)](https://github.com/Getlar/VigIL-Game-Validation/actions/workflows/github-code-scanning/codeql)&nbsp;[![PyTest](https://github.com/Getlar/VigIL-Game-Validation/actions/workflows/pytest.yml/badge.svg?branch=v0.0.1-CNN-mixed)](https://github.com/Getlar/VigIL-Game-Validation/actions/workflows/pytest.yml)&nbsp;


# MagMA - 

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

You will need `Python 3.9` installed on your system. Once installed, you can install the Lux AI season 2 environment and optionally the GPU version with:

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
pip install pettingzoo vec_noise stable_baselines3 sb3-contrib
```
#### Lux packages:
```bash
pip install --upgrade luxai_s2
```

#### Install JAX support: (Optional)
```bash
pip install -U "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install git+https://github.com/RoboEden/jux.git@dev
```

To test the existing implementation check out the [running docs](https://github.com/Getlar/VigIL-Game-Validation/blob/main/src/README.MD).

# Core Contributors

I would like to extend my heartfelt gratitude to [Gulyás László](https://github.com/lesIII) for their invaluable guidance and insightful mentorship throughout the course of this project.

I am also thankful to **Eötvös Lóránd University** for providing the necessary resources and environment that facilitated the development of this project.
