<p align="center">
    <img width="100%"  src="docs/images/logo.png"/>
</p>

[![Docker](https://github.com/MagmaMultiAgent/MagMA/actions/workflows/docker-publish.yml/badge.svg?branch=v0.0.3-SUT)](https://github.com/MagmaMultiAgent/MagMA/actions/workflows/docker-publish.yml)&nbsp;[![Pylint](https://github.com/MagmaMultiAgent/MagMA/actions/workflows/pylint.yml/badge.svg?branch=v0.0.3-SUT)](https://github.com/MagmaMultiAgent/MagMA/actions/workflows/pylint.yml)&nbsp;[![CodeQL Advanced](https://github.com/MagmaMultiAgent/MagMA/actions/workflows/codeql.yml/badge.svg?branch=v0.0.3-SUT)](https://github.com/MagmaMultiAgent/MagMA/actions/workflows/codeql.yml)&nbsp;[![PyTest](https://github.com/MagmaMultiAgent/MagMA/actions/workflows/pytest.yml/badge.svg?branch=v0.0.3-SUT)](https://github.com/MagmaMultiAgent/MagMA/actions/workflows/pytest.yml)&nbsp;![badge](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/Getlar/5a256487d767cc5d606d29bd521a18ae/raw/algo.json)&nbsp;![badge](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/Getlar/59b429b8fd9c4a525959c66dfa4ab97e/raw/TDK.json)&nbsp;![badge](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/Getlar/68de4e5ec35fcfe2a3f5583059a90521/raw/status.json)&nbsp;![GitHub issues](https://img.shields.io/github/issues/Getlar/VigIL-Game-Validation)&nbsp;![GitHub](https://img.shields.io/github/license/Getlar/VigIL-Game-Validation)&nbsp;![GitHub release (with filter)](https://img.shields.io/github/v/release/Getlar/VigIL-Game-Validation)&nbsp;

# MagMA - Single Unit Testbench

We started with a **benchmarking study** to assess the performance of various RL algorithms within a standardized environment to identify the optimal algorithm for our purposes. We utilized OpenAI’s Gym (not Gymnasium at the time) package ([Brockman et al. 2016](https://arxiv.org/abs/1606.01540)) for compatibility with the Lux package and PettingZoo ([J. Terry et al. 2021](https://arxiv.org/abs/2009.14471)) for environment parallelization. To access state-of-the- art RL algorithms, we conducted our simple benchmarking study using Stable Baselines 3 ([Raffin et al. 2021](https://jmlr.org/papers/v22/20-1364.html)), a reliable collection of reinforcement learning algorithms implemented in PyTorch. This toolkit provided us with a unified framework for all included algorithms and offered subprocess-level vectorization through Python’s Ray (**Ray/RLlib**) framework ([Moritz et al. 2018](https://arxiv.org/abs/1712.05889)), TensorBoard sup- port ([Abadi et al. 2016](https://arxiv.org/abs/1605.08695)), and easy out-of-the-box usability. Another significant advantage of **Stable Baselines 3** is the fidelity of its implementation to the original specifications in the respective research papers, which meant that only minimal parameter adjustments were necessary for usage. Due to hard- ware constraints, specifically the availability of only eight logical CPU cores on our local machine, we were limited to running a maximum of eight parallel environments through subprocess-level vectorization. A more detailed description of the testbench can be found in our [TDK thesis](https://github.com/MagmaMultiAgent/MagmaCore/tree/TDK/TDK).


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
