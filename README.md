<p align="center">
    <img width="100%"  src="docs/images/logo.png"/>
</p>

[![Docker](https://github.com/MagmaMultiAgent/MagMA/actions/workflows/docker-publish.yml/badge.svg?branch=main)](https://github.com/MagmaMultiAgent/MagMA/actions/workflows/docker-publish.yml)&nbsp;[![Pylint](https://github.com/MagmaMultiAgent/MagMA/actions/workflows/pylint.yml/badge.svg?branch=main)](https://github.com/MagmaMultiAgent/MagMA/actions/workflows/pylint.yml)&nbsp;[![CodeQL Advanced](https://github.com/MagmaMultiAgent/MagMA/actions/workflows/codeql.yml/badge.svg?branch=main)](https://github.com/MagmaMultiAgent/MagMA/actions/workflows/codeql.yml)&nbsp;[![PyTest](https://github.com/MagmaMultiAgent/MagMA/actions/workflows/pytest.yml/badge.svg?branch=main)](https://github.com/MagmaMultiAgent/MagMA/actions/workflows/pytest.yml)&nbsp;![badge](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/Getlar/5a256487d767cc5d606d29bd521a18ae/raw/algo.json)&nbsp;![badge](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/Getlar/59b429b8fd9c4a525959c66dfa4ab97e/raw/TDK.json)&nbsp;![badge](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/Getlar/68de4e5ec35fcfe2a3f5583059a90521/raw/status.json)&nbsp;![GitHub issues](https://img.shields.io/github/issues/MagmaMultiAgent/MagMA)&nbsp;![GitHub](https://img.shields.io/github/license/MagmaMultiAgent/MagMA)&nbsp;![GitHub release (with filter)](https://img.shields.io/github/v/release/MagmaMultiAgent/MagMA)&nbsp;

# MagMA Hybrid Approach (/w TS)

In this section, our primary focus will be on showcasing the effectiveness of our technique called trajectory separation to demonstrate its potential for improving the hybrid architecture. In addition, we will explore potential enhancements and limitations of the method, including examining the components of trajectory separation to determine the specific factors that contribute to the improvement of performance. We will also conduct a comparative analysis between the trajectory-separated hybrid, **pixel-to-pixel** architecture and existing solutions for the Lux AI competition. At the end of the section, an ablation study will be carried out, analyzing the methods in addition to trajectory separation.

In the subsequent configurations, factories, and units will be organized into distinct groups, each with aggregated outputs and trajectories. Groups can exhibit various characteristics, from global groups encompassing all entities of a specific type to individual groups in which each entity functions as its own separate group. In this study, we illustrate the significance of distinct trajectories within a multiagent setting and investigate various approaches for distributing rewards among the agents. All of the experiments shared a common training objective, which was to train the units to keep the factories alive until the end of the game, spanning 1,000 steps. In the following experiments, we will employ the metrics specified in Section 2.4.10. These metrics include the length of an episode, the aggregate amount of ice transferred to factories within a given episode, and the count of operational factories at each environment step, normalized by the maximum episode duration.

The experiments consist of three separate runs, each utilizing different seeds and running for a total of 102,400 steps (25 training cycles with a 4,096-step batch size). We felt the difference between variants could be sufficiently highlighted in this step range. Contrary to the **single-unit test bench** and the **monolithic approach** test in this scenario, **both players are active** and performing actions simultaneously. The **same model**, as detailed in Section 2.4.6, was used for both players and was trained on their collective trajectories. Following each training cycle, we ran an evaluation of the model by executing 12 distinct environments until the conclusion of the matches. These environments were seeded differently for each of the three runs. The evaluation process yielded 36 data points per training cycle for environment-specific metrics, specifically the metric of **Episode Length**. Additionally, by considering each player separately, there were 72 data points for player-specific metrics, including **Ice Transferred** and **Average Factory Alive**. The lines on the charts visually represent the metrics’ means, while the shaded area shows the standard deviation. In addition to the charts, tables are included to make comparisons easier. The columns **Final Ice Transferred** and Final **Episode Length** refer to the metrics collected at the last evaluation of the model. The mean value is presented, along with the standard deviation in brackets. **The X% of Episodes Finished** by metrics measure how long it took for the agents to learn how to keep the factories alive until the end of the game in terms of environment steps. The percentage indicates the ratio of evaluation environments that resulted in a finished episode, which is an episode where at least one factory survived from both players until 1,000 steps.



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
