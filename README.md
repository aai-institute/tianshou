<div align="center">
  <a href="http://tianshou.readthedocs.io"><img width="300px" height="auto" src="https://github.com/thu-ml/tianshou/raw/master/docs/_static/images/tianshou-logo.png"></a>
</div>

---

[![PyPI](https://img.shields.io/pypi/v/tianshou)](https://pypi.org/project/tianshou/) [![Conda](https://img.shields.io/conda/vn/conda-forge/tianshou)](https://github.com/conda-forge/tianshou-feedstock) [![Read the Docs](https://img.shields.io/readthedocs/tianshou)](https://tianshou.readthedocs.io/en/master) [![Read the Docs](https://img.shields.io/readthedocs/tianshou-docs-zh-cn?label=%E4%B8%AD%E6%96%87%E6%96%87%E6%A1%A3)](https://tianshou.readthedocs.io/zh/master/) [![Unittest](https://github.com/thu-ml/tianshou/actions/workflows/pytest.yml/badge.svg)](https://github.com/thu-ml/tianshou/actions) [![codecov](https://img.shields.io/codecov/c/gh/thu-ml/tianshou)](https://codecov.io/gh/thu-ml/tianshou) [![GitHub issues](https://img.shields.io/github/issues/thu-ml/tianshou)](https://github.com/thu-ml/tianshou/issues) [![GitHub stars](https://img.shields.io/github/stars/thu-ml/tianshou)](https://github.com/thu-ml/tianshou/stargazers) [![GitHub forks](https://img.shields.io/github/forks/thu-ml/tianshou)](https://github.com/thu-ml/tianshou/network) [![GitHub license](https://img.shields.io/github/license/thu-ml/tianshou)](https://github.com/thu-ml/tianshou/blob/master/LICENSE)


**Tianshou** ([天授](https://baike.baidu.com/item/%E5%A4%A9%E6%8E%88)) is a reinforcement learning platform based on pure PyTorch and [Gymnasium](http://github.com/Farama-Foundation/Gymnasium). Unlike other reinforcement learning libraries, which may have complex codebases,
unfriendly high-level APIs, or are not optimized for speed, Tianshou provides a high-performance, modularized framework 
and user-friendly interfaces for building deep reinforcement learning agents. One more aspect that sets Tianshou apart is its
generality: it supports online and offline RL, multi-agent RL, and model-based algorithms.

Tianshou aims at enabling concise implementations, both for researchers and practitioners, without sacrificing flexibility. 

Supported algorithms include:

- [Deep Q-Network (DQN)](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)
- [Double DQN](https://arxiv.org/pdf/1509.06461.pdf)
- [Dueling DQN](https://arxiv.org/pdf/1511.06581.pdf)
- [Branching DQN](https://arxiv.org/pdf/1711.08946.pdf)
- [Categorical DQN (C51)](https://arxiv.org/pdf/1707.06887.pdf)
- [Rainbow DQN (Rainbow)](https://arxiv.org/pdf/1710.02298.pdf)
- [Quantile Regression DQN (QRDQN)](https://arxiv.org/pdf/1710.10044.pdf)
- [Implicit Quantile Network (IQN)](https://arxiv.org/pdf/1806.06923.pdf)
- [Fully-parameterized Quantile Function (FQF)](https://arxiv.org/pdf/1911.02140.pdf)
- [Policy Gradient (PG)](https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf)
- [Natural Policy Gradient (NPG)](https://proceedings.neurips.cc/paper/2001/file/4b86abe48d358ecf194c56c69108433e-Paper.pdf)
- [Advantage Actor-Critic (A2C)](https://openai.com/blog/baselines-acktr-a2c/)
- [Trust Region Policy Optimization (TRPO)](https://arxiv.org/pdf/1502.05477.pdf)
- [Proximal Policy Optimization (PPO)](https://arxiv.org/pdf/1707.06347.pdf)
- [Deep Deterministic Policy Gradient (DDPG)](https://arxiv.org/pdf/1509.02971.pdf)
- [Twin Delayed DDPG (TD3)](https://arxiv.org/pdf/1802.09477.pdf)
- [Soft Actor-Critic (SAC)](https://arxiv.org/pdf/1812.05905.pdf)
- [Randomized Ensembled Double Q-Learning (REDQ)](https://arxiv.org/pdf/2101.05982.pdf)
- [Discrete Soft Actor-Critic (SAC-Discrete)](https://arxiv.org/pdf/1910.07207.pdf)
- [Vanilla Imitation Learning](https://en.wikipedia.org/wiki/Apprenticeship_learning)
- [Batch-Constrained deep Q-Learning (BCQ)](https://arxiv.org/pdf/1812.02900.pdf)
- [Conservative Q-Learning (CQL)](https://arxiv.org/pdf/2006.04779.pdf)
- [Twin Delayed DDPG with Behavior Cloning (TD3+BC)](https://arxiv.org/pdf/2106.06860.pdf)
- [Discrete Batch-Constrained deep Q-Learning (BCQ-Discrete)](https://arxiv.org/pdf/1910.01708.pdf)
- [Discrete Conservative Q-Learning (CQL-Discrete)](https://arxiv.org/pdf/2006.04779.pdf)
- [Discrete Critic Regularized Regression (CRR-Discrete)](https://arxiv.org/pdf/2006.15134.pdf)
- [Generative Adversarial Imitation Learning (GAIL)](https://arxiv.org/pdf/1606.03476.pdf)
- [Prioritized Experience Replay (PER)](https://arxiv.org/pdf/1511.05952.pdf)
- [Generalized Advantage Estimator (GAE)](https://arxiv.org/pdf/1506.02438.pdf)
- [Posterior Sampling Reinforcement Learning (PSRL)](https://www.ece.uvic.ca/~bctill/papers/learning/Strens_2000.pdf)
- [Intrinsic Curiosity Module (ICM)](https://arxiv.org/pdf/1705.05363.pdf)
- [Hindsight Experience Replay (HER)](https://arxiv.org/pdf/1707.01495.pdf)

Other noteworthy features:

- Elegant framework with dual APIs:
    * Tianshou's high-level API maximizes ease of use for application development while still retaining a high degree
      of flexibility.
    * The fundamental procedural API provides a maximum of flexibility for algorithm development without being
      overly verbose.
- State-of-the-art results in [MuJoCo benchmarks](https://github.com/thu-ml/tianshou/tree/master/examples/mujoco) for REINFORCE/A2C/TRPO/PPO/DDPG/TD3/SAC algorithms
- Support for vectorized environments (synchronous or asynchronous) for all algorithms (see [usage](https://tianshou.readthedocs.io/en/master/01_tutorials/07_cheatsheet.html#parallel-sampling))
- Support for super-fast vectorized environments based on [EnvPool](https://github.com/sail-sg/envpool/) for all algorithms (see [usage](https://tianshou.readthedocs.io/en/master/01_tutorials/07_cheatsheet.html#envpool-integration))
- Support for recurrent state representations in actor networks and critic networks (RNN-style training for POMDPs) (see [usage](https://tianshou.readthedocs.io/en/master/01_tutorials/07_cheatsheet.html#rnn-style-training))
- Support any type of environment state/action (e.g. a dict, a self-defined class, ...) [Usage](https://tianshou.readthedocs.io/en/master/01_tutorials/07_cheatsheet.html#user-defined-environment-and-different-state-representation)
- Support for customized training processes (see [usage](https://tianshou.readthedocs.io/en/master/01_tutorials/07_cheatsheet.html#customize-training-process))
- Support n-step returns estimation and prioritized experience replay for all Q-learning based algorithms; GAE, nstep and PER are highly optimized thanks to numba's just-in-time compilation and vectorized numpy operations
- Support for multi-agent RL (see [usage](https://tianshou.readthedocs.io/en/master/01_tutorials/07_cheatsheet.html#multi-agent-reinforcement-learning))
- Support for logging based on both [TensorBoard](https://www.tensorflow.org/tensorboard) and [W&B](https://wandb.ai/) 
- Support for multi-GPU training (see [usage](https://tianshou.readthedocs.io/en/master/01_tutorials/07_cheatsheet.html#multi-gpu))
- Comprehensive documentation, PEP8 code-style checking, type checking and thorough [tests](https://github.com/thu-ml/tianshou/actions)

In Chinese, Tianshou means divinely ordained, being derived to the gift of being born. 
Tianshou is a reinforcement learning platform, and the nature of RL is not learn from humans. 
So taking "Tianshou" means that there is no teacher to learn from, but rather to learn by oneself through constant interaction with the environment.

“天授”意指上天所授，引申为与生具有的天赋。天授是强化学习平台，而强化学习算法并不是向人类学习的，所以取“天授”意思是没有老师来教，而是自己通过跟环境不断交互来进行学习。

## Installation

Tianshou is currently hosted on [PyPI](https://pypi.org/project/tianshou/) and [conda-forge](https://github.com/conda-forge/tianshou-feedstock). It requires Python >= 3.11.

For installing the most recent version of Tianshou, the best way is clone the repository and install it with [poetry](https://python-poetry.org/)
(which you need to install on your system first)

```bash
git clone git@github.com:thu-ml/tianshou.git
cd tianshou
poetry install
```
You can also install the dev requirements by adding `--with dev` or the extras 
for say mujoco and acceleration by [envpool](https://github.com/sail-sg/envpool)
by adding `--extras "mujoco envpool"`

If you wish to install multiple extras, ensure that you include them in a single command. Sequential calls to `poetry install --extras xxx` will overwrite prior installations, leaving only the last specified extras installed.
Or you may install all the following extras by adding `--all-extras`. 
 
Available extras are:
- `atari` (for Atari environments)
- `box2d` (for Box2D environments)
- `classic_control` (for classic control (discrete) environments)
- `mujoco` (for MuJoCo environments)
- `mujoco-py` (for legacy mujoco-py environments[^1])
- `pybullet` (for pybullet environments)
- `robotics` (for gymnasium-robotics environments)
- `vizdoom` (for ViZDoom environments)
- `envpool` (for [envpool](https://github.com/sail-sg/envpool/)  integration)
- `argparse` (in order to be able to run the high level API examples)

[^1]: `mujoco-py` is a legacy package and is not recommended for new projects. 
It is only included for compatibility with older projects. 
Also note that there may be compatibility issues with macOS newer than 
Monterey.

Otherwise, you can install the latest release from PyPI (currently
far behind the master) with the following command:

```bash
$ pip install tianshou
```

If you are using Anaconda or Miniconda, you can install Tianshou from conda-forge:

```bash
$ conda install tianshou -c conda-forge
```

Alternatively to the poetry install, you can also install the latest source version through GitHub:

```bash
$ pip install git+https://github.com/thu-ml/tianshou.git@master --upgrade
```

Finally, you may check the installation via your Python console as follows:

```python
import tianshou
print(tianshou.__version__)
```

If no errors are reported, you have successfully installed Tianshou.

## Documentation

Tutorials and API documentation are hosted on [tianshou.readthedocs.io](https://tianshou.readthedocs.io/).

Find example scripts in the [test/](https://github.com/thu-ml/tianshou/blob/master/test) and [examples/](https://github.com/thu-ml/tianshou/blob/master/examples) folders.


## Why Tianshou?

### Comprehensive Functionality

| RL Platform                                                        | GitHub Stars                                                                                                                                    | # of Alg. <sup>(1)</sup> | Custom Env                     | Batch Training                    | RNN Support        | Nested Observation | Backend    |
| ------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------ | ------------------------------ | --------------------------------- | ------------------ | ------------------ | ---------- |
| [Baselines](https://github.com/openai/baselines)                   | [![GitHub stars](https://img.shields.io/github/stars/openai/baselines)](https://github.com/openai/baselines/stargazers)                         | 9                        | :heavy_check_mark: (gym)       | :heavy_minus_sign: <sup>(2)</sup> | :heavy_check_mark: | :x:                | TF1        |
| [Stable-Baselines](https://github.com/hill-a/stable-baselines)     | [![GitHub stars](https://img.shields.io/github/stars/hill-a/stable-baselines)](https://github.com/hill-a/stable-baselines/stargazers)           | 11                       | :heavy_check_mark: (gym)       | :heavy_minus_sign: <sup>(2)</sup> | :heavy_check_mark: | :x:                | TF1        |
| [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)   | [![GitHub stars](https://img.shields.io/github/stars/DLR-RM/stable-baselines3)](https://github.com/DLR-RM/stable-baselines3/stargazers)         | 7<sup> (3)</sup>         | :heavy_check_mark: (gym)       | :heavy_minus_sign: <sup>(2)</sup> | :x:                | :heavy_check_mark: | PyTorch    |
| [Ray/RLlib](https://github.com/ray-project/ray/tree/master/rllib/) | [![GitHub stars](https://img.shields.io/github/stars/ray-project/ray)](https://github.com/ray-project/ray/stargazers)                           | 16                       | :heavy_check_mark:             | :heavy_check_mark:                | :heavy_check_mark: | :heavy_check_mark: | TF/PyTorch |
| [SpinningUp](https://github.com/openai/spinningup)                 | [![GitHub stars](https://img.shields.io/github/stars/openai/spinningup)](https://github.com/openai/spinningupstargazers)                        | 6                        | :heavy_check_mark: (gym)       | :heavy_minus_sign: <sup>(2)</sup> | :x:                | :x:                | PyTorch    |
| [Dopamine](https://github.com/google/dopamine)                     | [![GitHub stars](https://img.shields.io/github/stars/google/dopamine)](https://github.com/google/dopamine/stargazers)                           | 7                        | :x:                            | :x:                               | :x:                | :x:                | TF/JAX     |
| [ACME](https://github.com/deepmind/acme)                           | [![GitHub stars](https://img.shields.io/github/stars/deepmind/acme)](https://github.com/deepmind/acme/stargazers)                               | 14                       | :heavy_check_mark: (dm_env)    | :heavy_check_mark:                | :heavy_check_mark: | :heavy_check_mark: | TF/JAX     |
| [keras-rl](https://github.com/keras-rl/keras-rl)                   | [![GitHub stars](https://img.shields.io/github/stars/keras-rl/keras-rl)](https://github.com/keras-rl/keras-rlstargazers)                        | 7                        | :heavy_check_mark: (gym)       | :x:                               | :x:                | :x:                | Keras      |
| [rlpyt](https://github.com/astooke/rlpyt)                          | [![GitHub stars](https://img.shields.io/github/stars/astooke/rlpyt)](https://github.com/astooke/rlpyt/stargazers)                               | 11                       | :x:                            | :heavy_check_mark:                | :heavy_check_mark: | :heavy_check_mark: | PyTorch    |
| [ChainerRL](https://github.com/chainer/chainerrl)                  | [![GitHub stars](https://img.shields.io/github/stars/chainer/chainerrl)](https://github.com/chainer/chainerrl/stargazers)                       | 18                       | :heavy_check_mark: (gym)       | :heavy_check_mark:                | :heavy_check_mark: | :x:                | Chainer    |
| [Sample Factory](https://github.com/alex-petrenko/sample-factory)  | [![GitHub stars](https://img.shields.io/github/stars/alex-petrenko/sample-factory)](https://github.com/alex-petrenko/sample-factory/stargazers) | 1<sup> (4)</sup>         | :heavy_check_mark: (gym)       | :heavy_check_mark:                | :heavy_check_mark: | :heavy_check_mark: | PyTorch    |
|                                                                    |                                                                                                                                                 |                          |                                |                                   |                    |                    |            |
| [Tianshou](https://github.com/thu-ml/tianshou)                     | [![GitHub stars](https://img.shields.io/github/stars/thu-ml/tianshou)](https://github.com/thu-ml/tianshou/stargazers)                           | 20                       | :heavy_check_mark: (Gymnasium) | :heavy_check_mark:                | :heavy_check_mark: | :heavy_check_mark: | PyTorch    |

<sup>(1): access date: 2021-08-08</sup>

<sup>(2): not all algorithms support this feature</sup>

<sup>(3): TQC and QR-DQN in [sb3-contrib](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib) instead of main repo</sup>

<sup>(4): super fast APPO!</sup>

### High Software Engineering Standards

| RL Platform                                                        | Documentation                                                                                                                                                        | Code Coverage                                                                                                                                                                                                                                                                                                                                 | Type Hints         | Last Update                                                                                                       |
| ------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------ | ----------------------------------------------------------------------------------------------------------------- |
| [Baselines](https://github.com/openai/baselines)                   | :x:                                                                                                                                                                  | :x:                                                                                                                                                                                                                                                                                                                                           | :x:                | ![GitHub last commit](https://img.shields.io/github/last-commit/openai/baselines?label=last%20update)             |
| [Stable-Baselines](https://github.com/hill-a/stable-baselines)     | [![Documentation Status](https://readthedocs.org/projects/stable-baselines/badge/?version=master)](https://stable-baselines.readthedocs.io/en/master/?badge=master)  | [![coverage](https://img.shields.io/badge/coverage-86%25-brightgreen.svg?style=flat)](https://www.codacy.com/app/baselines_janitors/stable-baselines?utm_source=github.com&utm_medium=referral&utm_content=hill-a/stable-baselines&utm_campaign=Badge_Coverage) <!-- https://github.com/thu-ml/tianshou/issues/249#issuecomment-895882193 --> | :x:                | ![GitHub last commit](https://img.shields.io/github/last-commit/hill-a/stable-baselines?label=last%20update)      |
| [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)   | [![Documentation Status](https://readthedocs.org/projects/stable-baselines/badge/?version=master)](https://stable-baselines3.readthedocs.io/en/master/?badge=master) | [![coverage report](https://gitlab.com/araffin/stable-baselines3/badges/master/coverage.svg)](https://gitlab.com/araffin/stable-baselines3/-/commits/master)                                                                                                                                                                                  | :heavy_check_mark: | ![GitHub last commit](https://img.shields.io/github/last-commit/DLR-RM/stable-baselines3?label=last%20update)     |
| [Ray/RLlib](https://github.com/ray-project/ray/tree/master/rllib/) | [![](https://readthedocs.org/projects/ray/badge/?version=master)](http://docs.ray.io/en/master/rllib.html)                                                           | :heavy_minus_sign:<sup>(1)</sup>                                                                                                                                                                                                                                                                                                              | :heavy_check_mark: | ![GitHub last commit](https://img.shields.io/github/last-commit/ray-project/ray?label=last%20update)              |
| [SpinningUp](https://github.com/openai/spinningup)                 | [![](https://img.shields.io/readthedocs/spinningup)](https://spinningup.openai.com/)                                                                                 | :x:                                                                                                                                                                                                                                                                                                                                           | :x:                | ![GitHub last commit](https://img.shields.io/github/last-commit/openai/spinningup?label=last%20update)            |
| [Dopamine](https://github.com/google/dopamine)                     | [![](https://img.shields.io/badge/docs-passing-green)](https://github.com/google/dopamine/tree/master/docs)                                                          | :x:                                                                                                                                                                                                                                                                                                                                           | :x:                | ![GitHub last commit](https://img.shields.io/github/last-commit/google/dopamine?label=last%20update)              |
| [ACME](https://github.com/deepmind/acme)                           | [![](https://img.shields.io/badge/docs-passing-green)](https://github.com/deepmind/acme/blob/master/docs/index.md)                                                   | :heavy_minus_sign:<sup>(1)</sup>                                                                                                                                                                                                                                                                                                              | :heavy_check_mark: | ![GitHub last commit](https://img.shields.io/github/last-commit/deepmind/acme?label=last%20update)                |
| [keras-rl](https://github.com/keras-rl/keras-rl)                   | [![Documentation](https://readthedocs.org/projects/keras-rl/badge/)](http://keras-rl.readthedocs.io/)                                                                | :heavy_minus_sign:<sup>(1)</sup>                                                                                                                                                                                                                                                                                                              | :x:                | ![GitHub last commit](https://img.shields.io/github/last-commit/keras-rl/keras-rl?label=last%20update)            |
| [rlpyt](https://github.com/astooke/rlpyt)                          | [![Docs](https://readthedocs.org/projects/rlpyt/badge/?version=latest&style=flat)](https://rlpyt.readthedocs.io/en/latest/)                                          | [![codecov](https://codecov.io/gh/astooke/rlpyt/graph/badge.svg)](https://codecov.io/gh/astooke/rlpyt)                                                                                                                                                                                                                                        | :x:                | ![GitHub last commit](https://img.shields.io/github/last-commit/astooke/rlpyt?label=last%20update)                |
| [ChainerRL](https://github.com/chainer/chainerrl)                  | [![Documentation Status](https://readthedocs.org/projects/chainerrl/badge/?version=latest)](http://chainerrl.readthedocs.io/en/latest/?badge=latest)                 | [![Coverage Status](https://coveralls.io/repos/github/chainer/chainerrl/badge.svg?branch=master)](https://coveralls.io/github/chainer/chainerrl?branch=master)                                                                                                                                                                                | :x:                | ![GitHub last commit](https://img.shields.io/github/last-commit/chainer/chainerrl?label=last%20update)            |
| [Sample Factory](https://github.com/alex-petrenko/sample-factory)  | [:heavy_minus_sign:](https://arxiv.org/abs/2006.11751)                                                                                                               | [![codecov](https://codecov.io/gh/alex-petrenko/sample-factory/branch/master/graph/badge.svg)](https://codecov.io/gh/alex-petrenko/sample-factory)                                                                                                                                                                                            | :x:                | ![GitHub last commit](https://img.shields.io/github/last-commit/alex-petrenko/sample-factory?label=last%20update) |
|                                                                    |                                                                                                                                                                      |                                                                                                                                                                                                                                                                                                                                               |                    |                                                                                                                   |
| [Tianshou](https://github.com/thu-ml/tianshou)                     | [![Read the Docs](https://img.shields.io/readthedocs/tianshou)](https://tianshou.readthedocs.io/en/master)                                                           | [![codecov](https://img.shields.io/codecov/c/gh/thu-ml/tianshou)](https://codecov.io/gh/thu-ml/tianshou)                                                                                                                                                                                                                                      | :heavy_check_mark: | ![GitHub last commit](https://img.shields.io/github/last-commit/thu-ml/tianshou?label=last%20update)              |

<sup>(1): it has continuous integration but the coverage rate is not available</sup>

### Reproducible, High-Quality Results

Tianshou is rigorously tested. In contrast to other RL platforms, **our tests include the full agent training procedure for all of the implemented algorithms**. Our tests would fail once if any of the agents failed to achieve a consistent level of performance on limited epochs. 
Our tests thus ensure reproducibility. 
Check out the [GitHub Actions](https://github.com/thu-ml/tianshou/actions) page for more detail.

Atari and MuJoCo benchmark results can be found in the [examples/atari/](examples/atari/) and [examples/mujoco/](examples/mujoco/) folders respectively. **Our MuJoCo results reach or exceed the level of performance of most existing benchmarks.**

### Policy Interface

All algorithms implement the following, highly general API:

- `__init__`: initialize the policy;
- `forward`: compute actions based on given observations;
- `process_buffer`: process initial buffer, which is useful for some offline learning algorithms
- `process_fn`: preprocess data from the replay buffer (since we have reformulated *all* algorithms to replay buffer-based algorithms);
- `learn`: learn from a given batch of data;
- `post_process_fn`: update the replay buffer from the learning process (e.g., prioritized replay buffer needs to update the weight);
- `update`: the main interface for training, i.e., `process_fn -> learn -> post_process_fn`.

The implementation of this API suffices for a new algorithm to be applicable within Tianshou,
making experimenation with new approaches particularly straightforward.

## Quick Start

Tianshou provides two API levels:
  * the high-level interface, which provides ease of use for end users seeking to run deep reinforcement learning applications
  * the procedural interface, which provides a maximum of control, especially for very advanced users and developers of reinforcement learning algorithms.

In the following, let us consider an example application using the *CartPole* gymnasium environment.
We shall apply the deep Q network (DQN) learning algorithm using both APIs.

### High-Level API

To get started, we need some imports.

```python
from tianshou.highlevel.config import SamplingConfig
from tianshou.highlevel.env import (
    EnvFactoryRegistered,
    VectorEnvType,
)
from tianshou.highlevel.experiment import DQNExperimentBuilder, ExperimentConfig
from tianshou.highlevel.params.policy_params import DQNParams
from tianshou.highlevel.trainer import (
    EpochTestCallbackDQNSetEps,
    EpochTrainCallbackDQNSetEps,
    EpochStopCallbackRewardThreshold
)
```

In the high-level API, the basis for an RL experiment is an `ExperimentBuilder`
with which we can build the experiment we then seek to run.
Since we want to use DQN, we use the specialization `DQNExperimentBuilder`.
The other imports serve to provide configuration options for our experiment.

The high-level API provides largely declarative semantics, i.e. the code is 
almost exclusively concerned with configuration that controls what to do
(rather than how to do it).

```python
experiment = (
    DQNExperimentBuilder(
        EnvFactoryRegistered(task="CartPole-v1", seed=0, venv_type=VectorEnvType.DUMMY),
        ExperimentConfig(
            persistence_enabled=False,
            watch=True,
            watch_render=1 / 35,
            watch_num_episodes=100,
        ),
        SamplingConfig(
            num_epochs=10,
            step_per_epoch=10000,
            batch_size=64,
            num_train_envs=10,
            num_test_envs=100,
            buffer_size=20000,
            step_per_collect=10,
            update_per_step=1 / 10,
        ),
    )
    .with_dqn_params(
        DQNParams(
            lr=1e-3,
            discount_factor=0.9,
            estimation_step=3,
            target_update_freq=320,
        ),
    )
    .with_model_factory_default(hidden_sizes=(64, 64))
    .with_epoch_train_callback(EpochTrainCallbackDQNSetEps(0.3))
    .with_epoch_test_callback(EpochTestCallbackDQNSetEps(0.0))
    .with_epoch_stop_callback(EpochStopCallbackRewardThreshold(195))
    .build()
)
experiment.run()
```

The experiment builder takes three arguments:
  * the environment factory for the creation of environments. In this case,
    we use an existing factory implementation for gymnasium environments.
  * the experiment configuration, which controls persistence and the overall
    experiment flow. In this case, we have configured that we want to observe
    the agent's behavior after it is trained (`watch=True`) for a number of
    episodes (`watch_num_episodes=100`). We have disabled persistence, because
    we do not want to save training logs, the agent or its configuration for 
    future use.
  * the sampling configuration, which controls fundamental training parameters,
    such as the total number of epochs we run the experiment for (`num_epochs=10`)  
    and the number of environment steps each epoch shall consist of
    (`step_per_epoch=10000`).
    Every epoch consists of a series of data collection (rollout) steps and 
    training steps.
    The parameter `step_per_collect` controls the amount of data that is 
    collected in each collection step and after each collection step, we
    perform a training step, applying a gradient-based update based on a sample
    of data (`batch_size=64`) taken from the buffer of data that has been 
    collected. For further details, see the documentation of `SamplingConfig`.

We then proceed to configure some of the parameters of the DQN algorithm itself
and of the neural network model we want to use.
A DQN-specific detail is the use of callbacks to configure the algorithm's
epsilon parameter for exploration. We want to use random exploration during rollouts 
(train callback), but we don't when evaluating the agent's performance in the test
environments (test callback).

Find the script in [examples/discrete/discrete_dqn_hl.py](examples/discrete/discrete_dqn_hl.py).
Here's a run (with the training time cut short):

<p align="center" style="text-algin:center">
  <img src="docs/_static/images/discrete_dqn_hl.gif">
</p>

Find many further applications of the high-level API in the `examples/` folder;
look for scripts ending with `_hl.py`.
Note that most of these examples require the extra package `argparse` 
(install it by adding `--extras argparse` when invoking poetry).

### Procedural API

Let us now consider an analogous example in the procedural API. 
Find the full script in [examples/discrete/discrete_dqn.py](https://github.com/thu-ml/tianshou/blob/master/examples/discrete/discrete_dqn.py).

First, import some relevant packages:

```python
import gymnasium as gym
import torch
from torch.utils.tensorboard import SummaryWriter
import tianshou as ts
```

Define some hyper-parameters:

```python
task = 'CartPole-v1'
lr, epoch, batch_size = 1e-3, 10, 64
train_num, test_num = 10, 100
gamma, n_step, target_freq = 0.9, 3, 320
buffer_size = 20000
eps_train, eps_test = 0.1, 0.05
step_per_epoch, step_per_collect = 10000, 10
```

Initialize the logger:

```python
logger = ts.utils.TensorboardLogger(SummaryWriter('log/dqn'))
# For other loggers, see https://tianshou.readthedocs.io/en/master/01_tutorials/05_logger.html
```

Make environments:

```python
# You can also try SubprocVectorEnv, which will use parallelization
train_envs = ts.env.DummyVectorEnv([lambda: gym.make(task) for _ in range(train_num)])
test_envs = ts.env.DummyVectorEnv([lambda: gym.make(task) for _ in range(test_num)])
```

Create the network as well as its optimizer:

```python
from tianshou.utils.net.common import Net

# Note: You can easily define other networks.
# See https://tianshou.readthedocs.io/en/master/01_tutorials/00_dqn.html#build-the-network
env = gym.make(task, render_mode="human")
state_shape = env.observation_space.shape or env.observation_space.n
action_shape = env.action_space.shape or env.action_space.n
net = Net(state_shape=state_shape, action_shape=action_shape, hidden_sizes=[128, 128, 128])
optim = torch.optim.Adam(net.parameters(), lr=lr)
```

Set up the policy and collectors:

```python
policy = ts.policy.DQNPolicy(
    model=net,
    optim=optim,
    discount_factor=gamma, 
    action_space=env.action_space,
    estimation_step=n_step,
    target_update_freq=target_freq
)
train_collector = ts.data.Collector(policy, train_envs, ts.data.VectorReplayBuffer(buffer_size, train_num), exploration_noise=True)
test_collector = ts.data.Collector(policy, test_envs, exploration_noise=True)  # because DQN uses epsilon-greedy method
```

Let's train it:

```python
result = ts.trainer.OffpolicyTrainer(
    policy=policy,
    train_collector=train_collector,
    test_collector=test_collector,
    max_epoch=epoch,
    step_per_epoch=step_per_epoch,
    step_per_collect=step_per_collect,
    episode_per_test=test_num,
    batch_size=batch_size,
    update_per_step=1 / step_per_collect,
    train_fn=lambda epoch, env_step: policy.set_eps(eps_train),
    test_fn=lambda epoch, env_step: policy.set_eps(eps_test),
    stop_fn=lambda mean_rewards: mean_rewards >= env.spec.reward_threshold,
    logger=logger,
).run()
print(f"Finished training in {result.timing.total_time} seconds")
```

Save/load the trained policy (it's exactly the same as loading a `torch.nn.module`):

```python
torch.save(policy.state_dict(), 'dqn.pth')
policy.load_state_dict(torch.load('dqn.pth'))
```

Watch the agent with 35 FPS:

```python
policy.eval()
policy.set_eps(eps_test)
collector = ts.data.Collector(policy, env, exploration_noise=True)
collector.collect(n_episode=1, render=1 / 35)
```

Inspect the data saved in TensorBoard:

```bash
$ tensorboard --logdir log/dqn
```

Please read the [documentation](https://tianshou.readthedocs.io) for advanced usage.

## Contributing

Tianshou is still under development. 
Further algorithms and features are continuously being added, and we always welcome contributions to help make Tianshou better. 
If you would like to contribute, please check out [this link](https://tianshou.org/en/master/04_contributing/04_contributing.html).

## Citing Tianshou

If you find Tianshou useful, please cite it in your publications.

```latex
@article{tianshou,
  author  = {Jiayi Weng and Huayu Chen and Dong Yan and Kaichao You and Alexis Duburcq and Minghao Zhang and Yi Su and Hang Su and Jun Zhu},
  title   = {Tianshou: A Highly Modularized Deep Reinforcement Learning Library},
  journal = {Journal of Machine Learning Research},
  year    = {2022},
  volume  = {23},
  number  = {267},
  pages   = {1--6},
  url     = {http://jmlr.org/papers/v23/21-1127.html}
}
```

## Acknowledgments

Tianshou is supported by [appliedAI Institute for Europe](https://www.appliedai-institute.de/en/),
who is committed to providing long-term support and development.

Tianshou was previously a reinforcement learning platform based on TensorFlow. You can check out the branch [`priv`](https://github.com/thu-ml/tianshou/tree/priv) for more detail. Many thanks to [Haosheng Zou](https://github.com/HaoshengZou)'s pioneering work for Tianshou before version 0.1.1.

We would like to thank [TSAIL](http://ml.cs.tsinghua.edu.cn/) and [Institute for Artificial Intelligence, Tsinghua University](http://ml.cs.tsinghua.edu.cn/thuai/) for providing such an excellent AI research platform.
