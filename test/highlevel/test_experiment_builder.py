from test.highlevel.env_factory import ContinuousTestEnvFactory, DiscreteTestEnvFactory

import pytest

from tianshou.highlevel.config import SamplingConfig
from tianshou.highlevel.experiment import (
    A2CExperimentBuilder,
    DDPGExperimentBuilder,
    DiscreteSACExperimentBuilder,
    DQNExperimentBuilder,
    ExperimentBuilder,
    ExperimentConfig,
    IQNExperimentBuilder,
    PGExperimentBuilder,
    PPOExperimentBuilder,
    REDQExperimentBuilder,
    SACExperimentBuilder,
    TD3ExperimentBuilder,
    TRPOExperimentBuilder,
)


@pytest.mark.parametrize(
    "builder_cls",
    [
        PPOExperimentBuilder,
        A2CExperimentBuilder,
        SACExperimentBuilder,
        DDPGExperimentBuilder,
        TD3ExperimentBuilder,
        # NPGExperimentBuilder,  # TODO test fails non-deterministically
        REDQExperimentBuilder,
        TRPOExperimentBuilder,
        PGExperimentBuilder,
    ],
)
def test_experiment_builder_continuous_default_params(builder_cls: type[ExperimentBuilder]) -> None:
    env_factory = ContinuousTestEnvFactory()
    sampling_config = SamplingConfig(
        num_epochs=1,
        step_per_epoch=100,
        num_train_envs=2,
        num_test_envs=2,
    )
    experiment_config = ExperimentConfig(persistence_enabled=False)
    builder = builder_cls(
        experiment_config=experiment_config,
        env_factory=env_factory,
        sampling_config=sampling_config,
    )
    experiment = builder.build()
    experiment.run(override_experiment_name="test")
    print(experiment)


@pytest.mark.parametrize(
    "builder_cls",
    [
        PPOExperimentBuilder,
        A2CExperimentBuilder,
        DQNExperimentBuilder,
        DiscreteSACExperimentBuilder,
        IQNExperimentBuilder,
    ],
)
def test_experiment_builder_discrete_default_params(builder_cls: type[ExperimentBuilder]) -> None:
    env_factory = DiscreteTestEnvFactory()
    sampling_config = SamplingConfig(
        num_epochs=1,
        step_per_epoch=100,
        num_train_envs=2,
        num_test_envs=2,
    )
    builder = builder_cls(
        experiment_config=ExperimentConfig(persistence_enabled=False),
        env_factory=env_factory,
        sampling_config=sampling_config,
    )
    experiment = builder.build()
    experiment.run(override_experiment_name="test")
    print(experiment)


def test_temp_builder_modification() -> None:
    env_factory = DiscreteTestEnvFactory()
    sampling_config = SamplingConfig(
        num_epochs=1,
        step_per_epoch=100,
        num_train_envs=2,
        num_test_envs=2,
    )
    builder = PPOExperimentBuilder(
        experiment_config=ExperimentConfig(persistence_enabled=False),
        env_factory=env_factory,
        sampling_config=sampling_config,
    )
    original_seed = builder.experiment_config.seed
    original_train_seed = builder.sampling_config.train_seed

    with builder.temp_config_mutation():
        builder.experiment_config.seed += 12345
        builder.sampling_config.train_seed += 456
        exp = builder.build()

    assert builder.experiment_config.seed == original_seed
    assert builder.sampling_config.train_seed == original_train_seed
    assert exp.config.seed == original_seed + 12345
    assert exp.sampling_config.train_seed == original_train_seed + 456
