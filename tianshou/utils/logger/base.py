from abc import ABC, abstractmethod
from collections.abc import Callable
from numbers import Number

import numpy as np

from tianshou.data import CollectStats, InfoStats, UpdateStats

LOG_DATA_TYPE = dict[str, int | Number | np.number | np.ndarray]


class BaseLogger(ABC):
    """The base class for any logger which is compatible with trainer.

    Try to overwrite write() method to use your own writer.

    :param train_interval: the log interval in log_train_data(). Default to 1000.
    :param test_interval: the log interval in log_test_data(). Default to 1.
    :param update_interval: the log interval in log_update_data(). Default to 1000.
    """

    def __init__(
        self,
        train_interval: int = 1000,
        test_interval: int = 1,
        update_interval: int = 1000,
        info_interval: int = 1,
    ) -> None:
        super().__init__()
        self.train_interval = train_interval
        self.test_interval = test_interval
        self.update_interval = update_interval
        self.info_interval = info_interval
        self.last_log_train_step = -1
        self.last_log_test_step = -1
        self.last_log_update_step = -1
        self.last_log_info_step = -1

    @abstractmethod
    def write(self, step_type: str, step: int, data: LOG_DATA_TYPE) -> None:
        """Specify how the writer is used to log data.

        :param str step_type: namespace which the data dict belongs to.
        :param step: stands for the ordinate of the data dict.
        :param data: the data to write with format ``{key: value}``.
        """

    def log_train_data(self, collect_result: CollectStats, step: int) -> None:
        """Use writer to log statistics generated during training.

        :param collect_result: a CollectStats object containing information of data collected in
            training stage, i.e., returns of collector.collect().
        :param step: stands for the timestep the collect_result being logged.
        """
        if (
            collect_result.n_collected_episodes > 0
            and step - self.last_log_train_step >= self.train_interval
        ):
            log_data = collect_result.to_dict()
            log_data = {f"train/{k}": v for k, v in log_data.items()}
            self.write("train/env_step", step, log_data)
            self.last_log_train_step = step

    def log_test_data(self, collect_result: CollectStats, step: int) -> None:
        """Use writer to log statistics generated during evaluating.

        :param collect_result: a CollectStats object containing information of data collected in
            evaluating stage, i.e., returns of collector.collect().
        :param step: stands for the timestep the collect_result being logged.
        """
        assert collect_result.n_collected_episodes > 0
        if step - self.last_log_test_step >= self.test_interval:
            log_data = collect_result.to_dict()
            log_data = {f"test/{k}": v for k, v in log_data.items()}
            self.write("test/env_step", step, log_data)
            self.last_log_test_step = step

    def log_update_data(self, update_result: UpdateStats, step: int) -> None:
        """Use writer to log statistics generated during updating.

        :param update_result: a LearnStats object containing information of data collected in
            updating stage, i.e., returns of policy.update().
        :param step: stands for the timestep the collect_result being logged.
        """
        if step - self.last_log_update_step >= self.update_interval:
            log_data = update_result.to_dict()
            log_data = {f"update/{k}": v for k, v in log_data.items()}
            self.write("update/gradient_step", step, log_data)
            self.last_log_update_step = step

    def log_info_data(self, info: InfoStats, step: int) -> None:
        """Use writer to log global statistics.

        :param epoch_result: a LearnStats object containing information of data collected in
            updating stage, i.e., returns of policy.update().
        :param step: stands for the timestep the collect_result being logged.
        """
        if step - self.last_log_info_step >= self.info_interval:
            log_data = info.to_dict()
            log_data = {f"info/{k}": v for k, v in log_data.items()}
            self.write("info/epoch", step, log_data)
            self.last_log_info_step = step

    @abstractmethod
    def save_data(
        self,
        epoch: int,
        env_step: int,
        gradient_step: int,
        save_checkpoint_fn: Callable[[int, int, int], str] | None = None,
    ) -> None:
        """Use writer to log metadata when calling ``save_checkpoint_fn`` in trainer.

        :param epoch: the epoch in trainer.
        :param env_step: the env_step in trainer.
        :param gradient_step: the gradient_step in trainer.
        :param function save_checkpoint_fn: a hook defined by user, see trainer
            documentation for detail.
        """

    @abstractmethod
    def restore_data(self) -> tuple[int, int, int]:
        """Return the metadata from existing log.

        If it finds nothing or an error occurs during the recover process, it will
        return the default parameters.

        :return: epoch, env_step, gradient_step.
        """


class LazyLogger(BaseLogger):
    """A logger that does nothing. Used as the placeholder in trainer."""

    def __init__(self) -> None:
        super().__init__()

    def write(self, step_type: str, step: int, data: LOG_DATA_TYPE) -> None:
        """The LazyLogger writes nothing."""

    def save_data(
        self,
        epoch: int,
        env_step: int,
        gradient_step: int,
        save_checkpoint_fn: Callable[[int, int, int], str] | None = None,
    ) -> None:
        pass

    def restore_data(self) -> tuple[int, int, int]:
        return 0, 0, 0
