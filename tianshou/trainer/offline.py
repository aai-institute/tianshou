import time
from collections import defaultdict
from typing import Any, Callable, Dict, Generator, Optional, Tuple, Union

import numpy as np
import tqdm

from tianshou.data import Collector, ReplayBuffer
from tianshou.policy import BasePolicy
from tianshou.trainer import gather_info, test_episode
from tianshou.utils import BaseLogger, LazyLogger, MovAvg, tqdm_config


def offline_trainer(
    policy: BasePolicy,
    buffer: ReplayBuffer,
    test_collector: Optional[Collector],
    max_epoch: int,
    update_per_epoch: int,
    episode_per_test: int,
    batch_size: int,
    test_fn: Optional[Callable[[int, Optional[int]], None]] = None,
    stop_fn: Optional[Callable[[float], bool]] = None,
    save_fn: Optional[Callable[[BasePolicy], None]] = None,
    save_checkpoint_fn: Optional[Callable[[int, int, int], None]] = None,
    resume_from_log: bool = False,
    reward_metric: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    logger: BaseLogger = LazyLogger(),
    verbose: bool = True,
) -> Dict[str, Union[float, str]]:
    """A wrapper for offline trainer procedure.

    The "step" in offline trainer means a gradient step.

    :param policy: an instance of the :class:`~tianshou.policy.BasePolicy` class.
    :param buffer: an instance of the :class:`~tianshou.data.ReplayBuffer` class.
        This buffer must be populated with experiences for offline RL.
    :param Collector test_collector: the collector used for testing. If it's None, then
        no testing will be performed.
    :param int max_epoch: the maximum number of epochs for training. The training
        process might be finished before reaching ``max_epoch`` if ``stop_fn`` is set.
    :param int update_per_epoch: the number of policy network updates, so-called
        gradient steps, per epoch.
    :param episode_per_test: the number of episodes for one policy evaluation.
    :param int batch_size: the batch size of sample data, which is going to feed in
        the policy network.
    :param function test_fn: a hook called at the beginning of testing in each epoch.
        It can be used to perform custom additional operations, with the signature ``f(
        num_epoch: int, step_idx: int) -> None``.
    :param function save_fn: a hook called when the undiscounted average mean reward in
        evaluation phase gets better, with the signature ``f(policy: BasePolicy) ->
        None``.
    :param function save_checkpoint_fn: a function to save training process, with the
        signature ``f(epoch: int, env_step: int, gradient_step: int) -> None``; you can
        save whatever you want. Because offline-RL doesn't have env_step, the env_step
        is always 0 here.
    :param bool resume_from_log: resume gradient_step and other metadata from existing
        tensorboard log. Default to False.
    :param function stop_fn: a function with signature ``f(mean_rewards: float) ->
        bool``, receives the average undiscounted returns of the testing result,
        returns a boolean which indicates whether reaching the goal.
    :param function reward_metric: a function with signature ``f(rewards: np.ndarray
        with shape (num_episode, agent_num)) -> np.ndarray with shape (num_episode,)``,
        used in multi-agent RL. We need to return a single scalar for each episode's
        result to monitor training in the multi-agent RL setting. This function
        specifies what is the desired metric, e.g., the reward of agent 1 or the
        average reward over all agents.
    :param BaseLogger logger: A logger that logs statistics during updating/testing.
        Default to a logger that doesn't log anything.
    :param bool verbose: whether to print the information. Default to True.

    :return: See :func:`~tianshou.trainer.gather_info`.
    """
    start_epoch, gradient_step = 0, 0
    if resume_from_log:
        start_epoch, _, gradient_step = logger.restore_data()
    stat: Dict[str, MovAvg] = defaultdict(MovAvg)
    start_time = time.time()

    if test_collector is not None:
        test_c: Collector = test_collector
        test_collector.reset_stat()
        test_result = test_episode(
            policy, test_c, test_fn, start_epoch, episode_per_test, logger,
            gradient_step, reward_metric
        )
        best_epoch = start_epoch
        best_reward, best_reward_std = test_result["rew"], test_result["rew_std"]
    if save_fn:
        save_fn(policy)

    for epoch in range(1 + start_epoch, 1 + max_epoch):
        policy.train()

        with tqdm.trange(update_per_epoch, desc=f"Epoch #{epoch}", **tqdm_config) as t:
            for _ in t:
                gradient_step += 1
                losses = policy.update(batch_size, buffer)
                data = {"gradient_step": str(gradient_step)}
                for k in losses.keys():
                    stat[k].add(losses[k])
                    losses[k] = stat[k].get()
                    data[k] = f"{losses[k]:.3f}"
                logger.log_update_data(losses, gradient_step)
                t.set_postfix(**data)

        logger.save_data(epoch, 0, gradient_step, save_checkpoint_fn)
        # test
        if test_collector is not None:
            test_result = test_episode(
                policy, test_c, test_fn, epoch, episode_per_test, logger,
                gradient_step, reward_metric
            )
            rew, rew_std = test_result["rew"], test_result["rew_std"]
            if best_epoch < 0 or best_reward < rew:
                best_epoch, best_reward, best_reward_std = epoch, rew, rew_std
                if save_fn:
                    save_fn(policy)
            if verbose:
                print(
                    f"Epoch #{epoch}: test_reward: {rew:.6f} ± {rew_std:.6f}, best_rew"
                    f"ard: {best_reward:.6f} ± {best_reward_std:.6f} in #{best_epoch}"
                )
            if stop_fn and stop_fn(best_reward):
                break

    if test_collector is None and save_fn:
        save_fn(policy)

    if test_collector is None:
        return gather_info(start_time, None, None, 0.0, 0.0)
    else:
        return gather_info(
            start_time, None, test_collector, best_reward, best_reward_std
        )


def offline_trainer_generator(
    policy: BasePolicy,
    buffer: ReplayBuffer,
    test_collector: Optional[Collector],
    max_epoch: int,
    update_per_epoch: int,
    episode_per_test: int,
    batch_size: int,
    test_fn: Optional[Callable[[int, Optional[int]], None]] = None,
    stop_fn: Optional[Callable[[float], bool]] = None,
    save_fn: Optional[Callable[[BasePolicy], None]] = None,
    save_checkpoint_fn: Optional[Callable[[int, int, int], None]] = None,
    resume_from_log: bool = False,
    reward_metric: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    logger: BaseLogger = LazyLogger(),
    verbose: bool = True,
) -> Generator[Tuple[int, Dict[str, Any], Dict[str, Any]], None, None]:
    """A wrapper for offline trainer procedure.
    Returns a generator that yields a 3-tuple (epoch, stats, info) of train results
    on every epoch.

    The "step" in offline trainer means a gradient step.

    :param policy: an instance of the :class:`~tianshou.policy.BasePolicy` class.
    :param buffer: an instance of the :class:`~tianshou.data.ReplayBuffer` class.
        This buffer must be populated with experiences for offline RL.
    :param Collector test_collector: the collector used for testing. If it's None, then
        no testing will be performed.
    :param int max_epoch: the maximum number of epochs for training. The training
        process might be finished before reaching ``max_epoch`` if ``stop_fn`` is set.
    :param int update_per_epoch: the number of policy network updates, so-called
        gradient steps, per epoch.
    :param episode_per_test: the number of episodes for one policy evaluation.
    :param int batch_size: the batch size of sample data, which is going to feed in
        the policy network.
    :param function test_fn: a hook called at the beginning of testing in each epoch.
        It can be used to perform custom additional operations, with the signature ``f(
        num_epoch: int, step_idx: int) -> None``.
    :param function save_fn: a hook called when the undiscounted average mean reward in
        evaluation phase gets better, with the signature ``f(policy: BasePolicy) ->
        None``.
    :param function save_checkpoint_fn: a function to save training process, with the
        signature ``f(epoch: int, env_step: int, gradient_step: int) -> None``; you can
        save whatever you want. Because offline-RL doesn't have env_step, the env_step
        is always 0 here.
    :param bool resume_from_log: resume gradient_step and other metadata from existing
        tensorboard log. Default to False.
    :param function stop_fn: a function with signature ``f(mean_rewards: float) ->
        bool``, receives the average undiscounted returns of the testing result,
        returns a boolean which indicates whether reaching the goal.
    :param function reward_metric: a function with signature ``f(rewards: np.ndarray
        with shape (num_episode, agent_num)) -> np.ndarray with shape (num_episode,)``,
        used in multi-agent RL. We need to return a single scalar for each episode's
        result to monitor training in the multi-agent RL setting. This function
        specifies what is the desired metric, e.g., the reward of agent 1 or the
        average reward over all agents.
    :param BaseLogger logger: A logger that logs statistics during updating/testing.
        Default to a logger that doesn't log anything.
    :param bool verbose: whether to print the information. Default to True.

    :return: See :func:`~tianshou.trainer.gather_info`.
    """
    start_epoch, gradient_step = 0, 0
    if resume_from_log:
        start_epoch, _, gradient_step = logger.restore_data()
    stat: Dict[str, MovAvg] = defaultdict(MovAvg)
    start_time = time.time()

    if test_collector is not None:
        test_c: Collector = test_collector
        test_collector.reset_stat()
        test_result = test_episode(
            policy, test_c, test_fn, start_epoch, episode_per_test, logger,
            gradient_step, reward_metric
        )
        best_epoch = start_epoch
        best_reward, best_reward_std = test_result["rew"], test_result["rew_std"]
    if save_fn:
        save_fn(policy)

    for epoch in range(1 + start_epoch, 1 + max_epoch):
        policy.train()

        with tqdm.trange(update_per_epoch, desc=f"Epoch #{epoch}", **tqdm_config) as t:
            for _ in t:
                gradient_step += 1
                losses = policy.update(batch_size, buffer)
                data = {"gradient_step": str(gradient_step)}
                for k in losses.keys():
                    stat[k].add(losses[k])
                    losses[k] = stat[k].get()
                    data[k] = f"{losses[k]:.3f}"
                logger.log_update_data(losses, gradient_step)
                t.set_postfix(**data)

        logger.save_data(epoch, 0, gradient_step, save_checkpoint_fn)
        # epoch_stat for yield clause
        epoch_stat: Dict[str, Any] = {k: v.get() for k, v in stat.items()}
        epoch_stat["gradient_step"] = gradient_step
        # test
        if test_collector is not None:
            test_result = test_episode(
                policy, test_c, test_fn, epoch, episode_per_test, logger,
                gradient_step, reward_metric
            )
            rew, rew_std = test_result["rew"], test_result["rew_std"]
            if best_epoch < 0 or best_reward < rew:
                best_epoch, best_reward, best_reward_std = epoch, rew, rew_std
                if save_fn:
                    save_fn(policy)
            if verbose:
                print(
                    f"Epoch #{epoch}: test_reward: {rew:.6f} ± {rew_std:.6f}, best_rew"
                    f"ard: {best_reward:.6f} ± {best_reward_std:.6f} in #{best_epoch}"
                )
            epoch_stat.update(
                {
                    "test_reward": rew,
                    "test_reward_std": rew_std,
                    "best_reward": best_reward,
                    "best_reward_std": best_reward_std,
                    "best_epoch": best_epoch
                }
            )

        if test_collector is None:
            info = gather_info(start_time, None, None, 0.0, 0.0)
        else:
            info = gather_info(
                start_time, None, test_collector, best_reward, best_reward_std
            )
        yield epoch, epoch_stat, info

        if test_collector is not None and stop_fn and stop_fn(best_reward):
            break

    if test_collector is None and save_fn:
        save_fn(policy)
