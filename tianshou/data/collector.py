import time
import warnings
from copy import copy
from dataclasses import dataclass
from typing import Any, Self, cast

import gymnasium as gym
import numpy as np
import torch

from tianshou.data import (
    Batch,
    CachedReplayBuffer,
    PrioritizedReplayBuffer,
    ReplayBuffer,
    ReplayBufferManager,
    SequenceSummaryStats,
    VectorReplayBuffer,
    to_numpy,
)
from tianshou.data.types import (
    ActBatchProtocol,
    ActStateBatchProtocol,
    ObsBatchProtocol,
    RolloutBatchProtocol,
)
from tianshou.env import BaseVectorEnv, DummyVectorEnv
from tianshou.policy import BasePolicy
from tianshou.utils.print import DataclassPPrintMixin


@dataclass(kw_only=True)
class CollectStatsBase(DataclassPPrintMixin):
    """The most basic stats, often used for offline learning."""

    n_collected_episodes: int = 0
    """The number of collected episodes."""
    n_collected_steps: int = 0
    """The number of collected steps."""


@dataclass(kw_only=True)
class CollectStats(CollectStatsBase):
    """A data structure for storing the statistics of rollouts."""

    collect_time: float = 0.0
    """The time for collecting transitions."""
    collect_speed: float = 0.0
    """The speed of collecting (env_step per second)."""
    returns: np.ndarray
    """The collected episode returns."""
    returns_stat: SequenceSummaryStats | None  # can be None if no episode ends during the collect step
    """Stats of the collected returns."""
    lens: np.ndarray
    """The collected episode lengths."""
    lens_stat: SequenceSummaryStats | None  # can be None if no episode ends during the collect step
    """Stats of the collected episode lengths."""

    @classmethod
    def with_autogenerated_stats(
        cls,
        returns: np.ndarray,
        lens: np.ndarray,
        n_collected_episodes: int = 0,
        n_collected_steps: int = 0,
        collect_time: float = 0.0,
        collect_speed: float = 0.0,
    ) -> Self:
        """Return a new instance with the stats autogenerated from the given lists."""
        returns_stat = SequenceSummaryStats.from_sequence(returns) if returns.size > 0 else None
        lens_stat = SequenceSummaryStats.from_sequence(lens) if lens.size > 0 else None
        return cls(
            n_collected_episodes=n_collected_episodes,
            n_collected_steps=n_collected_steps,
            collect_time=collect_time,
            collect_speed=collect_speed,
            returns=returns,
            returns_stat=returns_stat,
            lens=lens,
            lens_stat=lens_stat,
        )


class Collector:
    """Collector enables the policy to interact with different types of envs with exact number of steps or episodes.

    :param policy: an instance of the :class:`~tianshou.policy.BasePolicy` class.
    :param env: a ``gym.Env`` environment or an instance of the
        :class:`~tianshou.env.BaseVectorEnv` class.
    :param buffer: an instance of the :class:`~tianshou.data.ReplayBuffer` class.
        If set to None, will instantiate a :class:`~tianshou.data.VectorReplayBuffer`
        as the default buffer.
    :param exploration_noise: determine whether the action needs to be modified
        with the corresponding policy's exploration noise. If so, "policy.
        exploration_noise(act, batch)" will be called automatically to add the
        exploration noise into action. Default to False.

    .. note::

        Please make sure the given environment has a time limitation if using n_episode
        collect option.

    .. note::

        In past versions of Tianshou, the replay buffer passed to `__init__`
        was automatically reset. This is not done in the current implementation.
    """

    def __init__(
        self,
        policy: BasePolicy,
        env: gym.Env | BaseVectorEnv,
        buffer: ReplayBuffer | None = None,
        exploration_noise: bool = False,
    ) -> None:
        super().__init__()
        if isinstance(env, gym.Env) and not hasattr(env, "__len__"):
            warnings.warn("Single environment detected, wrap to DummyVectorEnv.")
            # Unfortunately, mypy seems to ignore the isinstance in lambda, maybe a bug in mypy
            self.env = DummyVectorEnv([lambda: env])
        else:
            self.env = env  # type: ignore
        self.env_num = len(self.env)
        self.exploration_noise = exploration_noise
        self.buffer = self._assign_buffer(buffer)
        self.policy = policy
        self._action_space = self.env.action_space

        self._pre_collect_obs_RO: np.ndarray | None = None
        self._pre_collect_info_R: list[dict] | None = None
        self._pre_collect_hidden_state_RH: np.ndarray | torch.Tensor | Batch | None = None

        self._is_closed = False
        self.collect_step, self.collect_episode, self.collect_time = 0, 0, 0.0

    def close(self) -> None:
        """Close the collector and the environment."""
        self.env.close()
        self._pre_collect_obs_RO = None
        self._pre_collect_info_R = None
        self._is_closed = True

    @property
    def is_closed(self) -> bool:
        """Return True if the collector is closed."""
        return self._is_closed

    def _assign_buffer(self, buffer: ReplayBuffer | None) -> ReplayBuffer:
        """Check if the buffer matches the constraint."""
        if buffer is None:
            buffer = VectorReplayBuffer(self.env_num, self.env_num)
        elif isinstance(buffer, ReplayBufferManager):
            assert buffer.buffer_num >= self.env_num
            if isinstance(buffer, CachedReplayBuffer):
                assert buffer.cached_buffer_num >= self.env_num
        else:  # ReplayBuffer or PrioritizedReplayBuffer
            assert buffer.maxsize > 0
            if self.env_num > 1:
                if isinstance(buffer, ReplayBuffer):
                    buffer_type = "ReplayBuffer"
                    vector_type = "VectorReplayBuffer"
                if isinstance(buffer, PrioritizedReplayBuffer):
                    buffer_type = "PrioritizedReplayBuffer"
                    vector_type = "PrioritizedVectorReplayBuffer"
                raise TypeError(
                    f"Cannot use {buffer_type}(size={buffer.maxsize}, ...) to collect "
                    f"{self.env_num} envs,\n\tplease use {vector_type}(total_size="
                    f"{buffer.maxsize}, buffer_num={self.env_num}, ...) instead.",
                )
        return buffer

    def reset(
        self,
        reset_buffer: bool = True,
        reset_stats: bool = True,
        gym_reset_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Reset the environment, statistics, and data needed to start the collection.

        :param reset_buffer: if true, reset the replay buffer attached
            to the collector.
        :param reset_stats: if true, reset the statistics attached to the collector.
        :param gym_reset_kwargs: extra keyword arguments to pass into the environment's
            reset function. Defaults to None (extra keyword arguments)
        """
        self.reset_env(gym_reset_kwargs=gym_reset_kwargs)
        if reset_buffer:
            self.reset_buffer()
        if reset_stats:
            self.reset_stat()
        self._is_closed = False

    def reset_stat(self) -> None:
        """Reset the statistic variables."""
        self.collect_step, self.collect_episode, self.collect_time = 0, 0, 0.0

    def reset_buffer(self, keep_statistics: bool = False) -> None:
        """Reset the data buffer."""
        self.buffer.reset(keep_statistics=keep_statistics)

    def reset_env(
        self,
        gym_reset_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Reset the environments and the initial obs, info, and hidden state of the collector."""
        gym_reset_kwargs = gym_reset_kwargs or {}
        self._pre_collect_obs_RO, self._pre_collect_info_R = self.env.reset(**gym_reset_kwargs)
        self._pre_collect_hidden_state_RH = None

    def _compute_action_policy_hidden(
        self,
        random: bool,
        ready_env_ids_R: np.ndarray,
        use_grad: bool,
        last_obs_RO: np.ndarray | None,
        last_info_R: list[dict],
        last_hidden_state_RH: np.ndarray | torch.Tensor | Batch | None = None,
    ) -> tuple[np.ndarray, np.ndarray, Batch, Batch | None]:
        """Returns the action, the normalized action, a "policy" entry, and the hidden state."""
        if random:
            try:
                act_normalized_RA = [self._action_space[i].sample() for i in ready_env_ids_R]
            # TODO: test whether envpool env explicitly
            except TypeError:  # envpool's action space is not for per-env
                act_normalized_RA = [self._action_space.sample() for _ in ready_env_ids_R]
            act_RA = self.policy.map_action_inverse(np.array(act_normalized_RA))
            policy_R = Batch()
            hidden_state_RH = None

        else:
            obs_batch_R = cast(ObsBatchProtocol, Batch(obs=last_obs_RO, info=last_info_R))

            with torch.set_grad_enabled(use_grad):
                act_batch_RA = self.policy(
                    obs_batch_R,
                    last_hidden_state_RH,
                )

            act_RA = to_numpy(act_batch_RA.act)
            if self.exploration_noise:
                act_RA = self.policy.exploration_noise(act_RA, obs_batch_R)
            act_normalized_RA = self.policy.map_action(act_RA)

            # TODO: cleanup the whole policy in batch thing
            # todo policy_R can also be none, check
            policy_R = act_batch_RA.get("policy", Batch())
            if not isinstance(policy_R, Batch):
                raise RuntimeError(
                    f"The policy result should be a {Batch}, but got {type(policy_R)}",
                )

            hidden_state_RH = act_batch_RA.get("state", None)
            # TODO: do we need the conditional? Would be better to just add hidden_state which could be None
            if hidden_state_RH is not None:
                policy_R.hidden_state = (
                    hidden_state_RH  # save state into buffer through policy attr
                )
        return act_RA, act_normalized_RA, policy_R, hidden_state_RH

    # TODO: reduce complexity, remove the noqa
    def collect(
        self,
        n_step: int | None = None,
        n_episode: int | None = None,
        random: bool = False,
        render: float | None = None,
        no_grad: bool = True,
        reset_before_collect: bool = False,
        gym_reset_kwargs: dict[str, Any] | None = None,
    ) -> CollectStats:
        """Collect a specified number of steps or episodes.

        To ensure an unbiased sampling result with the n_episode option, this function will
        first collect ``n_episode - env_num`` episodes, then for the last ``env_num``
        episodes, they will be collected evenly from each env.

        :param n_step: how many steps you want to collect.
        :param n_episode: how many episodes you want to collect.
        :param random: whether to use random policy for collecting data.
        :param render: the sleep time between rendering consecutive frames.
        :param no_grad: whether to retain gradient in policy.forward().
        :param reset_before_collect: whether to reset the environment before
            collecting data.
            It has only an effect if n_episode is not None, i.e.
             if one wants to collect a fixed number of episodes.
            (The collector needs the initial obs and info to function properly.)
        :param gym_reset_kwargs: extra keyword arguments to pass into the environment's
            reset function. Only used if reset_before_collect is True.

        .. note::

            One and only one collection number specification is permitted, either
            ``n_step`` or ``n_episode``.

        :return: The collected stats
        """
        # NAMING CONVENTION (mostly suffixes):
        # episode - An episode means a rollout until done (terminated or truncated). After an episode is completed,
        # the corresponding env is either reset or removed from the ready envs.
        # R - number ready env ids. Note that this might change when envs get idle.
        #     This can only happen in n_episode case, see explanation in the corresponding block.
        #     For n_step, we always use all envs to collect the data, while for n_episode,
        #     R will be at most n_episode at the beginning, but can decrease during the collection.
        # O - dimension(s) of observations
        # A - dimension(s) of actions
        # H - dimension(s) of hidden state
        # D - number of envs that reached done in the current collect iteration. Only relevant in n_episode case.
        # S - number of surplus envs, i.e. envs that are ready but won't be used in the next iteration.
        #     Only used in n_episode case. Then, R becomes R-S.

        use_grad = not no_grad
        gym_reset_kwargs = gym_reset_kwargs or {}

        # Input validation
        assert not self.env.is_async, "Please use AsyncCollector if using async venv."
        if n_step is not None:
            assert n_episode is None, (
                f"Only one of n_step or n_episode is allowed in Collector."
                f"collect, got n_step={n_step}, n_episode={n_episode}."
            )
            assert n_step > 0
            if n_step % self.env_num != 0:
                warnings.warn(
                    f"n_step={n_step} is not a multiple of #env ({self.env_num}), "
                    "which may cause extra transitions collected into the buffer.",
                )
            ready_env_ids_R = np.arange(self.env_num)
        elif n_episode is not None:
            assert n_episode > 0
            if self.env_num > n_episode:
                raise ValueError(
                    f"{n_episode=} should be larger than {self.env_num=} to "
                    f"collect at least one trajectory in each environment.",
                )
            ready_env_ids_R = np.arange(min(self.env_num, n_episode))
        else:
            raise TypeError(
                "Please specify at least one (either n_step or n_episode) "
                "in AsyncCollector.collect().",
            )

        start_time = time.time()

        if reset_before_collect:
            self.reset(reset_buffer=False, gym_reset_kwargs=gym_reset_kwargs)

        if self._pre_collect_obs_RO is None or self._pre_collect_info_R is None:
            raise ValueError(
                "Initial obs and info should not be None. "
                "Either reset the collector (using reset or reset_env) or pass reset_before_collect=True to collect.",
            )

        # get the first obs to be the current obs in the n_step case as
        # episodes as a new call to collect does not restart trajectories
        # (which we also really don't want)
        step_count = 0
        num_collected_episodes = 0
        episode_returns: list[float] = []
        episode_lens: list[int] = []
        episode_start_indices: list[int] = []

        last_obs_RO, last_info_R = self._pre_collect_obs_RO, self._pre_collect_info_R
        last_hidden_state_RH = self._pre_collect_hidden_state_RH
        while True:
            # todo check if we need this when using cur_rollout_batch
            # if len(cur_rollout_batch) != len(ready_env_ids):
            #     raise RuntimeError(
            #         f"The length of the collected_rollout_batch {len(cur_rollout_batch)}) is not equal to the length of ready_env_ids"
            #         f"{len(ready_env_ids)}. This should not happen and could be a bug!",
            #     )
            # restore the state: if the last state is None, it won't store

            # get the next action
            (
                act_RA,
                act_normalized_RA,
                policy_R,
                hidden_state_RH,
            ) = self._compute_action_policy_hidden(
                random=random,
                ready_env_ids_R=ready_env_ids_R,
                use_grad=use_grad,
                last_obs_RO=last_obs_RO,
                last_info_R=last_info_R,
                last_hidden_state_RH=last_hidden_state_RH,
            )

            obs_next_RO, rew_R, terminated_R, truncated_R, info_R = self.env.step(
                act_normalized_RA,
                ready_env_ids_R,
            )
            done_R = np.logical_or(terminated_R, truncated_R)

            current_iteration_batch = cast(
                RolloutBatchProtocol,
                Batch(
                    obs=last_obs_RO,
                    act=act_RA,
                    policy=policy_R,
                    obs_next=obs_next_RO,
                    rew=rew_R,
                    terminated=terminated_R,
                    truncated=truncated_R,
                    done=done_R,
                    info=info_R,
                ),
            )

            # TODO: only makes sense if render_mode is human.
            #  Also, doubtful whether it makes sense at all for true vectorized envs
            if render:
                self.env.render()
                if not np.isclose(render, 0):
                    time.sleep(render)

            # add data into the buffer
            ptr_R, ep_rew_R, ep_len_R, ep_idx_R = self.buffer.add(
                current_iteration_batch,
                buffer_ids=ready_env_ids_R,
            )

            # collect statistics
            num_episodes_done_this_iter = np.sum(done_R)
            num_collected_episodes += num_episodes_done_this_iter
            step_count += len(ready_env_ids_R)

            # preparing for the next iteration
            # obs_next, info and hidden_state will be modified inplace in the code below, so we copy to not affect the data in the buffer
            last_obs_RO = copy(obs_next_RO)
            last_info_R = copy(info_R)
            last_hidden_state_RH = copy(hidden_state_RH)

            # Preparing last_obs_RO, last_info_R, last_hidden_state_RH for the next while-loop iteration
            # Resetting envs that reached done, or removing some of them from the collection if needed (see below)
            if num_episodes_done_this_iter > 0:
                # TODO: adjust the whole index story, don't use np.where, just slice with boolean arrays
                # D - number of envs that reached done in the rollout above
                env_ind_local_D = np.where(done_R)[0]
                env_ind_global_D = ready_env_ids_R[env_ind_local_D]
                episode_lens.extend(ep_len_R[env_ind_local_D])
                episode_returns.extend(ep_rew_R[env_ind_local_D])
                episode_start_indices.extend(ep_idx_R[env_ind_local_D])
                # now we copy obs_next to obs, but since there might be
                # finished episodes, we have to reset finished envs first.

                obs_reset_DO, info_reset_D = self.env.reset(
                    env_ids=env_ind_global_D,
                    **gym_reset_kwargs,
                )

                # Set the hidden state to zero or None for the envs that reached done
                # TODO: does it have to be so complicated? We should have a single clear type for hidden_state instead of
                #  this complex logic
                self._reset_hidden_state_based_on_type(env_ind_local_D, last_hidden_state_RH)

                # preparing for the next iteration
                last_obs_RO[env_ind_local_D] = obs_reset_DO
                last_info_R[env_ind_local_D] = info_reset_D

                # Handling the case when we have more ready envs than desired and are not done yet
                #
                # This can only happen if we are collecting a fixed number of episodes
                # If we have more ready envs than there are remaining episodes to collect,
                # we will remove some of them for the next rollout
                # One effect of this is the following: only envs that have completed an episode
                # in the last step can ever be removed from the ready envs.
                # Thus, this guarantees that each env will contribute at least one episode to the
                # collected data (the buffer). This effect was previous called "avoiding bias in selecting environments"
                # However, it is not at all clear whether this is actually useful or necessary.
                # Additional naming convention:
                # S - number of surplus envs
                # TODO: can the whole block be removed? If we have too many episodes, we could just strip the last ones.
                #   Changing R to R-S highly increases the complexity of the code.
                if n_episode:
                    remaining_episodes_to_collect = n_episode - num_collected_episodes
                    surplus_env_num = len(ready_env_ids_R) - remaining_episodes_to_collect
                    if surplus_env_num > 0:
                        # R becomes R-S here, preparing for the next iteration in while loop
                        # Everything that was of length R needs to be filtered and become of length R-S
                        # Note that this won't be the last iteration, as one iteration equals one
                        # step and we still need to collect the remaining episodes to reach the breaking condition

                        # creating the mask
                        env_to_be_ignored_ind_local_S = env_ind_local_D[:surplus_env_num]
                        env_should_remain_R = np.ones_like(ready_env_ids_R, dtype=bool)
                        env_should_remain_R[env_to_be_ignored_ind_local_S] = False
                        # stripping the "idle" indices, shortening the relevant quantities from R to R-S
                        ready_env_ids_R = ready_env_ids_R[env_should_remain_R]
                        last_obs_RO = last_obs_RO[env_should_remain_R]
                        last_info_R = last_info_R[env_should_remain_R]
                        if hidden_state_RH is not None:
                            last_hidden_state_RH = last_hidden_state_RH[env_should_remain_R]

            if (n_step and step_count >= n_step) or (
                n_episode and num_collected_episodes >= n_episode
            ):
                break

        # generate statistics
        self.collect_step += step_count
        self.collect_episode += num_collected_episodes
        collect_time = max(time.time() - start_time, 1e-9)
        self.collect_time += collect_time

        if n_step:
            # persist for future collect iterations
            self._pre_collect_obs_RO = last_obs_RO
            self._pre_collect_info_R = last_info_R
            self._pre_collect_hidden_state_RH = last_hidden_state_RH
        elif n_episode:
            # reset envs and the _pre_collect fields
            self.reset_env(gym_reset_kwargs)  # todo still necessary?

        return CollectStats.with_autogenerated_stats(
            returns=np.array(episode_returns),
            lens=np.array(episode_lens),
            n_collected_episodes=num_collected_episodes,
            n_collected_steps=step_count,
            collect_time=collect_time,
            collect_speed=step_count / collect_time,
        )

    def _reset_hidden_state_based_on_type(
        self,
        env_ind_local_D: np.ndarray,
        last_hidden_state_RH: np.ndarray | torch.Tensor | Batch | None,
    ) -> None:
        if isinstance(last_hidden_state_RH, torch.Tensor):
            last_hidden_state_RH[env_ind_local_D].zero_()
        elif isinstance(last_hidden_state_RH, np.ndarray):
            last_hidden_state_RH[env_ind_local_D] = (
                None if last_hidden_state_RH.dtype == object else 0
            )
        elif isinstance(last_hidden_state_RH, Batch):
            last_hidden_state_RH.empty_(env_ind_local_D)
        # todo is this inplace magic and just working?


class AsyncCollector(Collector):
    """Async Collector handles async vector environment.

    The arguments are exactly the same as :class:`~tianshou.data.Collector`, please
    refer to :class:`~tianshou.data.Collector` for more detailed explanation.
    """

    def __init__(
        self,
        policy: BasePolicy,
        env: BaseVectorEnv,
        buffer: ReplayBuffer | None = None,
        exploration_noise: bool = False,
    ) -> None:
        # assert env.is_async
        warnings.warn("Using async setting may collect extra transitions into buffer.")
        super().__init__(
            policy,
            env,
            buffer,
            exploration_noise,
        )
        self._ready_env_ids: np.ndarray
        self._current_action_in_all_envs_EA: np.ndarray
        self._current_policy_in_all_envs_E: ActStateBatchProtocol | ActBatchProtocol | None
        self._current_obs_in_all_envs_EO: np.ndarray
        self._current_hidden_state_in_all_envs_EH: np.ndarray | torch.Tensor | Batch | None
        self._current_info_in_all_envs_E: list[dict]

    def reset(
        self,
        reset_buffer: bool = True,
        reset_stats: bool = True,
        gym_reset_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Reset the environment, statistics, and data needed to start the collection.

        :param reset_buffer: if true, reset the replay buffer attached
            to the collector.
        :param reset_stats: if true, reset the statistics attached to the collector.
        :param gym_reset_kwargs: extra keyword arguments to pass into the environment's
            reset function. Defaults to None (extra keyword arguments)
        """
        super().reset(
            reset_buffer=reset_buffer,
            reset_stats=reset_stats,
            gym_reset_kwargs=gym_reset_kwargs,
        )
        self._ready_env_ids = np.arange(self.env_num)
        # E denotes the number of parallel environments self.env_num
        self._current_obs_in_all_envs_EO = self._pre_collect_obs_RO
        self._current_info_in_all_envs_E = self._pre_collect_info_R
        self._current_hidden_state_in_all_envs_EH = self._pre_collect_hidden_state_RH
        self._current_action_in_all_envs_EA = np.empty(self.env_num)
        self._current_policy_in_all_envs_E = None

    def collect(
        self,
        n_step: int | None = None,
        n_episode: int | None = None,
        random: bool = False,
        render: float | None = None,
        no_grad: bool = True,
        reset_before_collect: bool = False,
        gym_reset_kwargs: dict[str, Any] | None = None,
    ) -> CollectStats:
        """Collect a specified number of steps or episodes with async env setting.

        This function doesn't collect exactly n_step or n_episode number of
        transitions. Instead, in order to support async setting, it may collect more
        than given n_step or n_episode transitions and save into buffer.

        :param n_step: how many steps you want to collect.
        :param n_episode: how many episodes you want to collect.
        :param random: whether to use random policy_R for collecting data. Default
            to False.
        :param render: the sleep time between rendering consecutive frames.
            Default to None (no rendering).
        :param no_grad: whether to retain gradient in policy_R.forward(). Default to
            True (no gradient retaining).
                :param reset_before_collect: whether to reset the environment before
            collecting data.
            It has only an effect if n_episode is not None, i.e.
             if one wants to collect a fixed number of episodes.
            (The collector needs the initial obs and info to function properly.)
        :param gym_reset_kwargs: extra keyword arguments to pass into the environment's
            reset function. Defaults to None (extra keyword arguments)

        .. note::

            One and only one collection number specification is permitted, either
            ``n_step`` or ``n_episode``.

        :return: A dataclass object
        """
        use_grad = not no_grad
        gym_reset_kwargs = gym_reset_kwargs or {}

        # collect at least n_step or n_episode
        if n_step is not None:
            assert n_episode is None, (
                "Only one of n_step or n_episode is allowed in Collector."
                f"collect, got n_step={n_step}, n_episode={n_episode}."
            )
            assert n_step > 0
        elif n_episode is not None:
            assert n_episode > 0
        else:
            raise TypeError(
                "Please specify at least one (either n_step or n_episode) "
                "in AsyncCollector.collect().",
            )

        if reset_before_collect:
            # first we need to step all envs to be able to interact with them
            if self.env.waiting_id:
                self.env.step(None, id=self.env.waiting_id)
            self.reset(reset_buffer=False, gym_reset_kwargs=gym_reset_kwargs)

        start_time = time.time()

        step_count = 0
        num_collected_episodes = 0
        episode_returns: list[float] = []
        episode_lens: list[int] = []
        episode_start_indices: list[int] = []

        ready_env_ids_R = self._ready_env_ids
        last_obs_RO, last_info_R = self._pre_collect_obs_RO, self._pre_collect_info_R
        last_hidden_state_RH = self._pre_collect_hidden_state_RH

        # Each iteration of the AsyncCollector is only stepping a subset of the
        # envs. The last observation/ hiddenstate of the ones not included in
        # the current iteration has to be retained.
        while True:
            # todo do we need this?
            # todo extend to all current attributes but some could be None at init
            if (
                not len(self._current_obs_in_all_envs_EO)
                == len(self._current_action_in_all_envs_EA)
                == self.env_num
            ):  # major difference
                raise ValueError(
                    f"{len(self._current_obs_in_all_envs_EO)=} and"
                    f"{len(self._current_action_in_all_envs_EA)=} have to equal"
                    f" {self.env_num=} as it tracks the current transition"
                    f"in all envs",
                )

            # get the next action
            (
                act_RA,
                act_normalized_RA,
                policy_R,
                hidden_state_RH,
            ) = self._compute_action_policy_hidden(
                random=random,
                ready_env_ids_R=ready_env_ids_R,
                use_grad=use_grad,
                last_obs_RO=last_obs_RO,
                last_info_R=last_info_R,
                last_hidden_state_RH=last_hidden_state_RH,
            )

            # save act_RA/policy_R/ hidden_state_RH before env.step
            self._current_action_in_all_envs_EA[ready_env_ids_R] = act_RA
            if self._current_policy_in_all_envs_E:
                self._current_policy_in_all_envs_E[ready_env_ids_R] = policy_R
            else:
                self._current_policy_in_all_envs_E = policy_R  # first iteration
            if hidden_state_RH is not None:
                if self._current_hidden_state_in_all_envs_EH is not None:
                    self._current_hidden_state_in_all_envs_EH[ready_env_ids_R] = hidden_state_RH
                else:
                    self._current_hidden_state_in_all_envs_EH = hidden_state_RH

            # step in env
            obs_next_RO, rew_R, terminated_R, truncated_R, info_R = self.env.step(
                act_normalized_RA,
                ready_env_ids_R,
            )
            done_R = np.logical_or(terminated_R, truncated_R)
            # Not all environments of the AsynCollector might have performed a step in this iteration.
            # Change batch_of_envs_with_step_in_this_iteration here to reflect that ready_env_ids_R has changed.
            # This means especially that R is potentially changing every iteration
            try:
                ready_env_ids_R = info_R["env_id"]
            except Exception:
                ready_env_ids_R = np.array([i["env_id"] for i in info_R])

            current_iteration_batch = cast(
                RolloutBatchProtocol,
                Batch(
                    obs=self._current_obs_in_all_envs_EO[ready_env_ids_R],
                    act=self._current_action_in_all_envs_EA[ready_env_ids_R],
                    policy=self._current_policy_in_all_envs_E[ready_env_ids_R],
                    obs_next=obs_next_RO,
                    rew=rew_R,
                    terminated=terminated_R,
                    truncated=truncated_R,
                    done=done_R,
                    info=info_R,
                ),
            )

            if render:
                self.env.render()
                if render > 0 and not np.isclose(render, 0):
                    time.sleep(render)

            # add data into the buffer
            ptr_R, ep_rew_R, ep_len_R, ep_idx_R = self.buffer.add(
                current_iteration_batch,
                buffer_ids=ready_env_ids_R,
            )

            # collect statistics
            num_episodes_done_this_iter = np.sum(done_R)
            step_count += len(ready_env_ids_R)
            num_collected_episodes += num_episodes_done_this_iter

            # preparing for the next iteration
            last_obs_RO = obs_next_RO
            last_info_R = info_R
            last_hidden_state_RH = self._current_hidden_state_in_all_envs_EH[ready_env_ids_R]

            if num_episodes_done_this_iter:
                env_ind_local_D = np.where(done_R)[0]
                env_ind_global_D = ready_env_ids_R[env_ind_local_D]
                episode_lens.extend(ep_len_R[env_ind_local_D])
                episode_returns.extend(ep_rew_R[env_ind_local_D])
                episode_start_indices.extend(ep_idx_R[env_ind_local_D])

                # now we copy obs_next_RO to obs, but since there might be
                # finished episodes, we have to reset finished envs first.
                obs_reset_DO, info_reset_D = self.env.reset(
                    env_ids=env_ind_global_D,
                    **gym_reset_kwargs,
                )
                last_obs_RO[env_ind_local_D] = obs_reset_DO
                last_info_R[env_ind_local_D] = info_reset_D

                self._reset_hidden_state_based_on_type(env_ind_local_D, last_hidden_state_RH)

            # update based on the current transition in all envs
            self._current_obs_in_all_envs_EO[ready_env_ids_R] = last_obs_RO
            # extremely ugly assignment, but hey, we gain explicit attributes, so who cares
            # this is a list, so loop over
            for idx, ready_env_id in enumerate(ready_env_ids_R):
                self._current_info_in_all_envs_E[ready_env_id] = last_info_R[idx]
            if self._current_hidden_state_in_all_envs_EH is not None:
                self._current_hidden_state_in_all_envs_EH[ready_env_ids_R] = last_hidden_state_RH
            else:
                self._current_hidden_state_in_all_envs_EH = last_hidden_state_RH

            if (n_step and step_count >= n_step) or (
                n_episode and num_collected_episodes >= n_episode
            ):
                break

        # generate statistics
        self.collect_step += step_count
        self.collect_episode += num_collected_episodes
        collect_time = max(time.time() - start_time, 1e-9)
        self.collect_time += collect_time

        # persist for future collect iterations
        self._ready_env_ids = ready_env_ids_R
        self._pre_collect_obs_RO = last_obs_RO
        self._pre_collect_info_R = last_info_R
        self._pre_collect_hidden_state_RH = last_hidden_state_RH

        return CollectStats(
            n_collected_episodes=num_collected_episodes,
            n_collected_steps=step_count,
            collect_time=collect_time,
            collect_speed=step_count / collect_time,
            returns=np.array(episode_returns),
            returns_stat=SequenceSummaryStats.from_sequence(episode_returns)
            if len(episode_returns) > 0
            else None,
            lens=np.array(episode_lens, int),
            lens_stat=SequenceSummaryStats.from_sequence(episode_lens)
            if len(episode_lens) > 0
            else None,
        )
