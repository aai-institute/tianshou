import torch
import numpy as np
import torch.nn.functional as F
from typing import Any, Dict

from tianshou.policy import QRDQNPolicy
from tianshou.data import Batch


class IQNPolicy(QRDQNPolicy):
    """Implementation of Implicit Quantile Network. arXiv:1806.06923.

    :param torch.nn.Module model: a model following the rules in
        :class:`~tianshou.policy.BasePolicy`. (s -> logits)
    :param torch.optim.Optimizer optim: a torch.optim for optimizing the model.
    :param float discount_factor: in [0, 1].
    :param int sample_size: the number of samples for policy evaluation.
        Default to 32.
    :param int online_sample_size: the number of samples for online model
        in training. Default to 8.
    :param int target_sample_size: the number of samples for target model
        in training. Default to 8.
    :param int estimation_step: the number of steps to look ahead. Default to 1.
    :param int target_update_freq: the target network update frequency (0 if
        you do not use the target network).
    :param bool reward_normalization: normalize the reward to Normal(0, 1).
        Default to False.

    .. seealso::

        Please refer to :class:`~tianshou.policy.QRDQNPolicy` for more detailed
        explanation.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optim: torch.optim.Optimizer,
        discount_factor: float = 0.99,
        sample_size: int = 32,
        online_sample_size: int = 8,
        target_sample_size: int = 8,
        estimation_step: int = 1,
        target_update_freq: int = 0,
        reward_normalization: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, optim, discount_factor, sample_size, estimation_step,
                         target_update_freq, reward_normalization, **kwargs)
        assert sample_size > 1, "sample_size should be greater than 1"
        assert online_sample_size > 1, "online_sample_size should be greater than 1"
        assert target_sample_size > 1, "target_sample_size should be greater than 1"
        self._sample_size = sample_size  # for policy eval
        self._online_sample_size = online_sample_size
        self._target_sample_size = target_sample_size
        # set sample size for online and target model
        self.model.sample_size = self._online_sample_size  # type: ignore
        if self._target:
            self.model_old.sample_size = self._target_sample_size  # type: ignore

    def train(self, mode: bool = True) -> "IQNPolicy":
        super().train(mode)
        self.model.sample_size = (self._online_sample_size if mode  # type: ignore
                                  else self._sample_size)  # type: ignore
        return self

    def sync_weight(self) -> None:
        """Synchronize the weight for the target network."""
        self.model_old.load_state_dict(self.model.state_dict())  # type: ignore
        self.model_old.sample_size = self._target_sample_size  # type: ignore

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
        if self._target and self._iter % self._freq == 0:
            self.sync_weight()
        self.optim.zero_grad()
        weight = batch.pop("weight", 1.0)
        out = self(batch)
        curr_dist, taus = out.logits, out.state
        act = batch.act
        curr_dist = curr_dist[np.arange(len(act)), act, :].unsqueeze(2)
        target_dist = batch.returns.unsqueeze(1)
        # calculate each element's difference between curr_dist and target_dist
        u = F.smooth_l1_loss(target_dist, curr_dist, reduction="none")
        huber_loss = (u * (
            taus.unsqueeze(2) - (target_dist - curr_dist).detach().le(0.).float()
        ).abs()).sum(-1).mean(1)
        loss = (huber_loss * weight).mean()
        # ref: https://github.com/ku2482/fqf-iqn-qrdqn.pytorch/
        # blob/master/fqf_iqn_qrdqn/agent/qrdqn_agent.py L130
        batch.weight = u.detach().abs().sum(-1).mean(1)  # prio-buffer
        loss.backward()
        self.optim.step()
        self._iter += 1
        return {"loss": loss.item()}
