import torch
import numpy as np
from typing import Any, Dict, Union

from tianshou.policy import C51Policy
from tianshou.data import Batch, ReplayBuffer


class RainbowPolicy(C51Policy):
    """Implementation of Categorical Deep Q-Network. arXiv:1707.06887.

    :param torch.nn.Module model: a model following the rules in
        :class:`~tianshou.policy.BasePolicy`. (s -> logits)
    :param torch.optim.Optimizer optim: a torch.optim for optimizing the model.
    :param float discount_factor: in [0, 1].
    :param int num_atoms: the number of atoms in the support set of the
        value distribution. Default to 51.
    :param float v_min: the value of the smallest atom in the support set.
        Default to -10.0.
    :param float v_max: the value of the largest atom in the support set.
        Default to 10.0.
    :param int estimation_step: the number of steps to look ahead. Default to 1.
    :param int target_update_freq: the target network update frequency (0 if
        you do not use the target network). Default to 0.
    :param bool reward_normalization: normalize the reward to Normal(0, 1).
        Default to False.

    .. seealso::

        Please refer to :class:`~tianshou.policy.DQNPolicy` for more detailed
        explanation.
    """

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
        self.model.sample_noise()
        self.model_old.sample_noise()
        return super().learn(batch, **kwargs)

    def exploration_noise(self, act: Union[np.ndarray, Batch], batch: Batch) -> Union[np.ndarray, Batch]:
        if self.training:
            self.model.sample_noise()
            return act
        else:
            return super().exploration_noise(act, batch)
