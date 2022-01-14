import copy
import math
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from tianshou.data import Batch, ReplayBuffer, to_torch
from tianshou.policy import SACPolicy
from tianshou.utils.net.continuous import ActorProb


class CQLPolicy(SACPolicy):

    def __init__(
        self,
        actor: ActorProb,
        actor_optim: torch.optim.Optimizer,
        critic1: torch.nn.Module,
        critic1_optim: torch.optim.Optimizer,
        critic2: torch.nn.Module,
        critic2_optim: torch.optim.Optimizer,
        cql_log_alpha: torch.Tensor,
        cql_alpha_optim: torch.optim.Optimizer,
        cql_weight: float = 1.0,
        tau: float = 0.005,
        gamma: float = 0.99,
        alpha: Union[float, Tuple[float, torch.Tensor, torch.optim.Optimizer]] = 0.2,
        temperature: float = 1.0,
        with_lagrange: bool = True,
        lagrange_threshold: float = 10.0,
        min_action: float = -1.0,
        max_action: float = 1.0,
        num_repeat_actions: int = 10,
        alpha_min: float = 0.0,
        alpha_max: float = 1e6,
        clip_grad: float = 1.0,
        device: Union[str, torch.device] = "cpu",
        **kwargs: Any
    ) -> None:
        super().__init__(
            actor, actor_optim, critic1, critic1_optim, critic2, critic2_optim, tau,
            gamma, alpha, **kwargs
        )
        # There are _target_entropy, _log_alpha, _alpha_optim in SACPolicy.
        self.device = device
        self.temperature = temperature
        self.with_lagrange = with_lagrange
        self.lagrange_threshold = lagrange_threshold

        self.cql_weight = cql_weight
        self.cql_log_alpha = cql_log_alpha.to(device)
        self.cql_alpha_optim = cql_alpha_optim

        self.min_action = min_action
        self.max_action = max_action

        self.num_repeat_actions = num_repeat_actions

        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.clip_grad = clip_grad

    def train(self, mode: bool = True) -> "CQLPolicy":
        """Set the module in training mode, except for the target network."""
        self.training = mode
        self.actor.train(mode)
        self.critic1.train(mode)
        self.critic2.train(mode)
        return self

    def sync_weight(self) -> None:
        """Soft-update the weight for the target network."""
        for net, net_old in [
            [self.critic1, self.critic1_old], [self.critic2, self.critic2_old]
        ]:
            for param, target_param in zip(net.parameters(), net_old.parameters()):
                target_param.data.copy_(
                    self._tau * param.data + (1 - self._tau) * target_param.data
                )

    def actor_pred(self, obs, epsilon=1e-6):
        # use obs to predict action
        # mu, sigma = self.actor.forward(obs)[0]
        # dist = torch.distributions.Normal(mu, sigma)
        # e = dist.rsample().to(self.device)
        # act_pred = torch.tanh(e)
        # # act_pred.shape: (batch_size, action_dim)
        # log_prob = (dist.log_prob(e) -
        #             torch.log(1 - act_pred.pow(2) + epsilon)).sum(1, keepdim=True)
        # return act_pred, log_prob
        batch = Batch(obs=obs, info=None)
        obs_result = self(batch)
        return obs_result.act, obs_result.log_prob

    def calc_actor_loss(self, obs):
        act_pred, log_pi = self.actor_pred(obs)
        q1 = self.critic1(obs, act_pred)
        q2 = self.critic2(obs, act_pred)
        min_Q = torch.min(q1, q2)
        actor_loss = (self._alpha * log_pi - min_Q).mean()
        # actor_loss.shape: (), log_pi.shape: (batch_size, 1)
        return actor_loss, log_pi

    def calc_pi_values(self, obs_pi, obs_q):
        act_pred, log_pi = self.actor_pred(obs_pi)

        q1 = self.critic1(obs_q, act_pred)
        q2 = self.critic2(obs_q, act_pred)

        return q1 - log_pi.detach(), q2 - log_pi.detach()

    def calc_random_values(self, obs, act):
        random_value1 = self.critic1(obs, act)
        random_log_prob1 = math.log(0.5 ** act.shape[-1])

        random_value2 = self.critic2(obs, act)
        random_log_prob2 = math.log(0.5 ** act.shape[-1])

        return random_value1 - random_log_prob1, random_value2 - random_log_prob2

    def process_fn(
        self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray
    ) -> Batch:
        return batch

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
        batch: Batch = to_torch(  # type: ignore
            batch, dtype=torch.float, device=self.device,
        )
        obs, act, rew, obs_next = batch.obs, batch.act, batch.rew, batch.obs_next
        batch_size = obs.shape[0]

        # compute actor loss and update actor
        # prevent alpha from being modified
        # current_alpha = copy.deepcopy(self._alpha)
        # TODO: fix alpha here?
        actor_loss, log_pi = self.calc_actor_loss(obs)
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # compute alpha loss
        if self._is_auto_alpha:
            log_pi = log_pi + self._target_entropy
            alpha_loss = -(self._log_alpha * log_pi.detach()).mean()
            self._alpha_optim.zero_grad()
            # update log_alpha
            alpha_loss.backward()
            self._alpha_optim.step()
            # update alpha
            self._alpha = self._log_alpha.detach().exp()

        # compute target_Q
        with torch.no_grad():
            act_next, new_log_pi = self.actor_pred(obs_next)

            target_Q1 = self.critic1_old(obs_next, act_next)
            target_Q2 = self.critic2_old(obs_next, act_next)

            target_Q = torch.min(target_Q1, target_Q2) - self._alpha * new_log_pi

            target_Q = \
                rew + self._gamma * (1 - batch.done) * target_Q.squeeze()
            # shape: (batch_size)

        # compute critic loss
        current_Q1 = self.critic1(obs, act).squeeze()
        current_Q2 = self.critic2(obs, act).squeeze()
        # shape: (batch_size)

        critic1_loss = F.mse_loss(current_Q1, target_Q)
        critic2_loss = F.mse_loss(current_Q2, target_Q)

        # CQL
        random_actions = torch.FloatTensor(
            batch_size * self.num_repeat_actions, act.shape[-1]
        ).uniform_(-self.min_action, self.max_action).to(self.device)
        tmp_obs = obs.unsqueeze(1) \
            .repeat(1, self.num_repeat_actions, 1) \
            .view(batch_size * self.num_repeat_actions, obs.shape[-1])
        tmp_obs_next = obs_next.unsqueeze(1) \
            .repeat(1, self.num_repeat_actions, 1) \
            .view(batch_size * self.num_repeat_actions, obs.shape[-1])
        # tmp_obs & tmp_obs_next: (batch_size * num_repeat, state_dim)

        current_pi_value1, current_pi_value2 = self.calc_pi_values(tmp_obs, tmp_obs)
        next_pi_value1, next_pi_value2 = self.calc_pi_values(tmp_obs_next, tmp_obs)

        random_value1, random_value2 = self.calc_random_values(tmp_obs, random_actions)

        for value in [
            current_pi_value1, current_pi_value2, next_pi_value1, next_pi_value2,
            random_value1, random_value2
        ]:
            value.reshape(batch_size, self.num_repeat_actions, 1)

        # cat q values
        cat_q1 = torch.cat([random_value1, current_pi_value1, next_pi_value1], 1)
        cat_q2 = torch.cat([random_value2, current_pi_value2, next_pi_value2], 1)
        # shape: (batch_size, 3 * num_repeat, 1)

        # TODO: - current_Q1.mean() * self.cql_weight ?
        cql1_scaled_loss = \
            torch.logsumexp(cat_q1 / self.temperature, dim=1).mean() * \
            self.cql_weight * self.temperature - current_Q1.mean() * \
            self.cql_weight
        cql2_scaled_loss = \
            torch.logsumexp(cat_q2 / self.temperature, dim=1).mean() * \
            self.cql_weight * self.temperature - current_Q2.mean() * \
            self.cql_weight
        # shape: (1)

        if self.with_lagrange:
            cql_alpha = torch.clamp(
                self.cql_log_alpha.exp(),
                self.alpha_min,
                self.alpha_max,
            )
            cql1_scaled_loss = \
                cql_alpha * (cql1_scaled_loss - self.lagrange_threshold)
            cql2_scaled_loss = \
                cql_alpha * (cql2_scaled_loss - self.lagrange_threshold)

            self.cql_alpha_optim.zero_grad()
            cql_alpha_loss = -(cql1_scaled_loss + cql2_scaled_loss) * 0.5
            cql_alpha_loss.backward(retain_graph=True)
            self.cql_alpha_optim.step()

        critic1_loss = critic1_loss + cql1_scaled_loss
        critic2_loss = critic2_loss + cql2_scaled_loss

        # update critic
        self.critic1_optim.zero_grad()
        critic1_loss.backward(retain_graph=True)
        # clip grad, prevent the vanishing gradient problem
        # TODO: It seems to be not necessary
        # clip_grad_norm_(self.critic1.parameters(), self.clip_grad)
        self.critic1_optim.step()

        self.critic2_optim.zero_grad()
        critic2_loss.backward()
        # clip_grad_norm_(self.critic2.parameters(), self.clip_grad)
        self.critic2_optim.step()

        self.sync_weight()

        result = {
            "loss/actor": actor_loss.item(),
            "loss/critic1": critic1_loss.item(),
            "loss/critic2": critic2_loss.item(),
        }
        if self._is_auto_alpha:
            result["loss/alpha"] = alpha_loss.item()
            result["alpha"] = self._alpha.item()
        if self.with_lagrange:
            result["loss/cql_alpha"] = cql_alpha_loss.item()
            result["cql_alpha"] = cql_alpha.item()
        return result
