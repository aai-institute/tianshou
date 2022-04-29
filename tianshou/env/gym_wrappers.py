import gym
import numpy as np


class ContinuousToDiscrete(gym.ActionWrapper):
    """Gym environment wrapper to take discrete action in a continous environment.

    Args:
        env (gym.Environment): gym envirionment with continous action space.
        action_per_branch (int): number of discrete actions in each dimension 
        of the action space.
    
    """

    def __init__(self, env: gym.Env, action_per_branch: int) -> None:
        super().__init__(env)
        self.action_per_branch = action_per_branch
        low = env.action_space.low
        high = env.action_space.high
        num_branches = env.action_space.shape[0]
        self.action_space = gym.spaces.MultiDiscrete(
            [action_per_branch] * num_branches
        )
        setattr(self.action_space, "n", num_branches)
        self.mesh = []
        for lo, hi in zip(low, high):
            self.mesh.append(np.linspace(lo, hi, action_per_branch))

    def action(self, act: np.ndarray) -> np.ndarray:
        # modify act
        act = np.array([self.mesh[i][a] for i, a in enumerate(act)])
        return act
