from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Self

import numpy as np
from gymnasium import spaces

from tianshou.utils.string import ToStringMixin


@dataclass(kw_only=True)
class ActionSpaceInfo(ToStringMixin):
    """A data structure for storing the different attributes of the action space."""

    action_shape: int | Sequence[int]
    """The shape of the action space."""
    min_action: float
    """The smallest allowable action or in the continuous case the lower bound for allowable action value."""
    max_action: float
    """The largest allowable action or in the continuous case the upper bound for allowable action value."""

    @property
    def action_dim(self) -> int:
        """Return the number of distinct actions an agent can take it its action space."""
        if isinstance(self.action_shape, int):
            return self.action_shape
        elif isinstance(self.action_shape, Sequence) and self.action_shape:
            return int(np.prod(self.action_shape))
        else:
            raise ValueError("Invalid action_shape: {self.action_shape}.")

    @classmethod
    def from_space(cls, space: spaces.Space) -> Self:
        """Return the attributes of the action space based on the instance type of the space."""
        if isinstance(space, spaces.Box):
            return cls(
                action_shape=space.shape,
                min_action=float(np.min(space.low)),
                max_action=float(np.max(space.high)),
            )
        elif isinstance(space, spaces.Discrete):
            return cls(
                action_shape=int(space.n),
                min_action=float(space.start),
                max_action=float(space.start + space.n - 1),
            )
        else:
            raise ValueError(
                f"Unsupported space type: {space.__class__}. Currently supported types are Discrete and Box.",
            )

    def _tostring_additional_entries(self) -> dict[str, Any]:
        return {"action_dim": self.action_dim}


@dataclass(kw_only=True)
class ObservationSpaceInfo(ToStringMixin):
    """A data structure for storing the different attributes of the observation space."""

    obs_shape: int | Sequence[int]
    """The shape of the observation space."""

    @property
    def obs_dim(self) -> int:
        """Return the number of distinct features or dimensions in the observation space."""
        if isinstance(self.obs_shape, int):
            return self.obs_shape
        elif isinstance(self.obs_shape, Sequence) and self.obs_shape:
            return int(np.prod(self.obs_shape))
        else:
            raise ValueError("Invalid obs_shape: {self.obs_shape}.")

    @classmethod
    def from_space(cls, space: spaces.Space) -> Self:
        """Return the attributes of the observation space based on the instance type of the space."""
        if isinstance(space, spaces.Box):
            return cls(
                obs_shape=space.shape,
            )
        elif isinstance(space, spaces.Discrete):
            return cls(
                obs_shape=int(space.n),
            )
        else:
            raise ValueError(
                f"Unsupported space type: {space.__class__}. Currently supported types are Discrete and Box.",
            )

    def _tostring_additional_entries(self) -> dict[str, Any]:
        return {"obs_dim": self.obs_dim}


@dataclass(kw_only=True)
class SpaceInfo(ToStringMixin):
    """A data structure for storing the attributes of both the action and observation space."""

    action_info: ActionSpaceInfo
    """Stores the attributes of the action space."""
    observation_info: ObservationSpaceInfo
    """Stores the attributes of the observation space."""

    @classmethod
    def from_env(cls, action_space: spaces.Space, observation_space: spaces.Space) -> Self:
        """Return the attributes of the action and observation space based on the instance type of each of the spaces."""
        action_info = ActionSpaceInfo.from_space(action_space)
        observation_info = ObservationSpaceInfo.from_space(observation_space)

        return cls(
            action_info=action_info,
            observation_info=observation_info,
        )
