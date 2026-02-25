#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

"""
A simple custom multi-agent environment built directly with TorchRL's EnvBase.

This serves as a concrete working example for users who want to implement their
own MARL environment and integrate it with BenchMARL.

Scenario
--------
N agents navigate a 2D arena and each tries to reach its own target.

- Observation: [agent_x, agent_y, target_x, target_y]  (shape: [n_agents, 4])
- Action: [dx, dy] velocity clipped to [-1, 1]          (shape: [n_agents, 2])
- Reward: negative Euclidean distance to the target      (shape: [n_agents, 1])
- Episode ends after ``max_steps`` steps.

All agents belong to a single group called "agents".
"""

import torch
from tensordict import TensorDict
from torchrl.data import Bounded, Composite, Unbounded
from torchrl.envs import EnvBase


class SimpleMultiAgentEnv(EnvBase):
    """Simple multi-agent navigation environment.

    Args:
        n_agents (int): Number of agents (default: 3).
        max_steps (int): Maximum number of steps per episode (default: 100).
        device (str): Torch device string (default: "cpu").
    """

    # Name used in group_map – must match the key in observation/action specs.
    AGENTS_GROUP = "agents"

    def __init__(
        self,
        n_agents: int = 3,
        max_steps: int = 100,
        device="cpu",
    ):
        super().__init__(device=device, batch_size=[])

        self.n_agents = n_agents
        self.max_steps = max_steps
        self._step_count = 0

        # Internal state (will be initialised in _reset)
        self._agent_pos = None
        self._target_pos = None

        self._make_specs()

    # ------------------------------------------------------------------
    # Spec definitions
    # ------------------------------------------------------------------

    def _make_specs(self):
        g = self.AGENTS_GROUP

        # Observations: 4-dimensional vector per agent
        self.observation_spec = Composite(
            {
                g: Composite(
                    observation=Unbounded(
                        shape=torch.Size([self.n_agents, 4]),
                        dtype=torch.float32,
                        device=self.device,
                    ),
                    shape=torch.Size([self.n_agents]),
                )
            },
            shape=torch.Size([]),
        )

        # Actions: 2-dimensional bounded vector per agent
        self.action_spec = Composite(
            {
                g: Composite(
                    action=Bounded(
                        low=-1.0,
                        high=1.0,
                        shape=torch.Size([self.n_agents, 2]),
                        dtype=torch.float32,
                        device=self.device,
                    ),
                    shape=torch.Size([self.n_agents]),
                )
            },
            shape=torch.Size([]),
        )

        # Rewards: scalar per agent
        self.reward_spec = Composite(
            {
                g: Composite(
                    reward=Unbounded(
                        shape=torch.Size([self.n_agents, 1]),
                        dtype=torch.float32,
                        device=self.device,
                    ),
                    shape=torch.Size([self.n_agents]),
                )
            },
            shape=torch.Size([]),
        )

    # ------------------------------------------------------------------
    # Core environment methods
    # ------------------------------------------------------------------

    def _reset(self, tensordict=None):
        self._step_count = 0

        # Randomise agent and target positions in [-1, 1]^2
        self._agent_pos = torch.rand(
            self.n_agents, 2, device=self.device
        ) * 2 - 1
        self._target_pos = torch.rand(
            self.n_agents, 2, device=self.device
        ) * 2 - 1

        obs = torch.cat([self._agent_pos, self._target_pos], dim=-1)

        return TensorDict(
            {
                self.AGENTS_GROUP: TensorDict(
                    {"observation": obs},
                    batch_size=[self.n_agents],
                ),
                "done": torch.zeros(1, dtype=torch.bool, device=self.device),
                "terminated": torch.zeros(
                    1, dtype=torch.bool, device=self.device
                ),
            },
            batch_size=[],
        )

    def _step(self, tensordict):
        action = tensordict[self.AGENTS_GROUP, "action"]

        # Move each agent (clamp to arena bounds)
        self._agent_pos = (self._agent_pos + action * 0.1).clamp(-1.0, 1.0)

        # Reward = negative L2 distance to target
        distances = torch.norm(
            self._agent_pos - self._target_pos, dim=-1, keepdim=True
        )
        rewards = -distances

        self._step_count += 1
        done = torch.tensor(
            [self._step_count >= self.max_steps],
            dtype=torch.bool,
            device=self.device,
        )

        obs = torch.cat([self._agent_pos, self._target_pos], dim=-1)

        return TensorDict(
            {
                self.AGENTS_GROUP: TensorDict(
                    {"observation": obs, "reward": rewards},
                    batch_size=[self.n_agents],
                ),
                "done": done,
                "terminated": done.clone(),
            },
            batch_size=[],
        )

    def _set_seed(self, seed):
        torch.manual_seed(seed)

    # ------------------------------------------------------------------
    # Helper used by the BenchMARL TaskClass
    # ------------------------------------------------------------------

    @property
    def group_map(self):
        """Maps group name -> list of agent names."""
        return {
            self.AGENTS_GROUP: [f"agent_{i}" for i in range(self.n_agents)]
        }
