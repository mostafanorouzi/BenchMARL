#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

"""
TorchRL EnvBase wrapper for GridSprayEnv.

``GridSprayEnv`` is a Gymnasium-based multi-agent coverage-path-planning
environment.  Its API returns plain Python / NumPy objects:

    obs_tuple, rewards, done, truncated, info = env.step(actions)
    obs_tuple, info                           = env.reset()

where
  - ``obs_tuple``  is a tuple/list of NumPy arrays, one per agent
  - ``rewards``    is a NumPy array of shape (n_agents,)
  - ``done``       is a single bool (True when all agents are done)
  - ``actions``    is a list of ints, one per agent

TorchRL / BenchMARL expect environments to speak TensorDict and use
spec-based APIs.  ``GridSprayEnvWrapper`` bridges the two worlds.

Usage
-----
    from multiagentcoverage.envs.grid_spray_env import GridSprayEnv
    from multiagentcoverage.envs.states import state_fn
    from multiagentcoverage.envs.rewards import return_home_reward_fn

    gym_env = GridSprayEnv(
        grid_size=10, num_agents=4, spray_capacity=300,
        max_steps=100, render=False,
        state_fn=state_fn, reward_fn=return_home_reward_fn,
    )
    torchrl_env = GridSprayEnvWrapper(gym_env, device="cpu")

    td = torchrl_env.reset()
    td["agents", "action"] = torchrl_env.action_spec.rand()
    td = torchrl_env.step(td)
"""

import numpy as np
import torch
from tensordict import TensorDict
from torchrl.data import Categorical, Composite, Unbounded
from torchrl.envs import EnvBase


class GridSprayEnvWrapper(EnvBase):
    """TorchRL ``EnvBase`` wrapper around ``GridSprayEnv``.

    All agents belong to a single group called ``"agents"``.

    Observation per agent : 1-D float32 vector of length ``obs_dim``
                            (determined at construction time from the env)
    Action per agent      : integer in ``[0, num_actions)`` (discrete)
    Reward per agent      : scalar float32

    Args:
        gym_env: A pre-constructed and freshly-reset ``GridSprayEnv``
                 instance.  The wrapper takes ownership of it.
        device (str): Torch device to place tensors on (default: ``"cpu"``).
    """

    AGENTS_GROUP = "agents"

    def __init__(self, gym_env, device="cpu"):
        super().__init__(device=device, batch_size=[])

        self._gym_env = gym_env
        self.n_agents = gym_env.n_agents
        self.num_actions = gym_env.num_actions  # 5 for GridSprayEnv

        # Determine per-agent observation dimension by calling reset once.
        obs_tuple, _ = gym_env.reset()
        self._obs_dim = int(np.asarray(obs_tuple[0]).shape[0])

        self._make_specs()

    # ------------------------------------------------------------------
    # Spec definitions
    # ------------------------------------------------------------------

    def _make_specs(self):
        g = self.AGENTS_GROUP

        # Per-agent 1-D float observation
        self.observation_spec = Composite(
            {
                g: Composite(
                    observation=Unbounded(
                        shape=torch.Size([self.n_agents, self._obs_dim]),
                        dtype=torch.float32,
                        device=self.device,
                    ),
                    shape=torch.Size([self.n_agents]),
                )
            },
            shape=torch.Size([]),
        )

        # Per-agent categorical (discrete) action
        # ``Categorical(n=k)`` stores actions as integers in [0, k-1]
        self.action_spec = Composite(
            {
                g: Composite(
                    action=Categorical(
                        n=self.num_actions,
                        shape=torch.Size([self.n_agents]),
                        dtype=torch.int64,
                        device=self.device,
                    ),
                    shape=torch.Size([self.n_agents]),
                )
            },
            shape=torch.Size([]),
        )

        # Per-agent scalar reward
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
        obs_tuple, _ = self._gym_env.reset()

        # Stack per-agent observations into a single tensor [n_agents, obs_dim]
        obs = torch.as_tensor(
            np.stack([np.asarray(o, dtype=np.float32) for o in obs_tuple]),
            dtype=torch.float32,
        ).to(self.device)

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
        # Extract actions: shape [n_agents] of int64, convert to Python list
        actions = tensordict[self.AGENTS_GROUP, "action"].cpu().numpy().tolist()

        obs_tuple, rewards_raw, done, _truncated, _info = self._gym_env.step(
            actions
        )

        # Observations
        obs = torch.as_tensor(
            np.stack([np.asarray(o, dtype=np.float32) for o in obs_tuple]),
            dtype=torch.float32,
        ).to(self.device)

        # Rewards: shape [n_agents, 1]
        rewards = torch.as_tensor(
            np.asarray(rewards_raw, dtype=np.float32),
            dtype=torch.float32,
        ).unsqueeze(-1).to(self.device)

        done_t = torch.tensor(
            [bool(done)], dtype=torch.bool, device=self.device
        )

        return TensorDict(
            {
                self.AGENTS_GROUP: TensorDict(
                    {"observation": obs, "reward": rewards},
                    batch_size=[self.n_agents],
                ),
                "done": done_t,
                "terminated": done_t.clone(),
            },
            batch_size=[],
        )

    def _set_seed(self, seed):
        # GridSprayEnv uses random.seed / np.random internally;
        # we reset with the seed to set the RNG state.
        self._gym_env.reset(seed=seed)

    # ------------------------------------------------------------------
    # Helper used by the BenchMARL TaskClass
    # ------------------------------------------------------------------

    @property
    def group_map(self):
        """Maps group name -> list of agent names."""
        return {
            self.AGENTS_GROUP: [
                f"agent_{i + 1}" for i in range(self.n_agents)
            ]
        }
