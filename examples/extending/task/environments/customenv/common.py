#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

"""
Example: integrating a fully custom MARL environment with BenchMARL.

This file defines two things:
  1. ``CustomEnvTask``  – the Task enum whose members enumerate the available tasks.
  2. ``CustomEnvClass`` – the TaskClass that knows how to build an EnvBase from
     a task name + YAML config.

The concrete environment is ``SimpleMultiAgentEnv`` (see ``my_custom_env.py``),
which is built from scratch on top of TorchRL's ``EnvBase``.
If you already have a PettingZoo-compatible environment you can wrap it with
``torchrl.envs.PettingZooWrapper`` instead – the TaskClass stays the same.
"""

from typing import Callable, Dict, List, Optional

from benchmarl.environments.common import Task, TaskClass
from benchmarl.utils import DEVICE_TYPING

from tensordict import TensorDictBase

from torchrl.data import Composite
from torchrl.envs import EnvBase

# Import the custom environment defined in my_custom_env.py
from .my_custom_env import SimpleMultiAgentEnv


class CustomEnvTask(Task):
    """Enum of tasks for the custom environment.

    Each member corresponds to a YAML file under ``conf/task/customenv/``.
    Add one ``TASK_NAME = None`` line per task you want to expose.
    """

    TASK_1 = None  # config loaded from conf/task/customenv/task_1.yaml
    TASK_2 = None  # config loaded from conf/task/customenv/task_2.yaml

    @staticmethod
    def associated_class():
        return CustomEnvClass


class CustomEnvClass(TaskClass):
    """TaskClass for the custom MARL environment.

    Implements all abstract methods required by BenchMARL.
    """

    def get_env_fun(
        self,
        num_envs: int,
        continuous_actions: bool,
        seed: Optional[int],
        device: DEVICE_TYPING,
    ) -> Callable[[], EnvBase]:
        # ``SimpleMultiAgentEnv`` is not vectorised, so ``num_envs`` is
        # ignored here – BenchMARL will automatically wrap it in a SerialEnv.
        # If your environment supports batching, pass ``num_envs`` through.
        return lambda: SimpleMultiAgentEnv(
            n_agents=self.config.get("n_agents", 3),
            max_steps=self.config.get("max_steps", 100),
            device=device,
        )

    def supports_continuous_actions(self) -> bool:
        # SimpleMultiAgentEnv has continuous velocity actions.
        return True

    def supports_discrete_actions(self) -> bool:
        # SimpleMultiAgentEnv does not have discrete actions.
        return False

    def has_render(self, env: EnvBase) -> bool:
        # SimpleMultiAgentEnv has no graphical renderer.
        return False

    def max_steps(self, env: EnvBase) -> int:
        # Maximum rollout length for evaluation.
        return self.config.get("max_steps", 100)

    def group_map(self, env: EnvBase) -> Dict[str, List[str]]:
        # Use the group_map defined on the environment itself.
        return env.group_map

    def observation_spec(self, env: EnvBase) -> Composite:
        # Return the observation part of the full observation spec.
        # (Here observation_spec already contains only "observation" keys.)
        return env.observation_spec.clone()

    def action_spec(self, env: EnvBase) -> Composite:
        # Full action spec – one (group_name, "action") entry per group.
        return env.full_action_spec.clone()

    def state_spec(self, env: EnvBase) -> Optional[Composite]:
        # No global state in this environment.
        return None

    def action_mask_spec(self, env: EnvBase) -> Optional[Composite]:
        # No action masks in this environment.
        return None

    def info_spec(self, env: EnvBase) -> Optional[Composite]:
        # No extra info in this environment.
        return None

    @staticmethod
    def env_name() -> str:
        # Must match the folder name under benchmarl/conf/task/
        return "customenv"

    @staticmethod
    def log_info(batch: TensorDictBase) -> Dict[str, float]:
        # Optionally return additional metrics to log each iteration.
        return {}
