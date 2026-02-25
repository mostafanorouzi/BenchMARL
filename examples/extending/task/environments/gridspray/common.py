#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

"""
BenchMARL Task enum and TaskClass for GridSprayEnv.

This module wires ``GridSprayEnvWrapper`` into BenchMARL's task system.

Steps to use
------------
1. Install the ``multiagentcoverage`` package (the one containing
   ``GridSprayEnv``).
2. Register ``GridSprayTask`` with BenchMARL at runtime (see
   ``run_grid_spray_env.py`` for a full example)::

       from benchmarl.environments import tasks as benchmarl_tasks
       from environments.gridspray.common import GridSprayTask
       benchmarl_tasks.append(GridSprayTask)

3. Load a task and run an experiment::

       task = GridSprayTask.GRID_SPRAY.get_from_yaml(path="conf/task/gridspray/grid_spray.yaml")
       experiment = Experiment(task=task, algorithm_config=..., ...)
       experiment.run()

Notes on ``state_fn`` and ``reward_fn``
----------------------------------------
``GridSprayEnv`` requires two *callable* arguments that are not serialisable
to YAML:
  - ``state_fn``  – computes per-agent observations from the env state
  - ``reward_fn`` – computes per-agent rewards from the env state

These must be supplied at Python level.  ``GridSprayClass`` imports them from
``multiagentcoverage`` by default; you can override ``get_env_fun`` to supply
your own functions.
"""

from typing import Callable, Dict, List, Optional

from benchmarl.environments.common import Task, TaskClass
from benchmarl.utils import DEVICE_TYPING

from tensordict import TensorDictBase

from torchrl.data import Composite
from torchrl.envs import EnvBase

from .grid_spray_wrapper import GridSprayEnvWrapper


class GridSprayTask(Task):
    """Enum of tasks for the GridSpray coverage environment.

    Each member maps to a YAML file under ``conf/task/gridspray/``.
    """

    GRID_SPRAY = None  # conf/task/gridspray/grid_spray.yaml

    @staticmethod
    def associated_class():
        return GridSprayClass


class GridSprayClass(TaskClass):
    """BenchMARL ``TaskClass`` for ``GridSprayEnv``.

    Implements all abstract methods required by BenchMARL.
    """

    def get_env_fun(
        self,
        num_envs: int,
        continuous_actions: bool,
        seed: Optional[int],
        device: DEVICE_TYPING,
    ) -> Callable[[], EnvBase]:
        """Build a factory function that returns a ``GridSprayEnvWrapper``.

        ``GridSprayEnv`` is not vectorised, so ``num_envs`` is ignored here –
        BenchMARL automatically wraps it in a ``SerialEnv``.

        The ``state_fn`` and ``reward_fn`` are imported from
        ``multiagentcoverage``.  Swap these imports for your own functions
        if you use different implementations.
        """
        # Import the user's environment and helper functions.
        # These come from the `multiagentcoverage` package that ships with
        # the GridSprayEnv project.
        try:
            from multiagentcoverage.envs.grid_spray_env import GridSprayEnv
            from multiagentcoverage.envs.states import state_fn
            from multiagentcoverage.envs.rewards import return_home_reward_fn
        except ImportError as exc:
            raise ImportError(
                "Could not import `multiagentcoverage`. "
                "Make sure the package is on your PYTHONPATH:\n"
                "    export PYTHONPATH=$PYTHONPATH:~/coverage_path_planning_marl\n"
                "or install it with `pip install -e .` from that directory."
            ) from exc

        cfg = self.config

        def _make():
            gym_env = GridSprayEnv(
                grid_size=cfg.get("grid_size", 10),
                num_agents=cfg.get("n_agents", 4),
                num_actions=cfg.get("num_actions", 5),
                spray_capacity=cfg.get("spray_capacity", 300),
                max_steps=cfg.get("max_steps", 100),
                coverage_size=cfg.get("coverage_size", 1),
                render=False,  # disable rendering during training
                state_fn=state_fn,
                reward_fn=return_home_reward_fn,
            )
            return GridSprayEnvWrapper(gym_env=gym_env, device=device)

        return _make

    def supports_continuous_actions(self) -> bool:
        # GridSprayEnv has discrete actions only (5 movement directions).
        return False

    def supports_discrete_actions(self) -> bool:
        return True

    def has_render(self, env: EnvBase) -> bool:
        # Rendering is disabled during training (render=False above).
        return False

    def max_steps(self, env: EnvBase) -> int:
        return self.config.get("max_steps", 100)

    def group_map(self, env: EnvBase) -> Dict[str, List[str]]:
        return env.group_map

    def observation_spec(self, env: EnvBase) -> Composite:
        return env.observation_spec.clone()

    def action_spec(self, env: EnvBase) -> Composite:
        return env.full_action_spec.clone()

    def state_spec(self, env: EnvBase) -> Optional[Composite]:
        # No shared global state tensor in this integration.
        # (GridSprayEnv does expose get_state() – see the README for how to
        # add it if you need centralised critics with a global state.)
        return None

    def action_mask_spec(self, env: EnvBase) -> Optional[Composite]:
        return None

    def info_spec(self, env: EnvBase) -> Optional[Composite]:
        return None

    @staticmethod
    def env_name() -> str:
        # Must match the folder name under benchmarl/conf/task/
        return "gridspray"

    @staticmethod
    def log_info(batch: TensorDictBase) -> Dict[str, float]:
        return {}
