#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

"""
End-to-end example: training MAPPO on the GridSprayEnv coverage environment.

Prerequisites
-------------
1. Install BenchMARL::

       pip install benchmarl   # or: pip install -e . from the repo root

2. Install the multiagentcoverage package::

       cd ~/coverage_path_planning_marl
       pip install -e .         # or: export PYTHONPATH=$PYTHONPATH:~/coverage_path_planning_marl

Run from the repository root::

    python examples/extending/task/run_grid_spray_env.py

What this script does
---------------------
1. Adds the local ``environments/`` folder to sys.path so Python can find
   the ``gridspray`` package.
2. Imports ``GridSprayTask`` and registers it with BenchMARL.
3. Loads the task YAML config.
4. Runs a short MAPPO training experiment.
"""

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# 1. Make the local ``environments/`` package importable.
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))

from environments.gridspray.common import GridSprayTask

# ---------------------------------------------------------------------------
# 2. Register GridSprayTask with BenchMARL.
#    This must happen BEFORE any hydra/task-registry usage.
# ---------------------------------------------------------------------------
from benchmarl.environments import tasks as benchmarl_tasks

if GridSprayTask not in benchmarl_tasks:
    benchmarl_tasks.append(GridSprayTask)

# ---------------------------------------------------------------------------
# 3. Load the task config from the local YAML file.
#    ``get_from_yaml`` reads the YAML and returns a GridSprayClass instance.
# ---------------------------------------------------------------------------
task = GridSprayTask.GRID_SPRAY.get_from_yaml(
    path=str(
        Path(__file__).parent
        / "conf"
        / "task"
        / "gridspray"
        / "grid_spray.yaml"
    )
)

print(f"Loaded task : {task}")
print(f"  grid_size     = {task.config['grid_size']}")
print(f"  n_agents      = {task.config['n_agents']}")
print(f"  max_steps     = {task.config['max_steps']}")

# ---------------------------------------------------------------------------
# 4. Set up and run a short MAPPO experiment.
# ---------------------------------------------------------------------------
from benchmarl.algorithms import MappoConfig
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models.mlp import MlpConfig

experiment_config = ExperimentConfig.get_from_yaml()
experiment_config.max_n_iters = 3
experiment_config.on_policy_collected_frames_per_batch = 200
experiment_config.on_policy_n_envs_per_worker = 1
experiment_config.evaluation = False
experiment_config.render = False
experiment_config.loggers = []

model_config = MlpConfig.get_from_yaml()
critic_model_config = MlpConfig.get_from_yaml()
algo_config = MappoConfig.get_from_yaml()

experiment = Experiment(
    algorithm_config=algo_config,
    model_config=model_config,
    critic_model_config=critic_model_config,
    seed=0,
    config=experiment_config,
    task=task,
)

experiment.run()
print("Training finished successfully!")
