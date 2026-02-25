#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

"""
End-to-end example: training MAPPO on a fully custom MARL environment.

Run from the repository root::

    python examples/extending/task/run_custom_env.py

The script:
1. Registers the custom environment with BenchMARL.
2. Loads task config from the local YAML file.
3. Creates an Experiment and runs a few iterations of MAPPO.

Prerequisites
-------------
    pip install benchmarl  # or: pip install -e .
"""

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# 1. Make sure BenchMARL can find the custom environment package.
#    (When you copy this pattern into your own project, adjust the path.)
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))

from environments.customenv.common import CustomEnvTask

# ---------------------------------------------------------------------------
# 2. Register the custom environment with BenchMARL.
#    Add the Task enum to the global ``tasks`` list so that the config
#    registry and Hydra pick it up automatically.
# ---------------------------------------------------------------------------
from benchmarl.environments import tasks as benchmarl_tasks

if CustomEnvTask not in benchmarl_tasks:
    benchmarl_tasks.append(CustomEnvTask)

# ---------------------------------------------------------------------------
# 3. Load a task from the local YAML file.
# ---------------------------------------------------------------------------
from benchmarl.environments.common import Task

task = CustomEnvTask.TASK_1.get_from_yaml(
    path=str(
        Path(__file__).parent / "conf" / "task" / "customenv" / "task_1.yaml"
    )
)

print(f"Loaded task: {task}")
print(f"  n_agents  = {task.config['n_agents']}")
print(f"  max_steps = {task.config['max_steps']}")

# ---------------------------------------------------------------------------
# 4. Build and run a short MAPPO experiment.
# ---------------------------------------------------------------------------
from benchmarl.algorithms import MappoConfig
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models.mlp import MlpConfig

experiment_config = ExperimentConfig.get_from_yaml()
# Keep the experiment short for this demo
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
