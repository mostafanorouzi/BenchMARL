# Creating a new task

In the following we will see how to:
1. Create new tasks from a **new** (custom) environment
2. Create new tasks from an **existing** environment

---

## Creating new tasks from a new environment

Below are the steps to create a new task and a new environment.

### Step 1 — Implement your custom TorchRL environment

BenchMARL requires environments to be
[`torchrl.envs.EnvBase`](https://pytorch.org/rl/stable/reference/envs.html#torchrl.envs.EnvBase)
objects.

If your environment is **not yet wrapped in TorchRL** you have two options:

| Your env type | Recommended wrapper |
|---|---|
| PettingZoo parallel env | [`torchrl.envs.PettingZooWrapper`](https://pytorch.org/rl/stable/reference/generated/torchrl.envs.PettingZooWrapper.html) |
| Any custom env | Subclass `torchrl.envs.EnvBase` directly |

A complete, working example of a custom `EnvBase` environment is provided in
[`environments/customenv/my_custom_env.py`](environments/customenv/my_custom_env.py).
It implements a simple multi-agent navigation task where N agents each navigate
to their own target. Read the file for the full implementation and comments.

Key points when subclassing `EnvBase`:
- Set `self.observation_spec`, `self.action_spec`, and `self.reward_spec` in `__init__`.
- Implement `_reset(tensordict)` returning the initial observation `TensorDict`.
- Implement `_step(tensordict)` returning the next obs / reward / done `TensorDict`.
- Implement `_set_seed(seed)` for reproducibility.
- Structure specs with a **group name** (e.g., `"agents"`) at the top level so
  that BenchMARL can identify agent groups.

```python
# Minimal spec structure (one group called "agents", n_agents agents)
from torchrl.data import Bounded, Composite, Unbounded

self.observation_spec = Composite(
    agents=Composite(
        observation=Unbounded(shape=[n_agents, obs_dim]),
        shape=[n_agents],
    ),
    shape=[],
)
self.action_spec = Composite(
    agents=Composite(
        action=Bounded(low=-1.0, high=1.0, shape=[n_agents, act_dim]),
        shape=[n_agents],
    ),
    shape=[],
)
```

### Step 2 — Create `CustomEnvTask` and `CustomEnvClass`

Create your `CustomEnvTask` (a `Task` enum) and `CustomEnvClass` (a `TaskClass`)
following [`environments/customenv/common.py`](environments/customenv/common.py).

```python
class CustomEnvTask(Task):
    TASK_1 = None   # config loaded from conf/task/customenv/task_1.yaml
    TASK_2 = None

    @staticmethod
    def associated_class():
        return CustomEnvClass

class CustomEnvClass(TaskClass):
    def get_env_fun(self, num_envs, continuous_actions, seed, device):
        return lambda: MyCustomEnv(
            n_agents=self.config["n_agents"],
            max_steps=self.config["max_steps"],
            device=device,
        )
    # … implement all other abstract methods …
```

### Step 3 — Add YAML configs

Create a `conf/task/customenv/` folder with one YAML file per task.
See [`conf/task/customenv/task_1.yaml`](conf/task/customenv/task_1.yaml) for an example:

```yaml
n_agents: 3
max_steps: 100
```

### Step 4 — Place files in BenchMARL (or keep them external)

**Option A – Inside BenchMARL** (permanent integration):
- Copy your `common.py` to `benchmarl/environments/customenv/common.py`.
- Copy your configs to `benchmarl/conf/task/customenv/`.
- Add `CustomEnvTask` to the `tasks` list in `benchmarl/environments/__init__.py`.

**Option B – External** (no changes to the library):
- Keep your files anywhere on your Python path.
- Register the env at runtime before creating the experiment:

```python
from benchmarl.environments import tasks as benchmarl_tasks
from my_project.customenv import CustomEnvTask

benchmarl_tasks.append(CustomEnvTask)
```

### Step 5 — Run

```bash
# Option A (env registered inside BenchMARL):
python benchmarl/run.py task=customenv/task_1 algorithm=mappo

# Option B (external registration, see run_custom_env.py):
python examples/extending/task/run_custom_env.py
```

A complete runnable script for Option B is provided in
[`run_custom_env.py`](run_custom_env.py).

### Step 6 (Optional) — Add a TaskConfig dataclass

Add a Python dataclass per task to enable typed config validation.
See [`environments/customenv/task_1.py`](environments/customenv/task_1.py):

```python
from dataclasses import dataclass, MISSING

@dataclass
class TaskConfig:
    n_agents: int = MISSING
    max_steps: int = MISSING
```

Then reference it in the YAML defaults:

```yaml
defaults:
  - customenv_task_1_config   # name = {env}_{task}_config
  - _self_

n_agents: 3
max_steps: 100
```

---

## Creating new tasks from an existing environment

If `customenv` with `task_1` and `task_2` already exists and you only want to
add `task_3`:

1. Add `TASK_3 = None` to `CustomEnvTask` in `environments/customenv/common.py`.
2. Add `conf/task/customenv/task_3.yaml`.
3. *(Optional)* Add `environments/customenv/task_3.py` with a `TaskConfig`
   dataclass and reference it in the YAML defaults.

---

## PettingZoo example

PR [#84](https://github.com/facebookresearch/BenchMARL/pull/84) contains an
example of adding your own PettingZoo task.

