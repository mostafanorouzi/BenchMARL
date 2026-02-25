# Creating a new task

In the following we will see how to:
1. Create new tasks from a **new** (custom) environment
2. Create new tasks from an **existing** environment
3. Wrap a **Gymnasium-based MARL environment** (e.g. `GridSprayEnv`)

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
| Custom Gymnasium-style MARL env | Subclass `torchrl.envs.EnvBase` directly (see below) |
| Built from scratch | Subclass `torchrl.envs.EnvBase` directly |

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

## Wrapping a Gymnasium-based MARL environment (GridSprayEnv example)

Many existing MARL environments expose a Gymnasium-style API where
all agents share a single `step()` call and observations/actions/rewards
are returned as tuples or arrays:

```python
# Typical Gymnasium-based MARL env interface
obs_tuple, rewards, done, truncated, info = env.step(actions)
obs_tuple, info = env.reset()
```

This pattern does **not** match TorchRL's `EnvBase` or PettingZoo's parallel
API, so you need a thin **wrapper class** to bridge them.

The [`GridSprayEnv`](https://github.com/mostafanorouzi/coverage_path_planning_marl)
environment (a coverage-path-planning scenario) is a concrete example of this
pattern:

| Property | Value |
|---|---|
| `observation_space` | `spaces.Tuple` — one `spaces.Box` per agent |
| `action_space` | `spaces.MultiDiscrete([5] * n_agents)` |
| `step()` returns | `(obs_tuple, rewards_array, all_done_bool, False, {})` |
| `reset()` returns | `(obs_tuple, {})` |

### Files for the GridSprayEnv integration

```
examples/extending/task/
├── environments/gridspray/
│   ├── __init__.py
│   ├── grid_spray_wrapper.py   ← TorchRL EnvBase wrapper
│   └── common.py               ← BenchMARL TaskClass + Task enum
├── conf/task/gridspray/
│   └── grid_spray.yaml         ← YAML config
└── run_grid_spray_env.py       ← end-to-end training script
```

### How the wrapper works

[`environments/gridspray/grid_spray_wrapper.py`](environments/gridspray/grid_spray_wrapper.py)
wraps any pre-constructed `GridSprayEnv` in a TorchRL `EnvBase`:

```python
class GridSprayEnvWrapper(EnvBase):
    AGENTS_GROUP = "agents"

    def __init__(self, gym_env, device="cpu"):
        super().__init__(device=device, batch_size=[])
        self._gym_env = gym_env
        self.n_agents = gym_env.n_agents
        obs_tuple, _ = gym_env.reset()
        self._obs_dim = np.asarray(obs_tuple[0]).shape[0]
        self._make_specs()

    def _make_specs(self):
        g = self.AGENTS_GROUP
        # Observations: float vector per agent
        self.observation_spec = Composite(
            {g: Composite(
                observation=Unbounded(shape=[self.n_agents, self._obs_dim]),
                shape=[self.n_agents],
            )}, shape=[],
        )
        # Discrete actions: integer in [0, num_actions)
        self.action_spec = Composite(
            {g: Composite(
                action=Categorical(n=self.num_actions, shape=[self.n_agents]),
                shape=[self.n_agents],
            )}, shape=[],
        )
        # Per-agent scalar rewards
        self.reward_spec = Composite(
            {g: Composite(
                reward=Unbounded(shape=[self.n_agents, 1]),
                shape=[self.n_agents],
            )}, shape=[],
        )

    def _reset(self, tensordict=None):
        obs_tuple, _ = self._gym_env.reset()
        obs = torch.as_tensor(np.stack([np.asarray(o) for o in obs_tuple]))
        return TensorDict({
            self.AGENTS_GROUP: TensorDict({"observation": obs}, batch_size=[self.n_agents]),
            "done": torch.zeros(1, dtype=torch.bool),
            "terminated": torch.zeros(1, dtype=torch.bool),
        }, batch_size=[])

    def _step(self, tensordict):
        actions = tensordict[self.AGENTS_GROUP, "action"].cpu().numpy().tolist()
        obs_tuple, rewards_raw, done, _, _ = self._gym_env.step(actions)
        obs = torch.as_tensor(np.stack([np.asarray(o) for o in obs_tuple]))
        rewards = torch.as_tensor(np.asarray(rewards_raw)).unsqueeze(-1)  # [n_agents, 1]
        done_t = torch.tensor([bool(done)], dtype=torch.bool)
        return TensorDict({
            self.AGENTS_GROUP: TensorDict(
                {"observation": obs, "reward": rewards},
                batch_size=[self.n_agents],
            ),
            "done": done_t, "terminated": done_t.clone(),
        }, batch_size=[])
```

### How the TaskClass works

[`environments/gridspray/common.py`](environments/gridspray/common.py)
constructs the `GridSprayEnv`, wraps it, and connects it to BenchMARL:

```python
class GridSprayClass(TaskClass):
    def get_env_fun(self, num_envs, continuous_actions, seed, device):
        from multiagentcoverage.envs.grid_spray_env import GridSprayEnv
        from multiagentcoverage.envs.states import state_fn
        from multiagentcoverage.envs.rewards import return_home_reward_fn

        cfg = self.config
        def _make():
            gym_env = GridSprayEnv(
                grid_size=cfg["grid_size"], num_agents=cfg["n_agents"],
                max_steps=cfg["max_steps"], render=False,
                state_fn=state_fn, reward_fn=return_home_reward_fn,
            )
            return GridSprayEnvWrapper(gym_env=gym_env, device=device)
        return _make

    def supports_continuous_actions(self): return False
    def supports_discrete_actions(self):   return True
    # … other abstract methods …
```

### Running the GridSprayEnv experiment

```bash
# Make sure multiagentcoverage is importable:
export PYTHONPATH=$PYTHONPATH:~/coverage_path_planning_marl

# Run from the repo root:
python examples/extending/task/run_grid_spray_env.py
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

