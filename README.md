# RoboNex Walking

## Setup
```bash
# Example
cd ~/humanoid_project
git clone https://github.com/Humanoid-Project/robonex-walking.git
git clone https://github.com/Humanoid-Project/robonex-description.git
git clone https://github.com/Humanoid-Project/robonex-common.git
source ./robonex-common/setup/setup_isaacsim.sh
python -m pip install -e ./robonex-walking/source/robonex_walking
```

Runs on the `isaacsim` conda env, not a `.venv`. `robonex-common` is pinned in
`source/robonex_walking/setup.py` — see [`robonex-common/setup/SETUP.md`](https://github.com/Humanoid-Project/robonex-common/blob/main/setup/SETUP.md).

| Variable | Required | Default | Description |
| --- | :---: | --- | --- |
| `ROBONEX_DESCRIPTION_ROOT` | No | Sibling `robonex-description` | Description checkout |

<br>

## Structure

```text
robonex-walking/
├── README.md
├── scripts/
│   ├── list_envs.py
│   ├── zero_agent.py
│   ├── random_agent.py
│   ├── export_policy_manifest.py
│   └── rsl_rl/
│       ├── train.py
│       ├── play.py
│       ├── cli_args.py
│       ├── eval_commands.py
│       ├── sim_joint_trace.py
│       ├── export_actor_cpu.py
│       └── eval/judge.py
├── tests/
│   └── pre-train.sh
└── source/robonex_walking/
```

| Task | Model |
| --- | --- |
| `RoboNex-Walking-V2-Edu-v0` | `robonex-description` ver2 closed-loop mesh USD |

<br>

## Test

### `tests/pre-train.sh`

Runs every test in `tests/` with pytest; extra arguments go to pytest.

```bash
# Example
cd ~/humanoid_project/robonex-walking
conda activate isaacsim

./tests/pre-train.sh
./tests/pre-train.sh -k slew
```

<br>

## Train

### `scripts/rsl_rl/train.py`

| Command | Option | Default | Description |
| --- | --- | --- | --- |
| - | `--task` | - | Gym task id |
| - | `--num_envs` | cfg (`512`) | Parallel environments |
| - | `--max_iterations` | cfg | PPO iterations |
| - | `--seed` | cfg | Environment seed |
| - | `--agent` | `rsl_rl_cfg_entry_point` | Agent config entry point |
| - | `--resume` | Off | Resume from a checkpoint |
| - | `--load_run` | - | Run folder to resume |
| - | `--checkpoint` | - | Checkpoint file to resume |
| - | `--experiment_name` | - | Parsed but not applied; the task config sets the log folder |
| - | `--run_name` | - | Run-name suffix |
| - | `--logger` | - | `wandb`, `tensorboard`, or `neptune` |
| - | `--log_project_name` | - | wandb / neptune project |
| - | `--video` | Off | Record videos |
| - | `--video_length` | `200` | Recorded steps |
| - | `--video_interval` | `2000` | Steps between recordings |
| - | `--distributed` | Off | Multi-GPU / multi-node training |
| - | `--export_io_descriptors` | Off | Export IO descriptors |

Logs go to `logs/rsl_rl/robonex_walking_v2_edu/<date>_<run_name>/`.

```bash
# Example
cd ~/humanoid_project/robonex-walking
conda activate isaacsim

python scripts/rsl_rl/train.py \
  --task RoboNex-Walking-V2-Edu-v0 \
  --num_envs 4096 \
  --max_iterations 1000 \
  --seed 42 \
  --run_name example \
  --headless
```

<br>

## Play

### `scripts/rsl_rl/play.py`

| Command | Option | Default | Description |
| --- | --- | --- | --- |
| - | `--task` | - | Gym task id |
| - | `--num_envs` | cfg | Parallel environments |
| - | `--seed` | cfg | Environment seed |
| - | `--agent` | `rsl_rl_cfg_entry_point` | Agent config entry point |
| - | `--load_run` | Latest | Run folder to load |
| - | `--checkpoint` | Latest | Checkpoint file |
| - | `--export_only` | Off | Export the policy and exit |
| - | `--real-time` | Off | Pace playback to wall clock |
| - | `--video` | Off | Record a video |
| - | `--video_length` | `200` | Recorded steps |
| - | `--disable_fabric` | Off | Use USD I/O instead of fabric |
| - | `--use_pretrained_checkpoint` | Off | Pre-trained checkpoint from Nucleus |

```bash
# Example
python scripts/rsl_rl/play.py \
  --task RoboNex-Walking-V2-Edu-v0 \
  --num_envs 1 \
  --load_run 2026-10-15_00-00-00_example

# Export only
python scripts/rsl_rl/play.py \
  --task RoboNex-Walking-V2-Edu-v0 \
  --num_envs 1 \
  --headless \
  --export_only \
  --checkpoint logs/rsl_rl/robonex_walking_v2_edu/2026-10-15_00-00-00_example/model_999.pt
```

| Output | Description |
| --- | --- |
| `<run>/exported/policy.onnx` | ONNX policy with observation normalizer |
| `<run>/exported/policy.pt` | TorchScript policy with observation normalizer |

<br>

## Evaluate

### `scripts/rsl_rl/eval_commands.py`

Evaluates one checkpoint on a fixed grid of velocity commands (`fwd_slow`, `fwd_max`, `fwd_goal`, `stop`, `back`,
`turn_*`, `strafe_*`, `diag*`).

| Command | Option | Default | Description |
| --- | --- | --- | --- |
| - | `--checkpoint` | `Required` | Checkpoint file |
| - | `--task` | `RoboNex-Walking-V2-Edu-v0` | Gym task id |
| - | `--num_envs` | `512` | Parallel environments |
| - | `--seed` | `0` | Environment seed |
| - | `--warmup_steps` | `120` | Steps before measuring |
| - | `--measure_steps` | `880` | Measured steps |
| - | `--cells` | All | Comma-separated command cells |
| - | `--out` | - | Result JSON path |
| - | `--training_cfg` | Off | Build the env from the checkpoint's `params/env.yaml` |
| - | `--strict` | Off | Exit when the env disagrees with the checkpoint's `params/env.yaml` |
| - | `--slew_limit` | Off | Deploy slew limiter (`6 rad/s`, `120 rad/s²`) |
| - | `--no_push` | Off | Disable push events |
| - | `--push_interval` | Training | Push interval `LOW HIGH` (s) |
| - | `--hip_yaw_kp_nominal` | Off | Nominal hip-yaw stiffness |
| - | `--gravity_bias_off` | Off | Zero the projected-gravity bias |
| - | `--obs_delay` | `0` | Joint observation delay (steps) |
| - | `--obs_delay_pin` | `-1` | Fix a trained random joint observation delay (steps) |
| - | `--actuation_delay` | Training | Actuation delay `LOW HIGH` (physics substeps) |
| - | `--joint_friction` | Training | Coulomb joint friction `LOW HIGH` (N·m) |
| - | `--floor_friction` | Training | Foot-floor friction |
| - | `--com_pin` | `0 0 0` | Base CoM offset `X Y Z` (m) |
| - | `--com_keep_training` | Off | Keep the training CoM randomization |
| - | `--load_pin` | No load | Base load `FX FY FZ TX TY TZ` (N, N·m) |
| - | `--load_keep_training` | Off | Keep the training base-load randomization |
| - | `--deploy_overspeed` | `10.0` | Deploy-stop joint speed (rad/s) |
| - | `--deploy_max_error_deg` | `25.0` | Deploy-stop tracking error (deg) |
| - | `--deploy_max_tilt_deg` | `40.0` | Deploy-stop trunk tilt (deg) |
| - | `--deploy_limit_margin` | `0.05` | Deploy-stop joint limit margin (rad) |

```bash
# Example
python scripts/rsl_rl/eval_commands.py \
  --checkpoint logs/rsl_rl/robonex_walking_v2_edu/2026-10-15_00-00-00_example/model_999.pt \
  --training_cfg \
  --slew_limit \
  --out eval/example.json \
  --headless
```

<br>

### `scripts/rsl_rl/eval/judge.py`

Checks an `eval_commands.py` JSON against the walking acceptance criteria and prints `PASS` or `FAIL`.

```bash
# Example
python scripts/rsl_rl/eval/judge.py eval/example.json
```

<br>

### `scripts/rsl_rl/sim_joint_trace.py`

Records a per-step joint trace in the same column layout as the hardware telemetry.

| Command | Option | Default | Description |
| --- | --- | --- | --- |
| - | `--checkpoint` | `Required` | Checkpoint file |
| - | `--out` | `Required` | Trace CSV path |
| - | `--task` | `RoboNex-Walking-V2-Edu-v0` | Gym task id |
| - | `--num_envs` | `64` | Parallel environments |
| - | `--env_index` | `0` | Env whose trace is written |
| - | `--seed` | `0` | Environment seed |
| - | `--vx` | `0.0` | Forward command (m/s) |
| - | `--vy` | `0.0` | Lateral command (m/s) |
| - | `--wz` | `0.0` | Yaw-rate command (rad/s) |
| - | `--scenario` | - | Command schedule `t:vx,vy,wz;...`; replaces `--vx/--vy/--wz` |
| - | `--warmup_steps` | `150` | Steps before recording |
| - | `--measure_steps` | `1650` | Recorded steps |
| - | `--registry_cfg` | Off | Use the current task config instead of the checkpoint's |
| - | `--keep_randomization` | Off | Keep domain randomization and pushes |
| - | `--slew_limit` | Off | Deploy slew limiter (`6 rad/s`, `120 rad/s²`) |
| - | `--set` | - | Override a float in the env config (`key=value`, repeatable) |
| - | `--obs_delay` | `0` | Joint observation delay (steps) |
| - | `--obs_delay_pin` | `-1` | Fix a trained random joint observation delay (steps) |
| - | `--actuation_delay` | Config | Actuation delay `LOW HIGH` (physics substeps) |
| - | `--com_pin` | `0 0 0` | Base CoM offset `X Y Z` (m) |
| - | `--com_keep_training` | Off | Keep the config CoM randomization |
| - | `--load_pin` | No load | Base load `FX FY FZ TX TY TZ` (N, N·m) |
| - | `--load_keep_training` | Off | Keep the config base-load randomization |

```bash
# Example
python scripts/rsl_rl/sim_joint_trace.py \
  --checkpoint logs/rsl_rl/robonex_walking_v2_edu/2026-10-15_00-00-00_example/model_999.pt \
  --scenario "0:0,0,0;10:0.3,0,0;20:0,0,0" \
  --slew_limit \
  --out trace/example.csv \
  --headless
```

| Output | Description |
| --- | --- |
| `<out>` | Per-step joint trace CSV |
| `<out>.meta.json` | Command, delays and config of the trace |

<br>

## Export

### `scripts/rsl_rl/export_actor_cpu.py`

Exports the actor and its observation normalizer to ONNX on the CPU, without Isaac Sim.

| Command | Option | Default | Description |
| --- | --- | --- | --- |
| - | `checkpoint` | `Required` | Checkpoint file |
| - | `--output` | `Required` | ONNX path |
| - | `--eps` | `0.01` | Normalizer eps |

```bash
# Example
python scripts/rsl_rl/export_actor_cpu.py \
  logs/rsl_rl/robonex_walking_v2_edu/2026-10-15_00-00-00_example/model_999.pt \
  --output exported/policy.onnx
```

<br>

### `scripts/export_policy_manifest.py`

| Command | Option | Default | Description |
| --- | --- | --- | --- |
| - | `policy` | `Required` | ONNX policy path |
| - | `--checkpoint` | `Required` | Checkpoint the ONNX came from |
| - | `--output` | `<policy_dir>/policy_manifest.json` | Manifest path |
| - | `--description-root` | Sibling checkout | `robonex-description` path |
| - | `--common-root` | Sibling checkout | `robonex-common` path |
| - | `--description-model` | Auto-detected | Free-base scene of the model the checkpoint was trained on |
| - | `--robot-model` | Auto-detected | `robonex-common` robot profile; must match the checkpoint |
| - | `--task` | Auto-detected | Task id stored in the manifest |
| - | `--actions-from-checkpoint` | Off | Kept for older commands; the checkpoint's actions are always used |

```bash
# Example
python scripts/export_policy_manifest.py \
  logs/rsl_rl/robonex_walking_v2_edu/2026-10-15_00-00-00_example/exported/policy.onnx \
  --checkpoint logs/rsl_rl/robonex_walking_v2_edu/2026-10-15_00-00-00_example/model_999.pt \
  --description-model ver2/mujoco/robot/edu/scene.xml
```

| Output | Description |
| --- | --- |
| `policy_manifest.json` | Policy, model bundle, `robonex-common` and training-source fingerprints |
| `export_receipt.json` | Checkpoint and training slew stage of the export |

<br>

## Tools

### `scripts/list_envs.py`

| Command | Option | Default | Description |
| --- | --- | --- | --- |
| - | `--keyword` | - | Filter registered tasks |

```bash
# Example
python scripts/list_envs.py --keyword Walking
```

<br>

### `scripts/zero_agent.py`

Sends a zero normalized action, i.e. the midpoint of each clipped joint range.

| Command | Option | Default | Description |
| --- | --- | --- | --- |
| - | `--task` | - | Gym task id |
| - | `--num_envs` | cfg | Parallel environments |
| - | `--disable_fabric` | Off | Use USD I/O instead of fabric |

```bash
# Example
python scripts/zero_agent.py \
  --task RoboNex-Walking-V2-Edu-v0 \
  --num_envs 1
```

<br>

### `scripts/random_agent.py`

| Command | Option | Default | Description |
| --- | --- | --- | --- |
| - | `--task` | - | Gym task id |
| - | `--num_envs` | cfg | Parallel environments |
| - | `--disable_fabric` | Off | Use USD I/O instead of fabric |

```bash
# Example
python scripts/random_agent.py \
  --task RoboNex-Walking-V2-Edu-v0 \
  --num_envs 1
```

<br>

## License

Licensed under the [PolyForm Noncommercial License 1.0.0](LICENSE). Copyright 2026 RoboNex. Files from the Isaac Lab
template keep their BSD 3-Clause headers. See [NOTICE](NOTICE).
