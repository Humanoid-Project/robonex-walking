import argparse
import collections
import json
import math
import pickle
import re
import sys
import threading
import time
import zipfile
from pathlib import Path

import numpy as np

try:
    import yaml
except ImportError as error:
    raise SystemExit(f"Missing PyYAML ({error}); run with robonex-description/.venv/bin/python")

from robonex_common.foot_roll import clip_foot_roll
from robonex_common.paths import DESCRIPTION_REPO_NAMES, resolve_repo
from robonex_common.runtime import ObservationHistory, gait_phase_at

ISAAC_LEG_ARTICULATION_ORDER = (
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "left_knee_pitch_joint",
    "right_knee_pitch_joint",
    "left_ankle_lower_joint",
    "left_ankle_upper_joint",
    "right_ankle_lower_joint",
    "right_ankle_upper_joint",
)

EXPECTED_TERMS = (
    ("joint_pos_rel", "delayed_joint_state"),
    ("joint_vel_rel", "delayed_joint_state"),
    ("imu_ang_vel", "delayed_imu_ang_vel"),
    ("projected_gravity", "delayed_imu_projected_gravity"),
    ("velocity_commands", "generated_commands"),
    ("gait_phase", "gait_phase"),
    ("actions", "last_action"),
)
TERM_SIZES = {"joint_pos_rel": 12, "joint_vel_rel": 12, "imu_ang_vel": 3, "projected_gravity": 3,
              "velocity_commands": 3, "gait_phase": 2, "actions": 12}

NORMALIZER_EPS = 1.0e-2


class _Loader(yaml.SafeLoader):
    pass


def _tuple(loader, node):
    return tuple(loader.construct_sequence(node, deep=True))


def _apply(loader, suffix, node):
    args = loader.construct_sequence(node, deep=True) if isinstance(node, yaml.SequenceNode) else []
    if suffix == "builtins.slice":
        return slice(*args)
    return {"__apply__": suffix, "args": args}


_Loader.add_constructor("tag:yaml.org,2002:python/tuple", _tuple)
_Loader.add_multi_constructor("tag:yaml.org,2002:python/object/apply:", _apply)


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as stream:
        return yaml.load(stream, Loader=_Loader)


class _Stub:
    def __init__(self, *args, **kwargs):
        self.args = args

    def __setstate__(self, state):
        self.state = state

    def __call__(self, *args, **kwargs):
        return _Stub(*args)


_STORAGE_DTYPES = {
    "FloatStorage": np.float32, "DoubleStorage": np.float64, "HalfStorage": np.float16,
    "LongStorage": np.int64, "IntStorage": np.int32, "ShortStorage": np.int16,
    "CharStorage": np.int8, "ByteStorage": np.uint8, "BoolStorage": np.bool_,
    "float32": np.float32, "float64": np.float64, "float16": np.float16, "int64": np.int64,
    "int32": np.int32, "int16": np.int16, "int8": np.int8, "uint8": np.uint8, "bool": np.bool_,
}


def _rebuild_tensor_v2(storage, offset, size, stride, *unused):
    size = tuple(int(v) for v in size)
    if not size:
        return np.array(storage[offset])
    item = storage.dtype.itemsize
    view = np.lib.stride_tricks.as_strided(storage[offset:], shape=size, strides=tuple(int(s) * item for s in stride))
    return np.array(view)


class _TorchZipUnpickler(pickle.Unpickler):
    def __init__(self, stream, archive, prefix):
        super().__init__(stream)
        self.archive = archive
        self.prefix = prefix

    def find_class(self, module, name):
        if module == "torch._utils" and name == "_rebuild_tensor_v2":
            return _rebuild_tensor_v2
        if module == "torch" and name in _STORAGE_DTYPES:
            return name
        if module == "collections" and name == "OrderedDict":
            return collections.OrderedDict
        if module.startswith(("torch", "numpy", "rsl_rl", "isaaclab", "robonex")):
            return _Stub
        try:
            return super().find_class(module, name)
        except (ImportError, AttributeError):
            return _Stub

    def persistent_load(self, pid):
        _, storage_type, key, _, _ = pid
        dtype = _STORAGE_DTYPES[storage_type if isinstance(storage_type, str) else str(storage_type).split(".")[-1]]
        return np.frombuffer(self.archive.read(f"{self.prefix}/data/{key}"), dtype=dtype)


def load_state_dict(path):
    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None:
        checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
        return {k: v.detach().cpu().numpy() for k, v in checkpoint["model_state_dict"].items()}
    with zipfile.ZipFile(path) as archive:
        pkl = next(n for n in archive.namelist() if n.endswith("data.pkl"))
        prefix = pkl[: -len("/data.pkl")]
        with archive.open(pkl) as stream:
            checkpoint = _TorchZipUnpickler(stream, archive, prefix).load()
    return {k: np.asarray(v) for k, v in checkpoint["model_state_dict"].items()}


class ActorMLP:
    def __init__(self, state, agent_cfg, observation_size, action_size):
        policy = agent_cfg["policy"]
        if policy.get("class_name") != "ActorCritic":
            raise RuntimeError(f"unsupported policy class {policy.get('class_name')}")
        if policy.get("activation") != "elu":
            raise RuntimeError(f"unsupported activation {policy.get('activation')}")
        if policy.get("state_dependent_std"):
            raise RuntimeError("state_dependent_std actors are not supported")
        indices = sorted(int(k.split(".")[1]) for k in state if k.startswith("actor.") and k.endswith(".weight"))
        self.layers = [(state[f"actor.{i}.weight"].astype(np.float32), state[f"actor.{i}.bias"].astype(np.float32))
                       for i in indices]
        hidden = [w.shape[0] for w, _ in self.layers[:-1]]
        if hidden != list(policy["actor_hidden_dims"]):
            raise RuntimeError(f"actor hidden dims {hidden} differ from agent.yaml {policy['actor_hidden_dims']}")
        if self.layers[0][0].shape[1] != observation_size or self.layers[-1][0].shape[0] != action_size:
            raise RuntimeError(
                f"actor maps {self.layers[0][0].shape[1]} -> {self.layers[-1][0].shape[0]}, "
                f"expected {observation_size} -> {action_size}")
        self.normalize = bool(policy.get("actor_obs_normalization"))
        if self.normalize:
            self.mean = state["actor_obs_normalizer._mean"].reshape(-1).astype(np.float32)
            self.std = state["actor_obs_normalizer._std"].reshape(-1).astype(np.float32)

    def __call__(self, observation):
        x = np.asarray(observation, dtype=np.float32)
        if self.normalize:
            x = (x - self.mean) / (self.std + np.float32(NORMALIZER_EPS))
        for index, (weight, bias) in enumerate(self.layers):
            x = weight @ x + bias
            if index < len(self.layers) - 1:
                x = np.where(x > 0.0, x, np.expm1(np.minimum(x, 0.0))).astype(np.float32)
        return x


def _func_name(term):
    return str(term.get("func") or term.get("class_type") or "").split(":")[-1]


def resolve_leg_order(run_dir, constants):
    path = Path(run_dir) / "params" / "joint_order.json"
    if path.is_file():
        order, source = tuple(json.loads(path.read_text(encoding="utf-8"))["action_joint_names"]), str(path)
    else:
        order = tuple(json.loads(Path(constants).read_text(encoding="utf-8"))["policy_joint_order"])
        source = f"{constants} policy_joint_order"
    if order != ISAAC_LEG_ARTICULATION_ORDER:
        diff = [(i, a, e) for i, (a, e) in enumerate(zip(order, ISAAC_LEG_ARTICULATION_ORDER)) if a != e]
        raise RuntimeError(f"leg joint order from {source} {order} differs from the player's articulation order "
                           f"{ISAAC_LEG_ARTICULATION_ORDER} (index, got, expected: {diff or 'lengths differ'})")
    return order, source


def upper_armature_values(upper_joints, upper_train_gains):
    if not upper_train_gains:
        raise RuntimeError("--upper-armature isaac: the run's env.yaml has no upper_* actuator groups (a legs run); "
                           "there is no trained upper-body armature to copy")
    if not upper_joints:
        raise RuntimeError("--upper-armature isaac: the scene has no upper-body actuators")
    values = {}
    for name in upper_joints:
        groups = [g for g, (_, _, _, _, exprs) in upper_train_gains.items() if any(re.match(e + "$", name) for e in exprs)]
        if len(groups) != 1:
            raise RuntimeError(f"--upper-armature isaac: {name} matches upper actuator groups {groups}, expected one")
        values[name] = float(upper_train_gains[groups[0]][3])
    return values


class RunContract:
    def __init__(self, run_dir, leg_order=ISAAC_LEG_ARTICULATION_ORDER):
        self.run_dir = Path(run_dir)
        self.leg_order = tuple(leg_order)
        env = load_yaml(self.run_dir / "params" / "env.yaml")
        agent = load_yaml(self.run_dir / "params" / "agent.yaml")
        self.env, self.agent = env, agent
        self.sim_dt = float(env["sim"]["dt"])
        self.decimation = int(env["decimation"])
        self.policy_dt = self.sim_dt * self.decimation
        self.model_variant = env.get("model_variant")
        policy = env["observations"]["policy"]
        self.history_length = int(policy["history_length"])
        if not policy.get("flatten_history_dim", True) or not policy.get("concatenate_terms", True):
            raise RuntimeError("policy observation must be concatenated and history-flattened")
        terms = [(k, v) for k, v in policy.items() if isinstance(v, dict) and "func" in v]
        found = tuple((k, _func_name(v)) for k, v in terms)
        expected = tuple((name, func) for name, func in EXPECTED_TERMS)
        if tuple(n for n, _ in found) != tuple(n for n, _ in expected) or any(
                not f.startswith(e) for (_, f), (_, e) in zip(found, expected)):
            raise RuntimeError(f"policy observation terms {found} differ from the supported layout {expected}")
        for name, term in terms:
            if term.get("scale") is not None or term.get("clip") is not None or term.get("modifiers"):
                raise RuntimeError(f"observation term {name} has a scale/clip/modifier this player does not apply")
            if int(term.get("history_length", 0)) not in (0, self.history_length):
                raise RuntimeError(f"observation term {name} has its own history length")
        self.terms = [(name, TERM_SIZES[name]) for name, _ in terms]
        p = dict(terms)
        jp = p["joint_pos_rel"]["params"]
        if jp.get("field") != "pos" or p["joint_vel_rel"]["params"].get("field") != "vel":
            raise RuntimeError("joint_pos_rel/joint_vel_rel fields are not pos/vel")
        self.joint_delay_range = (int(jp.get("min_delay_steps", 0)), int(jp.get("max_delay_steps", 1)))
        imu = env["scene"]["imu"]
        self.imu_delay_range = (int(imu.get("min_delay_steps", 0)), int(imu.get("max_delay_steps", 0)))
        if imu["offset"]["rot"] != (1.0, 0.0, 0.0, 0.0) or not str(imu["prim_path"]).endswith("/base_link"):
            raise RuntimeError("IMU is not on base_link with identity rotation")
        grav = p["projected_gravity"]["params"]
        if float(grav.get("offset_deg", 0.0)) > 0.0 or float(grav.get("ou_sigma_deg", 0.0)) > 0.0:
            print("Note: training biased projected_gravity; the player feeds the unbiased value", flush=True)
        gait = p["gait_phase"]["params"]
        self.gait_period = float(gait["period"])
        self.gait_deadband = float(gait.get("command_deadband", 0.05))
        self.gait_masked = gait.get("command_name") is not None
        if env["events"].get("randomize_gait_clock", {}).get("params", {}).get("offset_range", (0.0, 0.0)) != (0.0, 0.0):
            print("Note: training randomised the gait-clock offset; the player starts at phase 0", flush=True)

        action = env["actions"]["joint_pos"]
        obs_names = tuple(jp["asset_cfg"]["joint_names"])
        act_names = tuple(action["joint_names"])
        self.joint_order = self._resolve_order(act_names, bool(action.get("preserve_order")))
        obs_order = self._resolve_order(obs_names, bool(jp["asset_cfg"].get("preserve_order")))
        vel_order = self._resolve_order(tuple(p["joint_vel_rel"]["params"]["asset_cfg"]["joint_names"]),
                                        bool(p["joint_vel_rel"]["params"]["asset_cfg"].get("preserve_order")))
        if obs_order != self.joint_order or vel_order != self.joint_order:
            raise RuntimeError(f"observation joint order {obs_order} differs from action order {self.joint_order}")
        if action.get("use_default_offset"):
            raise RuntimeError("use_default_offset=True is not supported")
        self.offsets = np.array([action["offset"][n] for n in self.joint_order], dtype=np.float64)
        self.scales = np.array([action["scale"][n] for n in self.joint_order], dtype=np.float64)
        self.clip_low = np.array([action["clip"][n][0] for n in self.joint_order], dtype=np.float64)
        self.clip_high = np.array([action["clip"][n][1] for n in self.joint_order], dtype=np.float64)
        self.slew_enabled = bool(action.get("slew_enabled", True))
        self.slew_max_speed = float(action.get("max_speed", 6.0))
        self.slew_max_accel = float(action.get("max_accel", 120.0))
        if action.get("actuation_delay_substeps") is not None:
            print(f"Note: training used actuation_delay_substeps={action['actuation_delay_substeps']}; "
                  "the player applies targets without that delay", flush=True)
        self.roll_limit = float(action.get("foot_roll_limit", 0.0))
        self.roll_coeffs = tuple(float(c) for c in action.get("foot_roll_coeffs", ()))
        self.roll_pairs = []
        for upper, lower, sign in action.get("foot_roll_pairs", ()):
            iu, il = self.joint_order.index(upper), self.joint_order.index(lower)
            bounds = []
            for i in (iu, il):
                lo, hi = sign * self.clip_low[i], sign * self.clip_high[i]
                bounds.append((float(min(lo, hi)), float(max(lo, hi))))
            self.roll_pairs.append((iu, il, float(sign), bounds[0], bounds[1]))
        self.runner_clip = agent.get("clip_actions")
        self.runner_clip = None if self.runner_clip is None else float(self.runner_clip)
        self.motor_model = {}
        self.train_gains = {}
        for group, cfg in env["scene"]["robot"]["actuators"].items():
            exprs = [re.compile(e + "$") for e in cfg["joint_names_expr"]]
            for name in self.joint_order:
                if any(e.match(name) for e in exprs):
                    self.motor_model[name] = group
                    kp, kd = cfg["stiffness"], cfg["damping"]
                    self.train_gains[name] = (float(kp[name] if isinstance(kp, dict) else kp),
                                              float(kd[name] if isinstance(kd, dict) else kd),
                                              float(cfg.get("effort_limit_sim") or 0.0),
                                              float(cfg.get("armature") or 0.0))
        self.upper_train_gains = {}
        for group, cfg in env["scene"]["robot"]["actuators"].items():
            if group.startswith("upper_"):
                self.upper_train_gains[group] = (float(cfg["stiffness"]), float(cfg["damping"]),
                                                 float(cfg.get("effort_limit_sim") or 0.0),
                                                 float(cfg.get("armature") or 0.0), tuple(cfg["joint_names_expr"]))
        self.feet = tuple(env.get("feet_body_names", ("left_ankle_pitch_link", "right_ankle_pitch_link")))
        self.minimum_height = float(env["terminations"]["fall_down"]["params"]["minimum_height"])
        illegal = str(env["scene"]["illegal_contacts"]["prim_path"]).split("/Robot/", 1)[-1]
        self.illegal_regex = re.compile(illegal + "$")
        limits = env["commands"]["base_velocity"].get("limit_ranges") or env["commands"]["base_velocity"]["ranges"]
        self.command_limits = tuple(tuple(float(v) for v in limits[k]) for k in ("lin_vel_x", "lin_vel_y", "ang_vel_z"))
        frictions = env["events"].get("randomize_joint_friction", {}).get("params", {}).get("friction_range", {})
        self.train_friction = {n: tuple(frictions[n]) for n in self.joint_order if n in frictions}
        self.observation_size = sum(s for _, s in self.terms) * self.history_length

    def _resolve_order(self, names, preserve_order):
        if preserve_order:
            return tuple(names)
        if set(names) != set(self.leg_order):
            raise RuntimeError(f"joint set {names} is not the 12 ver3 leg joints; articulation order unknown")
        return self.leg_order


def slew_limit_step(position, velocity, target, dt, max_speed, max_accel):
    error = target - position
    braking = np.maximum(np.sqrt((max_accel * dt) ** 2 + 2.0 * max_accel * np.abs(error)) - max_accel * dt, 0.0)
    desired = np.sign(error) * np.minimum(braking, max_speed)
    change = np.clip(desired - velocity, -max_accel * dt, max_accel * dt)
    next_velocity = np.clip(velocity + change, -max_speed, max_speed)
    next_position = position + next_velocity * dt
    arrived = error * (target - next_position) <= 0.0
    next_position = np.where(arrived, target, next_position)
    next_velocity = np.where(arrived, 0.0, next_velocity)
    return next_position, next_velocity


class ActionPipeline:
    def __init__(self, contract, slew):
        self.c = contract
        self.slew = slew
        self.reset()

    def reset(self):
        self.slew_position = self.c.offsets.copy()
        self.slew_velocity = np.zeros(12)
        self.calls = 0
        self.runner_clipped = 0
        self.target_clipped = 0
        self.roll_clipped = 0

    def _clip_roll(self, targets):
        targets = targets.copy()
        changed = False
        for iu, il, sign, upper_range, lower_range in self.c.roll_pairs:
            upper, lower = clip_foot_roll(sign * float(targets[iu]), sign * float(targets[il]),
                                          upper_range, lower_range, self.c.roll_coeffs, self.c.roll_limit)
            upper, lower = sign * float(upper), sign * float(lower)
            changed |= upper != targets[iu] or lower != targets[il]
            targets[iu], targets[il] = upper, lower
        return targets, changed

    def apply(self, raw):
        raw = np.asarray(raw, dtype=np.float64)
        if not np.isfinite(raw).all():
            raise RuntimeError("policy output is not finite")
        clipped = raw if self.c.runner_clip is None else np.clip(raw, -self.c.runner_clip, self.c.runner_clip)
        scaled = clipped * self.c.scales + self.c.offsets
        targets = np.clip(scaled, self.c.clip_low, self.c.clip_high)
        self.calls += 1
        self.runner_clipped += int(np.count_nonzero(clipped != raw))
        self.target_clipped += int(np.count_nonzero(targets != scaled))
        roll = self.c.roll_limit > 0.0 and self.c.roll_pairs
        if roll:
            targets, changed = self._clip_roll(targets)
            self.roll_clipped += int(changed)
        if self.slew:
            previous = self.slew_position.copy()
            position, velocity = slew_limit_step(self.slew_position, self.slew_velocity, targets, self.c.policy_dt,
                                                 self.c.slew_max_speed, self.c.slew_max_accel)
            if roll:
                position, _ = self._clip_roll(position)
                velocity = (position - previous) / self.c.policy_dt
            self.slew_position, self.slew_velocity = position, velocity
            targets = position.copy()
        return clipped.astype(np.float32), targets


def quaternion_to_rpy(q):
    w, x, y, z = q
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


class MujocoPlant:
    def __init__(self, mujoco, model, contract):
        self.mj = mujoco
        self.model = model
        self.c = contract
        obj = mujoco.mjtObj

        def required(kind, name):
            index = mujoco.mj_name2id(model, kind, name)
            if index < 0:
                raise RuntimeError(f"MuJoCo name not found: {name}")
            return index

        self.joint_ids = np.array([required(obj.mjOBJ_JOINT, n) for n in contract.joint_order])
        self.qpos_adr = model.jnt_qposadr[self.joint_ids].astype(int)
        self.dof_adr = model.jnt_dofadr[self.joint_ids].astype(int)
        trn = model.actuator_trnid[:, 0]
        self.leg_actuators = []
        for name, jid in zip(contract.joint_order, self.joint_ids):
            matches = np.flatnonzero(trn == jid)
            if matches.size != 1:
                raise RuntimeError(f"expected one actuator on {name}, got {matches.tolist()}")
            self.leg_actuators.append(int(matches[0]))
        self.leg_actuators = np.array(self.leg_actuators)
        self.upper_actuators = np.array([a for a in range(model.nu) if a not in set(self.leg_actuators.tolist())],
                                        dtype=int)
        for a in range(model.nu):
            if model.actuator_trntype[a] != mujoco.mjtTrn.mjTRN_JOINT:
                raise RuntimeError(f"actuator {a} is not a joint actuator")
            if not np.isclose(model.actuator_biasprm[a, 1], -model.actuator_gainprm[a, 0]):
                raise RuntimeError(f"actuator {a} is not a position servo")
        self.root_id = required(obj.mjOBJ_JOINT, "root")
        if model.jnt_type[self.root_id] != mujoco.mjtJoint.mjJNT_FREE:
            raise RuntimeError("root is not a free joint")
        self.root_qadr = int(model.jnt_qposadr[self.root_id])
        actuated = set(int(trn[a]) for a in range(model.nu))
        self.passive_joints = [mujoco.mj_id2name(model, obj.mjOBJ_JOINT, j) for j in range(model.njnt)
                               if j != self.root_id and j not in actuated]
        self.upper_joints = [mujoco.mj_id2name(model, obj.mjOBJ_JOINT, int(trn[a])) for a in self.upper_actuators]
        for name in self.upper_joints:
            if not re.search(r"(shoulder|elbow|wrist|neck)", name):
                raise RuntimeError(f"non-leg actuator drives {name}, which is not an arm/neck joint")
        self.key_id = required(obj.mjOBJ_KEY, "home")
        home = model.key_qpos[self.key_id, self.qpos_adr]
        if not np.allclose(home, contract.offsets, atol=1e-6):
            raise RuntimeError(f"home keyframe leg pose {home} differs from the action offsets {contract.offsets}")
        self.upper_hold = np.array(model.key_ctrl[self.key_id, self.upper_actuators], dtype=np.float64)
        upper_home = np.array([model.key_qpos[self.key_id, model.jnt_qposadr[int(trn[a])]] for a in self.upper_actuators])
        if self.upper_hold.size and (np.max(np.abs(self.upper_hold)) > 1e-9 or np.max(np.abs(upper_home)) > 1e-9):
            raise RuntimeError("upper-body home targets/positions are not the default pose 0")
        if not np.allclose(model.key_ctrl[self.key_id, self.leg_actuators], contract.offsets, atol=1e-6):
            raise RuntimeError("home keyframe leg ctrl differs from the action offsets")
        self.base_id = required(obj.mjOBJ_BODY, "base_link")
        self.floor_id = required(obj.mjOBJ_GEOM, "floor")
        self.foot_ids = tuple(required(obj.mjOBJ_BODY, n) for n in contract.feet)
        self.illegal_bodies = {b for b in range(model.nbody)
                               if contract.illegal_regex.match(mujoco.mj_id2name(model, obj.mjOBJ_BODY, b) or "")}
        self.velocity = np.zeros(6)

    def upper_armature(self):
        m = self.model
        return {name: float(m.dof_armature[m.jnt_dofadr[m.actuator_trnid[a, 0]]])
                for name, a in zip(self.upper_joints, self.upper_actuators)}

    def set_upper_armature(self, values):
        m = self.model
        for name, a in zip(self.upper_joints, self.upper_actuators):
            m.dof_armature[m.jnt_dofadr[m.actuator_trnid[a, 0]]] = values[name]
        return self.upper_armature()

    def gain_table(self):
        m = self.model
        rows = []
        for name, a, jid in zip(self.c.joint_order, self.leg_actuators, self.joint_ids):
            dof = m.jnt_dofadr[jid]
            rows.append({"joint": name, "mujoco": [float(m.actuator_gainprm[a, 0]), float(-m.actuator_biasprm[a, 2]),
                                                   float(m.actuator_forcerange[a, 1]), float(m.dof_armature[dof])],
                         "isaac": list(self.c.train_gains.get(name, ())), "motor": self.c.motor_model.get(name)})
        for name, a in zip(self.upper_joints, self.upper_actuators):
            jid = m.actuator_trnid[a, 0]
            dof = m.jnt_dofadr[jid]
            isaac = None
            for group, (kp, kd, eff, arm, exprs) in self.c.upper_train_gains.items():
                if any(re.match(e + "$", name) for e in exprs):
                    isaac = [kp, kd, eff, arm]
            rows.append({"joint": name, "mujoco": [float(m.actuator_gainprm[a, 0]), float(-m.actuator_biasprm[a, 2]),
                                                   float(m.actuator_forcerange[a, 1]), float(m.dof_armature[dof])],
                         "isaac": isaac, "mujoco_dof_damping": float(m.dof_damping[dof])})
        return rows

    def imu(self, data):
        self.mj.mj_objectVelocity(self.model, data, self.mj.mjtObj.mjOBJ_XBODY, self.base_id, self.velocity, 1)
        rotation = data.xmat[self.base_id].reshape(3, 3)
        return self.velocity[:3].copy(), rotation.T @ np.array((0.0, 0.0, -1.0))

    def base_motion(self, data):
        v = np.zeros(6)
        self.mj.mj_objectVelocity(self.model, data, self.mj.mjtObj.mjOBJ_XBODY, self.base_id, v, 0)
        yaw = quaternion_to_rpy(data.xquat[self.base_id])[2]
        c, s = math.cos(yaw), math.sin(yaw)
        return c * v[3] + s * v[4], -s * v[3] + c * v[4], float(v[2])

    def contacts(self, data):
        feet = [False, False]
        illegal = None
        for i in range(data.ncon):
            con = data.contact[i]
            if con.geom1 == self.floor_id:
                other = con.geom2
            elif con.geom2 == self.floor_id:
                other = con.geom1
            else:
                continue
            body = int(self.model.geom_bodyid[other])
            for k, foot in enumerate(self.foot_ids):
                if body == foot:
                    feet[k] = True
            if body in self.illegal_bodies and con.dist <= 0.0:
                illegal = self.mj.mj_id2name(self.model, self.mj.mjtObj.mjOBJ_BODY, body)
        return feet, illegal

    def equality_error(self, data):
        if data.nefc == 0:
            return 0.0
        mask = np.asarray(data.efc_type[: data.nefc]) == int(self.mj.mjtConstraint.mjCNSTR_EQUALITY)
        return float(np.max(np.abs(np.asarray(data.efc_pos[: data.nefc])[mask]))) if np.any(mask) else 0.0


class Controller:
    def __init__(self, contract, actor, plant, slew, joint_delay, imu_delay):
        self.c = contract
        self.actor = actor
        self.plant = plant
        self.pipeline = ActionPipeline(contract, slew)
        self.history = ObservationHistory(contract.terms, contract.history_length)
        if self.history.expected_size != contract.observation_size:
            raise RuntimeError("observation size mismatch")
        self.joint_delay = int(joint_delay)
        self.imu_delay = int(imu_delay)
        self.command = np.zeros(3, dtype=np.float32)
        self.lock = threading.Lock()
        self.reset(None)

    def reset(self, data):
        self.pipeline.reset()
        self.history.reset()
        self.last_action = np.zeros(12, dtype=np.float32)
        self.step = 0
        self.joint_frames = collections.deque(maxlen=self.joint_delay + 1)
        self.imu_frames = collections.deque(maxlen=self.imu_delay + 1)
        if data is not None:
            self.record_imu(data, fill=True)

    def record_imu(self, data, fill=False):
        sample = self.plant.imu(data)
        if fill:
            self.imu_frames.clear()
            for _ in range(self.imu_frames.maxlen):
                self.imu_frames.append(sample)
        else:
            self.imu_frames.append(sample)

    def observation(self, data):
        q = data.qpos[self.plant.qpos_adr] - self.c.offsets
        dq = data.qvel[self.plant.dof_adr].copy()
        if not self.joint_frames:
            for _ in range(self.joint_frames.maxlen):
                self.joint_frames.append((q, dq))
        else:
            self.joint_frames.append((q, dq))
        q_obs, dq_obs = self.joint_frames[0]
        gyro, gravity = self.imu_frames[0]
        phase = gait_phase_at(self.step, period=self.c.gait_period, step_dt=self.c.policy_dt)
        if self.c.gait_masked and float(np.linalg.norm(self.command)) <= self.c.gait_deadband:
            phase = np.zeros_like(phase)
        values = {"joint_pos_rel": q_obs, "joint_vel_rel": dq_obs, "imu_ang_vel": gyro, "projected_gravity": gravity,
                  "velocity_commands": self.command, "gait_phase": phase, "actions": self.last_action}
        frame = np.concatenate([np.asarray(values[name], dtype=np.float32).reshape(-1) for name, _ in self.c.terms])
        return self.history.append(frame).observation()

    def act(self, data):
        obs = self.observation(data)
        raw = self.actor(obs)
        clipped, targets = self.pipeline.apply(raw)
        data.ctrl[self.plant.leg_actuators] = targets
        data.ctrl[self.plant.upper_actuators] = self.plant.upper_hold
        self.last_action = clipped
        self.step += 1
        return raw, targets


def parse_scenario(text):
    schedule = []
    for part in text.split(";"):
        start, values = part.split(":")
        command = tuple(float(v) for v in values.split(","))
        if len(command) != 3:
            raise ValueError(f"scenario segment {part!r} needs vx,vy,wz")
        schedule.append((float(start), command))
    schedule.sort()
    if schedule[0][0] != 0.0:
        raise ValueError("scenario must start at 0")
    return schedule


def command_at(schedule, t):
    return [command for start, command in schedule if start <= t + 1.0e-9][-1]


class Segments:
    def __init__(self, settle_s):
        self.settle_s = settle_s
        self.rows = []
        self.cur = None

    def command(self, command, t):
        command = tuple(round(float(v), 6) for v in command)
        if self.cur is not None and self.cur["command"] == command:
            return
        self.finish(t)
        self.cur = {"command": command, "start_s": t, "samples": []}

    def sample(self, t, vx, vy, wz):
        if self.cur is not None and t >= self.cur["start_s"] + self.settle_s:
            self.cur["samples"].append((vx, vy, wz))

    def finish(self, t):
        if self.cur is None:
            return
        s = np.array(self.cur["samples"]) if self.cur["samples"] else np.zeros((0, 3))
        cmd = self.cur["command"]
        row = {"command": list(cmd), "start_s": round(self.cur["start_s"], 4), "end_s": round(t, 4),
               "settle_s": self.settle_s, "samples": int(len(s))}
        if len(s):
            mean = s.mean(axis=0)
            row.update(mean_vx=float(mean[0]), mean_vy=float(mean[1]), mean_wz=float(mean[2]),
                       err_vx=float(mean[0] - cmd[0]), err_vy=float(mean[1] - cmd[1]), err_wz=float(mean[2] - cmd[2]))
        self.rows.append(row)
        self.cur = None


COMMAND_KEYS = {
    ord("W"): (0.05, 0.0, 0.0), ord("S"): (-0.05, 0.0, 0.0),
    ord("A"): (0.0, 0.0, 0.1), ord("D"): (0.0, 0.0, -0.1),
    ord("Q"): (0.0, 0.05, 0.0), ord("E"): (0.0, -0.05, 0.0),
}


def make_key_callback(controller, limits, reset_flag):
    def on_key(keycode):
        with controller.lock:
            if keycode == ord(" "):
                controller.command[:] = 0.0
            elif keycode == ord("R"):
                reset_flag.set()
            elif keycode in COMMAND_KEYS:
                for i, step in enumerate(COMMAND_KEYS[keycode]):
                    low, high = limits[i]
                    value = float(min(high, max(low, controller.command[i] + step)))
                    controller.command[i] = 0.0 if abs(value) < 1e-6 else value
            else:
                return
            vx, vy, wz = controller.command
        print(f"command  vx={vx:+.2f} m/s  vy={vy:+.2f} m/s  wz={wz:+.2f} rad/s", flush=True)
    return on_key


class HeadlessViewer:
    def is_running(self):
        return True

    def sync(self):
        return None


def simulate(mujoco, model, data, controller, args, viewer, reset_flag):
    plant = controller.plant
    contract = controller.c
    substeps = contract.policy_dt / model.opt.timestep
    if abs(substeps - round(substeps)) > 1e-6:
        raise RuntimeError(f"policy period {contract.policy_dt} is not a multiple of the MuJoCo dt {model.opt.timestep}")
    substeps = int(round(substeps))

    def reset():
        mujoco.mj_resetDataKeyframe(model, data, plant.key_id)
        mujoco.mj_forward(model, data)
        controller.reset(data)

    reset()
    schedule = parse_scenario(args.scenario) if args.scenario else None
    segments = Segments(args.settle_s)
    stats = {"root_z_min": math.inf, "roll_abs_max": 0.0, "pitch_abs_max": 0.0, "raw_abs_max": 0.0,
             "constraint_error_max": 0.0, "fall_time_s": None, "fall_cause": None, "left_contact": 0,
             "right_contact": 0, "physics_steps": 0, "policy_steps": 0, "upper_dev_max": 0.0}
    upper_qadr = model.jnt_qposadr[model.actuator_trnid[plant.upper_actuators, 0]] if plant.upper_actuators.size else []
    status, reason = "timeout", "duration_reached"
    wall_start = time.perf_counter()
    sim_start = data.time
    while True:
        if not viewer.is_running():
            status, reason = "viewer_closed", "viewer_closed"
            break
        if args.duration is not None and data.time >= args.duration - 1e-9:
            break
        if reset_flag.is_set():
            reset_flag.clear()
            reset()
            wall_start, sim_start = time.perf_counter(), data.time
        if schedule is not None:
            controller.command[:] = command_at(schedule, data.time)
        with controller.lock:
            command = controller.command.copy()
            raw, _ = controller.act(data)
        stats["policy_steps"] += 1
        stats["raw_abs_max"] = max(stats["raw_abs_max"], float(np.max(np.abs(raw))))
        segments.command(command, float(data.time))
        for _ in range(substeps):
            mujoco.mj_step(model, data)
            controller.record_imu(data)
            stats["physics_steps"] += 1
            if not (np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()):
                status, reason = "nonfinite", "simulation_state_is_not_finite"
                break
            z = float(data.qpos[plant.root_qadr + 2])
            stats["root_z_min"] = min(stats["root_z_min"], z)
            roll, pitch, _ = quaternion_to_rpy(data.xquat[plant.base_id])
            stats["roll_abs_max"] = max(stats["roll_abs_max"], abs(roll))
            stats["pitch_abs_max"] = max(stats["pitch_abs_max"], abs(pitch))
            stats["constraint_error_max"] = max(stats["constraint_error_max"], plant.equality_error(data))
            if len(upper_qadr):
                stats["upper_dev_max"] = max(stats["upper_dev_max"],
                                             float(np.max(np.abs(data.qpos[upper_qadr] - plant.upper_hold))))
            feet, illegal = plant.contacts(data)
            stats["left_contact"] += int(feet[0])
            stats["right_contact"] += int(feet[1])
            if stats["fall_time_s"] is None and (z < contract.minimum_height or illegal):
                stats["fall_time_s"] = float(data.time)
                stats["fall_cause"] = f"root_z {z:.3f} < {contract.minimum_height}" if z < contract.minimum_height \
                    else f"illegal contact {illegal}"
                if args.headless and args.stop_on_fall:
                    status, reason = "fell", stats["fall_cause"]
            if not args.headless:
                lag = (data.time - sim_start) - (time.perf_counter() - wall_start)
                if lag > 0.0:
                    time.sleep(lag)
        if status in ("nonfinite", "fell"):
            break
        if stats["fall_time_s"] is None:
            vx, vy, wz = plant.base_motion(data)
            segments.sample(float(data.time), vx, vy, wz)
        viewer.sync()
    segments.finish(float(data.time))
    n = max(1, stats["physics_steps"])
    roll, pitch, yaw = quaternion_to_rpy(data.xquat[plant.base_id])
    p = controller.pipeline
    return {
        "status": status, "reason": reason, "simulation_time_s": float(data.time),
        "wall_time_s": time.perf_counter() - wall_start,
        "fall_detected": stats["fall_time_s"] is not None, "fall_time_s": stats["fall_time_s"],
        "fall_cause": stats["fall_cause"],
        "base_height_min_m": stats["root_z_min"], "base_height_final_m": float(data.qpos[plant.root_qadr + 2]),
        "base_xy_final_m": [float(v) for v in data.qpos[plant.root_qadr:plant.root_qadr + 2]],
        "yaw_final_deg": math.degrees(yaw),
        "roll_abs_max_deg": math.degrees(stats["roll_abs_max"]), "pitch_abs_max_deg": math.degrees(stats["pitch_abs_max"]),
        "raw_action_abs_max": stats["raw_abs_max"], "constraint_error_abs_max": stats["constraint_error_max"],
        "upper_joint_abs_dev_max_rad": stats["upper_dev_max"],
        "left_foot_contact_fraction": stats["left_contact"] / n, "right_foot_contact_fraction": stats["right_contact"] / n,
        "runner_clip_fraction": p.runner_clipped / max(1, p.calls * 12),
        "target_clip_fraction": p.target_clipped / max(1, p.calls * 12),
        "foot_roll_clip_fraction": p.roll_clipped / max(1, p.calls),
        "physics_steps": stats["physics_steps"], "policy_steps": stats["policy_steps"],
        "segments": segments.rows,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Play a ver3 rsl_rl walking checkpoint in MuJoCo (sim-to-sim).")
    parser.add_argument("--checkpoint", type=Path, required=True, help="rsl_rl model_*.pt inside a run directory")
    parser.add_argument("--scene", choices=("full", "legs"), default="full")
    parser.add_argument("--vx", type=float, default=0.3)
    parser.add_argument("--vy", type=float, default=0.0)
    parser.add_argument("--wz", type=float, default=0.0)
    parser.add_argument("--scenario", type=str, help="T:VX,VY,WZ;... e.g. 0:0.3,0,0;8:0,0,0 (replaces --vx/--vy/--wz)")
    parser.add_argument("--duration", type=float, help="simulated seconds (required with --headless)")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--output", type=Path, help="JSON metrics path")
    parser.add_argument("--slew-limit", action=argparse.BooleanOptionalAction, default=None,
                        help="training's target slew limiter; default: as the run's env.yaml (slew_enabled)")
    parser.add_argument("--joint-friction", type=float, nargs=2, metavar=("RS02", "RS03"),
                        help="Coulomb friction (N*m) on the 12 leg joints by motor group, e.g. 0.14 0.47")
    parser.add_argument("--obs-delay", type=int, nargs=2, metavar=("JOINT_STEPS", "IMU_SUBSTEPS"),
                        help="fixed observation delays; default: the run's minimum (joint policy steps, IMU physics "
                             "steps of the Isaac dt, converted to MuJoCo steps)")
    parser.add_argument("--settle-s", type=float, default=1.0, help="seconds excluded at the start of each command")
    parser.add_argument("--stop-on-fall", action="store_true", help="headless: stop at the first fall")
    parser.add_argument("--description-root", type=Path, help="robonex-description checkout (default: sibling)")
    parser.add_argument("--upper-armature", choices=("model", "isaac"), default="model",
                        help="armature of the arm and neck joints: the MJCF's (model) or the run's env.yaml upper_* "
                             "actuator groups (isaac; placeholder values, full-variant runs only)")
    args = parser.parse_args(argv)
    if args.headless and args.duration is None:
        parser.error("--headless requires --duration")
    args.checkpoint = args.checkpoint.expanduser().resolve()
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint not found: {args.checkpoint}")
    if args.scenario:
        try:
            parse_scenario(args.scenario)
        except ValueError as error:
            parser.error(f"--scenario: {error}")
    if args.joint_friction is not None and any(not math.isfinite(v) or v < 0.0 for v in args.joint_friction):
        parser.error("--joint-friction needs two finite non-negative values")
    return args


def main(argv=None):
    args = parse_args(argv)
    import mujoco
    root = resolve_repo(DESCRIPTION_REPO_NAMES, "ROBONEX_DESCRIPTION_ROOT", args.description_root,
                        anchors=(Path(__file__),))
    leg_order, leg_order_source = resolve_leg_order(args.checkpoint.parent, root / "ver3" / "ver3_constants.json")
    contract = RunContract(args.checkpoint.parent, leg_order)
    state = load_state_dict(args.checkpoint)
    actor = ActorMLP(state, contract.agent, contract.observation_size, 12)
    scene = root / "ver3" / "mujoco" / "robot" / args.scene / "scene.xml"
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    plant = MujocoPlant(mujoco, model, contract)
    if args.upper_armature == "isaac":
        upper_armature = {"source": "isaac env.yaml scene.robot.actuators.upper_*.armature", "label": "placeholder",
                          "values": plant.set_upper_armature(upper_armature_values(plant.upper_joints,
                                                                                   contract.upper_train_gains))}
    else:
        upper_armature = {"source": "model", "label": "mjcf", "values": plant.upper_armature()}
    friction = None
    if args.joint_friction is not None:
        levels = dict(zip(("rs02", "rs03"), args.joint_friction))
        friction = {}
        for name, dof in zip(contract.joint_order, plant.dof_adr):
            model.dof_frictionloss[dof] = levels[contract.motor_model[name]]
            friction[name] = levels[contract.motor_model[name]]
    slew = contract.slew_enabled if args.slew_limit is None else args.slew_limit
    if args.obs_delay is None:
        joint_delay = contract.joint_delay_range[0]
        imu_delay = int(round(contract.imu_delay_range[0] * contract.sim_dt / model.opt.timestep))
    else:
        joint_delay, imu_delay = args.obs_delay
    controller = Controller(contract, actor, plant, slew, joint_delay, imu_delay)
    controller.command[:] = (args.vx, args.vy, args.wz)
    header = {
        "checkpoint": str(args.checkpoint), "trained_variant": contract.model_variant, "scene": str(scene),
        "mujoco_version": mujoco.__version__, "mujoco_dt_s": float(model.opt.timestep),
        "policy_dt_s": contract.policy_dt, "isaac_dt_s": contract.sim_dt, "decimation": contract.decimation,
        "nu": int(model.nu), "held_upper_actuators": len(plant.upper_actuators),
        "observation_terms": [f"{n}:{s}x{contract.history_length}" for n, s in contract.terms],
        "observation_size": contract.observation_size, "joint_order": list(contract.joint_order),
        "joint_order_source": leg_order_source,
        "slew": bool(slew), "joint_obs_delay_policy_steps": joint_delay, "imu_delay_mujoco_steps": imu_delay,
        "joint_friction": friction, "upper_armature": upper_armature, "runner_action_clip": contract.runner_clip,
        "command": [args.vx, args.vy, args.wz] if not args.scenario else args.scenario,
        "trained_command_limits": contract.command_limits,
    }
    reset_flag = threading.Event()
    if args.headless:
        result = simulate(mujoco, model, data, controller, args, HeadlessViewer(), reset_flag)
    else:
        import mujoco.viewer
        print("keys: W/S forward  A/D turn  Q/E strafe  SPACE stop  R reset", flush=True)
        print(f"command  vx={args.vx:+.2f} m/s  vy={args.vy:+.2f} m/s  wz={args.wz:+.2f} rad/s", flush=True)
        with mujoco.viewer.launch_passive(model, data, key_callback=make_key_callback(
                controller, contract.command_limits, reset_flag)) as viewer:
            result = simulate(mujoco, model, data, controller, args, viewer, reset_flag)
    result = {**header, **result}
    text = json.dumps(result, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("status", "fall_detected", "fall_time_s", "fall_cause",
                                             "base_height_min_m", "segments")}, indent=1))
    return 0 if result["status"] in ("timeout", "viewer_closed") else 1


if __name__ == "__main__":
    sys.exit(main())
