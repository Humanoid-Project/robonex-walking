import math

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils.version import get_isaac_sim_version


def reset_closed_loop_to_default(
    env,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.default_joint_pos[env_ids].clone()
    joint_vel = torch.zeros_like(asset.data.default_joint_vel[env_ids])
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    active_pos = joint_pos[:, asset_cfg.joint_ids]
    active_vel = joint_vel[:, asset_cfg.joint_ids]
    asset.set_joint_position_target(
        active_pos,
        joint_ids=asset_cfg.joint_ids,
        env_ids=env_ids,
    )
    asset.set_joint_velocity_target(
        active_vel,
        joint_ids=asset_cfg.joint_ids,
        env_ids=env_ids,
    )


def push_standing_by_setting_velocity(
    env,
    env_ids: torch.Tensor,
    velocity_range: dict[str, tuple[float, float]],
    command_name: str = "base_velocity",
    command_deadband: float = 0.05,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    command = env.command_manager.get_command(command_name)[env_ids]
    standing = env_ids[torch.linalg.norm(command, dim=1) < command_deadband]
    if standing.numel() == 0:
        return
    asset: Articulation = env.scene[asset_cfg.name]
    vel_w = asset.data.root_vel_w[standing].clone()
    for index, key in enumerate(("x", "y", "z", "roll", "pitch", "yaw")):
        low, high = velocity_range.get(key, (0.0, 0.0))
        vel_w[:, index] += torch.empty(standing.numel(), device=vel_w.device).uniform_(low, high)
    asset.write_root_velocity_to_sim(vel_w, env_ids=standing)


def randomize_gait_clock_offset(
    env,
    env_ids: torch.Tensor | None,
    offset_range: tuple[float, float] = (0.0, 0.0),
):
    low, high = float(offset_range[0]), float(offset_range[1])
    if not (math.isfinite(low) and math.isfinite(high)) or not 0.0 <= low <= high <= 1.0:
        raise ValueError(f"gait clock offset_range must satisfy 0 <= low <= high <= 1, got {offset_range}")
    offset = getattr(env, "_gait_clock_offset", None)
    if offset is None:
        if low == 0.0 and high == 0.0:
            return
        offset = torch.zeros(env.num_envs, device=env.device)
        env._gait_clock_offset = offset
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=offset.device)
    env_ids = torch.as_tensor(env_ids, device=offset.device)
    draw = low + (high - low) * torch.rand(len(env_ids), device=offset.device)
    offset[env_ids] = torch.remainder(draw, 1.0).to(offset.dtype)


def randomize_joint_coulomb_friction(
    env,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    friction_range: dict[str, tuple[float, float]],
    static_ratio: float = 1.0,
    viscous: float = 0.0,
):
    if get_isaac_sim_version().major < 5:
        raise RuntimeError("PhysX joint friction is a load-proportional coefficient before Isaac Sim 5.0, not N*m")
    if static_ratio < 1.0:
        raise ValueError(f"static_ratio {static_ratio} would put the static friction below the dynamic friction")
    asset: Articulation = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)
    joint_ids = asset_cfg.joint_ids
    if isinstance(joint_ids, slice):
        joint_ids = list(range(asset.num_joints))[joint_ids]
    names = [asset.joint_names[i] for i in joint_ids]
    low = torch.tensor([friction_range[name][0] for name in names], device=asset.device)
    high = torch.tensor([friction_range[name][1] for name in names], device=asset.device)
    dynamic = low + (high - low) * torch.rand(len(env_ids), len(names), device=asset.device)
    asset.write_joint_friction_coefficient_to_sim(
        joint_friction_coeff=dynamic * static_ratio,
        joint_dynamic_friction_coeff=dynamic,
        joint_viscous_friction_coeff=torch.full_like(dynamic, viscous),
        joint_ids=torch.tensor(joint_ids, dtype=torch.int, device=asset.device),
        env_ids=env_ids,
    )


def joint_zero_offset_bounds(names, offset_range):
    unknown = sorted(set(offset_range) - set(names))
    missing = [name for name in names if name not in offset_range]
    if unknown or missing:
        raise ValueError(f"joint zero offset_range keys must be the event's joints: unknown {unknown}, missing {missing}")
    low = torch.tensor([float(offset_range[name][0]) for name in names])
    high = torch.tensor([float(offset_range[name][1]) for name in names])
    if not bool(torch.all(torch.isfinite(low)) and torch.all(torch.isfinite(high))):
        raise ValueError(f"non-finite joint zero offset_range: {offset_range}")
    if bool(torch.any(high < low)):
        raise ValueError(f"joint zero offset_range upper bound below lower bound: {offset_range}")
    return low, high


def randomize_joint_zero_offset(
    env,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    offset_range: dict[str, tuple[float, float]],
):
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids
    if isinstance(joint_ids, slice):
        joint_ids = list(range(asset.num_joints))[joint_ids]
    joint_ids = [int(i) for i in joint_ids]
    names = [asset.joint_names[i] for i in joint_ids]
    low, high = joint_zero_offset_bounds(names, offset_range)
    offset = getattr(env, "_joint_zero_offset", None)
    if offset is None:
        if not bool(torch.any(low != 0.0) or torch.any(high != 0.0)):
            return
        offset = torch.zeros(env.scene.num_envs, asset.num_joints, device=asset.device)
        env._joint_zero_offset = offset
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=offset.device)
    env_ids = torch.as_tensor(env_ids, device=offset.device)
    low, high = low.to(offset.device), high.to(offset.device)
    draw = low.expand(len(env_ids), len(joint_ids)).clone()
    if bool(torch.any(high > low)):
        draw += (high - low) * torch.rand(len(env_ids), len(joint_ids), device=offset.device)
    columns = torch.tensor(joint_ids, dtype=torch.long, device=offset.device)
    offset[env_ids.unsqueeze(1), columns] = draw.to(offset.dtype)


class randomize_rigid_body_com_offset(ManagerTermBase):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.nominal_coms = None

    def __call__(
        self,
        env,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg,
        com_range: dict[str, tuple[float, float]],
    ):
        low = torch.tensor([float(com_range.get(key, (0.0, 0.0))[0]) for key in "xyz"])
        high = torch.tensor([float(com_range.get(key, (0.0, 0.0))[1]) for key in "xyz"])
        if bool(torch.any(high < low)):
            raise ValueError(f"com_range upper bound below lower bound: {com_range}")
        if self.nominal_coms is None and not bool(torch.any(low != 0.0) or torch.any(high != 0.0)):
            return
        asset: Articulation = env.scene[asset_cfg.name]
        if self.nominal_coms is None:
            self.nominal_coms = asset.root_physx_view.get_coms().clone()
        if env_ids is None:
            env_ids = torch.arange(env.scene.num_envs, device="cpu")
        else:
            env_ids = env_ids.cpu()
        body_ids = asset_cfg.body_ids
        if isinstance(body_ids, slice):
            body_ids = list(range(asset.num_bodies))[body_ids]
        body_ids = torch.tensor(body_ids, dtype=torch.long, device="cpu")
        offset = low.expand(len(env_ids), 1, 3).clone()
        if bool(torch.any(high > low)):
            offset += (high - low) * torch.rand(len(env_ids), 1, 3)
        coms = asset.root_physx_view.get_coms().clone()
        rows = env_ids[:, None]
        coms[rows, body_ids] = self.nominal_coms[rows, body_ids]
        coms[rows, body_ids, :3] += offset.to(coms.dtype)
        asset.root_physx_view.set_coms(coms, env_ids)


_LOAD_AXES = ("x", "y", "z")


def _sustained_load_bounds(force_range, torque_range, application_height_m):
    ranges = (force_range, torque_range)
    bounds = torch.tensor([[float(r.get(axis, (0.0, 0.0))[i]) for r in ranges for axis in _LOAD_AXES] for i in (0, 1)])
    height = torch.tensor(float(application_height_m))
    if not bool(torch.all(torch.isfinite(bounds))) or not bool(torch.isfinite(height)):
        raise ValueError(f"non-finite sustained load: force {force_range}, torque {torque_range}, "
                         f"height {application_height_m}")
    if bool(torch.any(bounds[1] < bounds[0])):
        raise ValueError(f"sustained load upper bound below lower bound: force {force_range}, torque {torque_range}")
    return bounds[0], bounds[1]


def apply_sustained_base_load(
    env,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    force_range: dict[str, tuple[float, float]],
    torque_range: dict[str, tuple[float, float]],
    application_height_m: float = 0.0,
):
    low, high = _sustained_load_bounds(force_range, torque_range, application_height_m)
    if not bool(torch.any(low != 0.0) or torch.any(high != 0.0)):
        return
    asset: Articulation = env.scene[asset_cfg.name]
    body_ids = asset_cfg.body_ids
    if isinstance(body_ids, slice):
        body_ids = list(range(asset.num_bodies))[body_ids]
    body_ids = [int(b) for b in body_ids]
    if len(body_ids) != 1:
        raise ValueError(f"the sustained load acts on exactly one body, got body_ids {body_ids}")
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)
    low = low.to(asset.device)
    high = high.to(asset.device)
    wrench = low.expand(len(env_ids), 6).clone()
    if bool(torch.any(high > low)):
        wrench += (high - low) * torch.rand(len(env_ids), 6, device=asset.device)
    force = wrench[:, :3]
    lever = torch.tensor([0.0, 0.0, float(application_height_m)], device=asset.device).expand_as(force)
    torque = wrench[:, 3:] + torch.linalg.cross(lever, force, dim=-1)
    asset.permanent_wrench_composer.set_forces_and_torques(
        forces=force.unsqueeze(1).contiguous(),
        torques=torque.unsqueeze(1).contiguous(),
        body_ids=body_ids,
        env_ids=env_ids,
    )


def neck_target_bounds(names, limits, margin, range_scale):
    unknown = sorted(set(limits) - set(names))
    missing = [name for name in names if name not in limits]
    if unknown or missing:
        raise ValueError(f"neck target limits keys must be the event's joints: unknown {unknown}, missing {missing}")
    scale, margin = float(range_scale), float(margin)
    if not (math.isfinite(scale) and 0.0 <= scale <= 1.0):
        raise ValueError(f"neck target range_scale must be in [0, 1], got {range_scale}")
    if not (math.isfinite(margin) and margin >= 0.0):
        raise ValueError(f"neck target margin must be finite and >= 0, got {margin}")
    lower = torch.tensor([float(limits[name][0]) for name in names]) + margin
    upper = torch.tensor([float(limits[name][1]) for name in names]) - margin
    if not bool(torch.all(torch.isfinite(lower)) and torch.all(torch.isfinite(upper))) or bool(torch.any(upper <= lower)):
        raise ValueError(f"neck target limits {limits} leave no range after margin {margin}")
    mid = 0.5 * (lower + upper)
    half = 0.5 * (upper - lower) * scale
    return mid - half, mid + half


def neck_target_timing(resample_s, blend_s):
    low, high = float(resample_s[0]), float(resample_s[1])
    blend = float(blend_s)
    if not (math.isfinite(low) and math.isfinite(high)) or not 0.0 < low <= high:
        raise ValueError(f"neck target resample_s must satisfy 0 < low <= high, got {resample_s}")
    if not (math.isfinite(blend) and blend >= 0.0):
        raise ValueError(f"neck target blend_s must be finite and >= 0, got {blend_s}")
    return low, high, blend


class randomize_neck_target(ManagerTermBase):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.prev = None

    def _setup(self, asset_cfg, limits, margin, range_scale, resample_s, blend_s):
        if float(range_scale) == 0.0:
            return None
        asset: Articulation = self._env.scene[asset_cfg.name]
        joint_ids = asset_cfg.joint_ids
        if isinstance(joint_ids, slice):
            joint_ids = list(range(asset.num_joints))[joint_ids]
        joint_ids = [int(i) for i in joint_ids]
        names = [asset.joint_names[i] for i in joint_ids]
        low, high = neck_target_bounds(names, limits, margin, range_scale)
        neck_target_timing(resample_s, blend_s)
        if self.prev is None:
            shape = (self._env.num_envs, len(joint_ids))
            device = asset.device
            self.prev = torch.zeros(shape, device=device)
            self.goal = torch.zeros(shape, device=device)
            self.target = torch.zeros(shape, device=device)
            self.elapsed = torch.zeros(self._env.num_envs, device=device)
            self.period = torch.zeros(self._env.num_envs, device=device)
        return asset, joint_ids, low.to(self.prev.device), high.to(self.prev.device)

    def _ids(self, env_ids):
        if env_ids is None:
            return torch.arange(self._env.num_envs, device=self.prev.device)
        return torch.as_tensor(env_ids, dtype=torch.long, device=self.prev.device)

    def _draw(self, count, low, high):
        return low + (high - low) * torch.rand(count, low.numel(), device=low.device)

    def _period(self, count, resample_s):
        low, high = float(resample_s[0]), float(resample_s[1])
        return low + (high - low) * torch.rand(count, device=self.prev.device)

    def reset(self, env_ids=None):
        params = self.cfg.params
        setup = self._setup(params["asset_cfg"], params["limits"], params.get("margin", 0.05), params.get("range_scale", 0.0),
                            params.get("resample_s", (1.0, 3.0)), params.get("blend_s", 1.0))
        if setup is None:
            return
        asset, joint_ids, low, high = setup
        ids = self._ids(env_ids)
        if params.get("start_at_default", False):
            start = asset.data.default_joint_pos[ids][:, joint_ids].to(self.prev.dtype)
            self.target[ids] = start
            self.goal[ids] = start
        else:
            self.target[ids] = self._draw(len(ids), low, high)
            self.goal[ids] = self._draw(len(ids), low, high)
        self.prev[ids] = self.target[ids]
        self.elapsed[ids] = 0.0
        self.period[ids] = self._period(len(ids), params.get("resample_s", (1.0, 3.0)))
        target = self.target[ids]
        if params.get("write_state_at_reset", True):
            asset.write_joint_state_to_sim(target, torch.zeros_like(target), joint_ids=joint_ids, env_ids=ids)
        asset.set_joint_position_target(target, joint_ids=joint_ids, env_ids=ids)

    def __call__(
        self,
        env,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg,
        limits: dict[str, tuple[float, float]],
        margin: float = 0.05,
        range_scale: float = 0.0,
        resample_s: tuple[float, float] = (1.0, 3.0),
        blend_s: float = 1.0,
        start_at_default: bool = False,
        write_state_at_reset: bool = True,
    ):
        setup = self._setup(asset_cfg, limits, margin, range_scale, resample_s, blend_s)
        if setup is None:
            return
        asset, joint_ids, low, high = setup
        ids = self._ids(env_ids)
        self.elapsed[ids] += float(env.step_dt)
        due = ids[self.elapsed[ids] >= self.period[ids]]
        if due.numel() > 0:
            self.prev[due] = self.target[due]
            self.goal[due] = self._draw(len(due), low, high)
            self.elapsed[due] = 0.0
            self.period[due] = self._period(len(due), resample_s)
        blend = float(blend_s)
        if blend > 0.0:
            alpha = torch.clamp(self.elapsed[ids] / blend, 0.0, 1.0).unsqueeze(1)
        else:
            alpha = torch.ones(len(ids), 1, device=self.prev.device)
        self.target[ids] = self.prev[ids] + (self.goal[ids] - self.prev[ids]) * alpha
        asset.set_joint_position_target(self.target[ids], joint_ids=joint_ids, env_ids=ids)
