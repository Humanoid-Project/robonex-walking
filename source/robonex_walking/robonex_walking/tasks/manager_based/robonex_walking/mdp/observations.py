from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase
from isaaclab.sensors import Imu, ImuCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def gait_phase(
    env: ManagerBasedRLEnv,
    period: float,
    command_name: str | None = None,
    command_deadband: float = 0.05,
) -> torch.Tensor:
    """Where the gait clock is, as (sin, cos) of the cycle phase.

    ``feet_gait`` rewards a contact pattern that matches this clock, and the
    clock is a function of time since reset. Without it in the observation the
    policy cannot tell which phase it is in, so the reward is unlearnable and
    sits at chance. H1 feeds it; G1 comments it out and leans on a much wider
    velocity command instead.

    Zeroed on a standing command. Every reward that consumes the clock is already
    gated off at zero command, but the observation was not, so the clock kept
    turning and the policy kept answering it: measured standing still after a
    0.3 m/s walk, the knees swung 37.4 deg peak-to-peak at 1.22 Hz against a
    1.25 Hz clock, one foot re-touching every cycle, for 26 W of mechanical power
    while the feet were supposed to be planted. Holding the pair at zero removes
    the cue instead of asking a later penalty to out-shout it. The width stays 2,
    so the deploy observation contract is unchanged.
    """
    global_phase = (env.episode_length_buf * env.step_dt) % period / period
    phase = torch.zeros(env.num_envs, 2, device=env.device)
    phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
    phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)
    if command_name is not None:
        command = env.command_manager.get_command(command_name)
        moving = (torch.linalg.norm(command, dim=1) > command_deadband).unsqueeze(-1)
        phase = phase * moving
    return phase


class delayed_joint_state(ManagerTermBase):
    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._asset_cfg = cfg.params["asset_cfg"]
        self._max_delay = int(cfg.params.get("max_delay_steps", 1))
        self._min_delay = int(cfg.params.get("min_delay_steps", 0))
        if not 0 <= self._min_delay <= self._max_delay:
            raise ValueError("joint observation delay needs 0 <= min_delay_steps <= max_delay_steps, got %d..%d" % (self._min_delay, self._max_delay))
        shared = getattr(env, "_joint_observation_delay", None)
        if shared is None:
            shared = torch.randint(self._min_delay, self._max_delay + 1, (env.num_envs,), device=env.device)
            env._joint_observation_delay = shared
        self._delay = shared
        self._frames = None
        self._counter = None

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        env_ids = torch.as_tensor(env_ids, device=self.device)
        self._delay[env_ids] = torch.randint(self._min_delay, self._max_delay + 1, (len(env_ids),), device=self.device)

    def __call__(self, env: ManagerBasedRLEnv, asset_cfg, field: str = "pos", max_delay_steps: int = 1,
                 min_delay_steps: int = 0):
        asset = env.scene[asset_cfg.name]
        if field == "pos":
            current = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
            zero_offset = getattr(env, "_joint_zero_offset", None)
            if zero_offset is not None:
                current = current - zero_offset[:, asset_cfg.joint_ids]
        else:
            current = asset.data.joint_vel[:, asset_cfg.joint_ids] - asset.data.default_joint_vel[:, asset_cfg.joint_ids]
        if self._frames is None:
            self._frames = current.unsqueeze(0).repeat(self._max_delay + 1, 1, 1)
        elif self._counter != env.common_step_counter:
            self._frames = torch.cat([current.unsqueeze(0), self._frames[:-1]], dim=0)
        else:
            self._frames[0] = current
        self._counter = env.common_step_counter
        fresh = env.episode_length_buf == 0
        if torch.any(fresh):
            self._frames[:, fresh] = current[fresh]
        return self._frames[self._delay, torch.arange(current.shape[0], device=current.device)]


class DelayedImu(Imu):
    def _initialize_impl(self):
        super()._initialize_impl()
        if self.cfg.history_length <= 0:
            raise ValueError(
                f"DelayedImu needs history_length > 0 so its ring advances every physics substep, got {self.cfg.history_length}"
            )
        if not 0 <= self.cfg.min_delay_steps <= self.cfg.max_delay_steps:
            raise ValueError(
                f"imu delay steps must satisfy 0 <= min <= max, got {self.cfg.min_delay_steps}..{self.cfg.max_delay_steps}"
            )
        slots = self.cfg.max_delay_steps + 1
        self._ring_w = torch.zeros(slots, self._num_envs, 3, device=self._device)
        self._ring_g = torch.zeros(slots, self._num_envs, 3, device=self._device)
        self._ring_g[..., 2] = -1.0
        self._head = 0
        self._fresh = torch.ones(self._num_envs, dtype=torch.bool, device=self._device)
        self._delay = self._draw(self._num_envs)
        self._envs = torch.arange(self._num_envs, device=self._device)

    def _draw(self, count):
        return torch.randint(self.cfg.min_delay_steps, self.cfg.max_delay_steps + 1, (count,), device=self._device)

    def update(self, dt, force_recompute=False):
        self._head = (self._head + 1) % self._ring_w.shape[0]
        super().update(dt, force_recompute)

    def reset(self, env_ids=None):
        super().reset(env_ids)
        if env_ids is None:
            env_ids = slice(None)
            count = self._num_envs
        else:
            count = len(env_ids)
        self._fresh[env_ids] = True
        self._delay[env_ids] = self._draw(count)

    def _update_buffers_impl(self, env_ids):
        super()._update_buffers_impl(env_ids)
        ang_vel = self._data.ang_vel_b[env_ids]
        gravity = self._data.projected_gravity_b[env_ids]
        self._ring_w[self._head, env_ids] = ang_vel
        self._ring_g[self._head, env_ids] = gravity
        fresh = torch.zeros_like(self._fresh)
        fresh[env_ids] = self._fresh[env_ids]
        if torch.any(fresh):
            self._ring_w[:, fresh] = self._data.ang_vel_b[fresh]
            self._ring_g[:, fresh] = self._data.projected_gravity_b[fresh]
            self._fresh[fresh] = False

    def delayed(self):
        self._update_outdated_buffers()
        index = (self._head - self._delay) % self._ring_w.shape[0]
        return self._ring_w[index, self._envs], self._ring_g[index, self._envs]


@configclass
class DelayedImuCfg(ImuCfg):
    class_type: type = DelayedImu
    min_delay_steps: int = 0
    max_delay_steps: int = 0


def delayed_imu_ang_vel(env: ManagerBasedRLEnv, asset_cfg) -> torch.Tensor:
    return env.scene[asset_cfg.name].delayed()[0]


def delayed_imu_projected_gravity(env: ManagerBasedRLEnv, asset_cfg) -> torch.Tensor:
    return env.scene[asset_cfg.name].delayed()[1]


class delayed_imu_projected_gravity_biased(ManagerTermBase):
    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        p = cfg.params
        self._offset_rad = float(p.get("offset_deg", 0.0)) * torch.pi / 180.0
        self._sigma_rad = float(p.get("ou_sigma_deg", 0.0)) * torch.pi / 180.0
        self._tau = float(p.get("ou_tau_s", 2.0))
        self._clip_rad = float(p.get("clip_deg", 8.0)) * torch.pi / 180.0
        if self._offset_rad < 0 or self._sigma_rad < 0 or self._tau <= 0 or self._clip_rad <= 0:
            raise ValueError("projected gravity bias needs offset_deg >= 0, ou_sigma_deg >= 0, ou_tau_s > 0, clip_deg > 0")
        self._active = self._offset_rad > 0 or self._sigma_rad > 0
        self._offset = torch.zeros(env.num_envs, 2, device=env.device)
        self._ou = torch.zeros(env.num_envs, 2, device=env.device)
        self._counter = None
        self.reset()

    def reset(self, env_ids=None):
        if not self._active:
            return
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        env_ids = torch.as_tensor(env_ids, device=self.device)
        n = len(env_ids)
        self._offset[env_ids] = (torch.rand(n, 2, device=self.device) * 2.0 - 1.0) * self._offset_rad
        self._ou[env_ids] = torch.randn(n, 2, device=self.device) * self._sigma_rad

    def bias(self):
        return torch.clamp(self._offset + self._ou, -self._clip_rad, self._clip_rad)

    def __call__(self, env: ManagerBasedRLEnv, asset_cfg, offset_deg: float = 0.0, ou_sigma_deg: float = 0.0,
                 ou_tau_s: float = 2.0, clip_deg: float = 8.0):
        gravity = env.scene[asset_cfg.name].delayed()[1]
        if not self._active:
            return gravity
        if self._counter != env.common_step_counter:
            a = float(torch.exp(torch.tensor(-env.step_dt / self._tau)))
            self._ou = a * self._ou + self._sigma_rad * (1.0 - a * a) ** 0.5 * torch.randn_like(self._ou)
            self._counter = env.common_step_counter
        b = self.bias()
        cr, sr = torch.cos(b[:, 0]), torch.sin(b[:, 0])
        cp, sp = torch.cos(b[:, 1]), torch.sin(b[:, 1])
        x, y, z = gravity[:, 0], gravity[:, 1], gravity[:, 2]
        y1 = cr * y - sr * z
        z1 = sr * y + cr * z
        x2 = cp * x + sp * z1
        z2 = -sp * x + cp * z1
        return torch.stack([x2, y1, z2], dim=-1)


def base_height_above_feet(env: ManagerBasedRLEnv, asset_cfg) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    feet_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    return (asset.data.root_pos_w[:, 2] - feet_z.min(dim=1).values).unsqueeze(-1)


def foot_contact(env: ManagerBasedRLEnv, sensor_cfg, weight_n: float, threshold_n: float = 20.0) -> torch.Tensor:
    if weight_n <= 0 or threshold_n < 0:
        raise ValueError("foot_contact needs weight_n > 0 and threshold_n >= 0")
    normal = env.scene.sensors[sensor_cfg.name].data.net_forces_w[:, sensor_cfg.body_ids, 2].clamp(min=0.0)
    return torch.cat([(normal > threshold_n).float(), normal / weight_n], dim=-1)


def joint_friction(env: ManagerBasedRLEnv, asset_cfg) -> torch.Tensor:
    return env.scene[asset_cfg.name].data.joint_dynamic_friction_coeff[:, asset_cfg.joint_ids]


def randomization_delays(env: ManagerBasedRLEnv, imu_cfg, action_name: str = "joint_pos") -> torch.Tensor:
    joint = getattr(env, "_joint_observation_delay", None)
    if joint is None:
        raise RuntimeError("no joint observation delay draw on the env; the policy group must use delayed_joint_state")
    imu = env.scene[imu_cfg.name]._delay
    actuation = env.action_manager.get_term(action_name)._delay_substeps
    return torch.stack([joint.float(), imu.float(), actuation.float()], dim=-1)


class last_push(ManagerTermBase):
    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        if float(cfg.params.get("cap_s", 2.0)) <= 0:
            raise ValueError("last_push needs cap_s > 0")
        if not hasattr(env, "_last_push_dv"):
            env._last_push_dv = torch.zeros(env.num_envs, 2, device=env.device)
            env._last_push_step = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else torch.as_tensor(env_ids, device=self.device)
        self._env._last_push_dv[ids] = 0.0
        self._env._last_push_step[ids] = -1

    def __call__(self, env: ManagerBasedRLEnv, cap_s: float = 2.0):
        step = env._last_push_step
        age = ((env.common_step_counter - step).float() * env.step_dt).clamp(0.0, cap_s)
        age = torch.where(step < 0, torch.full_like(age, cap_s), age)
        return torch.cat([env._last_push_dv, age.unsqueeze(-1)], dim=-1)


def _body_shape_ranges(asset) -> list[tuple[int, int]]:
    counts = []
    for link_path in asset.root_physx_view.link_paths[0]:
        counts.append(asset._physics_sim_view.create_rigid_body_view(link_path).max_shapes)
    if sum(counts) != asset.root_physx_view.max_shapes:
        raise RuntimeError(f"shape count per body {counts} does not add up to {asset.root_physx_view.max_shapes}")
    ranges, start = [], 0
    for count in counts:
        ranges.append((start, start + count))
        start += count
    return ranges


class _EpisodeDraw(ManagerTermBase):
    width = 0

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._asset_cfg = cfg.params["asset_cfg"]
        self._value = torch.zeros(env.num_envs, self.width, device=env.device)
        self._dirty = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids=None):
        if env_ids is None:
            self._dirty[:] = True
        else:
            self._dirty[torch.as_tensor(env_ids, device=self.device)] = True

    def _read(self, asset, env_ids_cpu: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def __call__(self, env: ManagerBasedRLEnv, asset_cfg):
        if torch.any(self._dirty):
            ids = self._dirty.nonzero(as_tuple=False).squeeze(-1)
            asset = env.scene[asset_cfg.name]
            self._value[ids] = self._read(asset, ids.cpu()).to(self._value.device, self._value.dtype)
            self._dirty[ids] = False
        return self._value


def _single_body(asset_cfg, asset) -> int:
    body_ids = asset_cfg.body_ids
    if isinstance(body_ids, slice):
        body_ids = list(range(asset.num_bodies))[body_ids]
    if len(body_ids) != 1:
        raise ValueError(f"expected exactly one body, got {body_ids}")
    return int(body_ids[0])


class base_mass_delta(_EpisodeDraw):
    width = 1

    def _read(self, asset, env_ids_cpu):
        body = _single_body(self._asset_cfg, asset)
        masses = asset.root_physx_view.get_masses()
        return (masses[env_ids_cpu, body] - asset.data.default_mass[env_ids_cpu, body]).unsqueeze(-1)


class base_com_offset(_EpisodeDraw):
    width = 3

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        asset = env.scene[self._asset_cfg.name]
        self._nominal = asset.root_physx_view.get_coms()[:, _single_body(self._asset_cfg, asset), :3].clone()

    def _read(self, asset, env_ids_cpu):
        coms = asset.root_physx_view.get_coms()[env_ids_cpu, _single_body(self._asset_cfg, asset), :3]
        return coms - self._nominal[env_ids_cpu]


class foot_friction(_EpisodeDraw):
    width = 4

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        asset = env.scene[self._asset_cfg.name]
        body_ids = self._asset_cfg.body_ids
        if isinstance(body_ids, slice) or len(body_ids) != 2:
            raise ValueError(f"foot_friction needs the two foot bodies, got {body_ids}")
        ranges = _body_shape_ranges(asset)
        self._shapes = [ranges[int(b)] for b in body_ids]
        if any(hi <= lo for lo, hi in self._shapes):
            raise RuntimeError(f"a foot body has no collision shape: {self._shapes}")

    def _read(self, asset, env_ids_cpu):
        props = asset.root_physx_view.get_material_properties()[env_ids_cpu]
        static = torch.stack([props[:, lo:hi, 0].mean(dim=1) for lo, hi in self._shapes], dim=-1)
        dynamic = torch.stack([props[:, lo:hi, 1].mean(dim=1) for lo, hi in self._shapes], dim=-1)
        return torch.cat([static, dynamic], dim=-1)
