from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


__all__ = ["compute_symmetric_states"]


_CACHE: dict[int, dict] = {}


_VECTOR_SIGNS = {
    "imu_ang_vel": (-1.0, 1.0, -1.0),
    "projected_gravity": (1.0, -1.0, 1.0),
    "velocity_commands": (1.0, -1.0, -1.0),
    "gait_phase": (-1.0, -1.0),
    "base_lin_vel": (1.0, -1.0, 1.0),
    "base_ang_vel": (-1.0, 1.0, -1.0),
    "projected_gravity_clean": (1.0, -1.0, 1.0),
    "base_height_above_feet": (1.0,),
    "base_mass_delta": (1.0,),
    "base_com_offset": (1.0, -1.0, 1.0),
    "randomization_delays": (1.0, 1.0, 1.0),
    "last_push": (1.0, -1.0, 1.0),
}

_JOINT_TERMS = ("joint_pos_rel", "joint_vel_rel", "actions", "joint_pos_clean", "joint_vel_clean")

_JOINT_MAGNITUDE_TERMS = ("joint_friction",)

_FOOT_PAIR_TERMS = {
    "foot_contact": (1, 0, 3, 2),
    "foot_friction": (1, 0, 3, 2),
}


def _mirror_name(name: str) -> str:
    if name.startswith("l_"):
        return "r_" + name[2:]
    if name.startswith("r_"):
        return "l_" + name[2:]
    if name.startswith("left_"):
        return "right_" + name[5:]
    if name.startswith("right_"):
        return "left_" + name[6:]
    raise ValueError(f"joint {name!r} has no left/right prefix; the mirror map is undefined")


def _joint_permutation(env: ManagerBasedRLEnv) -> torch.Tensor:
    action_term = env.action_manager.get_term("joint_pos")
    names = list(action_term._joint_names)
    index = {name: i for i, name in enumerate(names)}
    perm = [index[_mirror_name(name)] for name in names]
    offsets = torch.as_tensor(action_term._offset, dtype=torch.float32, device=env.device)
    if offsets.dim() == 0:
        offsets = offsets.expand(len(names))
    elif offsets.dim() > 1:
        offsets = offsets[0]
    residual = (offsets + offsets[perm]).abs().max().item()
    if residual > 1e-6:
        raise RuntimeError(
            f"action offsets are not mirror-antisymmetric (max |off_i + off_mirror| = {residual:.3e}); "
            "the default pose is not left-right symmetric and the sign map below is invalid"
        )
    scales = torch.as_tensor(action_term._scale, dtype=torch.float32, device=env.device)
    if scales.dim() == 0:
        scales = scales.expand(len(names))
    elif scales.dim() > 1:
        scales = scales[0]
    scale_residual = (scales - scales[perm]).abs().max().item()
    if scale_residual > 1e-9:
        raise RuntimeError(f"action scales differ across the mirror pair (max diff {scale_residual:.3e})")
    return torch.tensor(perm, dtype=torch.long, device=env.device)


def _check_joint_term_order(env: ManagerBasedRLEnv, group: str, term: str) -> None:
    cfgs = env.observation_manager._group_obs_term_cfgs[group]
    names = env.observation_manager.active_terms[group]
    cfg = cfgs[names.index(term)]
    asset_cfg = cfg.params.get("asset_cfg")
    if asset_cfg is None:
        return
    asset = env.scene[asset_cfg.name]
    obs_names = [asset.joint_names[i] for i in asset_cfg.joint_ids]
    action_names = list(env.action_manager.get_term("joint_pos")._joint_names)
    if obs_names != action_names:
        raise RuntimeError(
            f"observation term {term!r} resolves joints as {obs_names}, the action term as "
            f"{action_names}; the mirror permutation is built from the action order and would "
            "be applied to the wrong observation slots"
        )


def _check_pair_order(env: ManagerBasedRLEnv, group: str, term: str) -> None:
    cfgs = env.observation_manager._group_obs_term_cfgs[group]
    names = env.observation_manager.active_terms[group]
    params = cfgs[names.index(term)].params
    entity = params.get("asset_cfg", params.get("sensor_cfg"))
    bodies = list(getattr(entity, "body_names", None) or [])
    if len(bodies) != 2 or not bodies[0].startswith(("l_", "left_")) or bodies[1] != _mirror_name(bodies[0]):
        raise RuntimeError(f"observation term {term!r} must list its bodies as [l_*, r_*], got {bodies}")
    if not getattr(entity, "preserve_order", False):
        raise RuntimeError(f"observation term {term!r} needs preserve_order=True so its slots follow [l_*, r_*]")


def _group_layout(env: ManagerBasedRLEnv, group: str, perm: torch.Tensor):
    manager = env.observation_manager
    terms = manager.active_terms[group]
    dims = [int(d[0]) for d in manager.group_obs_term_dim[group]]
    history = manager._group_obs_term_cfgs[group][0].history_length or 1
    blocks: list[tuple[slice, torch.Tensor, torch.Tensor]] = []
    cursor = 0
    for name, width in zip(terms, dims):
        if width % history:
            raise RuntimeError(f"term {name!r} width {width} is not a multiple of history {history}")
        frame = width // history
        if name in _JOINT_TERMS or name in _JOINT_MAGNITUDE_TERMS:
            if frame != perm.numel():
                raise RuntimeError(f"term {name!r} frame dim {frame}, expected {perm.numel()} joints")
            _check_joint_term_order(env, group, name)
            index = perm
            sign = torch.full((frame,), -1.0 if name in _JOINT_TERMS else 1.0, device=env.device)
        elif name in _VECTOR_SIGNS:
            sign = torch.tensor(_VECTOR_SIGNS[name], dtype=torch.float32, device=env.device)
            if sign.numel() != frame:
                raise RuntimeError(f"term {name!r} frame dim {frame}, sign map has {sign.numel()}")
            index = torch.arange(frame, device=env.device)
        elif name in _FOOT_PAIR_TERMS:
            index = torch.tensor(_FOOT_PAIR_TERMS[name], dtype=torch.long, device=env.device)
            if index.numel() != frame:
                raise RuntimeError(f"term {name!r} frame dim {frame}, pair map has {index.numel()}")
            _check_pair_order(env, group, name)
            sign = torch.ones(frame, device=env.device)
        else:
            raise RuntimeError(f"observation term {name!r} has no mirror rule")
        blocks.append((slice(cursor, cursor + width), index, sign))
        cursor += width
    total = manager.group_obs_dim[group][0]
    if cursor != total:
        raise RuntimeError(f"group {group!r} layout covers {cursor} of {total} values")
    return blocks, history


def _layout(env: ManagerBasedRLEnv) -> dict:
    key = id(env)
    if key not in _CACHE:
        perm = _joint_permutation(env)
        groups = {}
        for group in env.observation_manager.active_terms:
            groups[group] = _group_layout(env, group, perm)
        names = list(env.action_manager.get_term("joint_pos")._joint_names)
        pairs = ", ".join(f"{names[i]}->-{names[j]}" for i, j in enumerate(perm.tolist()))
        print(f"[symmetry] mirror map resolved: {pairs}", flush=True)
        for group, (blocks, history) in groups.items():
            print(f"[symmetry]   group {group!r}: {len(blocks)} terms, history {history}", flush=True)
        _CACHE[key] = {"perm": perm, "groups": groups}
    return _CACHE[key]


def _mirror_group(obs: torch.Tensor, blocks, history: int, perm: torch.Tensor) -> torch.Tensor:
    out = obs.clone()
    for span, index, sign in blocks:
        chunk = out[:, span]
        view = chunk.view(chunk.shape[0], history, -1)
        out[:, span] = (view[:, :, index] * sign).reshape(chunk.shape)
    return out


@torch.no_grad()
def compute_symmetric_states(env, obs=None, actions=None):
    unwrapped = env.unwrapped
    layout = _layout(unwrapped)
    perm = layout["perm"]

    if obs is not None:
        batch = obs.batch_size[0]
        obs_aug = obs.repeat(2)
        for group, (blocks, history) in layout["groups"].items():
            if group not in obs.keys():
                continue
            obs_aug[group][:batch] = obs[group][:]
            obs_aug[group][batch:] = _mirror_group(obs[group], blocks, history, perm)
    else:
        obs_aug = None

    if actions is not None:
        batch = actions.shape[0]
        actions_aug = torch.zeros(batch * 2, actions.shape[1], device=actions.device, dtype=actions.dtype)
        actions_aug[:batch] = actions
        actions_aug[batch:] = -actions[:, perm]
    else:
        actions_aug = None

    return obs_aug, actions_aug
