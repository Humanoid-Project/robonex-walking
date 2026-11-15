import ast
import importlib.util
import math
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "source/robonex_walking/robonex_walking/tasks/manager_based/robonex_walking"
SYMMETRY = TASK / "mdp/symmetry.py"
OBS = TASK / "mdp/observations.py"
REWARDS = TASK / "mdp/rewards.py"
CFG = TASK / "robonex_walking_v2_env_cfg.py"

JOINT_ORDER = [
    "l_hip_yaw_joint", "r_hip_yaw_joint", "l_hip_pitch_joint", "r_hip_pitch_joint",
    "l_hip_roll_joint", "r_hip_roll_joint", "l_knee_pitch_joint", "r_knee_pitch_joint",
    "l_ankle_lower_joint", "l_ankle_upper_joint", "r_ankle_lower_joint", "r_ankle_upper_joint",
]
OFFSETS = [0.0, 0.0, 0.1, -0.1, 0.0, 0.0, -0.3857, 0.3857, -0.2057, 0.2057, 0.2057, -0.2057]
SCALES = [0.25] * 12
POLICY = [("joint_pos_rel", 12), ("joint_vel_rel", 12), ("imu_ang_vel", 3), ("projected_gravity", 3),
          ("velocity_commands", 3), ("gait_phase", 2), ("actions", 12)]
CRITIC = [("base_lin_vel", 3), ("joint_pos_clean", 12), ("joint_vel_clean", 12), ("base_ang_vel", 3),
          ("projected_gravity_clean", 3), ("base_height_above_feet", 1), ("foot_contact", 4),
          ("joint_friction", 12), ("foot_friction", 4), ("base_mass_delta", 1), ("base_com_offset", 3),
          ("randomization_delays", 3), ("last_push", 3)]
JOINT_PARAM_TERMS = {"joint_pos_rel", "joint_vel_rel", "joint_pos_clean", "joint_vel_clean", "joint_friction"}
FOOT_TERMS = {"base_height_above_feet", "foot_contact", "foot_friction"}
HISTORY = 5
PERM = [1, 0, 3, 2, 5, 4, 7, 6, 10, 11, 8, 9]


def _symmetry():
    spec = importlib.util.spec_from_file_location("robonex_symmetry_priv", SYMMETRY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module._CACHE.clear()
    return module


class _Entity:
    def __init__(self, name="robot", joint_ids=None, body_names=None, preserve_order=False):
        self.name = name
        self.joint_ids = list(range(12)) if joint_ids is None else joint_ids
        self.body_names = body_names
        self.preserve_order = preserve_order


class _TermCfg:
    history_length = HISTORY

    def __init__(self, params=None):
        self.params = {} if params is None else params


def _term_cfg(name):
    if name in JOINT_PARAM_TERMS:
        return _TermCfg({"asset_cfg": _Entity()})
    if name in FOOT_TERMS:
        key = "sensor_cfg" if name == "foot_contact" else "asset_cfg"
        return _TermCfg({key: _Entity(body_names=["l_foot", "r_foot"], preserve_order=True)})
    return _TermCfg()


class _Env:
    device = "cpu"

    def __init__(self, critic=CRITIC):
        term = types.SimpleNamespace(_joint_names=list(JOINT_ORDER), _offset=torch.tensor(OFFSETS),
                                     _scale=torch.tensor(SCALES))
        self.action_manager = types.SimpleNamespace(get_term=lambda name: term)
        groups = {"policy": POLICY, "critic": critic}
        self.observation_manager = types.SimpleNamespace(
            active_terms={g: [n for n, _ in t] for g, t in groups.items()},
            group_obs_term_dim={g: [(d * HISTORY,) for _, d in t] for g, t in groups.items()},
            group_obs_dim={g: (sum(d for _, d in t) * HISTORY,) for g, t in groups.items()},
            _group_obs_term_cfgs={g: [_term_cfg(n) for n, _ in t] for g, t in groups.items()},
        )
        self.scene = {"robot": types.SimpleNamespace(joint_names=list(JOINT_ORDER))}

    @property
    def unwrapped(self):
        return self


def _starts(terms):
    starts, cursor = {}, 0
    for name, dim in terms:
        starts[name] = cursor
        cursor += dim * HISTORY
    return starts


def test_critic_width_is_320_and_the_critic_input_555():
    env = _Env()
    critic = env.observation_manager.group_obs_dim["critic"][0]
    policy = env.observation_manager.group_obs_dim["policy"][0]
    assert (policy, critic, policy + critic) == (235, 320, 555)
    layout = _symmetry()._layout(env)
    blocks, history = layout["groups"]["critic"]
    assert history == HISTORY and len(blocks) == len(CRITIC)


def test_critic_mirror_is_an_involution():
    module = _symmetry()
    env = _Env()
    layout = module._layout(env)
    for group in ("policy", "critic"):
        blocks, history = layout["groups"][group]
        obs = torch.randn(9, env.observation_manager.group_obs_dim[group][0])
        once = module._mirror_group(obs, blocks, history, layout["perm"])
        torch.testing.assert_close(module._mirror_group(once, blocks, history, layout["perm"]), obs)
        assert not torch.allclose(once, obs)


def test_each_critic_term_gets_its_rule():
    module = _symmetry()
    env = _Env()
    layout = module._layout(env)
    blocks, history = layout["groups"]["critic"]
    width = env.observation_manager.group_obs_dim["critic"][0]
    obs = torch.arange(width, dtype=torch.float32).unsqueeze(0) + 1.0
    out = module._mirror_group(obs, blocks, history, layout["perm"])[0]
    s = _starts(CRITIC)

    def frame(name, dim, k=0):
        lo = s[name] + k * dim
        return obs[0, lo:lo + dim], out[lo:lo + dim]

    for k in range(HISTORY):
        src, dst = frame("joint_pos_clean", 12, k)
        torch.testing.assert_close(dst, -src[PERM])
        src, dst = frame("joint_vel_clean", 12, k)
        torch.testing.assert_close(dst, -src[PERM])
        src, dst = frame("joint_friction", 12, k)
        torch.testing.assert_close(dst, src[PERM])
        for name in ("foot_contact", "foot_friction"):
            src, dst = frame(name, 4, k)
            torch.testing.assert_close(dst, src[[1, 0, 3, 2]])
    signs = {
        "base_lin_vel": [1, -1, 1], "base_ang_vel": [-1, 1, -1], "projected_gravity_clean": [1, -1, 1],
        "base_height_above_feet": [1], "base_mass_delta": [1], "base_com_offset": [1, -1, 1],
        "randomization_delays": [1, 1, 1], "last_push": [1, -1, 1],
    }
    for name, sign in signs.items():
        for k in range(HISTORY):
            src, dst = frame(name, len(sign), k)
            torch.testing.assert_close(dst, src * torch.tensor(sign, dtype=torch.float32))


def test_policy_mirror_is_unchanged_by_the_generalised_blocks():
    module = _symmetry()
    env = _Env(critic=[("base_lin_vel", 3)])
    layout = module._layout(env)
    blocks, history = layout["groups"]["policy"]
    obs = torch.randn(4, 235)
    out = module._mirror_group(obs, blocks, history, layout["perm"])
    view = obs.view(4, -1)
    s = _starts(POLICY)
    expected = obs.clone()
    for name, dim in POLICY:
        for k in range(HISTORY):
            lo = s[name] + k * dim
            chunk = view[:, lo:lo + dim]
            if name in ("joint_pos_rel", "joint_vel_rel", "actions"):
                expected[:, lo:lo + dim] = -chunk[:, PERM]
            else:
                sign = torch.tensor(module._VECTOR_SIGNS[name])
                expected[:, lo:lo + dim] = chunk * sign
    torch.testing.assert_close(out, expected)


@pytest.mark.parametrize("bodies,preserve", [(["r_foot", "l_foot"], True), (["l_foot", "r_foot"], False),
                                             (["l_foot"], True)])
def test_foot_pair_order_is_enforced(bodies, preserve):
    module = _symmetry()
    env = _Env()
    names = env.observation_manager.active_terms["critic"]
    cfg = env.observation_manager._group_obs_term_cfgs["critic"][names.index("foot_contact")]
    cfg.params["sensor_cfg"] = _Entity(body_names=bodies, preserve_order=preserve)
    with pytest.raises(RuntimeError, match="foot_contact"):
        module._layout(env)


def test_joint_friction_order_drift_is_rejected():
    module = _symmetry()
    env = _Env()
    names = env.observation_manager.active_terms["critic"]
    cfg = env.observation_manager._group_obs_term_cfgs["critic"][names.index("joint_friction")]
    reordered = list(range(12))
    reordered[0], reordered[2] = reordered[2], reordered[0]
    cfg.params["asset_cfg"] = _Entity(joint_ids=reordered)
    with pytest.raises(RuntimeError, match="wrong observation slots"):
        module._layout(env)


def _class_terms(tree, name):
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
    return node, [t.targets[0].id for t in node.body if isinstance(t, ast.Assign)]


def test_cfg_term_order_matches_this_test_and_every_term_has_a_rule():
    tree = ast.parse(CFG.read_text())
    obs = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ObservationsCfg")
    critic = next(n for n in obs.body if isinstance(n, ast.ClassDef) and n.name == "CriticCfg")
    base = [t.targets[0].id for t in critic.body if isinstance(t, ast.Assign)]
    _, extra = _class_terms(tree, "PrivilegedCriticCfg")
    assert base + extra == [n for n, _ in CRITIC]
    module = _symmetry()
    for name in base + extra:
        assert (name in module._VECTOR_SIGNS or name in module._JOINT_TERMS
                or name in module._JOINT_MAGNITUDE_TERMS or name in module._FOOT_PAIR_TERMS), name


def test_flat_critic_group_holds_only_base_lin_vel():
    tree = ast.parse(CFG.read_text())
    groups = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "CriticCfg"]
    assert len(groups) == 1
    terms = [t.targets[0].id for t in groups[0].body if isinstance(t, ast.Assign)]
    assert terms == ["base_lin_vel"]
    tops = [n.name for n in tree.body if isinstance(n, ast.ClassDef)]
    assert tops.index("RoboNexWalkingV2EduPrivCriticEnvCfg") > tops.index("RoboNexWalkingV2EduEnvCfg")


class _Base:
    def __init__(self, cfg, env):
        self.cfg = cfg
        self._env = env
        self.num_envs = env.num_envs
        self.device = "cpu"


def _load(path, names, extra=None):
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if getattr(n, "name", None) in names]
    ns = {"torch": torch, "ManagerTermBase": _Base, "ManagerBasedRLEnv": object, "Articulation": object,
          "SceneEntityCfg": lambda *a, **k: None}
    ns.update(extra or {})
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), ns)
    return ns


def test_base_height_contact_friction_and_delays():
    ns = _load(OBS, {"base_height_above_feet", "foot_contact", "joint_friction", "randomization_delays"})
    data = types.SimpleNamespace(
        root_pos_w=torch.tensor([[0.0, 0.0, 0.95], [0.0, 0.0, 0.80]]),
        body_pos_w=torch.tensor([[[0, 0, 0.07], [0, 0, 0.12], [0, 0, 0.5]], [[0, 0, 0.20], [0, 0, 0.06], [0, 0, 0.5]]]),
        joint_dynamic_friction_coeff=torch.arange(24.0).view(2, 12),
    )
    sensor = types.SimpleNamespace(data=types.SimpleNamespace(
        net_forces_w=torch.tensor([[[0, 0, 241.6], [0, 0, 5.0]], [[0, 0, -3.0], [1, 2, 120.8]]])))
    imu = types.SimpleNamespace(_delay=torch.tensor([2, 6]))
    action = types.SimpleNamespace(_delay_substeps=torch.tensor([0, 3]))
    class _Scene(dict):
        sensors = {"contact_forces": sensor}

    env = types.SimpleNamespace(
        num_envs=2, device="cpu", scene=_Scene(robot=types.SimpleNamespace(data=data), imu=imu),
        action_manager=types.SimpleNamespace(get_term=lambda name: action), _joint_observation_delay=torch.tensor([1, 1]),
    )
    feet = types.SimpleNamespace(name="robot", body_ids=[0, 1])
    torch.testing.assert_close(ns["base_height_above_feet"](env, feet), torch.tensor([[0.88], [0.74]]))
    contact = ns["foot_contact"](env, types.SimpleNamespace(name="contact_forces", body_ids=[0, 1]), weight_n=241.6)
    torch.testing.assert_close(contact, torch.tensor([[1.0, 0.0, 1.0, 5.0 / 241.6], [0.0, 1.0, 0.0, 0.5]]))
    with pytest.raises(ValueError):
        ns["foot_contact"](env, types.SimpleNamespace(name="contact_forces", body_ids=[0, 1]), weight_n=0.0)
    joints = types.SimpleNamespace(name="robot", joint_ids=PERM)
    torch.testing.assert_close(ns["joint_friction"](env, joints), data.joint_dynamic_friction_coeff[:, PERM])
    delays = ns["randomization_delays"](env, types.SimpleNamespace(name="imu"))
    torch.testing.assert_close(delays, torch.tensor([[1.0, 2.0, 0.0], [1.0, 6.0, 3.0]]))
    del env._joint_observation_delay
    with pytest.raises(RuntimeError, match="delayed_joint_state"):
        ns["randomization_delays"](env, types.SimpleNamespace(name="imu"))


def _yaw_quat(q):
    w, x, y, z = q.unbind(-1)
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    zero = torch.zeros_like(yaw)
    return torch.stack([torch.cos(yaw / 2), zero, zero, torch.sin(yaw / 2)], dim=-1)


def _quat_apply_inverse(q, v):
    w, xyz = q[:, :1], -q[:, 1:]
    t = 2.0 * torch.linalg.cross(xyz, v, dim=-1)
    return v + w * t + torch.linalg.cross(xyz, t, dim=-1)


def test_push_record_in_yaw_frame_and_last_push_reset():
    yaw = math.pi / 2
    quat = torch.tensor([[1.0, 0, 0, 0], [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)], [1.0, 0, 0, 0]])
    data = types.SimpleNamespace(root_vel_w=torch.zeros(3, 6), root_quat_w=quat)

    def fake_push(env, env_ids, velocity_range, asset_cfg):
        vel = data.root_vel_w[env_ids].clone()
        vel[:, 0] += 0.3
        vel[:, 1] += -0.2
        data.root_vel_w[env_ids] = vel

    ns = _load(REWARDS, {"_record_push", "_mark_sway_push", "push_robot_with_sway_marker"},
               {"_push_by_setting_velocity": fake_push, "quat_apply_inverse": _quat_apply_inverse,
                "yaw_quat": _yaw_quat})
    env = types.SimpleNamespace(num_envs=3, device="cpu", common_step_counter=100, step_dt=0.02,
                                scene={"robot": types.SimpleNamespace(data=data)})
    obs = _load(OBS, {"last_push"})["last_push"]
    term = obs(types.SimpleNamespace(params={"cap_s": 2.0}), env)
    asset_cfg = types.SimpleNamespace(name="robot")
    ns["push_robot_with_sway_marker"](env, torch.tensor([0, 1]), {}, asset_cfg)
    env.common_step_counter = 125
    out = term(env, 2.0)
    torch.testing.assert_close(out[0], torch.tensor([0.3, -0.2, 0.5]))
    torch.testing.assert_close(out[1], torch.tensor([-0.2, -0.3, 0.5]), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(out[2], torch.tensor([0.0, 0.0, 2.0]))
    env.common_step_counter = 1000
    assert term(env, 2.0)[0, 2].item() == 2.0
    term.reset(torch.tensor([0]))
    torch.testing.assert_close(term(env, 2.0)[0], torch.tensor([0.0, 0.0, 2.0]))
    assert env._sway_last_push_step.tolist() == [100, 100, -1]
    ns["push_robot_with_sway_marker"](env, torch.tensor([], dtype=torch.long), {}, asset_cfg)


def test_episode_draws_read_once_per_reset():
    calls = []
    masses = torch.tensor([[11.0, 1.0], [10.7, 1.0]])
    coms = torch.zeros(2, 2, 7)
    coms[1, 0, :3] = torch.tensor([0.01, -0.02, 0.0])
    shapes = torch.zeros(2, 5, 3)
    shapes[0, :, 0] = torch.tensor([0.9, 0.4, 0.4, 0.7, 0.7])
    shapes[0, :, 1] = torch.tensor([0.8, 0.3, 0.3, 0.6, 0.6])
    shapes[1, :, 0] = 0.5
    shapes[1, :, 1] = 0.5
    view = types.SimpleNamespace(
        get_masses=lambda: (calls.append("m"), masses.clone())[1],
        get_coms=lambda: coms.clone(),
        get_material_properties=lambda: shapes.clone(),
        link_paths=[["base", "l_foot", "r_foot"]],
        max_shapes=5,
    )
    counts = {"base": 1, "l_foot": 2, "r_foot": 2}
    sim_view = types.SimpleNamespace(create_rigid_body_view=lambda p: types.SimpleNamespace(max_shapes=counts[p]))
    asset = types.SimpleNamespace(root_physx_view=view, _physics_sim_view=sim_view, num_bodies=3,
                                  data=types.SimpleNamespace(default_mass=torch.tensor([[10.0, 1.0], [10.0, 1.0]])))
    env = types.SimpleNamespace(num_envs=2, device="cpu", scene={"robot": asset})
    ns = _load(OBS, {"_body_shape_ranges", "_EpisodeDraw", "_single_body", "base_mass_delta", "base_com_offset",
                     "foot_friction"})
    base = types.SimpleNamespace(name="robot", body_ids=[0])
    mass = ns["base_mass_delta"](types.SimpleNamespace(params={"asset_cfg": base}), env)
    torch.testing.assert_close(mass(env, base), torch.tensor([[1.0], [0.7]]), atol=1e-6, rtol=1e-6)
    mass(env, base)
    assert calls == ["m"]
    masses[0, 0] = 10.2
    mass.reset(torch.tensor([0]))
    torch.testing.assert_close(mass(env, base), torch.tensor([[0.2], [0.7]]), atol=1e-6, rtol=1e-6)
    assert calls == ["m", "m"]
    com = ns["base_com_offset"](types.SimpleNamespace(params={"asset_cfg": base}), env)
    coms[1, 0, :3] = torch.tensor([0.03, -0.02, 0.0])
    com.reset()
    torch.testing.assert_close(com(env, base), torch.tensor([[0.0, 0.0, 0.0], [0.02, 0.0, 0.0]]))
    feet = types.SimpleNamespace(name="robot", body_ids=[1, 2])
    fric = ns["foot_friction"](types.SimpleNamespace(params={"asset_cfg": feet}), env)
    torch.testing.assert_close(fric(env, feet), torch.tensor([[0.4, 0.7, 0.3, 0.6], [0.5, 0.5, 0.5, 0.5]]))
    view.max_shapes = 6
    with pytest.raises(RuntimeError, match="does not add up"):
        ns["foot_friction"](types.SimpleNamespace(params={"asset_cfg": feet}), env)


ISAAC_DICT = Path.home() / "IsaacLab/source/isaaclab/isaaclab/utils/dict.py"


def _update():
    from collections.abc import Iterable, Mapping, Sized

    node = next(n for n in ast.parse(ISAAC_DICT.read_text()).body
                if getattr(n, "name", None) == "update_class_from_dict")
    ns = {"Iterable": Iterable, "Mapping": Mapping, "Sized": Sized, "Any": object, "string_to_callable": lambda s: s}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "dict", "exec"), ns)
    return ns["update_class_from_dict"]


def _parse(text):
    from hydra.core.override_parser.overrides_parser import OverridesParser

    override = OverridesParser.create().parse_overrides([text])[0]
    data = override.value()
    for key in reversed(override.key_or_group.split(".")[1:]):
        data = {key: data}
    return data


def _critic(**terms):
    return types.SimpleNamespace(observations=types.SimpleNamespace(critic=types.SimpleNamespace(**terms)))


@pytest.mark.skipif(not ISAAC_DICT.exists(), reason="Isaac Lab source not present")
def test_hydra_can_tune_or_drop_a_privileged_term_but_not_add_one():
    pytest.importorskip("hydra")
    update = _update()
    env = _critic(foot_contact=types.SimpleNamespace(params={"threshold_n": 20.0, "weight_n": 241.6}),
                  joint_friction=types.SimpleNamespace(params={}))
    update(env, _parse("env.observations.critic.foot_contact.params.threshold_n=10.0"))
    assert env.observations.critic.foot_contact.params["threshold_n"] == 10.0
    update(env, _parse("env.observations.critic.joint_friction=null"))
    assert env.observations.critic.joint_friction is None
    with pytest.raises((KeyError, AttributeError, ValueError)):
        update(env, _parse("env.observations.critic.joint_friction.params.foo=1.0"))
    with pytest.raises(KeyError):
        update(_critic(base_lin_vel=types.SimpleNamespace()), _parse("env.observations.critic.last_push.params.cap_s=2.0"))
