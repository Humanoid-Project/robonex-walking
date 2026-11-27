import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "source/robonex_walking/robonex_walking/tasks/manager_based/robonex_walking"
MODULE = TASK / "mdp/symmetry.py"
CONSTANTS = ROOT.parent / "robonex-description/ver3/ver3_constants.json"
HAS_COMMON = importlib.util.find_spec("robonex_common") is not None

pytestmark = pytest.mark.skipif(not (CONSTANTS.exists() and HAS_COMMON), reason="ver3 constants or robonex_common missing")

EXPECTED_PERM = [1, 0, 3, 2, 5, 4, 7, 6, 10, 11, 8, 9]
TERMS = [
    ("joint_pos_rel", 12),
    ("joint_vel_rel", 12),
    ("imu_ang_vel", 3),
    ("projected_gravity", 3),
    ("velocity_commands", 3),
    ("gait_phase", 2),
    ("actions", 12),
    ("foot_contact", 4),
]
HISTORY = 5
FEET = ["left_ankle_pitch_link", "right_ankle_pitch_link"]


def _load_symmetry():
    spec = importlib.util.spec_from_file_location("robonex_symmetry_v3", MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module._CACHE.clear()
    return module


def _load_contract():
    pkg = "robonex_v3_symmetry_test"
    if pkg not in sys.modules:
        mod = types.ModuleType(pkg)
        mod.__path__ = [str(TASK)]
        sys.modules[pkg] = mod
    for name in ("robot_contract_v2", "robot_contract_v3"):
        spec = importlib.util.spec_from_file_location(f"{pkg}.{name}", TASK / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    return module


def _order():
    return list(json.loads(CONSTANTS.read_text())["policy_joint_order"])


class _Entity:
    def __init__(self, name="robot", joint_ids=None, body_names=None, preserve_order=False):
        self.name = name
        self.joint_ids = joint_ids
        self.body_names = body_names
        self.preserve_order = preserve_order


class _TermCfg:
    history_length = HISTORY

    def __init__(self, params=None):
        self.params = {} if params is None else params


def _env(feet=FEET, preserve_order=True):
    contract = _load_contract()
    order = _order()
    action = types.SimpleNamespace(
        _joint_names=list(order),
        _offset=torch.tensor([contract.ACTION_OFFSETS[n] for n in order]),
        _scale=torch.tensor([contract.ACTION_SCALES[n] for n in order]),
    )
    joints = _TermCfg({"asset_cfg": _Entity(joint_ids=list(range(len(order))))})
    contact = _TermCfg({"sensor_cfg": _Entity("contact_forces", body_names=list(feet), preserve_order=preserve_order)})
    cfgs = [joints, joints, _TermCfg(), _TermCfg(), _TermCfg(), _TermCfg(), _TermCfg(), contact]
    manager = types.SimpleNamespace(
        active_terms={"policy": [n for n, _ in TERMS]},
        group_obs_term_dim={"policy": [(d * HISTORY,) for _, d in TERMS]},
        group_obs_dim={"policy": (sum(d for _, d in TERMS) * HISTORY,)},
        _group_obs_term_cfgs={"policy": cfgs},
    )
    return types.SimpleNamespace(
        device="cpu",
        action_manager=types.SimpleNamespace(get_term=lambda name: action),
        observation_manager=manager,
        scene={"robot": types.SimpleNamespace(joint_names=list(order))},
    )


def test_mirror_name_maps_g1_prefixes():
    module = _load_symmetry()
    assert module._mirror_name("left_hip_roll_joint") == "right_hip_roll_joint"
    assert module._mirror_name("right_ankle_pitch_link") == "left_ankle_pitch_link"
    assert module._mirror_name("l_hip_roll_joint") == "r_hip_roll_joint"
    for name in ("neck_yaw_joint", "lefthip_joint", "head_link"):
        with pytest.raises(ValueError):
            module._mirror_name(name)


def test_ver3_articulation_order_gives_the_ver2_permutation():
    module = _load_symmetry()
    assert module._joint_permutation(_env()).tolist() == EXPECTED_PERM


def test_ver3_contract_offsets_are_mirror_antisymmetric():
    module = _load_symmetry()
    env = _env()
    perm = module._joint_permutation(env)
    offsets = env.action_manager.get_term("joint_pos")._offset
    torch.testing.assert_close(offsets + offsets[perm], torch.zeros_like(offsets))


def test_pair_order_accepts_left_then_right():
    module = _load_symmetry()
    module._check_pair_order(_env(), "policy", "foot_contact")
    layout = module._layout(_env())
    blocks, history = layout["groups"]["policy"]
    width = sum(d for _, d in TERMS) * HISTORY
    obs = torch.randn(4, width)
    once = module._mirror_group(obs, blocks, history, layout["perm"])
    twice = module._mirror_group(once, blocks, history, layout["perm"])
    torch.testing.assert_close(twice, obs)
    assert not torch.allclose(once, obs)


@pytest.mark.parametrize("feet,preserve", [(list(reversed(FEET)), True), (FEET, False),
                                           (["left_ankle_pitch_link", "right_ankle_roll_link"], True)])
def test_pair_order_rejects_wrong_feet(feet, preserve):
    module = _load_symmetry()
    with pytest.raises(RuntimeError):
        module._check_pair_order(_env(feet=feet, preserve_order=preserve), "policy", "foot_contact")
