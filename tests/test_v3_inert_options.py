import ast
import json
import math
import types
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "source/robonex_walking/robonex_walking/tasks/manager_based/robonex_walking"
EVENTS = TASK / "mdp/events.py"
REWARDS = TASK / "mdp/rewards.py"
V2_CFG = TASK / "robonex_walking_v2_env_cfg.py"
V3_CFG = TASK / "robonex_walking_v3_env_cfg.py"
V3_CONTRACT = TASK / "robot_contract_v3.py"
CONSTANTS = ROOT.parent / "robonex-description/ver3/ver3_constants.json"
ARMS = tuple(f"{side}_{role}_joint" for side in ("left", "right")
             for role in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll"))
LEGS = tuple(f"{side}_{role}_joint" for side in ("left", "right")
             for role in ("hip_yaw", "hip_pitch", "hip_roll", "knee_pitch", "ankle_upper", "ankle_lower"))
NECK = ("neck_yaw_joint", "neck_pitch_joint")
DT = 0.02


class FakeTermBase:
    def __init__(self, cfg, env):
        self.cfg = cfg
        self._env = env


def exec_nodes(path, names, ns):
    nodes = [n for n in ast.parse(path.read_text()).body if getattr(n, "name", None) in names]
    assert {n.name for n in nodes} == set(names), names
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), ns)
    return ns


def arm_limits():
    if CONSTANTS.exists():
        lim = json.loads(CONSTANTS.read_text())["provisional_limits"]
        return {n: tuple(lim[n]) for n in ARMS}
    return {n: ((-1.745329, 1.745329) if "shoulder_pitch" in n else (-0.785398, 0.785398)) for n in ARMS}


def cfg_tree():
    tree = ast.parse(V3_CFG.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RoboNexWalkingV3EnvCfg")
    consts = {n.targets[0].id: ast.literal_eval(n.value) for n in tree.body
              if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id.isupper()
              and not n.targets[0].id.startswith("_")}
    return tree, cls, consts


def method(cls, name):
    return next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)


class FakeAsset:
    def __init__(self, names, num_envs):
        self.joint_names = list(names)
        self.num_joints = len(names)
        self.device = "cpu"
        self.data = types.SimpleNamespace(default_joint_pos=torch.zeros(num_envs, self.num_joints),
                                          joint_pos_target=torch.zeros(num_envs, self.num_joints),
                                          joint_pos=torch.zeros(num_envs, self.num_joints))
        self.target_calls = []
        self.state_calls = []

    def set_joint_position_target(self, target, joint_ids=None, env_ids=None):
        self.target_calls.append(list(joint_ids))
        self.data.joint_pos_target[env_ids[:, None], torch.as_tensor(joint_ids)] = target

    def write_joint_state_to_sim(self, position, velocity, joint_ids=None, env_ids=None):
        self.state_calls.append(list(joint_ids))
        self.data.joint_pos[env_ids[:, None], torch.as_tensor(joint_ids)] = position


def arm_term(range_scale, num_envs=128):
    ns = exec_nodes(EVENTS, ("neck_target_bounds", "neck_target_timing", "randomize_neck_target"),
                    {"torch": torch, "math": math, "SceneEntityCfg": object, "Articulation": object,
                     "ManagerTermBase": FakeTermBase})
    names = list(LEGS) + list(ARMS) + list(NECK)
    asset = FakeAsset(names, num_envs)
    ids = [names.index(n) for n in ARMS]
    params = {"asset_cfg": types.SimpleNamespace(name="robot", joint_ids=ids), "limits": arm_limits(), "margin": 0.05,
              "range_scale": range_scale, "resample_s": (1.0, 3.0), "blend_s": 1.0, "start_at_default": False,
              "write_state_at_reset": True}
    env = types.SimpleNamespace(scene={"robot": asset}, num_envs=num_envs, step_dt=DT, device="cpu")
    return ns, ns["randomize_neck_target"](types.SimpleNamespace(params=params), env), env, asset, params, ids


class ArmTargetTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_bounds_are_the_arm_limits_minus_margin(self):
        ns, *_ = arm_term(1.0)
        low, high = ns["neck_target_bounds"](list(ARMS), arm_limits(), 0.05, 1.0)
        for i, name in enumerate(ARMS):
            self.assertAlmostEqual(float(low[i]), arm_limits()[name][0] + 0.05, places=6)
            self.assertAlmostEqual(float(high[i]), arm_limits()[name][1] - 0.05, places=6)

    def test_active_term_writes_only_the_arm_columns_inside_the_range(self):
        ns, term, env, asset, params, ids = arm_term(0.3)
        low, high = ns["neck_target_bounds"](list(ARMS), arm_limits(), 0.05, 0.3)
        term.reset(None)
        for _ in range(300):
            term(env, torch.arange(env.num_envs), **params)
            self.assertTrue(bool(torch.all(term.target >= low - 1e-6) and torch.all(term.target <= high + 1e-6)))
        others = [j for j in range(asset.num_joints) if j not in ids]
        self.assertTrue(bool(torch.all(asset.data.joint_pos_target[:, others] == 0.0)))
        self.assertGreater(float(asset.data.joint_pos_target[:, ids].abs().max()), 0.1)
        self.assertTrue(all(c == ids for c in asset.target_calls + asset.state_calls))

    def test_inert_default_writes_nothing(self):
        _, term, env, asset, params, _ = arm_term(0.0)
        term.reset(None)
        term(env, torch.arange(env.num_envs), **params)
        self.assertEqual(asset.target_calls + asset.state_calls, [])
        self.assertIsNone(term.prev)

    def test_cfg_term_is_inert_full_only_and_reuses_the_neck_term(self):
        tree, cls, consts = cfg_tree()
        self.assertEqual(consts["ARM_TARGET_RANGE_SCALE"], 0.0)
        self.assertEqual(consts["ARM_TARGET_MARGIN_RAD"], 0.05)
        src = ast.unparse(method(cls, "_arm_target_term"))
        self.assertIn("func=mdp.randomize_neck_target", src)
        self.assertIn("mode='interval'", src)
        self.assertIn("interval_range_s=(step_s, step_s)", src)
        self.assertIn("joint_names=list(v3.ARM_JOINTS)", src)
        self.assertIn("v3.ARM_LIMITS", src)
        self.assertIn("'range_scale': ARM_TARGET_RANGE_SCALE", src)
        self.assertNotIn("ranges", src.replace("range_scale", ""))
        post = method(cls, "__post_init__")
        hits = [n for n in ast.walk(post) if isinstance(n, ast.If) and "arm_target" in ast.unparse(n)]
        self.assertEqual(len(hits), 1)
        self.assertEqual(ast.unparse(hits[0].test), "self.model_variant == 'full'")
        self.assertEqual([n for n in post.body if not isinstance(n, ast.If) and "arm_target" in ast.unparse(n)], [])
        self.assertNotIn("arm_target", V2_CFG.read_text())

    def test_contract_exposes_arm_limits_and_hand_bodies(self):
        src = V3_CONTRACT.read_text()
        self.assertIn("ARM_LIMITS = {name: _limits[name] for name in ARM_JOINTS}", src)
        self.assertIn('HAND_BODIES = ("left_wrist_roll_link", "right_wrist_roll_link")', src)


class FakeMassBase:
    calls = []

    def __init__(self, cfg, env):
        self.cfg = cfg

    def __call__(self, *args):
        FakeMassBase.calls.append(args)


def payload_class():
    ns = {"torch": torch, "SceneEntityCfg": object, "_isaaclab_events": types.SimpleNamespace(
        randomize_rigid_body_mass=FakeMassBase)}
    return exec_nodes(EVENTS, ("randomize_payload_mass",), ns)["randomize_payload_mass"]


class HandPayloadTest(unittest.TestCase):
    def setUp(self):
        FakeMassBase.calls = []

    def call(self, params, operation="add"):
        term = payload_class()(types.SimpleNamespace(params={}), None)
        term("env", None, "asset_cfg", params, operation)
        return FakeMassBase.calls

    def test_zero_add_range_never_reaches_isaaclab(self):
        self.assertEqual(self.call((0.0, 0.0)), [])
        self.assertEqual(self.call([0, 0]), [])
        self.assertEqual(self.call((-0.0, 0.0)), [])

    def test_a_nonzero_range_or_another_operation_is_passed_on_unchanged(self):
        calls = self.call((0.0, 0.5))
        self.assertEqual(calls, [("env", None, "asset_cfg", (0.0, 0.5), "add", "uniform", True, 1e-6)])
        self.assertEqual(len(self.call((0.0, 0.0), operation="abs")), 2)

    def test_signature_matches_isaaclab_term(self):
        node = next(n for n in ast.parse(EVENTS.read_text()).body if getattr(n, "name", None) == "randomize_payload_mass")
        call = next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == "__call__")
        names = [a.arg for a in call.args.args]
        self.assertEqual(names, ["self", "env", "env_ids", "asset_cfg", "mass_distribution_params", "operation",
                                 "distribution", "recompute_inertia", "min_mass"])
        self.assertEqual([ast.literal_eval(d) for d in call.args.defaults], ["uniform", True, 1e-6])

    def test_cfg_term_is_inert_in_both_variants(self):
        tree, cls, consts = cfg_tree()
        self.assertEqual(consts["HAND_PAYLOAD_KG"], (0.0, 0.0))
        src = ast.unparse(method(cls, "_hand_payload_term"))
        self.assertIn("func=mdp.randomize_payload_mass", src)
        self.assertIn("mode='reset'", src)
        self.assertIn("body_names=list(v3.HAND_BODIES)", src)
        self.assertIn("'mass_distribution_params': HAND_PAYLOAD_KG", src)
        self.assertIn("'operation': 'add'", src)
        post = method(cls, "__post_init__")
        top = [ast.unparse(n) for n in post.body if not isinstance(n, ast.If)]
        self.assertIn("self.events.hand_payload = self._hand_payload_term()", top)
        self.assertNotIn("hand_payload", V2_CFG.read_text())


class Env:
    def __init__(self, n):
        self.common_step_counter = 0
        self.command = torch.zeros(n, 3)
        self.vel = torch.zeros(n, 3)
        self.asset = types.SimpleNamespace(data=types.SimpleNamespace(root_ang_vel_b=torch.zeros(n, 3)))
        self.scene = {"robot": self.asset}
        self.command_manager = types.SimpleNamespace(get_command=lambda name: self.command)


def reward_classes():
    ns = {"torch": torch, "ManagerTermBase": FakeTermBase, "ManagerBasedRLEnv": object, "Articulation": object,
          "SceneEntityCfg": lambda *a, **k: types.SimpleNamespace(name="robot")}
    ns["_root_lin_vel_yaw_frame"] = lambda env, cfg: env.vel
    exec_nodes(REWARDS, ("track_lin_vel_y_cycle_exp", "track_ang_vel_z_cycle_exp"), ns)
    return ns["track_lin_vel_y_cycle_exp"], ns["track_ang_vel_z_cycle_exp"]


class TrackAngVelZCycleTest(unittest.TestCase):
    def step(self, term, env, wz, k):
        env.common_step_counter = k
        env.asset.data.root_ang_vel_b[:, 2] = wz
        return term(env, std=0.25, window_steps=40)

    def test_yaw_oscillation_around_the_command_is_not_penalised_but_a_bias_is(self):
        _, z_cls = reward_classes()
        env = Env(2)
        env.command[:, 2] = 0.2
        term = z_cls(types.SimpleNamespace(params={}), env)
        out = None
        for k in range(80):
            wobble = 0.4 * math.sin(2 * math.pi * k / 40)
            out = self.step(term, env, torch.tensor([0.2 + wobble, 0.1 + wobble]), k)
        self.assertGreater(float(out[0]), 0.99)
        self.assertAlmostEqual(float(out[1]), math.exp(-(0.1 / 0.25) ** 2), places=3)
        self.assertLess(math.exp(-(0.4 / 0.25) ** 2), 0.1)

    def test_command_change_and_reset_clear_the_window(self):
        _, z_cls = reward_classes()
        env = Env(2)
        term = z_cls(types.SimpleNamespace(params={}), env)
        for k in range(40):
            self.step(term, env, 0.0, k)
        env.command[0, 2] = 0.3
        out = self.step(term, env, 0.3, 40)
        self.assertAlmostEqual(float(out[0]), 1.0, places=6)
        self.assertAlmostEqual(float(out[1]), math.exp(-(0.3 / 40 / 0.25) ** 2), places=6)
        term.reset(torch.tensor([1]))
        out = self.step(term, env, 0.0, 41)
        self.assertAlmostEqual(float(out[1]), 1.0, places=6)
        self.assertEqual(int(term._count[1]), 1)

    def test_same_step_call_does_not_advance_and_nan_is_zeroed(self):
        _, z_cls = reward_classes()
        env = Env(1)
        term = z_cls(types.SimpleNamespace(params={}), env)
        env.asset.data.root_ang_vel_b[:, 2] = float("nan")
        a = term(env, std=0.25, window_steps=40)
        b = term(env, std=0.25, window_steps=40)
        self.assertTrue(torch.equal(a, b))
        self.assertEqual(float(a[0]), 1.0)
        self.assertEqual(int(term._count[0]), 1)

    def test_equals_the_lateral_term_on_the_same_signal(self):
        y_cls, z_cls = reward_classes()
        torch.manual_seed(1)
        ey, ez = Env(16), Env(16)
        ty = y_cls(types.SimpleNamespace(params={}), ey)
        tz = z_cls(types.SimpleNamespace(params={}), ez)
        for k in range(200):
            if k % 50 == 0:
                cmd = torch.rand(16) - 0.5
                ey.command[:, 1], ez.command[:, 2] = cmd, cmd
            sig = torch.randn(16) * 0.5
            ey.vel[:, 1] = sig
            ez.asset.data.root_ang_vel_b[:, 2] = sig
            ey.common_step_counter = ez.common_step_counter = k
            if k == 120:
                ty.reset(torch.tensor([3, 7]))
                tz.reset(torch.tensor([3, 7]))
            self.assertTrue(torch.equal(ty(ey, std=0.25, window_steps=40), tz(ez, std=0.25, window_steps=40)))

    def test_cfg_term_is_weight_zero_in_both_variants(self):
        tree, cls, consts = cfg_tree()
        self.assertEqual(consts["YAW_CYCLE_WEIGHT"], 0.0)
        self.assertEqual(consts["YAW_CYCLE_STD"], 0.25)
        self.assertEqual(consts["YAW_CYCLE_WINDOW_STEPS"], 40)
        post = method(cls, "__post_init__")
        top = [ast.unparse(n) for n in post.body if not isinstance(n, ast.If)]
        hit = [t for t in top if t.startswith("self.rewards.track_ang_vel_z_cycle = RewTerm(")]
        self.assertEqual(len(hit), 1)
        self.assertIn("func=mdp.track_ang_vel_z_cycle_exp", hit[0])
        self.assertIn("weight=YAW_CYCLE_WEIGHT", hit[0])
        self.assertNotIn("track_ang_vel_z_cycle", V2_CFG.read_text())


if __name__ == "__main__":
    unittest.main()
