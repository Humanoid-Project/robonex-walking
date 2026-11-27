import ast
import json
import math
import os
import types
import unittest
from pathlib import Path

import torch

ROOT = Path(os.environ.get("ROBONEX_WALKING_ROOT", Path(__file__).resolve().parents[1]))
TASK = ROOT / "source/robonex_walking/robonex_walking/tasks/manager_based/robonex_walking"
EVENTS = TASK / "mdp/events.py"
V2_CFG = TASK / "robonex_walking_v2_env_cfg.py"
V3_CFG = TASK / "robonex_walking_v3_env_cfg.py"
V3_CONTRACT = TASK / "robot_contract_v3.py"
DESC = Path(os.environ.get("ROBONEX_DESCRIPTION_ROOT", ROOT.parent / "robonex-description"))
CONSTANTS = DESC / "ver3/ver3_constants.json"
NECK = ("neck_yaw_joint", "neck_pitch_joint")
DT = 0.02


class FakeTermBase:
    def __init__(self, cfg, env):
        self.cfg = cfg
        self._env = env


def load_events():
    names = {"neck_target_bounds", "neck_target_timing", "randomize_neck_target"}
    nodes = [n for n in ast.parse(EVENTS.read_text()).body if getattr(n, "name", None) in names]
    assert {n.name for n in nodes} == names
    ns = {"torch": torch, "math": math, "SceneEntityCfg": object, "Articulation": object, "ManagerTermBase": FakeTermBase}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(EVENTS), "exec"), ns)
    return ns


def joint_names():
    if CONSTANTS.exists():
        c = json.loads(CONSTANTS.read_text())
        return list(c["leg_joints"]) + [n for n in c["upper_body_joints"]]
    legs = [f"{s}_{r}_joint" for s in ("left", "right")
            for r in ("hip_yaw", "hip_roll", "hip_pitch", "knee_pitch", "ankle_upper", "ankle_lower")]
    arms = [f"{s}_{r}_joint" for s in ("left", "right")
            for r in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll")]
    return legs + arms + list(NECK)


def neck_limits():
    if CONSTANTS.exists():
        lim = json.loads(CONSTANTS.read_text())["provisional_limits"]
        return {n: tuple(lim[n]) for n in NECK}
    return {"neck_yaw_joint": (-0.523599, 0.523599), "neck_pitch_joint": (-1.27409, 1.27409)}


class FakeAsset:
    def __init__(self, names, num_envs, default=None):
        self.joint_names = list(names)
        self.num_joints = len(names)
        self.device = "cpu"
        default = torch.zeros(num_envs, self.num_joints) if default is None else default
        self.data = types.SimpleNamespace(default_joint_pos=default,
                                          joint_pos_target=torch.zeros(num_envs, self.num_joints),
                                          joint_pos=default.clone())
        self.target_calls = []
        self.state_calls = []

    def set_joint_position_target(self, target, joint_ids=None, env_ids=None):
        self.target_calls.append((list(joint_ids), env_ids.clone()))
        self.data.joint_pos_target[env_ids[:, None], torch.as_tensor(joint_ids)] = target

    def write_joint_state_to_sim(self, position, velocity, joint_ids=None, env_ids=None):
        self.state_calls.append((list(joint_ids), env_ids.clone()))
        self.data.joint_pos[env_ids[:, None], torch.as_tensor(joint_ids)] = position


def make(num_envs=64, names=None, **overrides):
    ns = load_events()
    names = joint_names() if names is None else names
    asset = FakeAsset(names, num_envs)
    neck_ids = [names.index(n) for n in NECK if n in names]
    params = {
        "asset_cfg": types.SimpleNamespace(name="robot", joint_ids=neck_ids),
        "limits": neck_limits(),
        "margin": 0.05,
        "range_scale": 1.0,
        "resample_s": (1.0, 3.0),
        "blend_s": 1.0,
        "start_at_default": False,
        "write_state_at_reset": True,
    }
    params.update(overrides)
    env = types.SimpleNamespace(scene={"robot": asset}, num_envs=num_envs, step_dt=DT, device="cpu")
    term = ns["randomize_neck_target"](types.SimpleNamespace(params=params), env)
    return ns, term, env, asset, params, neck_ids


def run(term, env, params, steps):
    out = []
    for _ in range(steps):
        term(env, torch.arange(env.num_envs), **params)
        out.append(term.target.clone())
    return torch.stack(out)


class NeckTargetTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_bounds_are_contract_limits_minus_margin_scaled_about_the_middle(self):
        ns = load_events()
        lim = neck_limits()
        low, high = ns["neck_target_bounds"](list(NECK), lim, 0.05, 1.0)
        for i, n in enumerate(NECK):
            self.assertAlmostEqual(float(low[i]), lim[n][0] + 0.05, places=6)
            self.assertAlmostEqual(float(high[i]), lim[n][1] - 0.05, places=6)
        low, high = ns["neck_target_bounds"](list(NECK), lim, 0.05, 0.5)
        for i, n in enumerate(NECK):
            self.assertAlmostEqual(float(high[i] - low[i]), 0.5 * (lim[n][1] - lim[n][0] - 0.1), places=6)
        with self.assertRaises(ValueError):
            ns["neck_target_bounds"](["neck_yaw_joint"], lim, 0.05, 1.0)
        with self.assertRaises(ValueError):
            ns["neck_target_bounds"](list(NECK), lim, 0.05, 1.5)
        with self.assertRaises(ValueError):
            ns["neck_target_bounds"](list(NECK), lim, 0.6, 1.0)
        with self.assertRaises(ValueError):
            ns["neck_target_timing"]((0.0, 1.0), 0.5)

    def test_explicit_ranges_scale_toward_zero(self):
        ns = load_events()
        lim = neck_limits()
        ranges = {"neck_yaw_joint": (-0.2368, 0.2368), "neck_pitch_joint": (-0.26, 0.52)}
        low, high = ns["neck_target_bounds"](list(NECK), lim, 0.05, 1.0, ranges)
        torch.testing.assert_close(low, torch.tensor([-0.2368, -0.26]))
        torch.testing.assert_close(high, torch.tensor([0.2368, 0.52]))
        low, high = ns["neck_target_bounds"](list(NECK), lim, 0.05, 0.5, ranges)
        torch.testing.assert_close(low, torch.tensor([-0.1184, -0.13]))
        torch.testing.assert_close(high, torch.tensor([0.1184, 0.26]))
        low, high = ns["neck_target_bounds"](list(reversed(NECK)), lim, 0.05, 1.0, ranges)
        torch.testing.assert_close(low, torch.tensor([-0.26, -0.2368]))

    def test_explicit_ranges_are_checked(self):
        ns = load_events()
        lim = neck_limits()
        good = {"neck_yaw_joint": (-0.2368, 0.2368), "neck_pitch_joint": (-0.26, 0.52)}
        bad = [
            dict(good, neck_pitch_joint=(0.1, 0.52)),
            dict(good, neck_pitch_joint=(-0.52, -0.1)),
            dict(good, neck_pitch_joint=(0.0, 0.0)),
            dict(good, neck_yaw_joint=(-lim["neck_yaw_joint"][1], 0.2)),
            dict(good, neck_pitch_joint=(-0.26, lim["neck_pitch_joint"][1] - 0.04)),
            dict(good, neck_pitch_joint=(-0.26, float("nan"))),
            {"neck_yaw_joint": (-0.2, 0.2)},
            dict(good, head_joint=(-0.1, 0.1)),
        ]
        for ranges in bad:
            with self.assertRaises(ValueError, msg=str(ranges)):
                ns["neck_target_bounds"](list(NECK), lim, 0.05, 1.0, ranges)
        ns["neck_target_bounds"](list(NECK), lim, 0.05, 1.0, dict(good, neck_pitch_joint=(0.0, 0.52)))

    def test_no_ranges_keeps_the_symmetric_bounds(self):
        ns = load_events()
        low, high = ns["neck_target_bounds"](list(NECK), neck_limits(), 0.05, 0.5, None)
        if CONSTANTS.exists():
            torch.testing.assert_close(low, torch.tensor([-0.2368, -0.61205]), atol=1e-4, rtol=0)
            torch.testing.assert_close(high, torch.tensor([0.2368, 0.61205]), atol=1e-4, rtol=0)
        same = ns["neck_target_bounds"](list(NECK), neck_limits(), 0.05, 0.5)
        torch.testing.assert_close(same[0], low)
        torch.testing.assert_close(same[1], high)

    def test_term_with_ranges_draws_inside_them(self):
        ranges = {"neck_yaw_joint": (-0.2368, 0.2368), "neck_pitch_joint": (-0.26, 0.52)}
        _, term, env, asset, params, neck_ids = make(num_envs=256, ranges=ranges, resample_s=(0.3, 0.5))
        term.reset(None)
        targets = torch.cat([term.target.clone().unsqueeze(0), run(term, env, params, 500)])
        for i, n in enumerate(NECK):
            lo, hi = ranges[n]
            self.assertGreaterEqual(float(targets[..., i].min()), lo - 1e-6)
            self.assertLessEqual(float(targets[..., i].max()), hi + 1e-6)
            self.assertLess(float(targets[..., i].min()), lo + 0.1 * (hi - lo))
            self.assertGreater(float(targets[..., i].max()), hi - 0.1 * (hi - lo))
        self.assertGreater(float(targets[..., 1].mean()), 0.05)

    def test_inert_scale_ignores_invalid_ranges(self):
        _, term, env, asset, params, _ = make(range_scale=0.0, ranges={"neck_yaw_joint": (0.1, 0.2)})
        term.reset(None)
        term(env, torch.arange(env.num_envs), **params)
        self.assertIsNone(term.prev)
        self.assertEqual(asset.target_calls + asset.state_calls, [])

    def test_targets_stay_inside_the_range_and_cover_it(self):
        _, term, env, asset, params, neck_ids = make()
        term.reset(None)
        targets = torch.cat([term.target.clone().unsqueeze(0), run(term, env, params, 1500)])
        lim = neck_limits()
        for i, n in enumerate(NECK):
            lo, hi = lim[n][0] + 0.05, lim[n][1] - 0.05
            self.assertGreaterEqual(float(targets[..., i].min()), lo - 1e-6)
            self.assertLessEqual(float(targets[..., i].max()), hi + 1e-6)
            self.assertLess(float(targets[..., i].min()), lo + 0.1 * (hi - lo))
            self.assertGreater(float(targets[..., i].max()), hi - 0.1 * (hi - lo))
        sent = asset.data.joint_pos_target[:, neck_ids]
        self.assertTrue(torch.equal(sent, term.target))

    def test_only_neck_columns_are_written(self):
        _, term, env, asset, params, neck_ids = make()
        sentinel = torch.randn_like(asset.data.joint_pos_target)
        asset.data.joint_pos_target[:] = sentinel
        term.reset(torch.tensor([1, 5, 7]))
        run(term, env, params, 200)
        others = [j for j in range(asset.num_joints) if j not in neck_ids]
        self.assertTrue(torch.equal(asset.data.joint_pos_target[:, others], sentinel[:, others]))
        self.assertFalse(torch.equal(asset.data.joint_pos_target[:, neck_ids], sentinel[:, neck_ids]))
        self.assertTrue(all(ids == neck_ids for ids, _ in asset.target_calls + asset.state_calls))
        self.assertEqual([n for n in asset.joint_names if asset.joint_names.index(n) in neck_ids], list(NECK))

    def test_blend_is_linear_and_reaches_the_goal(self):
        _, term, env, asset, params, _ = make(num_envs=16, resample_s=(2.0, 2.0), blend_s=1.0)
        term.reset(None)
        start, goal = term.target.clone(), term.goal.clone()
        steps = run(term, env, params, 99)
        for k in (1, 25, 49, 50, 75):
            alpha = min((k + 1) * DT / 1.0, 1.0)
            torch.testing.assert_close(steps[k], start + (goal - start) * alpha, atol=1e-5, rtol=0)
        torch.testing.assert_close(steps[60], goal, atol=1e-6, rtol=0)
        step_max = (goal - start).abs() * DT / 1.0 + 1e-6
        self.assertTrue(bool(torch.all((steps[1:50] - steps[:49]).abs() <= step_max)))

    def test_no_jump_when_a_resample_interrupts_a_blend(self):
        _, term, env, asset, params, _ = make(num_envs=256, resample_s=(0.3, 0.5), blend_s=1.0)
        term.reset(None)
        traj = torch.cat([term.target.clone().unsqueeze(0), run(term, env, params, 500)])
        lim = neck_limits()
        span = torch.tensor([lim[n][1] - lim[n][0] - 0.1 for n in NECK])
        self.assertTrue(bool(torch.all((traj[1:] - traj[:-1]).abs() <= span * DT / 1.0 + 1e-6)))

    def test_resample_periods_are_per_env_within_the_interval(self):
        _, term, env, asset, params, _ = make(num_envs=512, resample_s=(1.0, 3.0))
        term.reset(None)
        self.assertGreaterEqual(float(term.period.min()), 1.0)
        self.assertLessEqual(float(term.period.max()), 3.0)
        self.assertGreater(float(term.period.std()), 0.3)

    def test_reset_draws_fresh_and_writes_state(self):
        _, term, env, asset, params, neck_ids = make(num_envs=32)
        term.reset(None)
        run(term, env, params, 30)
        before = term.target.clone()
        ids = torch.tensor([3, 4])
        asset.state_calls.clear()
        term.reset(ids)
        self.assertEqual(len(asset.state_calls), 1)
        torch.testing.assert_close(asset.data.joint_pos[ids][:, neck_ids], term.target[ids])
        torch.testing.assert_close(asset.data.joint_pos_target[ids][:, neck_ids], term.target[ids])
        self.assertTrue(bool(torch.all(term.elapsed[ids] == 0.0)))
        keep = torch.tensor([i for i in range(32) if i not in (3, 4)])
        self.assertTrue(torch.equal(term.target[keep], before[keep]))

    def test_start_at_default_holds_default_until_the_first_resample(self):
        _, term, env, asset, params, neck_ids = make(num_envs=8, start_at_default=True, resample_s=(1.0, 1.0),
                                                     write_state_at_reset=False)
        term.reset(None)
        self.assertTrue(bool(torch.all(term.target == 0.0)))
        self.assertEqual(asset.state_calls, [])
        traj = run(term, env, params, 48)
        self.assertTrue(bool(torch.all(traj == 0.0)))
        traj = run(term, env, params, 30)
        self.assertGreater(float(traj[-1].abs().max()), 0.0)

    def test_inert_scale_writes_nothing_and_allocates_nothing(self):
        _, term, env, asset, params, _ = make(range_scale=0.0)
        before = asset.data.joint_pos_target.clone()
        term.reset(None)
        run_params = dict(params)
        for _ in range(10):
            term(env, torch.arange(env.num_envs), **run_params)
        self.assertIsNone(term.prev)
        self.assertEqual(asset.target_calls + asset.state_calls, [])
        self.assertTrue(torch.equal(asset.data.joint_pos_target, before))

    def test_isaaclab_interval_of_one_step_fires_every_step(self):
        time_left = torch.full((4,), DT, dtype=torch.float32)
        for _ in range(100):
            time_left -= DT
            fired = (time_left < 1e-6).nonzero().flatten()
            self.assertEqual(len(fired), 4)
            time_left[fired] = torch.full((4,), DT)


def v3_post_init():
    tree = ast.parse(V3_CFG.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RoboNexWalkingV3EnvCfg")
    return tree, cls


class NeckTargetConfigTest(unittest.TestCase):
    def test_default_is_inert(self):
        tree, cls = v3_post_init()
        consts = {n.targets[0].id: ast.literal_eval(n.value) for n in tree.body
                  if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id.startswith("NECK_")}
        self.assertEqual(consts["NECK_TARGET_RANGE_SCALE"], 0.0)
        self.assertEqual(consts["NECK_TARGET_MARGIN_RAD"], 0.05)
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_neck_target_term")
        src = ast.unparse(method)
        self.assertIn("'range_scale': NECK_TARGET_RANGE_SCALE", src)
        self.assertIn("mode='interval'", src)
        self.assertIn("interval_range_s=(step_s, step_s)", src)
        self.assertIn("v3.NECK_LIMITS", src)
        self.assertIn("joint_names=list(v3.NECK_JOINTS)", src)
        self.assertIn("'ranges': dict(NECK_TARGET_RANGES)", src)
        self.assertEqual(consts["NECK_TARGET_RANGES"],
                         {"neck_yaw_joint": (-0.2368, 0.2368), "neck_pitch_joint": (-0.26, 0.52)})

    def test_ranges_fit_the_contract_limits(self):
        tree, _ = v3_post_init()
        consts = {n.targets[0].id: ast.literal_eval(n.value) for n in tree.body
                  if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id.startswith("NECK_")}
        ns = load_events()
        low, high = ns["neck_target_bounds"](list(NECK), neck_limits(), consts["NECK_TARGET_MARGIN_RAD"], 1.0,
                                             consts["NECK_TARGET_RANGES"])
        self.assertTrue(bool(torch.all(low < 0.0) and torch.all(high > 0.0)))

    def test_term_is_added_only_for_the_full_variant(self):
        _, cls = v3_post_init()
        post = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__post_init__")
        hits = [n for n in ast.walk(post) if isinstance(n, ast.If) and "neck_target" in ast.unparse(n)]
        self.assertEqual(len(hits), 1)
        self.assertEqual(ast.unparse(hits[0].test), "self.model_variant == 'full'")
        outside = [n for n in post.body if not isinstance(n, ast.If) and "neck_target" in ast.unparse(n)]
        self.assertEqual(outside, [])
        legs = next(n for n in ast.parse(V3_CFG.read_text()).body
                    if isinstance(n, ast.ClassDef) and n.name == "RoboNexWalkingV3LegsEnvCfg")
        self.assertIn("model_variant: str = 'legs'", ast.unparse(legs))
        self.assertNotIn("neck", ast.unparse(legs))

    def test_flat_v2_config_has_no_neck_term(self):
        self.assertNotIn("neck", V2_CFG.read_text())

    def test_legs_variant_asset_has_no_neck_and_inert_term_is_safe(self):
        legs = [n for n in joint_names() if n in json.loads(CONSTANTS.read_text())["leg_joints"]] if CONSTANTS.exists() \
            else joint_names()[:12]
        _, term, env, asset, params, neck_ids = make(names=legs, range_scale=0.0)
        self.assertEqual(neck_ids, [])
        term.reset(None)
        term(env, torch.arange(env.num_envs), **params)
        self.assertEqual(asset.target_calls, [])

    def test_contract_exposes_neck_limits_from_the_constants(self):
        src = V3_CONTRACT.read_text()
        self.assertIn("NECK_LIMITS = {name: _limits[name] for name in NECK_JOINTS}", src)
        self.assertIn('_limits = {k: tuple(v) for k, v in CONSTANTS["provisional_limits"].items()}', src)


if __name__ == "__main__":
    unittest.main()
