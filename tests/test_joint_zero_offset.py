import ast
import importlib.util
import math
import pathlib
import sys
import tempfile
import types
import unittest
from collections.abc import Sequence
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_sent_tracking_excess import ACTIONS, FakeAsset, FakeJointPositionAction, load_nodes

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "source/robonex_walking/robonex_walking/tasks/manager_based/robonex_walking"
EVENTS = TASK / "mdp/events.py"
OBS = TASK / "mdp/observations.py"
CFG = TASK / "robonex_walking_v2_env_cfg.py"
EFFECTS = ROOT / "scripts/rsl_rl/deploy_effects.py"
EVAL = ROOT / "scripts/rsl_rl/eval_commands.py"

LEG = (
    "l_hip_yaw_joint", "l_hip_pitch_joint", "l_hip_roll_joint", "l_knee_pitch_joint",
    "l_ankle_upper_joint", "l_ankle_lower_joint",
    "r_hip_yaw_joint", "r_hip_pitch_joint", "r_hip_roll_joint", "r_knee_pitch_joint",
    "r_ankle_upper_joint", "r_ankle_lower_joint",
)
JOINT_IDS = [15, 2, 9, 0, 11, 4, 13, 6, 1, 8, 3, 10]
NUM_JOINTS = 16
W195_DEG = {"hip_yaw": 1.0, "hip_roll": 1.0, "hip_pitch": 1.5, "knee_pitch": 2.0, "ankle_upper": 2.0, "ankle_lower": 2.0}


def w195_range():
    out = {}
    for name in LEG:
        r = math.radians(W195_DEG[name[2:-6]])
        out[name] = (-r, r)
    return out


def zero_range():
    return {name: (0.0, 0.0) for name in LEG}


def load_event():
    ns = {"torch": torch, "SceneEntityCfg": object, "Articulation": object}
    load_nodes(EVENTS, {"joint_zero_offset_bounds", "randomize_joint_zero_offset"}, ns)
    return ns["randomize_joint_zero_offset"]


class Scene(dict):
    num_envs = 0


def fake_env(num_envs=64, joint_ids=JOINT_IDS):
    names = ["passive_%d" % i for i in range(NUM_JOINTS)]
    for name, i in zip(LEG, joint_ids):
        names[i] = name
    asset = types.SimpleNamespace(
        num_joints=NUM_JOINTS, joint_names=names, device="cpu",
        data=types.SimpleNamespace(default_joint_pos=torch.randn(num_envs, NUM_JOINTS)),
    )
    scene = Scene(robot=asset)
    scene.num_envs = num_envs
    env = types.SimpleNamespace(scene=scene, num_envs=num_envs)
    asset_cfg = types.SimpleNamespace(name="robot", joint_ids=list(joint_ids))
    return env, asset, asset_cfg


class EventTest(unittest.TestCase):
    def test_draw_shape_order_and_ranges(self):
        event = load_event()
        torch.manual_seed(0)
        env, asset, asset_cfg = fake_env(4096)
        default = asset.data.default_joint_pos.clone()
        event(env, None, asset_cfg, w195_range())
        d = env._joint_zero_offset
        self.assertEqual(tuple(d.shape), (4096, NUM_JOINTS))
        passive = [i for i in range(NUM_JOINTS) if i not in JOINT_IDS]
        self.assertTrue(bool((d[:, passive] == 0).all()))
        for name, i in zip(LEG, JOINT_IDS):
            r = math.radians(W195_DEG[name[2:-6]])
            col = d[:, i]
            self.assertLessEqual(float(col.abs().max()), r + 1e-7, name)
            self.assertGreater(float(col.abs().max()), 0.95 * r, name)
            self.assertLess(abs(float(col.mean())), 0.05 * r, name)
        torch.testing.assert_close(asset.data.default_joint_pos, default, rtol=0, atol=0)

    def test_permuted_asset_order_keeps_each_joint_on_its_own_range(self):
        event = load_event()
        ids = list(reversed(JOINT_IDS))
        env, asset, asset_cfg = fake_env(256, ids)
        rng = {name: (0.01 * k, 0.01 * k) for k, name in enumerate(LEG)}
        event(env, None, asset_cfg, rng)
        for k, (name, i) in enumerate(zip(LEG, ids)):
            self.assertEqual(asset.joint_names[i], name)
            self.assertTrue(bool((env._joint_zero_offset[:, i] == torch.tensor(0.01 * k)).all()), name)

    def test_zero_range_is_inert_and_draws_no_random_numbers(self):
        event = load_event()
        env, _, asset_cfg = fake_env(8)
        torch.manual_seed(3)
        event(env, None, asset_cfg, zero_range())
        event(env, torch.tensor([1, 2]), asset_cfg, zero_range())
        after = torch.rand(1)
        torch.manual_seed(3)
        self.assertFalse(hasattr(env, "_joint_zero_offset"))
        torch.testing.assert_close(after, torch.rand(1), rtol=0, atol=0)

    def test_fixed_pin_draws_no_random_numbers(self):
        event = load_event()
        env, _, asset_cfg = fake_env(8)
        rng = zero_range()
        rng["l_knee_pitch_joint"] = (0.02, 0.02)
        torch.manual_seed(5)
        event(env, None, asset_cfg, rng)
        after = torch.rand(1)
        torch.manual_seed(5)
        torch.testing.assert_close(after, torch.rand(1), rtol=0, atol=0)
        self.assertTrue(bool((env._joint_zero_offset[:, JOINT_IDS[3]] == torch.tensor(0.02)).all()))

    def test_reset_redraws_only_the_reset_envs(self):
        event = load_event()
        env, _, asset_cfg = fake_env(16)
        event(env, None, asset_cfg, w195_range())
        before = env._joint_zero_offset.clone()
        ids = torch.tensor([3, 7])
        event(env, ids, asset_cfg, w195_range())
        after = env._joint_zero_offset
        keep = [i for i in range(16) if i not in (3, 7)]
        torch.testing.assert_close(after[keep], before[keep], rtol=0, atol=0)
        self.assertFalse(bool(torch.equal(after[ids][:, JOINT_IDS], before[ids][:, JOINT_IDS])))

    def test_bad_ranges_raise(self):
        event = load_event()
        env, _, asset_cfg = fake_env(4)
        missing = zero_range()
        del missing["r_hip_yaw_joint"]
        typo = zero_range()
        typo["l_knee_joint"] = (0.0, 0.0)
        flipped = zero_range()
        flipped["l_knee_pitch_joint"] = (0.1, -0.1)
        for bad in (missing, typo, flipped):
            with self.assertRaises(ValueError):
                event(env, None, asset_cfg, bad)


def load_obs():
    node = next(n for n in ast.parse(OBS.read_text()).body if getattr(n, "name", None) == "delayed_joint_state")

    class Base:
        def __init__(self, cfg, env):
            self.cfg = cfg
            self._env = env
            self.num_envs = env.num_envs
            self.device = "cpu"

    ns = {"torch": torch, "ManagerTermBase": Base, "ManagerBasedRLEnv": object}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "observations", "exec"), ns)
    return ns["delayed_joint_state"]


class ObservationTest(unittest.TestCase):
    def setUp(self):
        self.cls = load_obs()
        n = 3
        self.data = types.SimpleNamespace(
            joint_pos=torch.zeros(n, NUM_JOINTS), default_joint_pos=torch.full((n, NUM_JOINTS), 0.5),
            joint_vel=torch.zeros(n, NUM_JOINTS), default_joint_vel=torch.zeros(n, NUM_JOINTS))
        self.env = types.SimpleNamespace(num_envs=n, device="cpu", scene={"robot": types.SimpleNamespace(data=self.data)},
                                         common_step_counter=0, episode_length_buf=torch.zeros(n, dtype=torch.long))
        self.asset_cfg = types.SimpleNamespace(name="robot", joint_ids=JOINT_IDS)
        self.d = torch.zeros(n, NUM_JOINTS)
        self.d[:, JOINT_IDS] = torch.randn(n, 12) * 0.03

    def term(self, delay):
        cfg = types.SimpleNamespace(params={"asset_cfg": self.asset_cfg, "max_delay_steps": 2})
        term = self.cls(cfg, self.env)
        term._delay = torch.full((self.env.num_envs,), delay)
        return term

    def step(self, term, field, k):
        self.env.common_step_counter = k
        self.env.episode_length_buf = torch.full((self.env.num_envs,), k, dtype=torch.long)
        self.data.joint_pos = torch.full_like(self.data.joint_pos, 0.1 * k)
        self.data.joint_vel = torch.full_like(self.data.joint_vel, -0.2 * k)
        return term(self.env, self.asset_cfg, field=field)

    def test_absent_offset_is_unchanged(self):
        out = self.step(self.term(0), "pos", 1)
        torch.testing.assert_close(out, torch.full((3, 12), 0.1) - 0.5, rtol=0, atol=0)

    def test_position_subtracts_d_before_the_delay_buffer(self):
        self.env._joint_zero_offset = self.d
        term = self.term(1)
        outs = [self.step(term, "pos", k) for k in range(1, 5)]
        for k, out in zip(range(2, 5), outs[1:]):
            expected = torch.full((3, 12), 0.1 * (k - 1)) - 0.5 - self.d[:, JOINT_IDS]
            torch.testing.assert_close(out, expected, rtol=0, atol=1e-6)
        self.env.episode_length_buf = torch.zeros(3, dtype=torch.long)
        self.env.common_step_counter = 5
        self.data.joint_pos = torch.full_like(self.data.joint_pos, 0.9)
        fresh = term(self.env, self.asset_cfg, field="pos")
        torch.testing.assert_close(fresh, torch.full((3, 12), 0.9) - 0.5 - self.d[:, JOINT_IDS], rtol=0, atol=1e-6)

    def test_velocity_ignores_d(self):
        self.env._joint_zero_offset = self.d
        torch.testing.assert_close(self.step(self.term(0), "vel", 2), torch.full((3, 12), -0.4), rtol=0, atol=0)


def make_action(delay, num_envs=32):
    ns = {"torch": torch, "JointPositionAction": FakeJointPositionAction, "ManagerBasedEnv": object,
          "Sequence": Sequence, "SlewLimitedJointPositionActionCfg": object, "Ver2JointPositionActionCfg": object}
    load_nodes(ACTIONS, {"slew_limit_step", "foot_roll", "clip_foot_roll",
                         "SlewLimitedJointPositionAction", "Ver2JointPositionAction"}, ns)
    asset = FakeAsset(num_envs, NUM_JOINTS)
    env = types.SimpleNamespace(scene={"robot": asset}, joint_ids=JOINT_IDS, num_envs=num_envs, step_dt=0.02,
                                cfg=types.SimpleNamespace(decimation=8))
    cfg = types.SimpleNamespace(
        max_speed=6.0, max_accel=120.0, slew_enabled=True, foot_roll_limit=math.radians(12.0),
        foot_roll_coeffs=(-0.001195, 0.498159, 0.491024, -0.105182, -0.031960, 0.006356),
        foot_roll_pairs=(("l_ankle_upper_joint", "l_ankle_lower_joint", 1.0),
                         ("r_ankle_upper_joint", "r_ankle_lower_joint", -1.0)),
        actuation_delay_substeps=delay,
    )
    return ns["Ver2JointPositionAction"](cfg, env), asset, env


def rollout(term, asset, steps=12, seed=0, decimation=8):
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(steps):
        term.process_actions(6.0 * torch.randn(term.num_envs, 12, generator=g))
        sent = []
        for _ in range(decimation):
            term.apply_actions()
            sent.append(asset.data.joint_pos_target[:, JOINT_IDS].clone())
        out.append((term.processed_actions.clone(), torch.stack(sent, dim=1)))
    return out


def offset(n):
    d = torch.zeros(n, NUM_JOINTS)
    d[:, JOINT_IDS] = (torch.rand(n, 12) * 2 - 1) * math.radians(2.0)
    return d


class ActionTest(unittest.TestCase):
    def check(self, delay):
        torch.manual_seed(0)
        base, base_asset, _ = make_action(delay)
        torch.manual_seed(0)
        shifted, asset, env = make_action(delay)
        env._joint_zero_offset = offset(shifted.num_envs)
        d = env._joint_zero_offset[:, JOINT_IDS]
        low, high = shifted._clip[..., 0], shifted._clip[..., 1]
        outside = 0
        pairs = list(zip(rollout(base, base_asset, steps=40), rollout(shifted, asset, steps=40)))
        for (p0, s0), (p1, s1) in pairs:
            torch.testing.assert_close(p1, p0, rtol=0, atol=0)
            torch.testing.assert_close(s1, s0 + d.unsqueeze(1), rtol=0, atol=1e-6)
        for (_, _), (p1, s1) in pairs[-10:]:
            self.assertTrue(bool(((p1 >= low - 1e-6) & (p1 <= high + 1e-6)).all()))
            outside += int(((s1[:, -1] > high + 1e-6) | (s1[:, -1] < low - 1e-6)).sum())
        self.assertGreater(outside, 0)

    def test_no_delay_adds_d_after_clip_slew_and_roll(self):
        self.check(None)

    def test_delay_path_adds_d_to_held_and_new_targets(self):
        self.check((0, 8))

    def test_absent_or_zero_offset_is_bit_identical(self):
        for delay in (None, (0, 8)):
            torch.manual_seed(1)
            a, asset_a, _ = make_action(delay)
            torch.manual_seed(1)
            b, asset_b, env_b = make_action(delay)
            env_b._joint_zero_offset = torch.zeros(b.num_envs, NUM_JOINTS)
            for (p0, s0), (p1, s1) in zip(rollout(a, asset_a), rollout(b, asset_b)):
                torch.testing.assert_close(p1, p0, rtol=0, atol=0)
                torch.testing.assert_close(s1, s0, rtol=0, atol=0)

    def test_reset_keeps_the_measured_frame_default(self):
        term, asset, env = make_action((0, 8))
        asset.data.default_joint_pos[:] = torch.randn(term.num_envs, NUM_JOINTS)
        env._joint_zero_offset = offset(term.num_envs)
        default = asset.data.default_joint_pos.clone()
        term.reset()
        torch.testing.assert_close(term._delay_last, default[:, JOINT_IDS], rtol=0, atol=0)
        torch.testing.assert_close(term._slew_position, default[:, JOINT_IDS], rtol=0, atol=0)
        torch.testing.assert_close(asset.data.default_joint_pos, default, rtol=0, atol=0)


class ConfigTest(unittest.TestCase):
    def test_event_defaults_to_zero_on_every_leg_joint_and_reset_is_untouched(self):
        text = CFG.read_text()
        self.assertIn("JOINT_ZERO_OFFSET_RANGE = {name: (0.0, 0.0) for name in LEG_JOINTS}", text)
        tree = ast.parse(text)
        events = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "EventCfg")
        terms = {n.targets[0].id: n.value for n in events.body if isinstance(n, ast.Assign)}
        kw = {k.arg: k.value for k in terms["randomize_joint_zero"].keywords}
        self.assertEqual(ast.unparse(kw["func"]), "mdp.randomize_joint_zero_offset")
        self.assertEqual(ast.literal_eval(kw["mode"]), "reset")
        params = ast.unparse(kw["params"])
        self.assertIn("SceneEntityCfg('robot', joint_names=LEG_JOINTS)", params)
        self.assertIn("dict(JOINT_ZERO_OFFSET_RANGE)", params)
        reset_kw = {k.arg: k.value for k in terms["reset_leg_joints"].keywords}
        self.assertEqual(ast.unparse(reset_kw["func"]), "mdp.reset_closed_loop_to_default")
        reset_fn = next(n for n in ast.parse(EVENTS.read_text()).body
                        if getattr(n, "name", None) == "reset_closed_loop_to_default")
        self.assertNotIn("zero_offset", ast.unparse(reset_fn))
        event_fn = next(n for n in ast.parse(EVENTS.read_text()).body
                        if getattr(n, "name", None) == "randomize_joint_zero_offset")
        self.assertNotIn("default_joint_pos", ast.unparse(event_fn))

    def test_privileged_clean_joint_state_stays_true(self):
        tree = ast.parse(CFG.read_text())
        priv = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "PrivilegedCriticCfg")
        funcs = {n.targets[0].id: {k.arg: ast.unparse(k.value) for k in n.value.keywords}.get("func")
                 for n in priv.body if isinstance(n, ast.Assign)}
        self.assertEqual(funcs["joint_pos_clean"], "mdp.joint_pos_rel")


def effects():
    spec = importlib.util.spec_from_file_location("deploy_effects_joint_zero", EFFECTS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def zero_term(rng):
    return types.SimpleNamespace(func="zero", mode="reset", params={"asset_cfg": "robot", "offset_range": dict(rng)})


class PinTest(unittest.TestCase):
    def setUp(self):
        self.fx = effects()

    def cfg(self, term):
        return types.SimpleNamespace(events=types.SimpleNamespace(randomize_joint_zero=term))

    def test_parse(self):
        self.assertIsNone(self.fx.parse_joint_zero_pin(None))
        self.assertEqual(self.fx.parse_joint_zero_pin(["l_knee_pitch_joint=1.5", "r_hip_roll_joint=-1"]),
                         {"l_knee_pitch_joint": 1.5, "r_hip_roll_joint": -1.0})
        for bad in (["l_knee_pitch_joint"], ["=1"], ["a=1", "a=2"], ["a=nan"]):
            with self.assertRaises(ValueError):
                self.fx.parse_joint_zero_pin(bad)

    def test_default_pins_a_trained_range_to_zero(self):
        cfg = self.cfg(zero_term(w195_range()))
        record = self.fx.apply_joint_zero_pin(cfg)
        self.assertEqual(cfg.events.randomize_joint_zero.params["offset_range"], zero_range())
        self.assertEqual((record["source"], record["nominal"], record["event_present"]), ("default_nominal", True, True))

    def test_pin_sets_named_joints_in_radians(self):
        cfg = self.cfg(zero_term(w195_range()))
        record = self.fx.apply_joint_zero_pin(cfg, {"l_knee_pitch_joint": 2.0})
        rng = cfg.events.randomize_joint_zero.params["offset_range"]
        self.assertEqual(rng["l_knee_pitch_joint"], (math.radians(2.0),) * 2)
        self.assertEqual(rng["r_knee_pitch_joint"], (0.0, 0.0))
        self.assertEqual((record["source"], record["nominal"]), ("pin", False))

    def test_keep_training(self):
        cfg = self.cfg(zero_term(w195_range()))
        record = self.fx.apply_joint_zero_pin(cfg, keep_training=True)
        self.assertEqual(cfg.events.randomize_joint_zero.params["offset_range"], w195_range())
        self.assertEqual((record["source"], record["nominal"]), ("training", False))
        with self.assertRaises(ValueError):
            self.fx.apply_joint_zero_pin(cfg, {"l_knee_pitch_joint": 1.0}, keep_training=True)

    def test_dropped_event_is_restored_from_the_template_only_for_a_nonzero_pin(self):
        cfg = self.cfg(None)
        record = self.fx.apply_joint_zero_pin(cfg, None, zero_term(zero_range()))
        self.assertIsNone(cfg.events.randomize_joint_zero)
        self.assertEqual((record["event_present"], record["nominal"]), (False, True))
        record = self.fx.apply_joint_zero_pin(cfg, {"r_ankle_upper_joint": -1.0}, zero_term(zero_range()))
        self.assertEqual(cfg.events.randomize_joint_zero.params["offset_range"]["r_ankle_upper_joint"],
                         (math.radians(-1.0),) * 2)
        self.assertTrue(record["event_present"])

    def test_unknown_joint_or_no_event_raises(self):
        with self.assertRaises(KeyError):
            self.fx.apply_joint_zero_pin(self.cfg(zero_term(zero_range())), {"l_knee_joint": 1.0})
        with self.assertRaises(KeyError):
            self.fx.apply_joint_zero_pin(self.cfg(None), {"l_knee_pitch_joint": 1.0}, None)


def load_check():
    names = {"_scalar_param", "_diff_config", "_json_safe", "_differs", "_check_action_path", "check_training_config"}
    tree = ast.parse(EVAL.read_text())
    nodes = [n for n in tree.body if getattr(n, "name", None) in names
             or (isinstance(n, ast.Assign) and getattr(n.targets[0], "id", None) in ("_MISSING", "_SKIP"))]
    ns = {"pathlib": pathlib}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "eval_commands", "exec"), ns)
    return ns["check_training_config"]


class StrictTest(unittest.TestCase):
    def run_check(self, saved_events, live_events, zero):
        check = load_check()
        with tempfile.TemporaryDirectory() as tmp:
            params = Path(tmp) / "params"
            params.mkdir()
            (params / "env.yaml").write_text(yaml.safe_dump({"events": saved_events}))
            applied = {"deploy_effects": {"joint_zero_offset": zero}, "action_scale": {}, "action_offset": {},
                       "events": live_events}
            return [i for i in check(str(Path(tmp) / "model_1.pt"), applied) if i.startswith("events")]

    def test_pre_w195_run_on_new_code_is_not_a_mismatch(self):
        live = {"push": {"x": 1.0}, "randomize_joint_zero": {"params": {"offset_range": {"a": [0.0, 0.0]}}}}
        self.assertEqual(self.run_check({"push": {"x": 1.0}}, live, {"nominal": True}), [])
        live_none = {"push": {"x": 1.0}, "randomize_joint_zero": None}
        self.assertEqual(self.run_check({"push": {"x": 1.0}}, live_none, {"nominal": True}), [])

    def test_nonzero_pin_on_a_pre_w195_run_is_reported(self):
        live = {"push": {"x": 1.0}, "randomize_joint_zero": {"params": {"offset_range": {"a": [0.03, 0.03]}}}}
        self.assertEqual(len(self.run_check({"push": {"x": 1.0}}, live, {"nominal": False})), 1)

    def test_w195_run_pinned_to_zero_is_reported_like_the_com_pin(self):
        saved = {"randomize_joint_zero": {"params": {"offset_range": {"a": [-0.03, 0.03]}}}}
        live = {"randomize_joint_zero": {"params": {"offset_range": {"a": [0.0, 0.0]}}}}
        self.assertEqual(len(self.run_check(saved, live, {"nominal": True})), 2)


if __name__ == "__main__":
    unittest.main()
