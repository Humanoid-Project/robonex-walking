import ast
import importlib.util
import math
import pathlib
import tempfile
import types
import unittest
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "source/robonex_walking/robonex_walking/tasks/manager_based/robonex_walking"
OBS = TASK / "mdp/observations.py"
REWARDS = TASK / "mdp/rewards.py"
EVENTS = TASK / "mdp/events.py"
CFG = TASK / "robonex_walking_v2_env_cfg.py"
EVAL = ROOT / "scripts/rsl_rl/eval_commands.py"
TRACE = ROOT / "scripts/rsl_rl/sim_joint_trace.py"
METRICS = ROOT / "scripts/rsl_rl/transition_metrics.py"

PERIOD = 0.8
DT = 0.02


def load_nodes(path, names, namespace):
    nodes = [n for n in ast.parse(path.read_text()).body if getattr(n, "name", None) in names]
    found = {n.name for n in nodes}
    assert found == set(names), set(names) - found
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def load_metrics():
    spec = importlib.util.spec_from_file_location("transition_metrics_under_test", METRICS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def clock_namespace():
    ns = {"torch": torch, "math": math, "ManagerBasedRLEnv": object, "Articulation": object, "ContactSensor": object,
          "SceneEntityCfg": object}
    load_nodes(OBS, {"gait_clock_phase", "gait_phase"}, ns)
    load_nodes(REWARDS, {"_bounded_square", "_quat_apply", "_saturate", "feet_gait_centred", "feet_clearance_clock_l2"}, ns)
    return ns


def cfg_constants():
    out = {}
    for node in ast.parse(CFG.read_text()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                out[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                pass
    return out


def cfg_term(cls_name, name):
    tree = ast.parse(CFG.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls_name)
    node = next(n for n in cls.body if isinstance(n, ast.Assign) and n.targets[0].id == name)
    return {k.arg: k.value for k in node.value.keywords}


def cfg_params(cls_name, name):
    kw = cfg_term(cls_name, name)
    return {ast.literal_eval(k): ast.unparse(v) for k, v in zip(kw["params"].keys, kw["params"].values)}


class Scene(types.SimpleNamespace):
    def __init__(self, robot, **kw):
        super().__init__(**kw)
        self.robot = robot

    def __getitem__(self, key):
        return self.robot


class ClockEnv:
    def __init__(self, lengths, offset=None, command=0.3):
        n = len(lengths)
        self.num_envs, self.device, self.step_dt = n, "cpu", DT
        self.episode_length_buf = torch.as_tensor(lengths, dtype=torch.long)
        if offset is not None:
            self._gait_clock_offset = torch.as_tensor(offset, dtype=torch.float32)
        cmd = torch.zeros(n, 3)
        cmd[:, 0] = command
        self.command_manager = types.SimpleNamespace(get_command=lambda _: cmd)
        g = torch.Generator().manual_seed(3)
        contact_time = torch.rand(n, 2, generator=g) - 0.5
        sensor = types.SimpleNamespace(data=types.SimpleNamespace(current_contact_time=contact_time))
        quat = torch.zeros(n, 2, 4)
        quat[..., 0] = 1.0
        body_pos = torch.rand(n, 2, 3, generator=g) * 0.1 + torch.tensor([0.0, 0.0, 0.07])
        robot = types.SimpleNamespace(data=types.SimpleNamespace(body_pos_w=body_pos, body_quat_w=quat))
        self.scene = Scene(robot, sensors={"contact_forces": sensor}, env_origins=torch.zeros(n, 3))


SOLE = [[-0.0591, -0.0588, -0.0654], [-0.0591, 0.0588, -0.0654], [0.1441, -0.0588, -0.0654], [0.1441, 0.0588, -0.0654]]
SENSOR = types.SimpleNamespace(name="contact_forces", body_ids=[0, 1])
ASSET = types.SimpleNamespace(name="robot", body_ids=[0, 1])


def readers(ns, env):
    return {
        "gait_phase": ns["gait_phase"](env, PERIOD, command_name="base_velocity"),
        "feet_gait": ns["feet_gait_centred"](env, PERIOD, [0.0, 0.5], SENSOR, threshold=0.5375,
                                             command_name="base_velocity"),
        "feet_clearance": ns["feet_clearance_clock_l2"](env, 0.053, ASSET, PERIOD, [0.0, 0.5], SOLE,
                                                        stance_fraction=0.5375, command_name="base_velocity"),
    }


class InertDefaultsTest(unittest.TestCase):
    def test_code_defaults_are_off(self):
        consts = cfg_constants()
        self.assertEqual(consts["GAIT_CLOCK_OFFSET_RANGE"], (0.0, 0.0))
        self.assertEqual(consts["STOP_GRACE_S"], 0.0)
        kw = cfg_term("EventCfg", "randomize_gait_clock")
        self.assertEqual(ast.unparse(kw["func"]), "mdp.randomize_gait_clock_offset")
        self.assertEqual(ast.literal_eval(kw["mode"]), "reset")
        self.assertEqual(cfg_params("EventCfg", "randomize_gait_clock"), {"offset_range": "GAIT_CLOCK_OFFSET_RANGE"})
        for name in ("stand_still", "stand_still_pose"):
            self.assertEqual(cfg_params("RewardsCfg", name)["stop_grace_s"], "STOP_GRACE_S")

    def test_grace_param_is_only_on_the_two_standing_terms(self):
        tree = ast.parse(CFG.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RewardsCfg")
        holders = [n.targets[0].id for n in cls.body if isinstance(n, ast.Assign) and "stop_grace_s" in ast.unparse(n)]
        self.assertEqual(sorted(holders), ["stand_still", "stand_still_pose"])
        callers = [n.name for n in ast.parse(REWARDS.read_text()).body
                   if isinstance(n, ast.FunctionDef) and n.name != "_stop_grace" and "_stop_grace(" in ast.unparse(n)]
        self.assertEqual(sorted(callers), ["stand_still_airborne", "stand_still_pose_l1"])

    def test_inert_event_creates_no_state_and_leaves_the_clock_bitwise(self):
        ns = {"torch": torch, "math": math}
        load_nodes(EVENTS, {"randomize_gait_clock_offset"}, ns)
        env = ClockEnv(list(range(0, 1000, 7)))
        ns["randomize_gait_clock_offset"](env, None, (0.0, 0.0))
        self.assertFalse(hasattr(env, "_gait_clock_offset"))
        clock = clock_namespace()
        legacy = (env.episode_length_buf * env.step_dt) % PERIOD / PERIOD
        self.assertTrue(torch.equal(clock["gait_clock_phase"](env, PERIOD), legacy))

    def test_offset_range_is_validated(self):
        ns = {"torch": torch, "math": math}
        load_nodes(EVENTS, {"randomize_gait_clock_offset"}, ns)
        for bad in ((-0.1, 0.5), (0.5, 0.2), (0.0, 1.5), (0.0, float("nan"))):
            with self.assertRaises(ValueError):
                ns["randomize_gait_clock_offset"](ClockEnv([0, 1]), None, bad)


class ClockOffsetTest(unittest.TestCase):
    def test_draw_is_uniform_and_per_env(self):
        ns = {"torch": torch, "math": math}
        load_nodes(EVENTS, {"randomize_gait_clock_offset"}, ns)
        torch.manual_seed(0)
        env = ClockEnv([0] * 40000)
        ns["randomize_gait_clock_offset"](env, None, (0.0, 1.0))
        offset = env._gait_clock_offset
        self.assertTrue(bool(((offset >= 0.0) & (offset < 1.0)).all()))
        counts = torch.histc(offset, bins=8, min=0.0, max=1.0)
        expected = 40000 / 8
        chi2 = float(((counts - expected) ** 2 / expected).sum())
        self.assertLess(chi2, 24.32)
        before = offset.clone()
        ns["randomize_gait_clock_offset"](env, torch.tensor([3, 5]), (0.0, 1.0))
        changed = (offset != before).nonzero().flatten().tolist()
        self.assertEqual(changed, [3, 5])

    def test_offset_shifts_every_reader_identically(self):
        ns = clock_namespace()
        lengths = [0, 13, 250, 500, 777, 999]
        k = [0, 7, 13, 20, 33, 39]
        shifted = ClockEnv(lengths, offset=[v / 40.0 for v in k])
        reference = ClockEnv([a + b for a, b in zip(lengths, k)])
        reference.scene = shifted.scene
        got, want = readers(ns, shifted), readers(ns, reference)
        for name in got:
            torch.testing.assert_close(got[name], want[name], atol=1e-5, rtol=0.0, msg=name)
        moved = readers(ns, ClockEnv(lengths))
        self.assertFalse(torch.allclose(moved["gait_phase"], got["gait_phase"]))

    def test_zero_command_still_zeroes_the_shifted_clock(self):
        ns = clock_namespace()
        env = ClockEnv([3, 100], offset=[0.3, 0.7], command=0.0)
        out = readers(ns, env)
        self.assertEqual(out["gait_phase"].abs().sum().item(), 0.0)
        self.assertEqual(out["feet_gait"].abs().sum().item(), 0.0)
        self.assertEqual(out["feet_clearance"].abs().sum().item(), 0.0)

    def test_no_reader_computes_the_clock_itself(self):
        for path in sorted((TASK / "mdp").glob("*.py")):
            for node in ast.parse(path.read_text()).body:
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name != "gait_clock_phase":
                    text = ast.unparse(node)
                    self.assertNotIn("episode_length_buf * env.step_dt", text, (path.name, node.name))
        trace = TRACE.read_text()
        self.assertIn('"clock_offset"', trace)
        self.assertIn('state["clock_offset"]', trace)
        self.assertIn("gait_clock_phase(unwrapped, period)", EVAL.read_text())


class GraceEnv:
    def __init__(self, n=3):
        self.num_envs, self.device, self.step_dt = n, "cpu", DT
        self.common_step_counter = 0
        self.episode_length_buf = torch.full((n,), 50, dtype=torch.long)
        self.command = torch.zeros(n, 3)
        self.command_manager = types.SimpleNamespace(get_command=lambda _: self.command)
        pos = torch.full((n, 2), 0.1)
        default = torch.zeros(n, 2)
        robot = types.SimpleNamespace(data=types.SimpleNamespace(joint_pos=pos, default_joint_pos=default))
        self.scene = {"robot": robot}

    def step(self):
        self.common_step_counter += 1
        self.episode_length_buf += 1


class StopGraceTest(unittest.TestCase):
    def setUp(self):
        self.ns = {"torch": torch, "math": math, "ManagerBasedRLEnv": object, "Articulation": object,
                   "SceneEntityCfg": lambda *a, **k: types.SimpleNamespace(name="robot", joint_ids=[0, 1])}
        load_nodes(REWARDS, {"_stop_grace", "stand_still_airborne", "stand_still_pose_l1"}, self.ns)
        self.ns["_contact_mask"] = lambda env, cfg, threshold: torch.tensor([[True, False]] * env.num_envs)
        self.asset = types.SimpleNamespace(name="robot", joint_ids=[0, 1])

    def run_terms(self, grace, steps=80, walk=10):
        env = GraceEnv()
        env.command[0, 0] = 0.3
        env.command[2, 1] = 0.2
        out = {"stand_still": [], "stand_still_pose": []}
        for i in range(steps):
            env.step()
            if i == walk:
                env.command[:] = 0.0
            out["stand_still"].append(self.ns["stand_still_airborne"](env, sensor_cfg=None, stop_grace_s=grace))
            out["stand_still_pose"].append(self.ns["stand_still_pose_l1"](env, asset_cfg=self.asset, stop_grace_s=grace))
        return {k: torch.stack(v) for k, v in out.items()}

    def test_grace_zeroes_both_terms_for_exactly_t(self):
        base = self.run_terms(0.0)
        for grace, n in ((0.76, 38), (0.5, 25), (0.02, 1)):
            out = self.run_terms(grace)
            for name in out:
                walked = out[name][:, 0]
                zero = (walked == 0).nonzero().flatten().tolist()
                self.assertEqual(zero, list(range(10)) + list(range(10, 10 + n)), (name, grace))
                self.assertTrue(torch.equal(out[name][10 + n:, 0], base[name][10 + n:, 0]), (name, grace))
                self.assertTrue(torch.equal(out[name][:, 1], base[name][:, 1]), (name, grace))
                self.assertTrue(torch.equal(out[name][:, 2], torch.where(
                    torch.arange(80) < 10 + n, torch.zeros(80), base[name][:, 2])), (name, grace))

    def test_zero_grace_is_the_old_term(self):
        out = self.run_terms(0.0)
        self.assertEqual(out["stand_still"][:10, 0].tolist(), [0.0] * 10)
        self.assertEqual(out["stand_still"][10:, 0].tolist(), [1.0] * 70)
        self.assertEqual(out["stand_still"][:, 1].tolist(), [1.0] * 80)
        bare = {"torch": torch, "ManagerBasedRLEnv": object, "SceneEntityCfg": self.ns["SceneEntityCfg"],
                "_contact_mask": self.ns["_contact_mask"]}
        load_nodes(REWARDS, {"stand_still_airborne"}, bare)
        env = GraceEnv()
        env.step()
        self.assertEqual(bare["stand_still_airborne"](env, sensor_cfg=None).tolist(), [1.0, 1.0, 1.0])

    def test_reset_and_second_stop(self):
        env = GraceEnv(1)
        grace = []
        schedule = [0.3] * 5 + [0.0] * 3 + [0.3] * 2 + [0.0] * 50
        for i, c in enumerate(schedule):
            env.step()
            env.command[0, 0] = c
            if i == 30:
                env.episode_length_buf[:] = 1
                env.command[0, 0] = 0.0
            grace.append(bool(self.ns["_stop_grace"](env, "base_velocity", 0.05, 0.1)[0]))
        self.assertEqual([i for i, g in enumerate(grace) if g], [5, 6, 7, 10, 11, 12, 13, 14])

    def test_negative_grace_is_refused(self):
        with self.assertRaises(ValueError):
            self.ns["_stop_grace"](GraceEnv(), "base_velocity", 0.05, -0.1)


class SwitchDrawTest(unittest.TestCase):
    def test_switch_phase_is_uniform(self):
        m = load_metrics()
        period_steps = m.steps_per_period(PERIOD, DT)
        self.assertEqual(period_steps, 40)
        g = torch.Generator().manual_seed(0)
        switch = m.draw_switch_steps(40000, 300, period_steps, g)
        self.assertEqual((int(switch.min()), int(switch.max())), (300, 339))
        for offset in (0.0, 0.37):
            phase = ((switch.double() * DT) % PERIOD / PERIOD + offset) % 1.0
            counts = torch.histc(phase.float(), bins=8, min=0.0, max=1.0)
            chi2 = float(((counts - 5000.0) ** 2 / 5000.0).sum())
            self.assertLess(chi2, 24.32, offset)
        self.assertEqual(len(set(switch.tolist())), 40)
        again = m.draw_switch_steps(40000, 300, period_steps, torch.Generator().manual_seed(0))
        self.assertTrue(torch.equal(switch, again))

    def test_period_must_be_whole_steps(self):
        with self.assertRaises(ValueError):
            load_metrics().steps_per_period(0.81, DT)


class TransitionMetricsTest(unittest.TestCase):
    def run_synthetic(self):
        m = load_metrics()
        switch = torch.tensor([5, 6, 7, 8])
        rec = m.TransitionRecorder(switch, DT, 110, ["a", "b"], push_step=100)
        pushes = []
        for step in range(rec.last_step() + 1):
            rel = step - switch
            phase = torch.full((4,), (step % 40) / 40.0)
            failed = torch.tensor([False, int(rel[1]) == 20, step == 2, False])
            contact = torch.ones(4, 2, dtype=torch.bool)
            if int(rel[0]) < 3:
                contact[0, 1] = False
            if int(rel[0]) == 60:
                contact[0, 0] = False
            if int(rel[1]) < 1000:
                contact[1, 1] = int(rel[1]) % 2 == 0
            error = torch.full((4, 2), 0.1)
            if int(rel[0]) == 10:
                error[0, 1] = 0.5
            if int(rel[0]) == 40:
                error[0, 0] = 0.6
            if int(rel[3]) == 29:
                error[3, 0] = 0.7
            foot_delta = torch.tensor([[0.02, 0.30]] * 4)
            com = torch.tensor([[0.03, 0.04]] * 4)
            rec.update(step, phase, failed, contact, error, foot_delta, com, 0.303)
            pushes += rec.push_due(step).nonzero().flatten().tolist()
        return rec.summary(), pushes

    def test_synthetic_trajectory(self):
        out, pushes = self.run_synthetic()
        self.assertEqual(sorted(pushes), [0, 3])
        self.assertEqual((out["envs"], out["valid_envs"], out["failed_before_switch"], out["falls"]), (4, 3, 1, 1))
        self.assertAlmostEqual(out["fall_fraction"], 1 / 3)
        self.assertAlmostEqual(out["fall_time_s"]["mean"], 0.4)
        t = out["time_to_double_support_s"]
        self.assertEqual(t["n"], 3)
        self.assertAlmostEqual(t["mean"], 0.06 / 3, places=6)
        self.assertAlmostEqual(t["p50"], 0.0, places=6)
        deg = 180.0 / math.pi
        td = out["peak_error_touchdown_deg"]
        self.assertAlmostEqual(td["b"]["mean"], (0.5 + 0.1 + 0.1) * deg / 3, places=3)
        self.assertAlmostEqual(td["a"]["mean"], 0.1 * deg, places=3)
        self.assertAlmostEqual(out["peak_error_touchdown_over_25deg"], 1 / 3)
        two = out["peak_error_2s_deg_max"]
        self.assertAlmostEqual(two["p50"], 0.6 * deg, places=3)
        self.assertAlmostEqual(out["peak_error_2s_over_25deg"], 2 / 3)
        self.assertEqual(out["fore_aft_offset_2s_m"]["n"], 2)
        self.assertAlmostEqual(out["fore_aft_offset_2s_m"]["mean"], 0.02, places=6)
        self.assertAlmostEqual(out["stance_width_2s_m"]["mean"], 0.30, places=6)
        self.assertAlmostEqual(out["lateral_offset_2s_m"]["mean"], 0.003, places=6)
        self.assertAlmostEqual(out["com_support_dist_2s_m"]["mean"], 0.05, places=6)
        self.assertAlmostEqual(out["com_support_fore_aft_2s_m"]["mean"], 0.03, places=6)
        lifts = out["foot_lifts_after_1s"]
        self.assertEqual(lifts["n"], 2)
        self.assertAlmostEqual(lifts["mean"], 0.5)
        self.assertAlmostEqual(out["re_step_fraction"], 0.5)
        bins = {row["bin"]: row["n"] for row in out["by_phase"]}
        self.assertEqual(bins, {0: 0, 1: 3, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0})
        self.assertAlmostEqual(out["by_phase"][1]["fall_fraction"], 1 / 3)

    def test_cells_are_opt_in(self):
        m = load_metrics()
        self.assertEqual(len(m.TRANSITION_CELLS), 10)
        self.assertEqual(m.expand_cells(["fwd_slow", "transition"])[1:], list(m.TRANSITION_CELLS))
        tree = ast.parse(EVAL.read_text())
        cells = next(n for n in tree.body if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", None) == "CELLS")
        default = ast.literal_eval(cells.value)
        self.assertEqual(len(default), 17)
        self.assertFalse(set(default) & set(m.TRANSITION_CELLS))
        commands = {tuple(c["command"]) for c in m.TRANSITION_CELLS.values()}
        self.assertEqual(commands, {(0.1, 0.0, 0.0), (0.2, 0.0, 0.0), (0.3, 0.0, 0.0), (0.0, 0.1, 0.0), (0.0, 0.0, 0.1)})
        text = EVAL.read_text()
        self.assertIn("transition cells need --no_push", text)
        self.assertIn('meta["transition"] = transition_definition(', text)


def load_check():
    names = {"_scalar_param", "_diff_config", "_json_safe", "_differs", "_check_action_path", "check_training_config"}
    tree = ast.parse(EVAL.read_text())
    nodes = [n for n in tree.body if getattr(n, "name", None) in names
             or (isinstance(n, ast.Assign) and getattr(n.targets[0], "id", None) in ("_MISSING", "_SKIP"))]
    ns = {"pathlib": pathlib}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "eval_commands", "exec"), ns)
    return ns["check_training_config"]


class EvalCheckTest(unittest.TestCase):
    def check(self, saved, live_events, live_specs):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            (run / "params").mkdir()
            (run / "params/env.yaml").write_text(yaml.safe_dump(saved))
            applied = {"action_scale": {}, "action_offset": {}, "action_clip": {}, "events": live_events,
                       "reward_specs": live_specs, "reward_terms": sorted(live_specs),
                       "deploy_effects": {"actuation_delay": {"source": "pin"}}}
            issues = load_check()(run / "model.pt", applied)
        return [i for i in issues if "missing" not in i and "clip" not in i and "foot_roll" not in i
                and "slew" not in i and "max_" not in i and "agent" not in i]

    def saved(self):
        return {"events": {"push_robot": {"mode": "interval"}},
                "rewards": {"stand_still": {"weight": -2.0, "params": {"threshold": 1.0}}}}

    def test_old_run_against_inert_new_keys_is_clean(self):
        live_events = {"push_robot": {"mode": "interval"},
                       "randomize_gait_clock": {"mode": "reset", "params": {"offset_range": [0.0, 0.0]}}}
        live_specs = {"stand_still": {"weight": -2.0, "params": {"threshold": 1.0, "stop_grace_s": 0.0}}}
        self.assertEqual(self.check(self.saved(), live_events, live_specs), [])

    def test_active_new_keys_are_reported(self):
        live_events = {"push_robot": {"mode": "interval"},
                       "randomize_gait_clock": {"mode": "reset", "params": {"offset_range": [0.0, 1.0]}}}
        live_specs = {"stand_still": {"weight": -2.0, "params": {"threshold": 1.0, "stop_grace_s": 0.75}}}
        issues = self.check(self.saved(), live_events, live_specs)
        self.assertTrue(any("randomize_gait_clock" in i for i in issues), issues)
        self.assertTrue(any("stop_grace_s" in i for i in issues), issues)


if __name__ == "__main__":
    unittest.main()
