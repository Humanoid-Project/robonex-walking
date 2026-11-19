import ast
import copy
import pathlib
import tempfile
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "scripts/rsl_rl/eval_commands.py"
NAMES = {"_scalar_param", "_diff_config", "_json_safe", "_differs", "_check_action_path", "check_training_config"}

JOINTS = ("l_ankle_upper_joint", "l_ankle_lower_joint")
COEFFS = [-0.001195, 0.498159, 0.491024, -0.105182, -0.03196, 0.006356]
PAIRS = [["l_ankle_upper_joint", "l_ankle_lower_joint", 1.0]]

ACTION = {
    "scale": {"l_ankle_upper_joint": 0.085508, "l_ankle_lower_joint": 0.118109},
    "offset": {"l_ankle_upper_joint": 0.23962121825843782, "l_ankle_lower_joint": -0.22908552614940247},
    "clip": {"l_ankle_upper_joint": [-0.646244, 0.496145], "l_ankle_lower_joint": [-0.583412, 0.462984]},
    "max_speed": 6.0,
    "max_accel": 120.0,
    "slew_enabled": True,
    "foot_roll_limit": 0.20943951023931956,
    "foot_roll_coeffs": COEFFS,
    "foot_roll_pairs": PAIRS,
    "actuation_delay_substeps": None,
}


def load():
    tree = ast.parse(EVAL.read_text())
    nodes = [n for n in tree.body if getattr(n, "name", None) in NAMES
             or (isinstance(n, ast.Assign) and getattr(n.targets[0], "id", None) in ("_MISSING", "_SKIP"))]
    ns = {"pathlib": pathlib}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "eval_commands", "exec"), ns)
    return ns["check_training_config"]


def applied(**changes):
    out = {
        "deploy_effects": {"slew_limit": False},
        "action_scale": dict(ACTION["scale"]),
        "action_offset": dict(ACTION["offset"]),
        "action_clip": {k: list(v) for k, v in ACTION["clip"].items()},
        "runner_clip": 14.0,
        "foot_roll": {"limit": ACTION["foot_roll_limit"], "coeffs": list(COEFFS), "pairs": copy.deepcopy(PAIRS)},
        "slew": {"enabled": True, "max_speed": 6.0, "max_accel": 120.0},
    }
    for key, value in changes.items():
        out[key] = value
    return out


class StrictActionPathTest(unittest.TestCase):
    def setUp(self):
        self.check = load()
        self.tmp = tempfile.TemporaryDirectory()
        self.run = Path(self.tmp.name)
        (self.run / "params").mkdir()
        self.checkpoint = self.run / "model_1999.pt"

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, action=None, agent=True, drop=()):
        action = copy.deepcopy(ACTION if action is None else action)
        for key in drop:
            action.pop(key)
        (self.run / "params/env.yaml").write_text(yaml.safe_dump({"actions": {"joint_pos": action}}))
        if agent:
            (self.run / "params/agent.yaml").write_text(yaml.safe_dump({"clip_actions": 14.0}))

    def test_matching_params_pass_strict(self):
        self.write()
        self.assertEqual(self.check(self.checkpoint, applied(), strict=True), [])

    def test_float32_clip_rounding_is_tolerated(self):
        self.write()
        live = applied()
        live["action_clip"]["l_ankle_upper_joint"][1] = 0.49614500999
        self.assertEqual(self.check(self.checkpoint, live, strict=True), [])

    def test_a_changed_target_clip_is_reported(self):
        self.write()
        live = applied()
        live["action_clip"]["l_ankle_lower_joint"] = [-0.583412, 0.40]
        issues = self.check(self.checkpoint, live)
        self.assertEqual(len(issues), 1)
        self.assertIn("clip[l_ankle_lower_joint] high", issues[0])

    def test_a_changed_foot_roll_clip_is_reported(self):
        self.write()
        for key, value in (("limit", 0.2), ("coeffs", COEFFS[:5] + [0.0]),
                           ("pairs", [["l_ankle_upper_joint", "l_ankle_lower_joint", -1.0]])):
            live = applied()
            live["foot_roll"][key] = value
            issues = self.check(self.checkpoint, live)
            self.assertEqual(len(issues), 1, key)
            self.assertIn("foot_roll_", issues[0])

    def test_a_changed_slew_stage_is_reported(self):
        self.write()
        for key, value in (("enabled", False), ("max_speed", 5.0), ("max_accel", 100.0)):
            live = applied()
            live["slew"][key] = value
            self.assertEqual(len(self.check(self.checkpoint, live)), 1, key)

    def test_deploy_limiter_on_a_run_trained_without_slew_is_not_a_mismatch(self):
        action = dict(ACTION, slew_enabled=False)
        self.write(action)
        live = applied(deploy_effects={"slew_limit": True})
        self.assertEqual(self.check(self.checkpoint, live, strict=True), [])
        live = applied()
        self.assertEqual(len(self.check(self.checkpoint, live)), 1)

    def test_a_changed_runner_clip_is_reported(self):
        self.write()
        issues = self.check(self.checkpoint, applied(runner_clip=10.0))
        self.assertEqual(len(issues), 1)
        self.assertIn("clip_actions", issues[0])

    def test_missing_fields_fail_only_strict(self):
        for drop in (("clip",), ("foot_roll_limit",), ("foot_roll_coeffs",), ("foot_roll_pairs",), ("slew_enabled",),
                     ("max_speed",), ("max_accel",), ("scale",), ("offset",)):
            self.write(drop=drop)
            self.assertEqual(self.check(self.checkpoint, applied()), [], drop)
            issues = self.check(self.checkpoint, applied(), strict=True)
            self.assertEqual(len(issues), 1, drop)
            self.assertIn("missing", issues[0])

    def test_missing_joint_clip_fails_strict(self):
        action = copy.deepcopy(ACTION)
        action["clip"].pop("l_ankle_lower_joint")
        self.write(action)
        self.assertEqual(self.check(self.checkpoint, applied()), [])
        self.assertEqual(len(self.check(self.checkpoint, applied(), strict=True)), 1)

    def test_missing_agent_yaml_fails_only_strict(self):
        self.write(agent=False)
        self.assertEqual(self.check(self.checkpoint, applied()), [])
        issues = self.check(self.checkpoint, applied(), strict=True)
        self.assertEqual(len(issues), 1)
        self.assertIn("agent.yaml", issues[0])


if __name__ == "__main__":
    unittest.main()
