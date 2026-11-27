import ast
import contextlib
import importlib.util
import io
import tempfile
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EFFECTS = ROOT / "scripts/rsl_rl/deploy_effects.py"
EVAL = ROOT / "scripts/rsl_rl/eval_commands.py"
TRACE = ROOT / "scripts/rsl_rl/sim_joint_trace.py"

DESC = "/data1/workspaces/robonex/humanoid_project/robonex-description"
V2 = DESC + "/ver2/isaac/edu/closed_loop_mesh/robonex_edu_closed_loop_mesh.usd"
V2_3 = "/home/polygon/humanoid_project/robonex-description/ver2-3/isaac/edu/closed_loop_mesh/robonex_edu_closed_loop_mesh.usd"
V2_2 = "/home/polygon/humanoid_project/robonex-description/ver2-2/isaac/edu/closed_loop_mesh/robonex_edu_closed_loop_mesh.usd"
NEW_URDF = "/home/polygon/humanoid_project/robonex-description/new_urdf/isaac/edu/closed_loop_mesh/robonex_edu_closed_loop_mesh.usd"
OLD_ROOT = "/home/polygon/humanoid_project/robonex-description/isaac/closed_loop_mesh/robonex_closed_loop_mesh.usd"
V3_FULL = "/data1/workspaces/robonex/v3ws/robonex-description/ver3/isaac/full/closed_loop_mesh/robonex_ver3_full_closed_loop_mesh.usd"
V3_LEGS = "/home/polygon/humanoid_project/robonex-description/ver3/isaac/legs/closed_loop_mesh/robonex_ver3_legs_closed_loop_mesh.usd"

ROLES = ("hip_yaw", "hip_pitch", "hip_roll", "knee_pitch", "ankle_upper", "ankle_lower")
V2_JOINTS = ["%s_%s_joint" % (side, role) for side in ("l", "r") for role in ROLES]
V3_JOINTS = ["%s_%s_joint" % (side, role) for side in ("left", "right") for role in ROLES]


def load_effects():
    spec = importlib.util.spec_from_file_location("deploy_effects_identity", EFFECTS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def saved_yaml(usd, joints):
    lines = ["scene:", "  robot:", "    spawn:", "      usd_path: %s" % usd, "actions:", "  joint_pos:",
             "    joint_names: !!python/tuple"]
    lines += ["    - %s" % name for name in joints]
    lines += ["    clip:", "      %s: !!python/tuple" % joints[0], "      - -1.0", "      - 1.0", "seed: 43"]
    return "\n".join(lines) + "\n"


def cfg(usd, joints):
    return types.SimpleNamespace(
        scene=types.SimpleNamespace(robot=types.SimpleNamespace(spawn=types.SimpleNamespace(usd_path=usd))),
        actions=types.SimpleNamespace(joint_pos=types.SimpleNamespace(joint_names=tuple(joints))))


class RobotIdentityTest(unittest.TestCase):
    def setUp(self):
        self.effects = load_effects()
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def checkpoint(self, text):
        run = Path(self.tmp.name) / ("run%d" % len(list(Path(self.tmp.name).iterdir())))
        (run / "params").mkdir(parents=True)
        if text is not None:
            (run / "params" / "env.yaml").write_text(text)
        return str(run / "model_999.pt")

    def check(self, live_usd, live_joints, trained_usd, trained_joints, cross_plant=False):
        checkpoint = self.checkpoint(saved_yaml(trained_usd, trained_joints))
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            record = self.effects.check_robot_identity(cfg(live_usd, live_joints), cfg(live_usd, live_joints),
                                                       checkpoint, cross_plant)
        return record, out.getvalue()

    def test_identity_of_known_paths(self):
        identity = self.effects.robot_identity
        self.assertEqual(identity(V2), ("ver2", "edu"))
        self.assertEqual(identity(V2_3), ("ver2", "edu"))
        self.assertEqual(identity(NEW_URDF), ("ver2", "edu"))
        self.assertEqual(identity(V2_2), ("ver2-2", "edu"))
        self.assertEqual(identity(V3_FULL), ("ver3", "full"))
        self.assertEqual(identity(V3_LEGS), ("ver3", "legs"))
        self.assertEqual(identity(V3_LEGS.replace("robonex-description", "robonex_description")), ("ver3", "legs"))
        self.assertIsNone(identity(OLD_ROOT))
        self.assertIsNone(identity(None))
        self.assertIsNone(identity("https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/x.usd"))

    def test_every_identity_names_a_registered_task(self):
        self.assertEqual(self.effects.TASK_BY_IDENTITY, {
            ("ver2", "edu"): "RoboNex-Walking-V2-Edu-v0",
            ("ver3", "full"): "RoboNex-Walking-V3-v0",
            ("ver3", "legs"): "RoboNex-Walking-V3-Legs-v0",
        })
        init = (ROOT / "source/robonex_walking/robonex_walking/tasks/manager_based/robonex_walking/__init__.py").read_text()
        for task in self.effects.TASK_BY_IDENTITY.values():
            self.assertIn('id="%s"' % task, init)

    def test_same_robot_passes_and_records_both_sides(self):
        record, printed = self.check(V2, V2_JOINTS, V2_3, V2_JOINTS)
        self.assertEqual(printed, "")
        self.assertEqual(record, {"robot_usd": V2, "trained_robot_usd": V2_3, "robot_identity": ["ver2", "edu"],
                                  "trained_robot_identity": ["ver2", "edu"], "cross_plant": False})

    def test_version_mismatch_refuses_even_with_cross_plant(self):
        for cross in (False, True):
            with self.assertRaises(SystemExit) as raised:
                self.check(V2, V2_JOINTS, V3_LEGS, V3_JOINTS, cross_plant=cross)
            self.assertIn("ver3/legs", str(raised.exception.code))
            self.assertIn("--task RoboNex-Walking-V3-Legs-v0", str(raised.exception.code))
            self.assertIn("action joints differ", str(raised.exception.code))
        with self.assertRaises(SystemExit) as raised:
            self.check(V3_FULL, V3_JOINTS, V2, V2_JOINTS, cross_plant=True)
        self.assertIn("--task RoboNex-Walking-V2-Edu-v0", str(raised.exception.code))

    def test_version_mismatch_refuses_with_matching_joint_names(self):
        with self.assertRaises(SystemExit) as raised:
            self.check(V2, V3_JOINTS, V3_FULL, V3_JOINTS, cross_plant=True)
        self.assertNotIn("action joints differ", str(raised.exception.code))

    def test_variant_mismatch_needs_cross_plant(self):
        with self.assertRaises(SystemExit) as raised:
            self.check(V3_FULL, V3_JOINTS, V3_LEGS, V3_JOINTS)
        self.assertIn("--cross_plant", str(raised.exception.code))
        self.assertIn("--task RoboNex-Walking-V3-Legs-v0", str(raised.exception.code))
        record, _ = self.check(V3_FULL, V3_JOINTS, V3_LEGS, V3_JOINTS, cross_plant=True)
        self.assertEqual(record["robot_identity"], ["ver3", "full"])
        self.assertEqual(record["trained_robot_identity"], ["ver3", "legs"])
        self.assertTrue(record["cross_plant"])

    def test_minor_version_mismatch_needs_cross_plant(self):
        with self.assertRaises(SystemExit) as raised:
            self.check(V2, V2_JOINTS, V2_2, V2_JOINTS)
        self.assertIn("no task builds the trained robot", str(raised.exception.code))
        record, _ = self.check(V2, V2_JOINTS, V2_2, V2_JOINTS, cross_plant=True)
        self.assertEqual(record["trained_robot_identity"], ["ver2-2", "edu"])

    def test_joint_name_mismatch_refuses_on_the_same_robot(self):
        with self.assertRaises(SystemExit) as raised:
            self.check(V3_LEGS, V3_JOINTS, V3_LEGS, V3_JOINTS[:-1] + ["right_ankle_roll_joint"], cross_plant=True)
        self.assertIn("action joints differ", str(raised.exception.code))
        self.assertIn("right_ankle_roll_joint", str(raised.exception.code))

    def test_joint_order_alone_is_not_a_mismatch(self):
        record, _ = self.check(V3_LEGS, V3_JOINTS, V3_LEGS, list(reversed(V3_JOINTS)))
        self.assertEqual(record["robot_identity"], ["ver3", "legs"])

    def test_unknown_trained_robot_warns_and_continues(self):
        record, printed = self.check(V2, V2_JOINTS, OLD_ROOT, V2_JOINTS)
        self.assertIn("trained robot unknown", printed)
        self.assertIsNone(record["trained_robot_identity"])
        self.assertEqual(record["robot_identity"], ["ver2", "edu"])

    def test_missing_saved_params_warn_and_continue(self):
        checkpoint = self.checkpoint(None)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            record = self.effects.check_robot_identity(cfg(V2, V2_JOINTS), cfg(V2, V2_JOINTS), checkpoint)
        self.assertIn("trained robot unknown", out.getvalue())
        self.assertIsNone(record["trained_robot_usd"])

    def test_saved_without_joint_names_checks_the_robot_only(self):
        checkpoint = self.checkpoint("scene:\n  robot:\n    spawn:\n      usd_path: %s\n" % V3_LEGS)
        with self.assertRaises(SystemExit):
            self.effects.check_robot_identity(cfg(V2, V2_JOINTS), cfg(V2, V2_JOINTS), checkpoint, True)

    def test_recorded_usd_is_the_live_one_after_overlays(self):
        checkpoint = self.checkpoint(saved_yaml(V3_LEGS, V3_JOINTS))
        record = self.effects.check_robot_identity(cfg(V3_FULL, V3_JOINTS), cfg(V3_LEGS, V3_JOINTS), checkpoint, True)
        self.assertEqual(record["robot_usd"], V3_LEGS)
        self.assertEqual(record["robot_identity"], ["ver3", "full"])


def main_of(path):
    return next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "main")


def flags(path):
    out = {}
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "parser.add_argument" and node.args:
            out[node.args[0].value] = {k.arg: k.value for k in node.keywords}
    return out


class ScriptWiringTest(unittest.TestCase):
    def test_both_scripts_check_the_registry_robot_before_building_the_env(self):
        for path in (EVAL, TRACE):
            main = main_of(path)
            calls = {}
            for node in ast.walk(main):
                if isinstance(node, ast.Call):
                    calls.setdefault(ast.unparse(node.func), []).append(node)
            self.assertEqual(len(calls["check_robot_identity"]), 1, path.name)
            check = calls["check_robot_identity"][0]
            self.assertEqual([ast.unparse(a) for a in check.args],
                             ["parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)", "env_cfg",
                              "args_cli.checkpoint", "args_cli.cross_plant"], path.name)
            self.assertLess(check.lineno, calls["gym.make"][0].lineno, path.name)
            overlay = calls["apply_training_env_cfg"][0]
            self.assertLess(overlay.lineno, check.lineno, path.name)
            self.assertEqual(ast.unparse(flags(path)["--cross_plant"]["action"]), "'store_true'", path.name)

    def test_meta_records_task_and_robot(self):
        for path in (EVAL, TRACE):
            meta = next(n for n in ast.walk(main_of(path)) if isinstance(n, ast.Assign)
                        and ast.unparse(n.targets[0]) == "meta")
            entries = {ast.literal_eval(k): ast.unparse(v) for k, v in zip(meta.value.keys, meta.value.values)}
            self.assertEqual(entries["task"], "args_cli.task", path.name)
            for key in ("robot_usd", "trained_robot_usd", "robot_identity", "trained_robot_identity", "cross_plant"):
                self.assertEqual(entries[key], "robot_check['%s']" % key, path.name)


if __name__ == "__main__":
    unittest.main()
