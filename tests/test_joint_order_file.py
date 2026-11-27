import ast
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "scripts/rsl_rl/train.py"
EFFECTS = ROOT / "scripts/rsl_rl/deploy_effects.py"
PLAYER = ROOT / "scripts/sim_to_sim/play_v3_mujoco.py"
DESC = ROOT.parent / "robonex-description"
TRAIN_NAMES = {"joint_order_record", "ver3_constants_path", "policy_order_problem", "write_joint_order"}

ORDER = ["left_hip_yaw_joint", "right_hip_yaw_joint", "left_hip_pitch_joint", "right_hip_pitch_joint",
         "left_hip_roll_joint", "right_hip_roll_joint", "left_knee_pitch_joint", "right_knee_pitch_joint",
         "left_ankle_lower_joint", "left_ankle_upper_joint", "right_ankle_lower_joint", "right_ankle_upper_joint"]
V2_USD = "/home/polygon/humanoid_project/robonex-description/ver2/isaac/edu/closed_loop_mesh/robonex_edu_closed_loop_mesh.usd"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_train():
    effects = load_module("deploy_effects_joint_order", EFFECTS)
    tree = ast.parse(TRAIN.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in TRAIN_NAMES]
    ns = {"json": json, "os": os, "Path": Path, "robot_identity": effects.robot_identity}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(TRAIN), "exec"), ns)
    return ns


def swapped(order):
    order = list(order)
    order[0], order[1] = order[1], order[0]
    return order


class TrainJointOrderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = load_train()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "robonex-description"
        (self.root / "ver3/isaac/full/closed_loop_mesh").mkdir(parents=True)
        self.usd = str(self.root / "ver3/isaac/full/closed_loop_mesh/robonex_ver3_full_closed_loop_mesh.usd")
        (self.root / "ver3/ver3_constants.json").write_text(json.dumps({"policy_joint_order": ORDER}))

    def tearDown(self):
        self.tmp.cleanup()

    def record(self, usd, names):
        return self.ns["joint_order_record"]("RoboNex-Walking-V3-v0", usd, tuple(names), ORDER + ["neck_yaw_joint"],
                                             ("base_link", "head_link"))

    def test_record_fields(self):
        record = self.record(self.usd, ORDER)
        self.assertEqual(set(record), {"task", "usd_path", "action_joint_names", "articulation_joint_names", "body_names"})
        self.assertEqual(record["action_joint_names"], ORDER)
        self.assertEqual(record["articulation_joint_names"][-1], "neck_yaw_joint")
        json.dumps(record)

    def test_ver3_order_equal_passes(self):
        self.assertIsNone(self.ns["policy_order_problem"](self.record(self.usd, ORDER)))

    def test_ver3_order_swapped_refuses(self):
        problem = self.ns["policy_order_problem"](self.record(self.usd, swapped(ORDER)))
        self.assertIn("differs", problem)
        self.assertIn("(0, 'right_hip_yaw_joint', 'left_hip_yaw_joint')", problem)

    def test_ver3_same_set_other_length_refuses(self):
        self.assertIsNotNone(self.ns["policy_order_problem"](self.record(self.usd, ORDER[:-1])))

    def test_ver3_without_constants_refuses(self):
        (self.root / "ver3/ver3_constants.json").unlink()
        self.assertIn("ver3_constants.json", self.ns["policy_order_problem"](self.record(self.usd, ORDER)))

    def test_ver2_and_unknown_robots_are_not_checked(self):
        self.assertIsNone(self.ns["policy_order_problem"](self.record(V2_USD, swapped(ORDER))))
        self.assertIsNone(self.ns["policy_order_problem"](self.record(None, swapped(ORDER))))

    def test_constants_path_on_the_server_layout(self):
        usd = "/data1/workspaces/robonex/v3ws/robonex-description/ver3/isaac/legs/closed_loop_mesh/x.usd"
        self.assertEqual(self.ns["ver3_constants_path"](usd),
                         Path("/data1/workspaces/robonex/v3ws/robonex-description/ver3/ver3_constants.json"))
        self.assertIsNone(self.ns["ver3_constants_path"](V2_USD))

    def test_write_joint_order(self):
        log_dir = Path(self.tmp.name) / "run"
        path = self.ns["write_joint_order"](str(log_dir), self.record(self.usd, ORDER))
        self.assertEqual(Path(path), log_dir / "params/joint_order.json")
        self.assertEqual(json.loads(Path(path).read_text())["action_joint_names"], ORDER)

    @unittest.skipUnless((DESC / "ver3/ver3_constants.json").is_file(), "robonex-description ver3 not present")
    def test_real_description_constants(self):
        usd = str(DESC / "ver3/isaac/full/closed_loop_mesh/robonex_ver3_full_closed_loop_mesh.usd")
        self.assertIsNone(self.ns["policy_order_problem"](self.record(usd, ORDER)))
        self.assertIsNotNone(self.ns["policy_order_problem"](self.record(usd, swapped(ORDER))))

    def test_main_writes_after_gym_make_and_before_the_runner(self):
        main = next(n for n in ast.parse(TRAIN.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "main")
        source = ast.unparse(main)
        self.assertLess(source.index("gym.make("), source.index("policy_order_problem(record)"))
        self.assertLess(source.index("write_joint_order(log_dir, record)"), source.index("wrapper(env"))


class PlayerLegOrderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.player = load_module("play_v3_mujoco_joint_order", PLAYER)
        except (ImportError, SystemExit) as error:
            raise unittest.SkipTest(f"player imports unavailable: {error}")

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.run = Path(self.tmp.name) / "run"
        (self.run / "params").mkdir(parents=True)
        self.constants = Path(self.tmp.name) / "ver3_constants.json"
        self.constants.write_text(json.dumps({"policy_joint_order": ORDER}))

    def tearDown(self):
        self.tmp.cleanup()

    def write_file(self, names):
        (self.run / "params/joint_order.json").write_text(json.dumps({"action_joint_names": names}))

    def test_order_constant_equals_the_fixture(self):
        self.assertEqual(self.player.ISAAC_LEG_ARTICULATION_ORDER, tuple(ORDER))

    def test_file_present(self):
        self.write_file(ORDER)
        order, source = self.player.resolve_leg_order(self.run, self.tmp.name + "/missing.json")
        self.assertEqual(order, tuple(ORDER))
        self.assertEqual(source, str(self.run / "params/joint_order.json"))

    def test_file_absent_uses_constants(self):
        order, source = self.player.resolve_leg_order(self.run, self.constants)
        self.assertEqual(order, tuple(ORDER))
        self.assertIn("policy_joint_order", source)
        self.assertIn(str(self.constants), source)

    def test_file_mismatch_raises(self):
        self.write_file(swapped(ORDER))
        with self.assertRaises(RuntimeError) as raised:
            self.player.resolve_leg_order(self.run, self.constants)
        self.assertIn("joint_order.json", str(raised.exception))

    def test_constants_mismatch_raises(self):
        self.constants.write_text(json.dumps({"policy_joint_order": swapped(ORDER)}))
        with self.assertRaises(RuntimeError):
            self.player.resolve_leg_order(self.run, self.constants)

    def test_file_wins_over_constants(self):
        self.write_file(ORDER)
        self.constants.write_text(json.dumps({"policy_joint_order": swapped(ORDER)}))
        self.assertEqual(self.player.resolve_leg_order(self.run, self.constants)[0], tuple(ORDER))


if __name__ == "__main__":
    unittest.main()
