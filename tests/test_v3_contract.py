import ast
import importlib.util
import json
import re
import sys
import types
import unittest
from pathlib import Path

TASK = Path(__file__).resolve().parents[1] / "source/robonex_walking/robonex_walking/tasks/manager_based/robonex_walking"
DESC = Path(__file__).resolve().parents[2] / "robonex-description"
PLAYER = Path(__file__).resolve().parents[1] / "scripts/sim_to_sim/play_v3_mujoco.py"
HAS_V3 = (DESC / "ver3/ver3_constants.json").exists()


def _load(name):
    pkg = "robonex_v3_contract_test"
    if pkg not in sys.modules:
        mod = types.ModuleType(pkg)
        mod.__path__ = [str(TASK)]
        sys.modules[pkg] = mod
    spec = importlib.util.spec_from_file_location(f"{pkg}.{name}", TASK / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@unittest.skipUnless(HAS_V3, "robonex-description ver3 not present")
class V3ContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.v2 = _load("robot_contract_v2")
        cls.v3 = _load("robot_contract_v3")
        cls.constants = json.loads((DESC / "ver3/ver3_constants.json").read_text())

    def test_leg_names_map_to_the_ver3_leg_joints(self):
        self.assertEqual(set(self.v3.LEG_JOINTS), set(self.constants["leg_joints"]))
        self.assertEqual(len(self.v3.LEG_JOINTS), 12)

    def test_action_contract_equals_ver2_legs(self):
        for v2_name in self.v2.LEG_JOINTS:
            name = self.v3.V2_TO_V3[v2_name]
            self.assertEqual(self.v3.ACTION_SCALES[name], self.v2.ACTION_SCALES[v2_name])
            self.assertEqual(self.v3.ACTION_OFFSETS[name], self.v2.ACTION_OFFSETS[v2_name])
            self.assertEqual(tuple(self.v3.ACTION_CLIPS[name]), tuple(self.v2.ACTION_CLIPS[v2_name]))

    def test_upper_joints_match_the_model(self):
        self.assertEqual(set(self.v3.UPPER_JOINTS), set(self.constants["upper_body_joints"]))

    def test_motor_ids_of_the_legs_are_unchanged(self):
        ids = self.constants["motor_id_by_joint"]
        common = {j.model_name: j.motor_id for j in __import__("robonex_common.joints", fromlist=["x"]).ACTUATED_JOINTS}
        for v2_name, motor_id in common.items():
            self.assertEqual(ids[self.v3.V2_TO_V3[v2_name]], motor_id)

    def test_default_pose_covers_every_joint_once(self):
        full = self.v3.closed_loop_default_joint_pos("full")
        legs = self.v3.closed_loop_default_joint_pos("legs")
        self.assertEqual(set(full) - set(legs), set(self.v3.UPPER_JOINTS))
        self.assertTrue(all(full[n] == 0.0 for n in self.v3.UPPER_JOINTS))
        self.assertTrue(any(k.endswith(":0") for k in full))
        self.assertFalse(any(k.endswith(("_x", "_y", "_z")) for k in full))

    def test_mass_and_heights(self):
        self.assertAlmostEqual(self.v3.MASS_KG, 29.711, places=3)
        self.assertAlmostEqual(self.v3.BASE_HEIGHT, self.v2.BASE_HEIGHT, places=6)

    def test_upper_actuator_patterns_select_each_upper_joint_once(self):
        roles = self.v3.UPPER_MOTOR
        pats = [".*%s_joint" % ("_" + r if r.startswith(("shoulder", "elbow", "wrist")) else r) for r in roles]
        for name in self.v3.UPPER_JOINTS:
            self.assertEqual(sum(bool(re.fullmatch(p, name)) for p in pats), 1, name)
        legs = [".*_hip_yaw_joint", ".*_ankle_upper_joint", ".*_ankle_lower_joint", ".*_hip_pitch_joint", ".*_hip_roll_joint", ".*_knee_pitch_joint"]
        for name in self.v3.LEG_JOINTS:
            self.assertEqual(sum(bool(re.fullmatch(p, name)) for p in legs), 1, name)
        for name in self.v3.UPPER_JOINTS:
            self.assertFalse(any(re.fullmatch(p, name) for p in legs), name)

    def test_policy_order_is_common_order_renamed_and_the_player_order(self):
        common = __import__("robonex_common.joints", fromlist=["x"]).POLICY_JOINT_ORDER
        renamed = tuple(self.v3.v3_name(n) for n in common)
        node = next(n for n in ast.parse(PLAYER.read_text()).body if isinstance(n, ast.Assign)
                    and getattr(n.targets[0], "id", None) == "ISAAC_LEG_ARTICULATION_ORDER")
        self.assertEqual(renamed, tuple(self.constants["policy_joint_order"]))
        self.assertEqual(renamed, ast.literal_eval(node.value))
        self.assertEqual(set(renamed), set(self.v3.LEG_JOINTS))

    def test_feet_and_roll_pairs(self):
        self.assertEqual(self.v3.FEET, ("left_ankle_pitch_link", "right_ankle_pitch_link"))
        self.assertEqual(self.v3.FOOT_ROLL_PAIRS[0][:2], ("left_ankle_upper_joint", "left_ankle_lower_joint"))


if __name__ == "__main__":
    unittest.main()
