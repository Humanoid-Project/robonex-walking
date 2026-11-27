import ast
import os
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRACE = ROOT / "scripts/rsl_rl/sim_joint_trace.py"
NAMES = {"upper_trace_joints", "upper_trace_header", "upper_trace_path"}

LEGS = ["left_hip_yaw_joint", "right_hip_yaw_joint", "left_hip_pitch_joint", "right_hip_pitch_joint",
        "left_hip_roll_joint", "right_hip_roll_joint", "left_knee_pitch_joint", "right_knee_pitch_joint",
        "left_ankle_lower_joint", "left_ankle_upper_joint", "right_ankle_lower_joint", "right_ankle_upper_joint"]
UPPER = ["left_shoulder_pitch_joint", "right_shoulder_pitch_joint", "neck_yaw_joint", "left_elbow_joint",
         "right_elbow_joint", "neck_pitch_joint"]
PASSIVE = ["left_ankle_lower_rod_joint", "left_foot_roll_joint"]


def load():
    tree = ast.parse(TRACE.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in NAMES]
    ns = {"os": os}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(TRACE), "exec"), ns)
    return ns


class UpperTraceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = load()

    def test_full_variant_keeps_articulation_order_and_skips_passive_joints(self):
        joints = LEGS[:2] + UPPER[:3] + PASSIVE + LEGS[2:] + UPPER[3:]
        actuated = LEGS + UPPER
        self.assertEqual(self.ns["upper_trace_joints"](joints, LEGS, actuated), UPPER)

    def test_legs_variant_has_none(self):
        self.assertEqual(self.ns["upper_trace_joints"](LEGS + PASSIVE, LEGS, LEGS), [])

    def test_header(self):
        header = self.ns["upper_trace_header"](["neck_yaw_joint", "left_elbow_joint"])
        self.assertEqual(header, ["t_s", "step", "neck_yaw.pos", "neck_yaw.vel", "neck_yaw.target", "neck_yaw.torque",
                                  "left_elbow.pos", "left_elbow.vel", "left_elbow.target", "left_elbow.torque"])
        self.assertEqual(len(self.ns["upper_trace_header"](UPPER)), 2 + 4 * len(UPPER))

    def test_path(self):
        self.assertEqual(self.ns["upper_trace_path"]("/x/trace_WA2.csv"), "/x/trace_WA2.upper.csv")
        self.assertEqual(self.ns["upper_trace_path"]("/x/trace"), "/x/trace.upper.csv")

    def test_main_wiring(self):
        text = TRACE.read_text()
        self.assertIn('parser.add_argument("--upper_joints", action="store_true"', text)
        main = ast.unparse(next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == "main"))
        self.assertIn("raise SystemExit('--upper_joints: the robot has no actuated joint", main)
        self.assertIn("writer.writerow(upper_trace_header(upper_names))", main)
        self.assertEqual(main.count("writer.writerow(header)"), 1)
        self.assertLess(main.index("gym.make("), main.index("if args_cli.upper_joints:"))
        self.assertLess(main.index("env.step(actions)"), main.index("upper_rows.append(upper_row)"))


if __name__ == "__main__":
    unittest.main()
