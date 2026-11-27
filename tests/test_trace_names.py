import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRACE = ROOT / "scripts/rsl_rl/sim_joint_trace.py"
NAMES = {"HIP_ROLL_LINKS", "trace_body_names"}

V2_BODIES = ["base_link", "l_hip_yaw_link", "l_hip_roll_link", "l_foot", "r_hip_yaw_link", "r_hip_roll_link", "r_foot"]
V3_BODIES = ["base_link", "left_hip_yaw_link", "left_hip_roll_link", "left_ankle_pitch_link", "right_hip_yaw_link",
             "right_hip_roll_link", "right_ankle_pitch_link", "left_shoulder_pitch_link", "head_link"]


def load():
    tree = ast.parse(TRACE.read_text())
    nodes = [n for n in tree.body if getattr(n, "name", None) in NAMES
             or (isinstance(n, ast.Assign) and getattr(n.targets[0], "id", None) in NAMES)]
    ns = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "sim_joint_trace", "exec"), ns)
    return ns["trace_body_names"]


def main_source():
    tree = ast.parse(TRACE.read_text())
    return ast.unparse(next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main"))


class TraceBodyNamesTest(unittest.TestCase):
    def setUp(self):
        self.names = load()

    def test_ver2_names(self):
        self.assertEqual(self.names(("l_foot", "r_foot"), V2_BODIES),
                         (["l_foot", "r_foot"], ["l_hip_roll_link", "r_hip_roll_link"]))

    def test_ver3_names(self):
        self.assertEqual(self.names(("left_ankle_pitch_link", "right_ankle_pitch_link"), V3_BODIES),
                         (["left_ankle_pitch_link", "right_ankle_pitch_link"], ["left_hip_roll_link", "right_hip_roll_link"]))

    def test_feet_order_follows_the_cfg(self):
        self.assertEqual(self.names(["right_ankle_pitch_link", "left_ankle_pitch_link"], V3_BODIES)[0],
                         ["right_ankle_pitch_link", "left_ankle_pitch_link"])

    def test_no_hip_roll_pair_refuses(self):
        bodies = [n for n in V3_BODIES if n != "right_hip_roll_link"]
        with self.assertRaises(SystemExit) as raised:
            self.names(("left_ankle_pitch_link", "right_ankle_pitch_link"), bodies)
        self.assertIn("left_hip_roll_link", str(raised.exception.code))
        self.assertIn("l_hip_roll_link", str(raised.exception.code))

    def test_mixed_hip_names_do_not_pair(self):
        bodies = ["l_hip_roll_link", "right_hip_roll_link", "l_foot", "r_foot"]
        with self.assertRaises(SystemExit):
            self.names(("l_foot", "r_foot"), bodies)

    def test_missing_or_extra_feet_refuse(self):
        with self.assertRaises(SystemExit):
            self.names(("l_foot", "r_foot"), V3_BODIES)
        with self.assertRaises(SystemExit):
            self.names(("left_ankle_pitch_link",), V3_BODIES)

    def test_main_reads_feet_from_the_env_cfg_and_keeps_the_csv_labels(self):
        source = main_source()
        self.assertIn("trace_body_names(getattr(unwrapped.cfg, 'feet_body_names', ('l_foot', 'r_foot')), body_names)",
                      source)
        self.assertIn("contact_ids = [contact_bodies.index(n) for n in feet]", source)
        self.assertNotIn("index(n) for n in ('l_foot', 'r_foot')", source)
        self.assertNotIn("stance_fraction", source)
        for label in ("'l_foot_x'", "'r_foot_fz'", "'l_hip_y'", "'r_hip_z'", "'l_foot_by'", "'l_sole_z'"):
            self.assertIn(label, source)


if __name__ == "__main__":
    unittest.main()
