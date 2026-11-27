import importlib.util
import json
import math
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source/robonex_walking"
HELPER = Path(__file__).resolve().parent / "_cfg_stub_build.py"
GOLDEN = Path(__file__).resolve().parent / "golden/v2_edu_cfg.json"
ISAACLAB = Path.home() / "IsaacLab/source/isaaclab"
ISAAC_DICT = ISAACLAB / "isaaclab/utils/dict.py"
DESC = ROOT.parent / "robonex-description"
CONSTANTS = DESC / "ver3/ver3_constants.json"
HAS_COMMON = importlib.util.find_spec("robonex_common") is not None

LEGS = tuple(f"{side}_{role}_joint" for side in ("left", "right")
             for role in ("hip_yaw", "hip_pitch", "hip_roll", "knee_pitch", "ankle_upper", "ankle_lower"))
ANKLES = ["left_ankle_upper_joint", "left_ankle_lower_joint", "right_ankle_upper_joint", "right_ankle_lower_joint"]
DEG = math.pi / 180.0
ZERO_DEG = {"hip_yaw": 1.0, "hip_pitch": 1.5, "hip_roll": 1.0, "knee_pitch": 2.0, "ankle_upper": 2.0, "ankle_lower": 2.0}
UPPER_GROUPS = {"upper_" + r for r in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll",
                                       "neck_yaw", "neck_pitch")}
NECK_RANGES = {"neck_yaw_joint": (-0.2368, 0.2368), "neck_pitch_joint": (-0.26, 0.52)}
INIT = "scene.robot.init_state.joint_pos"
V3_INERT = ("events.hand_payload.", "rewards.track_ang_vel_z_cycle.")
ARMS = [f"{side}_{role}_joint" for side in ("left", "right")
        for role in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll")]
USD = "scene.robot.spawn.usd_path"


def decode(obj):
    if isinstance(obj, dict):
        return {k: decode(v) for k, v in obj.items()}
    if isinstance(obj, list):
        if obj and obj[0] == "__tuple__":
            return tuple(decode(v) for v in obj[1:])
        return [decode(v) for v in obj]
    return obj


def flat(obj, prefix="", out=None):
    out = {} if out is None else out
    if isinstance(obj, dict):
        for k, v in obj.items():
            flat(v, f"{prefix}.{k}" if prefix else k, out)
    elif isinstance(obj, (list, tuple)) and obj and all(isinstance(x, (dict, list, tuple)) for x in obj):
        for i, v in enumerate(obj):
            flat(v, f"{prefix}[{i}]", out)
    else:
        out[prefix] = obj
    return out


def rename(obj, table):
    if isinstance(obj, dict):
        return {table.get(k, k): rename(v, table) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(rename(x, table) for x in obj)
    if isinstance(obj, str):
        return table.get(obj, obj)
    return obj


def diff(a, b):
    fa, fb = flat(a), flat(b)
    return {k: (fa.get(k, "<absent>"), fb.get(k, "<absent>")) for k in set(fa) | set(fb)
            if fa.get(k, "<absent>") != fb.get(k, "<absent>")}


def ver2_names(obj, path="", found=None):
    found = [] if found is None else found
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k.startswith(("l_", "r_")):
                found.append(f"{path}/{k}")
            ver2_names(v, f"{path}/{k}", found)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            ver2_names(v, f"{path}[{i}]", found)
    elif isinstance(obj, str) and obj.startswith(("l_", "r_")):
        found.append(f"{path}={obj}")
    return found


def expected_v2_to_legs():
    zero = {f"events.randomize_joint_zero.params.offset_range.{s}_{r}_joint": ((0.0, 0.0), (-d * DEG, d * DEG))
            for s in ("left", "right") for r, d in ZERO_DEG.items()}
    return {
        "actions.joint_pos.joint_names": (LEGS, list(LEGS)),
        "events.randomize_hip_yaw_stiffness.params.stiffness_distribution_params": ((0.25, 1.0), (1.0, 1.0)),
        **zero,
        "feet_body_names": ("<absent>", ("left_ankle_pitch_link", "right_ankle_pitch_link")),
        "model_variant": ("<absent>", "legs"),
        "observations.policy.joint_pos_rel.params.min_delay_steps": (0, 1),
        "observations.policy.joint_pos_rel.params.max_delay_steps": (1, 2),
        "observations.policy.joint_vel_rel.params.min_delay_steps": (0, 1),
        "observations.policy.joint_vel_rel.params.max_delay_steps": (1, 2),
        "rewards.feet_clearance.params.stance_fraction": (0.55, 0.625),
        "rewards.feet_contact_force.params.threshold": (360.1, 434.5),
        "rewards.feet_gait.params.threshold": (0.55, 0.625),
        "rewards.feet_width.params.lateral_deadband": (0.0, 0.44),
        "rewards.joint_load.params.asset_cfg.joint_names": (LEGS, ANKLES),
        "rewards.joint_load.weight": (0.0, -0.02),
        "scene.contact_forces.prim_path": ("{ENV_REGEX_NS}/Robot/.*_foot", "{ENV_REGEX_NS}/Robot/.*_ankle_pitch_link"),
        "scene.illegal_contacts.prim_path": (
            "{ENV_REGEX_NS}/Robot/(base_link|.*_knee_link|.*_ankle_link|.*_hip_.*_link)",
            "{ENV_REGEX_NS}/Robot/(base_link|.*_knee_link|.*_ankle_roll_link|.*_hip_.*_link"
            "|.*_shoulder_.*_link|.*_elbow_link|.*_wrist_roll_link|neck_.*_link|head_link)"),
        USD: ("robonex-description/ver2/isaac/edu/closed_loop_mesh/robonex_edu_closed_loop_mesh.usd",
              "robonex-description/ver3/isaac/legs/closed_loop_mesh/robonex_ver3_legs_closed_loop_mesh.usd"),
    }


@unittest.skipUnless(ISAAC_DICT.exists(), "Isaac Lab source not present")
@unittest.skipUnless(CONSTANTS.exists(), "robonex-description ver3 not present")
@unittest.skipUnless(HAS_COMMON, "robonex_common not importable")
class V3CfgBuildTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        proc = subprocess.run([sys.executable, str(HELPER), str(SOURCE), str(ISAACLAB)],
                              capture_output=True, text=True, timeout=300)
        if proc.returncode != 0:
            raise AssertionError(f"cfg build failed:\n{proc.stderr[-4000:]}")
        dumps = json.loads(proc.stdout)
        cls.raw_v2 = dumps["v2_before"]
        cls.v2 = decode(dumps["v2_before"])
        cls.v2_after = decode(dumps["v2_after"])
        cls.full = decode(dumps["v3"])
        cls.legs = decode(dumps["v3_legs"])
        cls.rename = dumps["rename"]
        cls.constants = json.loads(CONSTANTS.read_text())

    def test_building_v3_leaves_v2_unchanged(self):
        self.assertEqual(sorted(diff(self.v2, self.v2_after)), [])

    def test_v2_dump_equals_the_golden(self):
        golden = json.loads(GOLDEN.read_text())
        changed = sorted(diff(decode(golden), self.v2))
        self.assertEqual(changed, [], "V2 dump changed; regenerate tests/golden/v2_edu_cfg.json only for a V2 change")
        self.assertEqual(golden, self.raw_v2)

    def test_v3_dumps_carry_no_ver2_names(self):
        self.assertEqual(ver2_names(self.full), [])
        self.assertEqual(ver2_names(self.legs), [])

    def test_v2_mapped_and_v3_legs_differ_only_as_registered(self):
        got = {k: v for k, v in diff(rename(self.v2, self.rename), self.legs).items() if not k.startswith(INIT)}
        inert = {k: got.pop(k) for k in [k for k in got if k.startswith(V3_INERT)]}
        self.assertTrue(all(old == "<absent>" for old, _ in inert.values()))
        self.assertEqual(inert["events.hand_payload.mode"][1], "reset")
        self.assertTrue(inert["events.hand_payload.func"][1].endswith("mdp.events:randomize_payload_mass"))
        self.assertEqual(inert["events.hand_payload.params.operation"][1], "add")
        self.assertEqual(inert["events.hand_payload.params.mass_distribution_params"][1], (0.0, 0.0))
        self.assertEqual(inert["events.hand_payload.params.asset_cfg.body_names"][1],
                         ["left_wrist_roll_link", "right_wrist_roll_link"])
        self.assertTrue(inert["rewards.track_ang_vel_z_cycle.func"][1].endswith("mdp.rewards:track_ang_vel_z_cycle_exp"))
        self.assertEqual(inert["rewards.track_ang_vel_z_cycle.weight"][1], 0.0)
        self.assertEqual(inert["rewards.track_ang_vel_z_cycle.params.std"][1], 0.25)
        self.assertEqual(inert["rewards.track_ang_vel_z_cycle.params.window_steps"][1], 40)
        self.assertEqual(len(inert), 21)
        for term in ("hand_payload", "track_ang_vel_z_cycle"):
            self.assertNotIn(term, self.v2["events"])
            self.assertNotIn(term, self.v2["rewards"])
        expected = expected_v2_to_legs()
        self.assertEqual(sorted(got), sorted(expected))
        self.assertEqual(len(got), 29)
        for key, (old, new) in expected.items():
            if isinstance(new, tuple) and all(isinstance(x, float) for x in new):
                self.assertEqual(len(got[key][1]), len(new), key)
                for g, e in zip(got[key][1], new):
                    self.assertAlmostEqual(g, e, places=12, msg=key)
                self.assertEqual(got[key][0], old, key)
            else:
                self.assertEqual(got[key], (old, new), key)

    def test_full_and_legs_differ_only_in_the_upper_body_and_neck_term(self):
        got = diff(self.legs, self.full)
        self.assertEqual(got.pop("model_variant"), ("legs", "full"))
        self.assertEqual(got.pop(USD)[1],
                         "robonex-description/ver3/isaac/full/closed_loop_mesh/robonex_ver3_full_closed_loop_mesh.usd")
        groups, init, neck, arm, other = set(), set(), {}, {}, []
        for key, (legs, full) in got.items():
            if key.startswith("scene.robot.actuators.upper_") and legs == "<absent>":
                groups.add(key.split(".")[3])
            elif key.startswith(INIT + ".") and legs == "<absent>":
                init.add(key[len(INIT) + 1:])
            elif key.startswith("events.neck_target.") and legs == "<absent>":
                neck[key[len("events.neck_target."):]] = full
            elif key.startswith("events.arm_target.") and legs == "<absent>":
                arm[key[len("events.arm_target."):]] = full
            else:
                other.append(key)
        self.assertEqual(other, [])
        self.assertEqual(groups, UPPER_GROUPS)
        self.assertEqual(init, set(self.constants["upper_body_joints"]))
        self.assertTrue(all(self.full["scene"]["robot"]["init_state"]["joint_pos"][n] == 0.0 for n in init))
        self.assertEqual(neck["mode"], "interval")
        self.assertEqual(neck["params.range_scale"], 0.0)
        self.assertEqual(neck["params.asset_cfg.joint_names"], ["neck_yaw_joint", "neck_pitch_joint"])
        for name, rng in NECK_RANGES.items():
            self.assertEqual(neck[f"params.ranges.{name}"], rng)
        self.assertNotIn("neck_target", self.legs["events"])
        self.assertNotIn("neck_target", self.v2["events"])
        self.assertEqual(arm["mode"], "interval")
        self.assertEqual(arm["params.range_scale"], 0.0)
        self.assertTrue(arm["func"].endswith("mdp.events:randomize_neck_target"))
        self.assertEqual(arm["params.asset_cfg.joint_names"], ARMS)
        limits = self.constants["provisional_limits"]
        for name in ARMS:
            self.assertEqual(arm[f"params.limits.{name}"], tuple(limits[name]))
        self.assertNotIn("params.ranges", arm)
        self.assertNotIn("arm_target", self.legs["events"])
        self.assertNotIn("arm_target", self.v2["events"])


if __name__ == "__main__":
    unittest.main()
