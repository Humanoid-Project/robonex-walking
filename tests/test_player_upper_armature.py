import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLAYER = ROOT / "scripts/sim_to_sim/play_v3_mujoco.py"
DESC = ROOT.parent / "robonex-description"
FULL_SCENE = DESC / "ver3/mujoco/robot/full/scene.xml"
ARMS = [f"{side}_{role}_joint" for side in ("left", "right")
        for role in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll")]
UPPER = ARMS + ["neck_yaw_joint", "neck_pitch_joint"]
GAINS = {
    "upper_shoulder_pitch": (40.0, 2.0, 17.0, 0.0042, (".*_shoulder_pitch_joint",)),
    "upper_shoulder_roll": (40.0, 2.0, 42.0, 0.02, (".*_shoulder_roll_joint",)),
    "upper_shoulder_yaw": (40.0, 2.0, 14.0, 0.0042, (".*_shoulder_yaw_joint",)),
    "upper_elbow": (40.0, 2.0, 14.0, 0.0042, (".*_elbow_joint",)),
    "upper_wrist_roll": (10.0, 0.5, 5.5, 0.001, (".*_wrist_roll_joint",)),
    "upper_neck_yaw": (20.0, 1.0, 5.5, 0.001, (".*neck_yaw_joint",)),
    "upper_neck_pitch": (20.0, 1.0, 5.5, 0.001, (".*neck_pitch_joint",)),
}


def load_player():
    spec = importlib.util.spec_from_file_location("play_v3_mujoco_upper_armature", PLAYER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class UpperArmatureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.player = load_player()
        except (ImportError, SystemExit) as error:
            raise unittest.SkipTest(f"player imports unavailable: {error}")

    def test_values_come_from_the_matching_group(self):
        values = self.player.upper_armature_values(UPPER, GAINS)
        self.assertEqual(set(values), set(UPPER))
        self.assertEqual(values["left_shoulder_roll_joint"], 0.02)
        self.assertEqual(values["right_elbow_joint"], 0.0042)
        self.assertEqual(values["neck_pitch_joint"], 0.001)
        self.assertEqual(values["left_wrist_roll_joint"], 0.001)

    def test_legs_run_or_legs_scene_is_refused(self):
        with self.assertRaisesRegex(RuntimeError, "legs run"):
            self.player.upper_armature_values(UPPER, {})
        with self.assertRaisesRegex(RuntimeError, "no upper-body actuators"):
            self.player.upper_armature_values([], GAINS)

    def test_unmatched_or_ambiguous_joint_is_refused(self):
        with self.assertRaisesRegex(RuntimeError, "head_joint matches upper actuator groups \\[\\]"):
            self.player.upper_armature_values(UPPER + ["head_joint"], GAINS)
        doubled = dict(GAINS, upper_any=(1.0, 1.0, 1.0, 0.5, (".*_elbow_joint",)))
        with self.assertRaisesRegex(RuntimeError, "left_elbow_joint matches"):
            self.player.upper_armature_values(UPPER, doubled)

    def test_default_is_model_and_isaac_is_a_choice(self):
        base = ["--checkpoint", str(PLAYER)]
        self.assertEqual(self.player.parse_args(base).upper_armature, "model")
        self.assertEqual(self.player.parse_args(base + ["--upper-armature", "isaac"]).upper_armature, "isaac")
        with self.assertRaises(SystemExit):
            self.player.parse_args(base + ["--upper-armature", "zero"])


@unittest.skipUnless(FULL_SCENE.is_file(), "ver3 full MJCF not present")
class UpperArmatureMujocoTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import mujoco
        except ImportError as error:
            raise unittest.SkipTest(f"mujoco unavailable: {error}")
        try:
            cls.player = load_player()
        except (ImportError, SystemExit) as error:
            raise unittest.SkipTest(f"player imports unavailable: {error}")
        cls.mujoco = mujoco

    def test_only_upper_dofs_change(self):
        import numpy as np
        mj = self.mujoco
        model = mj.MjModel.from_xml_path(str(FULL_SCENE))
        plant = object.__new__(self.player.MujocoPlant)
        plant.model = model
        trn = model.actuator_trnid[:, 0]
        names = [mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, int(trn[a])) for a in range(model.nu)]
        plant.upper_actuators = np.array([a for a, n in enumerate(names) if n in UPPER], dtype=int)
        plant.upper_joints = [names[a] for a in plant.upper_actuators]
        self.assertEqual(set(plant.upper_joints), set(UPPER))
        before = model.dof_armature.copy()
        mjcf = plant.upper_armature()
        self.assertEqual(mjcf["left_shoulder_pitch_joint"], 0.0042)
        self.assertEqual({mjcf[n] for n in UPPER if "shoulder_pitch" not in n}, {0.0})
        values = self.player.upper_armature_values(plant.upper_joints, GAINS)
        after = plant.set_upper_armature(values)
        self.assertEqual(after, values)
        dof = {names[a]: int(model.jnt_dofadr[trn[a]]) for a in range(model.nu)}
        changed = {int(i) for i in np.flatnonzero(model.dof_armature != before)}
        self.assertEqual(changed, {dof[n] for n in UPPER if values[n] != mjcf[n]})
        self.assertEqual(len(changed), 10)


if __name__ == "__main__":
    unittest.main()
